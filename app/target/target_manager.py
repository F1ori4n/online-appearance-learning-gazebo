from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.reid.appearance_memory import AppearanceMemory
from app.reid.reidentifier import ReIdentifier
from app.target.target_reassignment import TargetReassignment
from app.target.target_state import ExperimentMode, TargetState
from app.tracking.track import Track
from app.tracking.track_confidence import TrackConfidenceEstimator


@dataclass
class TargetOutput:
    state: TargetState
    visible: bool
    target_track_id: Optional[int]
    target_track: Optional[Track]
    confidence: float
    reid_score: float
    candidate_track_id: Optional[int]
    event: Optional[str] = None
    reasons: list[str] = field(default_factory=list)


def join_events(*parts: Optional[str]) -> Optional[str]:
    return ";".join(part for part in parts if part) or None


class TargetManager:
    """
    Target manager with two online ReID initializations.

    online_reid creates one immutable anchor from the first valid target crop.
    calibration_reid collects several immutable anchors during a configured in
    simulation time window. After that window both modes use the exact same
    adaptive gallery update policy.
    """

    def __init__(self,
                 mode: str,
                 reidentifier: Optional[ReIdentifier],
                 memory: Optional[AppearanceMemory],
                 config: Optional[dict] = None, ):

        cfg = config or {}
        self.mode = ExperimentMode(mode)
        self.reidentifier = reidentifier
        self.memory = memory

        self.confidence_estimator = TrackConfidenceEstimator(
            min_yolo_conf=cfg.get("min_yolo_conf", 0.55),
            min_crop_height=cfg.get("min_crop_height", 45),
            max_center_jump_norm=cfg.get("max_center_jump_norm", 0.35),
            crowd_distance_norm=cfg.get("crowd_distance_norm", 0.22),
        )
        self.reassignment = (
            TargetReassignment(
                reidentifier,
                threshold=cfg.get("reassign_threshold", 0.80),
                margin=cfg.get("reassign_margin", 0.08),
            )
            if reidentifier is not None
            else None
        )

        self.max_missing_time = float(cfg.get("max_missing_time", 0.25))
        self.reid_stable_interval_sec = float(cfg.get("reid_stable_interval_sec", 1.0))
        self.reid_low_conf_interval_sec = float(cfg.get("reid_low_conf_interval_sec", 0.20))
        self.identity_check_threshold = float(cfg.get("identity_check_threshold", 0.72))

        self.memory_update_min_similarity = float(cfg.get("memory_update_min_similarity", 0.82))
        self.memory_update_max_similarity = float(cfg.get("memory_update_max_similarity", 0.97))
        self.memory_update_min_track_age_sec = float(cfg.get("memory_update_min_track_age_sec", 1.0))
        self.memory_update_min_yolo_conf = float(cfg.get("memory_update_min_yolo_conf", 0.70))
        self.memory_update_min_crop_height = int(cfg.get("memory_update_min_crop_height", 80))

        # Calibration window in simulation time (t=18 to t=20)
        self.calibration_start_time_sec = float(cfg.get("calibration_start_time_sec", 18.0))
        self.online_updates_start_time_sec = float(cfg.get("online_updates_start_time_sec", 20.0))
        self.calibration_embeddings = max(1, int(cfg.get("calibration_embeddings", 15)))

        if self.online_updates_start_time_sec <= self.calibration_start_time_sec:
            raise ValueError(
                "online_updates_start_time_sec must be greater than "
                "calibration_start_time_sec"
            )
        self.calibration_sample_interval_sec = (
                (self.online_updates_start_time_sec - self.calibration_start_time_sec)
                / max(1, self.calibration_embeddings - 1)
        )

        self.reassign_consecutive_frames = max(1, int(cfg.get("reassign_consecutive_frames", 3)))

        self.target_track_id: Optional[int] = None
        self.state = TargetState.UNINITIALIZED
        self.last_seen_time: Optional[float] = None
        self.previous_target_track: Optional[Track] = None
        self.target_stable_since: Optional[float] = None
        self.initialized = bool(self.memory is not None and self.memory.initialized)

        self.last_reid_check_time = -1e9
        self.last_target_reid_score = 0.0
        self.identity_suspect = False

        self.calibration_buffer: list = []
        self.next_calibration_sample_index = 0
        self.calibration_wait_event_emitted = False
        self.calibration_finished_event_emitted = False

        self._candidate_id: Optional[int] = None
        self._candidate_hits = 0

    @staticmethod
    def _find_track(tracks: list[Track], track_id: Optional[int]) -> Optional[Track]:
        if track_id is None:
            return None
        return next((t for t in tracks if t.track_id == track_id), None)

    @staticmethod
    def _choose_initial_track(tracks: list[Track]) -> Optional[Track]:
        valid = [t for t in tracks if t.valid_id and t.crop is not None]
        return max(valid, key=lambda t: t.conf) if valid else None

    def _extract(self, track: Track):
        return None if self.reidentifier is None else self.reidentifier.extract(track)

    def _reset_candidate(self) -> None:
        """Rest the candidate counting for the 3 frame confirmation rule"""
        self._candidate_id = None
        self._candidate_hits = 0

    def _initialize_online_memory(self, target: Track, now: float) -> tuple[float, Optional[str]]:
        """Online ReID from exactly one target observation."""
        if self.mode != ExperimentMode.ONLINE_REID:
            return self.last_target_reid_score, None
        if self.memory is None:
            return 0.0, "ONLINE_MEMORY_MISSING"
        if self.memory.initialized:
            self.initialized = True
            return self.last_target_reid_score, None

        if now < self.calibration_start_time_sec:
            return 0.0, "ONLINE_MEMORY_INIT_DELAYED_BY_START_TIME"

        emb = self._extract(target)
        if emb is None:
            self.initialized = False
            return 0.0, "ONLINE_MEMORY_INIT_SKIPPED_LOW_QUALITY"

        self.memory.initialize([emb], now, target.track_id)
        self.initialized = self.memory.initialized
        self.last_target_reid_score = 1.0 if self.initialized else 0.0
        self.last_reid_check_time = now
        self.target_stable_since = now
        return self.last_target_reid_score, "ONLINE_MEMORY_INITIALIZED_FROM_FIRST_CROP"

    def _collect_calibration_memory(self, target: Track, now: float) -> tuple[float, Optional[str]]:
        """Collect anchors only during the configured 2s simulation window."""
        if self.mode != ExperimentMode.CALIBRATION_REID:
            return self.last_target_reid_score, None
        if self.memory is None:
            return 0.0, "CALIBRATION_MEMORY_MISSING"
        if self.memory.initialized:
            self.initialized = True
            return self.last_target_reid_score, None

        if now < self.calibration_start_time_sec:
            if not self.calibration_wait_event_emitted:
                self.calibration_wait_event_emitted = True
                return 0.0, (
                    "CALIBRATION_WAITING:"
                    f"start={self.calibration_start_time_sec:.2f}"
                )
            return 0.0, None

        sample_event = None
        if self.next_calibration_sample_index < self.calibration_embeddings:
            due_time = (
                    self.calibration_start_time_sec
                    + self.next_calibration_sample_index
                    * self.calibration_sample_interval_sec
            )
            if now >= due_time:
                emb = self._extract(target)
                if emb is None:
                    sample_event = "CALIBRATION_SAMPLE_SKIPPED_LOW_QUALITY"
                else:
                    self.calibration_buffer.append(emb)
                    self.next_calibration_sample_index += 1
                    sample_event = (
                        "CALIBRATION_PROGRESS:"
                        f"{len(self.calibration_buffer)}/{self.calibration_embeddings}"
                    )

        window_finished = now >= self.online_updates_start_time_sec
        requested_count_reached = (len(self.calibration_buffer) >= self.calibration_embeddings)

        if not window_finished and not requested_count_reached:
            return 0.0, sample_event

        # At t=20 s initialize with all successfully collected views.
        if not self.calibration_buffer:
            emb = self._extract(target)
            if emb is not None:
                self.calibration_buffer.append(emb)

        if not self.calibration_buffer:
            self.initialized = False
            return 0.0, join_events(
                sample_event, "CALIBRATION_FINISH_FAILED_NO_VALID_CROPS"
            )

        self.memory.initialize(
            self.calibration_buffer,
            now,
            target.track_id,
        )
        self.initialized = self.memory.initialized
        self.last_target_reid_score = 1.0 if self.initialized else 0.0
        self.last_reid_check_time = now
        self.target_stable_since = now
        self.identity_suspect = False
        self.calibration_finished_event_emitted = True
        finish_kind = (
            "CALIBRATION_FINISHED"
            if len(self.calibration_buffer) >= self.calibration_embeddings
            else "CALIBRATION_FINISHED_PARTIAL"
        )
        return self.last_target_reid_score, join_events(
            sample_event,
            f"{finish_kind}:anchors={len(self.calibration_buffer)}",
        )

    def _try_memory_update(self,
                           target: Track,
                           emb,
                           score: float,
                           now: float,
                           low_confidence: bool, ) -> str:

        # Normal path starts at simulation time t=20 s.
        if now < self.online_updates_start_time_sec:
            return "MEMORY_UPDATE_BLOCKED_INITIALIZATION_WINDOW"

        stable_age = now - (self.target_stable_since or now)
        if low_confidence or self.identity_suspect:
            return "MEMORY_UPDATE_REJECTED_UNSTABLE_TRACK"
        if stable_age < self.memory_update_min_track_age_sec:
            return "MEMORY_UPDATE_REJECTED_TRACK_TOO_YOUNG"
        if target.conf < self.memory_update_min_yolo_conf:
            return "MEMORY_UPDATE_REJECTED_LOW_YOLO_CONF"
        if target.height < self.memory_update_min_crop_height:
            return "MEMORY_UPDATE_REJECTED_SMALL_CROP"
        if score < self.memory_update_min_similarity:
            return "MEMORY_UPDATE_REJECTED_LOW_SIMILARITY"
        if score > self.memory_update_max_similarity:
            return "MEMORY_UPDATE_REJECTED_REDUNDANT"

        if self.memory.maybe_update(emb, now, target.track_id, quality=target.conf):
            return "MEMORY_UPDATE_ACCEPTED"
        reason = str(getattr(self.memory, "last_update_reason", "rejected")).upper()
        return f"MEMORY_UPDATE_{reason}"

    def _periodic_check(self,
                        target: Track,
                        now: float,
                        low_confidence: bool, ) -> tuple[float, Optional[str]]:

        """
        Periodically verify identity and update memory
        """

        if self.memory is None or not self.memory.initialized:
            return self.last_target_reid_score, None

        frequent = low_confidence or self.identity_suspect
        interval = (
            self.reid_low_conf_interval_sec
            if frequent
            else self.reid_stable_interval_sec
        )
        if now - self.last_reid_check_time < interval:
            return self.last_target_reid_score, None

        self.last_reid_check_time = now
        kind = "REID_LOW_CONFIDENCE_CHECK" if frequent else "REID_PERIODIC_CHECK"
        emb = self._extract(target)
        if emb is None:
            return self.last_target_reid_score, join_events(
                kind, "REID_CHECK_SKIPPED_LOW_QUALITY"
            )

        score = float(self.memory.similarity(emb))
        was_suspect = self.identity_suspect
        self.identity_suspect = score < self.identity_check_threshold
        self.last_target_reid_score = score

        if self.identity_suspect:
            identity_event = f"REID_IDENTITY_LOW:score={score:.3f}"
        elif was_suspect:
            identity_event = f"REID_IDENTITY_RECOVERED:score={score:.3f}"
        else:
            identity_event = f"REID_IDENTITY_OK:score={score:.3f}"

        update_event = self._try_memory_update(
            target, emb, score, now, low_confidence
        )
        return score, join_events(kind, identity_event, update_event)

    def update(self, tracks: list[Track], current_time: float, frame_width: int) -> TargetOutput:
        # Initial target selection.
        if self.target_track_id is None:
            target = self._choose_initial_track(tracks)
            if target is None:
                return TargetOutput(
                    self.state, False, None, None, 0.0, 0.0, None
                )

            self.target_track_id = target.track_id
            self.last_seen_time = current_time
            self.previous_target_track = target
            self.target_stable_since = current_time

            # Tracker Only mode, no memory, no ReID
            if self.mode == ExperimentMode.TRACKER_ONLY:
                self.initialized = True
                self.state = TargetState.TRACKING
                return TargetOutput(
                    self.state, True, target.track_id, target,
                    target.conf, 0.0, None, "TARGET_ACQUIRED_TRACKER_ONLY"
                )
            # ReID modes must have memory and reidentifier
            if self.memory is None or self.reidentifier is None:
                raise RuntimeError("ReID modes require memory and reidentifier")

            if self.mode == ExperimentMode.ONLINE_REID:
                score, init_event = self._initialize_online_memory(
                    target, current_time
                )
                self.state = (
                    TargetState.TRACKING
                    if self.initialized
                    else TargetState.TRACKING_LOW_CONFIDENCE
                )
                return TargetOutput(
                    self.state, True, target.track_id, target,
                    target.conf, score, None,
                    join_events("TARGET_ACQUIRED_ONLINE_REID", init_event),
                )

            score, calibration_event = self._collect_calibration_memory(
                target, current_time
            )
            self.state = (
                TargetState.TRACKING
                if self.initialized
                else TargetState.CALIBRATING
            )
            return TargetOutput(
                self.state, True, target.track_id, target,
                target.conf, score, None,
                join_events(
                    "TARGET_ACQUIRED_CALIBRATION_REID",
                    calibration_event,
                ),
            )

        # Track has still the same ID.
        target = self._find_track(tracks, self.target_track_id)
        if target is not None:
            self._reset_candidate()
            self.last_seen_time = current_time
            confidence = self.confidence_estimator.evaluate(
                target, self.previous_target_track, tracks, frame_width
            )

            score = self.last_target_reid_score
            event = None
            if self.mode != ExperimentMode.TRACKER_ONLY:
                if self.memory is None or self.reidentifier is None:
                    raise RuntimeError("ReID modes require memory and reidentifier")

                # Initialize memory if not done
                if not self.memory.initialized:
                    if self.mode == ExperimentMode.ONLINE_REID:
                        score, event = self._initialize_online_memory(
                            target, current_time
                        )
                    else:
                        score, event = self._collect_calibration_memory(
                            target, current_time
                        )
                else:
                    # Periodic identity check + memory update
                    score, event = self._periodic_check(
                        target, current_time, confidence.low
                    )
                self.initialized = self.memory.initialized

            # Update state based on confidence
            if self.mode == ExperimentMode.CALIBRATION_REID and not self.initialized:
                self.state = TargetState.CALIBRATING
            elif (confidence.low or self.identity_suspect
                  or (self.mode == ExperimentMode.ONLINE_REID and not self.initialized)):
                self.state = TargetState.TRACKING_LOW_CONFIDENCE
                event = join_events(event, "TRACK_CONFIDENCE_LOW")
            else:
                self.state = TargetState.TRACKING

            self.previous_target_track = target
            return TargetOutput(
                self.state, True, self.target_track_id,
                target, confidence.score, score, None,
                event, confidence.reasons,
            )

        # Short ByteTrack grace period (target briefly missing).
        missing_time = (
            current_time - self.last_seen_time
            if self.last_seen_time is not None else 999.0
        )
        if missing_time < self.max_missing_time:
            self.state = TargetState.REID_VERIFYING
            return TargetOutput(
                self.state, False, self.target_track_id, None, 0.0,
                self.last_target_reid_score, None,
                "TARGET_TEMPORARILY_MISSING",
            )

        # Target lost: check every valid candidate at the configured interval.
        self.state = TargetState.REID_VERIFYING
        if (
                self.mode != ExperimentMode.TRACKER_ONLY
                and self.reassignment is not None
                and self.memory is not None
                and self.memory.initialized
        ):
            # Directly Trigger ReID every frame
            candidates = [
                t for t in tracks
                if t.valid_id and t.crop is not None
                   and t.track_id != self.target_track_id
            ]
            result = self.reassignment.try_reassign(
                candidates, old_track_id=self.target_track_id
            )
            candidate_id = result.new_track_id
            score = float(result.score)

            # Candidate found, check 3 frame consistency
            if result.success and result.track is not None:
                if self._candidate_id == result.track.track_id:
                    self._candidate_hits += 1
                else:
                    self._candidate_id = result.track.track_id
                    self._candidate_hits = 1

                # 3 frame rule passed, reassign ID
                if self._candidate_hits >= self.reassign_consecutive_frames:
                    old_id = self.target_track_id
                    hits = self._candidate_hits
                    self.target_track_id = result.track.track_id
                    self.last_seen_time = current_time
                    self.previous_target_track = result.track
                    self.target_stable_since = current_time
                    self.last_reid_check_time = current_time
                    self.last_target_reid_score = score
                    self.identity_suspect = False
                    self._reset_candidate()
                    self.state = TargetState.TARGET_REFOUND
                    return TargetOutput(
                        self.state, True, self.target_track_id, result.track, result.track.conf, score, candidate_id,
                        f"TRACK_REASSIGNED_CONFIRMED:{old_id}->{self.target_track_id}:score={score:.3f}:hits={hits}",
                    )

                # Not enough frames keep waiting
                return TargetOutput(
                    self.state, False, self.target_track_id, None, 0.0, score, candidate_id,
                    f"REID_LOST_CANDIDATE_CHECK:tid={candidate_id}:score={score:.3f}:hits={self._candidate_hits}/{self.reassign_consecutive_frames}",
                )

            # No candidate found
            self._reset_candidate()

            return TargetOutput(
                self.state, False, self.target_track_id, None,
                0.0, score, candidate_id, "REID_LOST_CHECK_REJECTED",
            )

        # Tracker Only mode, No ReID, just declare lost
        self.state = TargetState.TARGET_LOST
        return TargetOutput(
            self.state, False, self.target_track_id, None, 0.0, 0.0, None, "TARGET_LOST",
        )
