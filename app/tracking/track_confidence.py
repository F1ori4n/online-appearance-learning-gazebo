from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

from app.tracking.track import Track

@dataclass
class TrackConfidenceResult:
    low: bool  # True if ReID verification should be triggered
    score: float  # Overall reliability (0.0 to 1.0)
    reasons: list[str]  # Reason why score dropped


class TrackConfidenceEstimator:
    """
    Evaluates how reliable a current track is.

    Factors affecting confidence levels:
    - YOLO confidence
    - CROP Size
    - Movement Jumps
    """

    def __init__(self,
                 min_yolo_conf: float = 0.55,
                 min_crop_height: int = 55,
                 max_center_jump_norm: float = 0.35,
                 crowd_distance_norm: float = 0.22, ):

        self.min_yolo_conf = float(min_yolo_conf)
        self.min_crop_height = int(min_crop_height)
        self.max_center_jump_norm = float(max_center_jump_norm)
        self.crowd_distance_norm = float(crowd_distance_norm)

    def evaluate(self,
                 track: Optional[Track],
                 previous_track: Optional[Track],
                 all_tracks: Iterable[Track],
                 frame_width: int, ) -> TrackConfidenceResult:

        reasons: list[str] = []

        # Check if track exists, if not mark as missing
        if track is None:
            return TrackConfidenceResult(True, 0.0, ["missing_track"])

        score = float(track.conf)

        if not track.valid_id:
            reasons.append("invalid_track_id")
            score *= 0.4

        if track.conf < self.min_yolo_conf:
            reasons.append("low_yolo_conf")
            score *= 0.6

        if track.height < self.min_crop_height:
            reasons.append("small_crop")
            score *= 0.7

        if previous_track is not None and frame_width > 0:
            cx, cy = track.center
            pcx, pcy = previous_track.center
            jump = hypot(cx - pcx, cy - pcy) / frame_width
            if jump > self.max_center_jump_norm:
                reasons.append("large_bbox_jump")
                score *= 0.5

        # Another person close to the current target center can produce ID switches.
        if frame_width > 0:
            cx, cy = track.center
            for other in all_tracks:
                if other.track_id == track.track_id:
                    continue
                ox, oy = other.center
                dist = hypot(cx - ox, cy - oy) / frame_width
                if dist < self.crowd_distance_norm:
                    reasons.append("nearby_person")
                    score *= 0.75
                    break

        # keep score between [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        return TrackConfidenceResult(low=bool(reasons), score=score, reasons=reasons)
