from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.reid.reidentifier import ReIdentifier
from app.tracking.track import Track


@dataclass
class ReassignmentResult:
    track: Optional[Track]
    score: float
    old_track_id: Optional[int]
    new_track_id: Optional[int]
    success: bool


class TargetReassignment:
    """
    Reassigns Target from an old ByteTrack ID to a new one using ReID.
    """

    def __init__(self, reidentifier: ReIdentifier, threshold: float = 0.80, margin: float = 0.08):
        self.reidentifier = reidentifier
        self.threshold = float(threshold)
        self.margin = float(margin)

    def try_reassign(self, tracks: list[Track], old_track_id: Optional[int]) -> ReassignmentResult:

        # Rank candidates by ReID score and filter out old IDs
        ranked_candidates = [
            c for c in self.reidentifier.rank_candidates(tracks)
            if c.track.track_id != old_track_id]

        if not ranked_candidates:
            return ReassignmentResult(None, 0.0, old_track_id, None, False)

        best = ranked_candidates[0]
        # If there is no second candidate, margin check automatically satisfied
        second_score = ranked_candidates[1].score if len(ranked_candidates) > 1 else 0.0

        success = best.score >= self.threshold and (best.score - second_score) >= self.margin

        # new_track_id is set even on failure so the visualizer can show the best candidate.
        return ReassignmentResult(
            track=best.track if success else None,
            score=float(best.score),
            old_track_id=old_track_id,
            new_track_id=best.track.track_id,
            success=success, )
