from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from app.reid.appearance_memory import AppearanceMemory
from app.tracking.track import Track
from app.reid.extractor import FeatureExtractor


@dataclass
class ReIDCandidate:
    track: Track
    score: float
    embedding: np.ndarray


class ReIdentifier:
    """Feature extraction + ReID matching block."""

    def __init__(self,
                 extractor: FeatureExtractor,
                 memory: AppearanceMemory,
                 min_crop_height: int = 55,
                 min_yolo_conf: float = 0.55, ):

        self.extractor = extractor
        self.memory = memory
        self.min_crop_height = int(min_crop_height)
        self.min_yolo_conf = float(min_yolo_conf)

    def crop_quality_ok(self, track: Track) -> bool:
        """check if crop quality is good enough for ReID"""
        return (track.crop is not None
                and track.height >= self.min_crop_height
                and track.conf >= self.min_yolo_conf)

    def extract(self, track: Track) -> Optional[np.ndarray]:
        """extract features vector from track"""
        if not self.crop_quality_ok(track):
            return None
        return self.extractor.extract(track.crop)

    def score_track(self, track: Track) -> Optional[ReIDCandidate]:
        """Compare track against appearance memory"""
        emb = self.extract(track)
        if emb is None:
            return None
        return ReIDCandidate(track=track, score=self.memory.similarity(emb), embedding=emb)

    def rank_candidates(self, tracks: Iterable[Track]) -> list[ReIDCandidate]:
        """Sort all visible tracks by their ReID score"""
        candidates = []
        for track in tracks:
            cand = self.score_track(track)
            if cand is not None:
                candidates.append(cand)
        # Sort from highest score to lowest
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
