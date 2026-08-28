from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector, for stable cosine similarity."""
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 1e-12:
        return emb
    return emb / norm


def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@dataclass
class MemoryEntry:
    embedding: np.ndarray
    timestamp: float
    track_id: int
    quality: float


class AppearanceMemory:
    """
    Appearance memory with immutable anchors and an adaptive gallery.

    The initial learning phase creates anchor entries. These entries are never
    replaced, which protects the identity against gradual drift.
    Later, only reliable and non-redundant feature vectors are added to
    as adaptive entries.
    """

    def __init__(self,
                 max_size: int = 30,
                 update_threshold: float = 0.84, # Memory update threshold
                 update_max_similarity: float = 0.97,
                 min_update_interval: float = 0.50,
                 min_quality: float = 0.75, ):

        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if update_max_similarity <= update_threshold:
            raise ValueError("update_max_similarity must be greater than update_threshold")

        self.max_size = int(max_size)
        self.update_threshold = float(update_threshold)
        self.update_max_similarity = float(update_max_similarity)
        self.min_update_interval = float(min_update_interval)
        self.min_quality = float(min_quality)

        self.anchor_entries: List[MemoryEntry] = []
        self.adaptive_entries: List[MemoryEntry] = []
        self.last_update_time = -1e9
        self.last_update_reason = "not_attempted"

    @property
    def entries(self) -> List[MemoryEntry]:
        return [*self.anchor_entries, *self.adaptive_entries]

    @property
    def initialized(self) -> bool:
        return bool(self.anchor_entries) or bool(self.adaptive_entries)

    def initialize(self, embeddings: list[np.ndarray], timestamp: float, track_id: int) -> None:
        """
        Create the immutable anchor entries from the first observations
        """
        if not embeddings:
            self.last_update_reason = "initialize_without_embeddings"
            return

        normalized = [normalize_embedding(e) for e in embeddings]
        # Keep the initial observations as immutable anchors. If there are more
        # observations than the configured capacity, keep the most recent ones.
        normalized = normalized[-self.max_size:]
        self.anchor_entries = [
            MemoryEntry(embedding=emb, timestamp=timestamp, track_id=track_id, quality=1.0)
            for emb in normalized]

        self.adaptive_entries = []
        self.last_update_time = float(timestamp)
        self.last_update_reason = "initialized"

    def similarity(self, embedding: np.ndarray) -> float:
        """Return the highest similarity to any stored entry"""
        if not self.initialized:
            return 0.0

        emb = normalize_embedding(embedding)
        entries = self.entries

        # best match against all anchors
        return float(max(cosine_sim(emb, entry.embedding) for entry in entries))

    def maybe_update(self,
                     embedding: np.ndarray,
                     timestamp: float,
                     track_id: int,
                     quality: float = 1.0, ) -> bool:
        """
        Add a new feature vector to the adaptive gallery.
        """
        emb = normalize_embedding(embedding)
        timestamp = float(timestamp)
        quality = float(quality)

        if not self.initialized:
            self.initialize([emb], timestamp, track_id)
            self.last_update_reason = "accepted_initialization"
            return True

        # check Cooldown
        if timestamp - self.last_update_time < self.min_update_interval:
            self.last_update_reason = "rejected_cooldown"
            return False

        # Avoid low quality crops
        if quality < self.min_quality:
            self.last_update_reason = "rejected_low_quality"
            return False

        score = self.similarity(emb)

        # Low similarity Likely different person
        if score < self.update_threshold:
            self.last_update_reason = "rejected_low_similarity"
            return False

        # Too simular, view redundant
        if score > self.update_max_similarity:
            self.last_update_reason = "rejected_redundant"
            return False

        # Add new adaptive entry to gallery
        adaptive_capacity = max(0, self.max_size - len(self.anchor_entries))
        if adaptive_capacity <= 0:
            self.last_update_reason = "rejected_no_adaptive_capacity"
            return False

        self.adaptive_entries.append(
            MemoryEntry(
                embedding=emb,
                timestamp=timestamp,
                track_id=track_id,
                quality=quality,
            )
        )

        # Keep the highest quality and most recent adaptive entries.
        self.adaptive_entries = sorted(
            self.adaptive_entries,
            key=lambda entry: (entry.quality, entry.timestamp),
        )[-adaptive_capacity:]

        self.last_update_time = timestamp
        self.last_update_reason = "accepted_adaptive_view"
        return True

    def size(self) -> int:
        return len(self.anchor_entries) + len(self.adaptive_entries)

    def anchor_size(self) -> int:
        return len(self.anchor_entries)

    def adaptive_size(self) -> int:
        return len(self.adaptive_entries)
