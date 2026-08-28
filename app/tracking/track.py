from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Track:
    """
    Data object for a single tracked person in a frame.

    It holds the bounding box coordinates, the track ID
    assigned by ByteTrack and a image crop of the person.
    """

    track_id: int
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    conf: float                 # Detection confidence (0.0 - 1.0)
    crop: Optional[np.ndarray]      # crop of the person
    class_id: int = 0

    @property
    def valid_id(self) -> bool:
        return self.track_id is not None and self.track_id >= 0

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> int:
        return self.width * self.height
