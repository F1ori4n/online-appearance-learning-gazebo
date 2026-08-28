from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from app.tracking.track import Track


class ByteTrackTracker:
    """
    YOLO + ByteTrack wrapper to detect and track person bounding boxes.
    It processes frames one by one and returns a list of tracks.
    """

    def __init__(self,
                 model_path: str = "yolov8s.pt",
                 tracker_config: str = "bytetrack.yaml",
                 device=None,
                 person_class_id: int = 0,
                 yolo_conf: float = 0.25, ):

        self.model = YOLO(model_path)

        if device is not None:
            self.model.to(device)

        self.device = device
        self.tracker_config = tracker_config
        self.person_class_id = int(person_class_id)
        self.yolo_conf = float(yolo_conf)

    def process(self, frame: np.ndarray) -> List[Track]:
        """
        Processes a single frame and returns a list of tracks.
        """
        # persist=True makes ByteTrack keep IDs over multiple frames
        result = self.model.track(frame,
                                  persist=True,
                                  tracker=self.tracker_config,
                                  device=self.device,
                                  verbose=False,
                                  conf=self.yolo_conf, )[0]

        h, w = frame.shape[:2]
        tracks: list[Track] = []
        if result.boxes is None:
            return tracks

        for box in result.boxes:
            # Only track People
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id != self.person_class_id:
                continue

            # Convert YOLO coordinates to integer and limit them to image boundaries
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Save crop for the case that ReID is needed
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = None

            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            tracks.append(Track(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                conf=conf,
                crop=crop,
                class_id=cls_id))
        return tracks
