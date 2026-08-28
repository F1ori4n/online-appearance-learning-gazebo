from __future__ import annotations

import csv
from pathlib import Path


class EventLogger:
    """
    Logs the output of the Perception node for each processed frame.
    It logs the decision output of the Target manager.
    This includes, the current state, confidence, ReID scores, and event reason.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        # Check if parent directories exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in write mode and initialize CSV header
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "time", "mode", "state", "visible", "track_id", "confidence", "reid_score",
                "candidate_track_id", "num_tracks", "event", "reasons"
            ],
        )
        self.writer.writeheader()

    def log(self, **row) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()
