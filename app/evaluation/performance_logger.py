from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np


class PerformanceLogger:
    """
    Collect core perception timing and ReID activity.
    """

    FIELDNAMES = [
        "wall_elapsed_sec",
        "frame_count",
        "mode",
        "rolling_fps",
        "mean_fps",
        "callback_total_ms",
        "tracking_ms",
        "target_update_ms",
        "reid_calls_total",
        "reid_calls_since_last_sample",
    ]

    def __init__(self,
                 *,
                 mode: str,
                 log_path: str = "",
                 summary_path: str = "",
                 extractor=None, ) -> None:

        self.mode = str(mode)
        self.extractor = extractor

        self.log_path = Path(log_path).expanduser() if log_path else None
        if summary_path:
            self.summary_path = Path(summary_path).expanduser()
        elif self.log_path is not None:
            self.summary_path = self.log_path.parent / "performance_summary.json"
        else:
            self.summary_path = None

        self.start_time = time.perf_counter()
        self.frame_timestamps: deque[float] = deque(maxlen=120)

        self.callback_times_ms: deque[float] = deque(maxlen=10000)
        self.tracking_times_ms: deque[float] = deque(maxlen=10000)
        self.target_update_times_ms: deque[float] = deque(maxlen=10000)

        self.last_reid_call_count = 0
        self.finalized = False
        self.failed = False

        self.file = None
        self.writer = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.log_path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()
            self.file.flush()

    def record_frame(self,
                     *,
                     frame_count: int,
                     frame_timestamp: float,
                     callback_total_ms: float,
                     tracking_ms: float,
                     target_update_ms: float, ) -> None:
        """
        Record one complete perception callback
        """

        if self.failed or self.finalized:
            return

        try:
            self.frame_timestamps.append(float(frame_timestamp))
            self.callback_times_ms.append(float(callback_total_ms))
            self.tracking_times_ms.append(float(tracking_ms))
            self.target_update_times_ms.append(float(target_update_ms))

            # Log only every 10 frames to keep CSV small
            if frame_count % 10 == 0:
                self._write_sample(
                    frame_count=frame_count,
                    callback_total_ms=callback_total_ms,
                    tracking_ms=tracking_ms,
                    target_update_ms=target_update_ms,
                )
        except Exception as exc:
            self.failed = True
            print(f"Performance logging disabled after an error: {exc}")

    def _write_sample(self,
                      *,
                      frame_count: int,
                      callback_total_ms: float,
                      tracking_ms: float,
                      target_update_ms: float, ) -> None:
        """
        Write one row to CSV
        """
        elapsed = time.perf_counter() - self.start_time
        mean_fps = frame_count / elapsed if elapsed > 0 else 0.0

        # Query ReID calls
        reid_calls = int(self.extractor.extract_calls) if self.extractor else 0
        calls_since_last = reid_calls - self.last_reid_call_count
        self.last_reid_call_count = reid_calls

        if self.writer is None:
            return

        self.writer.writerow({
            "wall_elapsed_sec": f"{elapsed:.3f}",
            "frame_count": frame_count,
            "mode": self.mode,
            "rolling_fps": f"{self._rolling_fps():.4f}",
            "mean_fps": f"{mean_fps:.4f}",
            "callback_total_ms": f"{callback_total_ms:.4f}",
            "tracking_ms": f"{tracking_ms:.4f}",
            "target_update_ms": f"{target_update_ms:.4f}",
            "reid_calls_total": reid_calls,
            "reid_calls_since_last_sample": calls_since_last,
        })
        self.file.flush()

    def finalize(self, *, frame_count: int) -> None:
        """
        Compute and write the final summary statistics
        """
        if self.finalized:
            return
        self.finalized = True

        if self.file is not None:
            self.file.flush()
            self.file.close()

        if self.summary_path is None:
            return

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = max(0.0, time.perf_counter() - self.start_time)

        summary = {
            "mode": self.mode,
            "wall_duration_sec": elapsed,
            "frames": int(frame_count),
            "mean_fps": frame_count / elapsed if elapsed > 0 else 0.0,
            "callback_total_ms": self._summary(self.callback_times_ms),
            "tracking_ms": self._summary(self.tracking_times_ms),
            "target_update_ms": self._summary(self.target_update_times_ms),
            "total_reid_calls": int(self.extractor.extract_calls) if self.extractor else 0,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", )

    def _rolling_fps(self) -> float:
        """Calculate the FPS over the last 120 frames"""
        if len(self.frame_timestamps) < 2:
            return 0.0
        duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
        return ((len(self.frame_timestamps) - 1) / duration) if duration > 0 else 0.0

    @staticmethod
    def _summary(values) -> dict[str, float | None]:
        """
        Calculate summary (mean, median, p95, max)
        """
        values = list(values)
        if not values:
            return {"mean": None, "median": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }
