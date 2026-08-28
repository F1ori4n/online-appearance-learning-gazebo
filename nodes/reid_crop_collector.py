#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from app.tracking.bytetrack_tracker import ByteTrackTracker


class ReIDCropCollector(Node):
    """Collect person crops from a short, controlled simulation.
    The actor rotate in place while this node stores the exact ByteTrack/YOLO 
    crops used by the normal perception pipeline.
    """

    def __init__(self) -> None:
        super().__init__("reid_crop_collector")
        self.bridge = CvBridge()

        # Deckare parameters for calibration
        self.declare_parameter("model_path", "yolov8s.pt")
        self.declare_parameter("tracker_config", "bytetrack.yaml")
        self.declare_parameter("output_dir", "evaluation/reid_dataset/raw/remy")
        self.declare_parameter("capture_start_sec", 19.5)
        self.declare_parameter("capture_end_sec", 24.5)
        self.declare_parameter("stop_sec", 25.0)
        self.declare_parameter("save_interval_sec", 0.08)
        self.declare_parameter("min_yolo_conf", 0.70)
        self.declare_parameter("min_crop_height", 160)
        self.declare_parameter("save_context", False)

        # Parse parameters
        self.output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.capture_start_sec = float(self.get_parameter("capture_start_sec").value)
        self.capture_end_sec = float(self.get_parameter("capture_end_sec").value)
        self.stop_sec = float(self.get_parameter("stop_sec").value)
        self.save_interval_sec = float(self.get_parameter("save_interval_sec").value)
        self.min_yolo_conf = float(self.get_parameter("min_yolo_conf").value)
        self.min_crop_height = int(self.get_parameter("min_crop_height").value)
        self.save_context = bool(self.get_parameter("save_context").value)

        # Validate timing parameters
        if not (0.0 <= self.capture_start_sec < self.capture_end_sec <= self.stop_sec):
            raise ValueError("Expected 0 <= capture_start_sec < capture_end_sec <= stop_sec")

        # Check if GPU available
        if torch.accelerator.is_available():
            self.device = torch.accelerator.current_accelerator()
        else:
            self.device = torch.device("cpu")

        # initialize ByteTrack tracker
        self.tracker = ByteTrackTracker(
            model_path=str(self.get_parameter("model_path").value),
            tracker_config=str(self.get_parameter("tracker_config").value),
            device=self.device,
        )

        # Subscribe to camera feed
        self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, 10
        )

        # State variables
        self.first_stamp_sec: Optional[float] = None
        self.first_wall_time = time.monotonic()
        self.last_save_elapsed = -1e9
        self.saved_count = 0
        self.done = False
        self.last_multiple_warning = -1e9
        self.metadata_path = self.output_dir / "metadata.csv"

        self.get_logger().info(
            "ReID crop collector started: "
            f"capture={self.capture_start_sec:.1f}-{self.capture_end_sec:.1f}s, "
            f"stop={self.stop_sec:.1f}s, output={self.output_dir}"
        )

    @staticmethod
    def _message_stamp_sec(msg: Image) -> Optional[float]:
        stamp = msg.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if value > 0.0 else None

    def _elapsed(self, msg: Image) -> float:
        stamp_sec = self._message_stamp_sec(msg)
        if stamp_sec is None:
            return time.monotonic() - self.first_wall_time
        if self.first_stamp_sec is None:
            self.first_stamp_sec = stamp_sec
        return max(0.0, stamp_sec - self.first_stamp_sec)

    def _finish(self) -> None:
        if self.done:
            return
        self.done = True
        self.get_logger().info(
            f"Crop collection finished: saved={self.saved_count}, "
            f"output={self.output_dir}"
        )
        if rclpy.ok():
            rclpy.shutdown()

    def rgb_callback(self, msg: Image) -> None:
        if self.done:
            return

        elapsed = self._elapsed(msg)

        # stop logic based on simulation time
        if elapsed >= self.stop_sec:
            self._finish()
            return
        
        # Only save within the defined capture window
        if elapsed < self.capture_start_sec or elapsed > self.capture_end_sec:
            return

        # Only save a picture ever save interval
        if elapsed - self.last_save_elapsed < self.save_interval_sec:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        tracks = self.tracker.process(frame)

        # Filter valid tracks based on input parameters
        valid = [
            track
            for track in tracks
            if track.valid_id
            and track.crop is not None
            and float(track.conf) >= self.min_yolo_conf
            and int(track.height) >= self.min_crop_height
        ]

        if not valid:
            return

        track = valid[0]
        crop = track.crop
        if crop is None or crop.size == 0:
            return

        self.saved_count += 1
        self.last_save_elapsed = elapsed
        
        # save as: crop_0001.jpg, crop_0002.jpg, etc.
        filename = f"crop_{self.saved_count:04d}.jpg"
        crop_path = self.output_dir / filename

        if not cv2.imwrite(str(crop_path), crop):
            self.get_logger().warning(f"Failed to save crop: {crop_path}")
            return


def main() -> None:
    rclpy.init()
    node = ReIDCropCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
