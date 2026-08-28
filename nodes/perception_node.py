#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node

# ROS Messages
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32, String

from app.evaluation.event_logger import EventLogger
from app.evaluation.performance_logger import PerformanceLogger
from app.reid.appearance_memory import AppearanceMemory
from app.reid.reidentifier import ReIdentifier
from app.target.target_manager import TargetManager
from app.tracking.bytetrack_tracker import ByteTrackTracker
from app.reid.extractor import FeatureExtractor


class PerceptionNode(Node):
    """
    Perception-only node for direct cmd_vel following.

    Responsibilities:
    - Detect people using YOLO + ByteTrack
    - Identify who is being tracked
    - publish target state for follow node
    """

    def __init__(self):
        super().__init__("person_perception_node")
        self.bridge = CvBridge()

        self.declare_parameter("model_path", "yolov8s.pt")
        self.declare_parameter("mode", "online_reid")  # tracker_only | online_reid | calibration_reid
        self.declare_parameter("log_path", "logs/experiment.csv")
        self.declare_parameter("tracker_config", "bytetrack.yaml")
        self.declare_parameter("reid_model", "")

        # Output paths for performance logger
        self.declare_parameter("performance_log_path", "")
        self.declare_parameter("performance_summary_path", "")

        # Target and ReID Parameters
        self.declare_parameter("reassign_threshold", 0.80)
        self.declare_parameter("reassign_margin", 0.08)
        self.declare_parameter("reassign_consecutive_frames", 3)
        self.declare_parameter("min_yolo_conf", 0.55)
        self.declare_parameter("min_crop_height", 90)
        self.declare_parameter("identity_check_threshold", 0.72)

        # memory update parameters
        self.declare_parameter("memory_update_threshold", 0.84)
        self.declare_parameter("memory_update_max_similarity", 0.97)
        self.declare_parameter("memory_update_cooldown_sec", 0.50)
        self.declare_parameter("memory_update_min_track_age_sec", 1.0)
        self.declare_parameter("memory_update_min_yolo_conf", 0.75)
        self.declare_parameter("memory_update_min_crop_height", 160)

        # ReID intervals
        self.declare_parameter("reid_stable_interval_sec", 1.0)
        self.declare_parameter("reid_low_conf_interval_sec", 0.20)
        self.declare_parameter("max_missing_time", 0.25)

        # ReID initialization and Calibration
        self.declare_parameter("gallery_max_size", 30)
        self.declare_parameter("calibration_start_time_sec", 18.0)
        self.declare_parameter("online_updates_start_time_sec", 20.0)
        self.declare_parameter("calibration_embeddings", 15)
        self.declare_parameter("run_memory_path", "")

        self.mode = str(self.get_parameter("mode").value)
        self.reid_model = str(self.get_parameter("reid_model").value)
        self.frame_count = 0
        self.start_wall_time = time.time()
        self.reassign_threshold = float(self.get_parameter("reassign_threshold").value)

        run_memory_path = str(self.get_parameter("run_memory_path").value).strip()
        self.run_memory_path = (Path(run_memory_path).expanduser() if run_memory_path else None)

        # Hardware acceleration
        if torch.accelerator.is_available():
            self.device = torch.accelerator.current_accelerator()
            self.get_logger().info("Using GPU acceleration.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().warning("No GPU acceleration available. Using CPU.")

        # Initialize Tracker
        self.tracker = ByteTrackTracker(
            model_path=self.get_parameter("model_path").value,
            tracker_config=str(self.get_parameter("tracker_config").value),
            device=self.device,
        )

        # Initialize AppearanceMemory + ReID if not track_only
        self.extractor = None
        if self.mode == "tracker_only":
            reidentifier = None
            memory = None
        else:
            memory = AppearanceMemory(
                max_size=int(self.get_parameter("gallery_max_size").value),
                update_threshold=float(self.get_parameter("memory_update_threshold").value),
                update_max_similarity=float(self.get_parameter("memory_update_max_similarity").value),
                min_update_interval=float(self.get_parameter("memory_update_cooldown_sec").value),
                min_quality=float(self.get_parameter("memory_update_min_yolo_conf").value),
            )
            self.extractor = FeatureExtractor(
                device=self.device,
                model_name=self.reid_model,
                weights_path=None,
            )
            reidentifier = ReIdentifier(
                self.extractor,
                memory,
                min_crop_height=int(self.get_parameter("min_crop_height").value),
                min_yolo_conf=float(self.get_parameter("min_yolo_conf").value),
            )

        self.memory = memory

        # Initialize TargetManager
        self.target_manager = TargetManager(
            mode=self.mode,
            reidentifier=reidentifier,
            memory=memory,
            config={
                "calibration_start_time_sec": float(self.get_parameter("calibration_start_time_sec").value),
                "online_updates_start_time_sec": float(self.get_parameter("online_updates_start_time_sec").value),
                "calibration_embeddings": int(self.get_parameter("calibration_embeddings").value),
                "reid_stable_interval_sec": float(self.get_parameter("reid_stable_interval_sec").value),
                "reid_low_conf_interval_sec": float(self.get_parameter("reid_low_conf_interval_sec").value),
                "identity_check_threshold": float(self.get_parameter("identity_check_threshold").value),
                "memory_update_min_similarity": float(self.get_parameter("memory_update_threshold").value),
                "memory_update_max_similarity": float(self.get_parameter("memory_update_max_similarity").value),
                "memory_update_min_track_age_sec": float(self.get_parameter("memory_update_min_track_age_sec").value),
                "memory_update_min_yolo_conf": float(self.get_parameter("memory_update_min_yolo_conf").value),
                "memory_update_min_crop_height": int(self.get_parameter("memory_update_min_crop_height").value),
                "max_missing_time": float(self.get_parameter("max_missing_time").value),
                "reassign_threshold": float(self.get_parameter("reassign_threshold").value),
                "reassign_margin": float(self.get_parameter("reassign_margin").value),
                "min_yolo_conf": float(self.get_parameter("min_yolo_conf").value),
                "min_crop_height": int(self.get_parameter("min_crop_height").value),
                "reassign_consecutive_frames": int(self.get_parameter("reassign_consecutive_frames").value),
            },
        )

        # Initialize Logging
        self.logger = EventLogger(self.get_parameter("log_path").value)
        self.performance = PerformanceLogger(
            mode=self.mode,
            extractor=self.extractor,
            log_path=str(self.get_parameter("performance_log_path").value),
            summary_path=str(self.get_parameter("performance_summary_path").value),
        )

        # Subscribe to TurteBot 4 camera feed
        self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, 10)

        # Publishers
        self.control_pub = self.create_publisher(Vector3Stamped, "/target_person/control", 10)
        self.visible_pub = self.create_publisher(Bool, "/target_person/visible", 10)
        self.yolo_vis_pub = self.create_publisher(Image, "/yolo_vis", 10)
        self.image_size = None

        # Startup Info
        reid_description = ("disabled" if self.mode == "tracker_only" else f"torchreid:{self.reid_model}")
        strategy = {
            "tracker_only": "tracking only",
            "online_reid": "one-crop bootstrap, then adaptive online gallery",
            "calibration_reid": (
                "in-simulation 2 s multi-view calibration, then adaptive online gallery"
            ), }.get(self.mode, self.mode)

        self.get_logger().info(
            f"PerceptionNode started in mode={self.mode!r}; "
            f"strategy={strategy}; ReID={reid_description}; "
            f"calibration_window="
            f"{self.get_parameter('calibration_start_time_sec').value}-"
            f"{self.get_parameter('online_updates_start_time_sec').value}s; "
            f"calibration_embeddings={self.get_parameter('calibration_embeddings').value}; "
            f"checks stable={self.get_parameter('reid_stable_interval_sec').value}s, "
            f"low-conf={self.get_parameter('reid_low_conf_interval_sec').value}s"
        )

    def export_reid_memory(self) -> None:
        """Export ReID memory as a .npz file for offline evaluation."""
        if self.run_memory_path is None or self.memory is None:
            return
        entries = list(self.memory.entries)
        if not entries:
            self.get_logger().warning("No ReID memory entries available for export.")
            return

        self.run_memory_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_count = self.memory.anchor_size()
        kinds = np.asarray(
            ["anchor" if index < anchor_count else "adaptive"
             for index in range(len(entries))]
        )
        np.savez_compressed(
            self.run_memory_path,
            embeddings=np.stack([entry.embedding for entry in entries]).astype(np.float32),
            entry_kind=kinds,
            timestamps=np.asarray([entry.timestamp for entry in entries], dtype=np.float64),
            track_ids=np.asarray([entry.track_id for entry in entries], dtype=np.int64),
            qualities=np.asarray([entry.quality for entry in entries], dtype=np.float32),
            mode=np.asarray(self.mode),
            model_name=np.asarray(self.reid_model),
            weight_source=np.asarray(getattr(self.extractor, "weight_source", "library")),
            reassign_threshold=np.asarray(self.reassign_threshold, dtype=np.float32),
            calibration_start_time_sec=np.asarray(float(self.get_parameter("calibration_start_time_sec").value),
                                                  dtype=np.float32, ),
            online_updates_start_time_sec=np.asarray(float(self.get_parameter("online_updates_start_time_sec").value),
                                                     dtype=np.float32, ),
            configured_calibration_embeddings=np.asarray(int(self.get_parameter("calibration_embeddings").value),
                                                         dtype=np.int32, ),
        )
        self.get_logger().info(
            f"Exported ReID memory: anchors={anchor_count}, "
            f"adaptive={self.memory.adaptive_size()}, path={self.run_memory_path}"
        )

    def publish_yolo_vis(self, frame, tracks, output):
        """
        Visualization:
        - Blue: Locked target (shows ReID score)
        - Yellow: Candidate for reassignment (shows ReID score)
        """
        vis = frame.copy()
        target_id = output.target_track_id
        candidate_id = output.candidate_track_id

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            is_locked = bool(output.visible and target_id is not None and track.track_id == target_id)
            is_maybe = bool(candidate_id is not None and track.track_id == candidate_id and not is_locked)

            # Color coding
            if is_locked:
                color = (255, 0, 0)  # Blue = Target
                label = f"ReID: {output.reid_score:.2f}"
            elif is_maybe:
                color = (0, 255, 255)  # Yellow = Candidate
                label = f"ReID: {output.reid_score:.2f}"
            else:
                continue

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if label:
                # Draw text background
                text_pos = (x1, max(18, y1 - 7))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (text_pos[0] - 2, text_pos[1] - th - 4), (text_pos[0] + tw + 2, text_pos[1] + 3),
                              (0, 0, 0), -1)
                cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.yolo_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))

    def rgb_callback(self, msg: Image):
        callback_start = time.perf_counter()
        self.frame_count += 1
        wall_now = time.time()
        strategy_now = self.get_clock().now().nanoseconds / 1e9

        # Decode image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Process image YOLO + ByteTrack
        stage_start = time.perf_counter()
        tracks = self.tracker.process(frame)
        tracking_ms = (time.perf_counter() - stage_start) * 1000.0

        # Update Target Manager
        stage_start = time.perf_counter()
        output = self.target_manager.update(tracks, strategy_now, frame.shape[1])
        target_update_ms = (time.perf_counter() - stage_start) * 1000.0

        # Publish target status
        visible = bool(output.visible and output.target_track is not None)
        self.visible_pub.publish(Bool(data=visible))

        height, width = frame.shape[:2]
        if self.image_size != (width, height):
            self.image_size = (width, height)
            self.get_logger().info(f"RGB resolution detected: {width}x{height}")

        # Create control signal
        control = Vector3Stamped()
        control.header = msg.header
        control.header.frame_id = "target_control"
        if visible:
            x1, y1, x2, y2 = output.target_track.bbox
            cx = 0.5 * (float(x1) + float(x2))
            x_offset = (cx - 0.5 * width) / max(1.0, 0.5 * width)
            control.vector.x = float(x_offset)
            control.vector.y = float(y2 - y1)
            control.vector.z = max(float(output.confidence), 1e-6)
        else:
            control.vector.x = 0.0
            control.vector.y = 0.0
            control.vector.z = 0.0
        self.control_pub.publish(control)

        # Log state and performance
        self.logger.log(
            time=wall_now - self.start_wall_time,
            mode=self.mode,
            state=output.state.value,
            visible=output.visible,
            track_id=output.target_track_id,
            confidence=output.confidence,
            reid_score=output.reid_score,
            candidate_track_id=output.candidate_track_id,
            num_tracks=len(tracks),
            event=output.event or "",
            reasons=";".join(output.reasons),
        )

        self.performance.record_frame(
            frame_count=self.frame_count,
            frame_timestamp=callback_start,
            callback_total_ms=(time.perf_counter() - callback_start) * 1000.0,
            tracking_ms=tracking_ms,
            target_update_ms=target_update_ms,
        )

        # Publish debug visualization
        self.publish_yolo_vis(frame, tracks, output)


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.performance.finalize(frame_count=node.frame_count)
        node.export_reid_memory()
        if node.logger is not None and hasattr(node.logger, "close"):
            try:
                node.logger.close()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
