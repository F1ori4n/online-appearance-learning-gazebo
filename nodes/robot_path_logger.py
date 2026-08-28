#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import signal
import sys
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32, Int32, String


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class RobotPathLogger(Node):
    """
    Minimal timed logger for repeatable person-following experiments.
    """

    def __init__(self) -> None:
        super().__init__("robot_path_logger")

        self.declare_parameter("output_dir", "results/path_run")
        self.declare_parameter("sample_rate_hz", 10.0)
        self.declare_parameter("end_sim_time_sec", 210.0)
        self.declare_parameter("duration_sec", 0.0)
        self.declare_parameter("scenario", "world_2")
        self.declare_parameter("mode", "online_reid")
        self.declare_parameter("reid_model", "resnet50")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate_hz = max(1.0, float(self.get_parameter("sample_rate_hz").value))
        self.end_sim_time_sec = float(self.get_parameter("end_sim_time_sec").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.scenario = str(self.get_parameter("scenario").value)
        self.mode = str(self.get_parameter("mode").value)
        self.reid_model = str(self.get_parameter("reid_model").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        # kinematic variables
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0

        # Time and path variables
        self.first_clock_sec: Optional[float] = None
        self.last_sample_clock_sec: Optional[float] = None
        self.last_path_x: Optional[float] = None
        self.last_path_y: Optional[float] = None
        self.path_length_m = 0.0
        self.start_pose: Optional[tuple[float, float, float]] = None
        self.rows = 0
        self.finished = False
        self.finish_reason = "RUNNING"
        self._finalized = False

        # CSV Initialisation
        self.csv_path = self.output_dir / "robot_path.csv"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "sim_time_sec",
                "elapsed_sec",
                "robot_x",
                "robot_y",
                "robot_yaw",
                "path_length_m",
                "cmd_linear",
                "cmd_angular",
            ],
        )
        self.writer.writeheader()

        # Subscribe to Odom and cmd vel
        self.create_subscription(Odometry, odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, cmd_vel_topic, self.cmd_cb, 10)

        self.create_timer(1.0 / self.sample_rate_hz, self.sample)
        self.get_logger().info(
            f"Path logger started: output={self.output_dir}, "
            f"end_sim_time={self.end_sim_time_sec:.1f}s, duration={self.duration_sec:.1f}s"
        )

    def odom_cb(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.robot_x = float(pose.position.x)
        self.robot_y = float(pose.position.y)
        self.robot_yaw = yaw_from_quaternion(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )

    def cmd_cb(self, msg: TwistStamped) -> None:
        self.cmd_linear = float(msg.twist.linear.x)
        self.cmd_angular = float(msg.twist.angular.z)


    def current_sim_time(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def sample(self) -> None:
        if self.finished:
            return

        now = self.current_sim_time()
        if now <= 0.0:
            return

        if self.first_clock_sec is None:
            self.first_clock_sec = now
            self.last_sample_clock_sec = now

        elapsed = now - self.first_clock_sec

        # Exit conditions
        if 0.0 < self.end_sim_time_sec <= now:
            self.request_finish("COMPLETED_BY_SIM_TIME")
            return
        if 0.0 < self.duration_sec <= elapsed:
            self.request_finish("COMPLETED_BY_DURATION")
            return

        # log path
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return

        if self.start_pose is None:
            self.start_pose = (self.robot_x, self.robot_y, self.robot_yaw)
            self.last_path_x = self.robot_x
            self.last_path_y = self.robot_y

        if self.last_path_x is not None and self.last_path_y is not None:
            step = math.hypot(self.robot_x - self.last_path_x, self.robot_y - self.last_path_y)
            # Reject impossible odometry jumps.
            if step < 1.0:
                self.path_length_m += step
        self.last_path_x = self.robot_x
        self.last_path_y = self.robot_y

        self.rows += 1
        self.writer.writerow(
            {
                "sim_time_sec": f"{now:.3f}",
                "elapsed_sec": f"{elapsed:.3f}",
                "robot_x": f"{self.robot_x:.5f}",
                "robot_y": f"{self.robot_y:.5f}",
                "robot_yaw": f"{self.robot_yaw:.5f}",
                "path_length_m": f"{self.path_length_m:.5f}",
                "cmd_linear": f"{self.cmd_linear:.5f}",
                "cmd_angular": f"{self.cmd_angular:.5f}",
            }
        )
        if self.rows % int(self.sample_rate_hz) == 0:
            self.csv_file.flush()

        self.last_sample_clock_sec = now

    def request_finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finish_reason = reason
        self.finished = True
        self.get_logger().info(f"Finishing path logger: {reason}")

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self.csv_file.flush()
        self.csv_file.close()

        end_sim = self.last_sample_clock_sec
        elapsed = 0.0
        if self.first_clock_sec is not None and end_sim is not None:
            elapsed = max(0.0, end_sim - self.first_clock_sec)

        final_pose = None
        displacement = None
        if self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None:
            final_pose = {"x": self.robot_x, "y": self.robot_y, "yaw": self.robot_yaw}
            if self.start_pose is not None:
                displacement = math.hypot(
                    self.robot_x - self.start_pose[0], self.robot_y - self.start_pose[1]
                )

        summary = {
            "result": self.finish_reason,
            "scenario": self.scenario,
            "mode": self.mode,
            "reid_model": self.reid_model,
            "sample_rate_hz": self.sample_rate_hz,
            "first_sim_time_sec": self.first_clock_sec,
            "last_sim_time_sec": end_sim,
            "logged_duration_sec": elapsed,
            "samples": self.rows,
            "path_length_m": self.path_length_m,
            "start_pose": None
            if self.start_pose is None
            else {"x": self.start_pose[0], "y": self.start_pose[1], "yaw": self.start_pose[2]},
            "final_pose": final_pose,
            "displacement_m": displacement,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.get_logger().info(
            f"Saved {self.rows} samples, path={self.path_length_m:.2f}m to {self.output_dir}"
        )


def main() -> int:
    rclpy.init()
    node = RobotPathLogger()

    def stop_handler(_signum: int, _frame: object) -> None:
        node.request_finish("INTERRUPTED")

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.request_finish("INTERRUPTED")
    finally:
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
