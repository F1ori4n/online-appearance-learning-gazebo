#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

Point2D = Tuple[float, float]
TimedPoint2D = Tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    """Clamps a value between a lower and upper bound"""
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    """Normalizes an angle"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Converts a quaternion orientation to a yaw angle"""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CmdVelFollowNode(Node):
    """
    Translates the Perception nodes control signal into robot motion

    It ensures the target remains within the camera frame by using a visual
    servoing controller with a central dead zone. If the target is lost, it
    uses a breadcrumb trail to backtrack and search for the person.
    """

    # Control Rates and Timeouts
    CONTROL_RATE = 30.0
    TARGET_TIMEOUT = 0.40
    ODOM_TIMEOUT = 0.80
    SCAN_TIMEOUT = 0.60

    # Flicker protection
    LOSS_CONFIRM_TIME = 0.15
    REACQUIRE_STABLE_TIME = 0.30

    # Normal visual following (keep target in frame, not necessarily center)
    DRIVE_SPEED = 0.46
    MAX_ANGULAR = 0.95
    ANGULAR_GAIN = 1.35
    ANGULAR_D_GAIN = 0.03
    OFFSET_ALPHA = 0.20
    EDGE_EXPONENT = 1.10
    MAX_OFFSET_DERIV = 1.40
    TURN_SLOWDOWN = 0.08
    MIN_TURN_SPEED_FACTOR = 0.78
    CENTER_TOLERANCE = 0.18 # The robot only starts turning when the target is more than 18% away from the center

    # Distance control based on bbox height
    REFERENCE_IMAGE_HEIGHT = 240
    SLOW_HEIGHT_AT_REFERENCE = 175.0
    HOLD_ENTER_HEIGHT_AT_REFERENCE = 210.0
    HOLD_EXIT_HEIGHT_AT_REFERENCE = 195.0
    REVERSE_HEIGHT_AT_REFERENCE = 232.0
    MIN_BBOX_HEIGHT_AT_REFERENCE = 55.0
    REVERSE_SPEED = -0.08

    # Camera Model and Range Estimation
    CAMERA_HFOV = math.radians(69.0)
    RANGE_SCALE_AT_REFERENCE = 360.0
    MIN_ESTIMATED_RANGE = 0.65
    MAX_ESTIMATED_RANGE = 4.50

    # Breadcrumb path memory
    TARGET_HISTORY_SECONDS = 10.0
    TARGET_HISTORY_MAX_POINTS = 80
    BREADCRUMB_PERIOD = 0.25
    BREADCRUMB_MIN_SPACING = 0.18
    RECOVERY_WAYPOINT_SPACING = 0.30
    RECOVERY_WAYPOINT_RADIUS = 0.28

    # Recovery and search.
    RECOVERY_MIN_TIMEOUT = 14.0
    RECOVERY_MAX_TIMEOUT = 35.0
    RECOVERY_TIME_FACTOR = 1.35
    RECOVERY_TIME_MARGIN = 2.0
    RECOVERY_MIN_SPEED_FACTOR = 0.90
    RECOVERY_HEADING_GAIN = 1.45
    RECOVERY_MAX_ANGULAR = 0.78
    RECOVERY_TURN_IN_PLACE_ANGLE = 0.90
    RECOVERY_WAYPOINT_STALL_TIME = 4.5
    RECOVERY_WAYPOINT_MIN_PROGRESS = 0.08
    DIRECTIONAL_SEARCH_TIME = 4.0
    FULL_SEARCH_TIME = 26.0
    SEARCH_ANGULAR = 0.42
    SEARCH_HEADING_GAIN = 1.20

    # Lidar sectors and local obstacle avoidance
    FRONT_CENTER_SECTOR = math.radians(20.0)
    FRONT_CORNER_SECTOR = math.radians(58.0)
    SIDE_SECTOR = math.radians(100.0)
    FRONT_AVOID_DISTANCE = 0.60
    CORNER_AVOID_DISTANCE = 0.45
    THIN_OBSTACLE_DISTANCE = 0.38
    CRITICAL_DISTANCE = 0.28
    SIDE_CLEARANCE = 0.23
    MIN_AVOID_FORWARD_SPEED = 0.08
    OBSTACLE_TURN_GAIN = 0.55
    SIDE_AVOID_GAIN = 0.30
    AVOID_COMMIT_TIME = 0.55
    VISIBLE_MAX_AVOID_TURN = 0.12
    VISIBLE_MAX_SIDE_BIAS = 0.08

    # Target Masking
    TARGET_MASK_HALF_ANGLE = math.radians(14.0)
    TARGET_MASK_RANGE_TOLERANCE = 0.55
    TARGET_MASK_KEEP_CLOSER_MARGIN = 0.40

    # Escape and stall handling.
    ESCAPE_REVERSE_SPEED = -0.10
    ESCAPE_ANGULAR = 0.72
    ESCAPE_DURATION = 0.75
    STALL_WINDOW = 0.90
    STALL_MIN_COMMAND = 0.12
    STALL_MIN_PROGRESS = 0.035
    TURN_ESCAPE_DISTANCE = 0.42
    TURN_ESCAPE_MIN_COMMAND = 0.14

    MAX_LINEAR_ACCEL = 2.20
    MAX_ANGULAR_ACCEL = 1.20

    def __init__(self) -> None:
        super().__init__("cmd_vel_follow_node")

        # Interface settings
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("image_height", 480)  # Picture height
        self.declare_parameter("debug", True)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.image_height = max(1, int(self.get_parameter("image_height").value))
        self.debug = bool(self.get_parameter("debug").value)

        # publisher to send control commands to robot
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)

        # Subscribe to target, odom and laser scanner
        self.create_subscription(Vector3Stamped, "/target_person/control", self.target_control_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)

        # Target states
        self.visible_signal = False
        self.offset: Optional[float] = None
        self.filtered_offset = 0.0
        self.previous_effective_offset = 0.0
        self.last_seen_offset = 0.0
        self.bbox_height_px = 0.0
        self.last_target_measurement = 0.0
        self.false_since: Optional[float] = None
        self.true_since: Optional[float] = None
        self.loss_active = False
        self.lost_since: Optional[float] = None
        self.distance_hold = False

        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_odom_time = 0.0

        # Resolution-scaled to current image height
        height_scale = self.image_height / float(self.REFERENCE_IMAGE_HEIGHT)
        self.slow_height_px = self.SLOW_HEIGHT_AT_REFERENCE * height_scale
        self.hold_enter_height_px = self.HOLD_ENTER_HEIGHT_AT_REFERENCE * height_scale
        self.hold_exit_height_px = self.HOLD_EXIT_HEIGHT_AT_REFERENCE * height_scale
        self.reverse_height_px = self.REVERSE_HEIGHT_AT_REFERENCE * height_scale
        self.min_bbox_height_px = self.MIN_BBOX_HEIGHT_AT_REFERENCE * height_scale
        self.rgb_range_scale = self.RANGE_SCALE_AT_REFERENCE * height_scale

        # Lidar states
        self.last_scan_time = 0.0
        self.front_range = math.inf
        self.front_left_range = math.inf
        self.front_right_range = math.inf
        self.left_range = math.inf
        self.right_range = math.inf
        self.front_raw = math.inf
        self.front_left_raw = math.inf
        self.front_right_raw = math.inf
        self.masked_target_rays = 0
        self.last_scan_warning_time = 0.0

        # Avoidance and escape
        self.avoid_direction = 0.0
        self.avoid_until = 0.0
        self.escape_until = 0.0
        self.stall_anchor_x = 0.0
        self.stall_anchor_y = 0.0
        self.stall_anchor_time = 0.0

        # Breadcrumb recovery
        self.target_history: Deque[TimedPoint2D] = deque(maxlen=self.TARGET_HISTORY_MAX_POINTS)
        self.last_breadcrumb_time = 0.0
        self.recovery_waypoints: Deque[Point2D] = deque()
        self.recovery_started: Optional[float] = None
        self.recovery_deadline: Optional[float] = None
        self.recovery_current_waypoint: Optional[Point2D] = None
        self.recovery_best_distance = math.inf
        self.recovery_last_progress_time = 0.0
        self.search_started: Optional[float] = None
        self.expected_search_heading: Optional[float] = None
        self.search_direction = 1.0

        # Command state.
        self.mode = "IDLE"
        self.last_control_time = time.monotonic()
        self.last_linear = 0.0
        self.last_angular = 0.0

        self.create_timer(1.0 / self.CONTROL_RATE, self.control_loop)
        self.create_timer(2.0, self.status_log)
        self.get_logger().info("follower started")
        self.get_logger().info(
            f"Configured RGB image height: {self.image_height}px; "
            f"bbox thresholds slow={self.slow_height_px:.1f}, "
            f"hold_enter={self.hold_enter_height_px:.1f}, "
            f"hold_exit={self.hold_exit_height_px:.1f}, "
            f"reverse={self.reverse_height_px:.1f}px"
        )

    # Input callbacks
    def target_control_cb(self, msg: Vector3Stamped) -> None:
        """Receives the combined target signal"""
        now = time.monotonic()
        # Check that target is visible
        visible = float(msg.vector.z) > 0.0

        # Handle visibility transitions
        if visible != self.visible_signal:
            if visible:
                self.true_since = now
                self.false_since = None
            else:
                self.false_since = now
                self.true_since = None
        self.visible_signal = visible

        # if visible save current values
        if visible:
            self.offset = float(msg.vector.x)
            self.last_seen_offset = self.offset
            self.bbox_height_px = max(0.0, float(msg.vector.y))
            self.last_target_measurement = now

    def odom_cb(self, msg: Odometry) -> None:
        """Receives Odometry and updates position and orientation."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = float(p.x)
        self.robot_y = float(p.y)
        self.robot_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.last_odom_time = time.monotonic()

    def scan_cb(self, msg: LaserScan) -> None:
        """Divide Laser signal into areas"""
        sectors = {"f": [], "fl": [], "fr": [], "l": [], "r": []}
        now = time.monotonic()

        target_fresh = (
                self.visible_signal
                and self.offset is not None
                and self.bbox_height_px >= self.min_bbox_height_px
                and now - self.last_target_measurement <= self.TARGET_TIMEOUT
        )
        target_bearing = 0.0
        target_range = math.inf
        if target_fresh:
            target_bearing = -float(self.offset) * 0.5 * self.CAMERA_HFOV
            target_range = self.estimate_target_range()

        angle = float(msg.angle_min)
        min_valid = max(0.02, float(msg.range_min))
        max_valid = float(msg.range_max) if msg.range_max > 0.0 else math.inf
        masked = 0

        for raw in msg.ranges:
            distance = float(raw)
            if math.isfinite(distance) and min_valid <= distance <= max_valid:
                a = normalize_angle(angle)

                if target_fresh:
                    in_target_cone = abs(normalize_angle(a - target_bearing)) <= self.TARGET_MASK_HALF_ANGLE
                    near_target_range = abs(distance - target_range) <= self.TARGET_MASK_RANGE_TOLERANCE
                    not_clearly_closer = distance >= target_range - self.TARGET_MASK_KEEP_CLOSER_MARGIN
                    if in_target_cone and near_target_range and not_clearly_closer:
                        masked += 1
                        angle += float(msg.angle_increment)
                        continue

                if abs(a) <= self.FRONT_CENTER_SECTOR:
                    sectors["f"].append(distance)
                elif self.FRONT_CENTER_SECTOR < a <= self.FRONT_CORNER_SECTOR:
                    sectors["fl"].append(distance)
                elif -self.FRONT_CORNER_SECTOR <= a < -self.FRONT_CENTER_SECTOR:
                    sectors["fr"].append(distance)
                elif self.FRONT_CORNER_SECTOR < a <= self.SIDE_SECTOR:
                    sectors["l"].append(distance)
                elif -self.SIDE_SECTOR <= a < -self.FRONT_CORNER_SECTOR:
                    sectors["r"].append(distance)
            angle += float(msg.angle_increment)

        self.front_range = self.near_average(sectors["f"])
        self.front_left_range = self.near_average(sectors["fl"])
        self.front_right_range = self.near_average(sectors["fr"])
        self.left_range = self.near_average(sectors["l"])
        self.right_range = self.near_average(sectors["r"])

        # Raw minima are used only for the critical thin-obstacle layer.
        self.front_raw = min(sectors["f"], default=math.inf)
        self.front_left_raw = min(sectors["fl"], default=math.inf)
        self.front_right_raw = min(sectors["fr"], default=math.inf)
        self.masked_target_rays = masked
        self.last_scan_time = now

    @staticmethod
    def near_average(values) -> float:
        if not values:
            return math.inf
        ordered = sorted(values)
        count = min(2, len(ordered))
        return float(sum(ordered[:count]) / count)

    # Common state helpers
    def odom_fresh(self, now: float) -> bool:
        return self.last_odom_time > 0.0 and now - self.last_odom_time <= self.ODOM_TIMEOUT

    def scan_fresh(self, now: float) -> bool:
        return self.last_scan_time > 0.0 and now - self.last_scan_time <= self.SCAN_TIMEOUT

    # State helpers
    def confirmed_visible(self, now: float) -> bool:
        return (self.last_target_measurement > 0.0
                and self.visible_signal
                and self.offset is not None
                and now - self.last_target_measurement <= self.TARGET_TIMEOUT)

    def clear_loss_after_stable_reacquisition(self, now: float) -> None:
        """Clear loss after taget stable reacquisition"""
        if not self.loss_active or self.true_since is None:
            return
        if now - self.true_since < self.REACQUIRE_STABLE_TIME:
            return
        self.loss_active = False
        self.lost_since = None
        self.recovery_waypoints.clear()
        self.recovery_deadline = None
        self.recovery_current_waypoint = None
        self.search_started = None

    def estimate_target_range(self) -> float:
        """Estimates the distance to the target based on bounding box height"""
        if self.bbox_height_px < self.min_bbox_height_px:
            return math.inf
        return clamp(self.rgb_range_scale / max(1.0, self.bbox_height_px),
                     self.MIN_ESTIMATED_RANGE,
                     self.MAX_ESTIMATED_RANGE, )

    # Target path estimation
    def estimate_target_world_position(self, now: float) -> Optional[Point2D]:
        """Estimates the position of the target based on offset and distance"""
        if not self.odom_fresh(now) or self.offset is None:
            return None
        estimated_range = self.estimate_target_range()
        if not math.isfinite(estimated_range):
            return None
        # Viewing direction of the robot + approximated angel of the target
        relative_bearing = -float(self.offset) * 0.5 * self.CAMERA_HFOV
        world_bearing = self.robot_yaw + relative_bearing
        return (
            self.robot_x + estimated_range * math.cos(world_bearing),
            self.robot_y + estimated_range * math.sin(world_bearing),
        )

    def record_target_breadcrumb(self, now: float) -> None:
        """Records current target breadcrumb"""
        if now - self.last_breadcrumb_time < self.BREADCRUMB_PERIOD:
            return
        point = self.estimate_target_world_position(now)
        if point is None:
            return
        # Remove points that are older than TARGET_HISTORY_SECONDS
        while self.target_history and now - self.target_history[0][2] > self.TARGET_HISTORY_SECONDS:
            self.target_history.popleft()

        if self.target_history:
            lx, ly, _ = self.target_history[-1]
            if math.hypot(point[0] - lx, point[1] - ly) < self.BREADCRUMB_MIN_SPACING:
                return

        self.target_history.append((point[0], point[1], now))
        self.last_breadcrumb_time = now

    def begin_loss_episode(self, now: float) -> None:
        self.loss_active = True
        self.lost_since = now
        self.prepare_recovery_path(now)
        self.get_logger().warning("Target lost; starting recovery.")

    def prepare_recovery_path(self, now: float) -> None:
        """Prepares recovery path to last Target position"""
        self.recovery_waypoints.clear()
        self.recovery_started = now
        self.recovery_deadline = None
        self.recovery_current_waypoint = None
        self.recovery_best_distance = math.inf
        self.recovery_last_progress_time = now
        self.search_started = None

        # Only use recent points
        recent = [(x, y) for x, y, stamp in self.target_history if now - stamp <= self.TARGET_HISTORY_SECONDS]

        fallback_heading = self.robot_yaw - self.last_seen_offset * 0.5 * self.CAMERA_HFOV
        self.expected_search_heading = fallback_heading
        self.search_direction = -1.0 if self.last_seen_offset > 0.05 else 1.0

        if not recent or not self.odom_fresh(now):
            return

        # Find closest point to robot and start path from there
        nearest_index = min(range(len(recent)),
                            key=lambda i: math.hypot(recent[i][0] - self.robot_x, recent[i][1] - self.robot_y), )
        candidate_path = recent[nearest_index:]

        # simplify path
        simplified: list[Point2D] = []
        for point in candidate_path:
            if not simplified or math.hypot(
                    point[0] - simplified[-1][0], point[1] - simplified[-1][1]
            ) >= self.RECOVERY_WAYPOINT_SPACING:
                simplified.append(point)
        if candidate_path and (not simplified or simplified[-1] != candidate_path[-1]):
            simplified.append(candidate_path[-1])

        # Add waypoints that are not reached yet
        for point in simplified:
            if math.hypot(point[0] - self.robot_x, point[1] - self.robot_y) > self.RECOVERY_WAYPOINT_RADIUS:
                self.recovery_waypoints.append(point)

        # Calculate Search direction from last movement direction
        if len(recent) >= 2:
            dx = recent[-1][0] - recent[-2][0]
            dy = recent[-1][1] - recent[-2][1]
            if math.hypot(dx, dy) > 0.05:
                self.expected_search_heading = math.atan2(dy, dx)

        relative = normalize_angle(self.expected_search_heading - self.robot_yaw)
        if abs(relative) > 0.08:
            self.search_direction = 1.0 if relative > 0.0 else -1.0
        elif self.last_seen_offset > 0.05:
            self.search_direction = -1.0
        elif self.last_seen_offset < -0.05:
            self.search_direction = 1.0

        recovery_path_length = 0.0
        px, py = self.robot_x, self.robot_y
        for wx, wy in self.recovery_waypoints:
            recovery_path_length += math.hypot(wx - px, wy - py)
            px, py = wx, wy

        if self.recovery_waypoints:
            nominal = recovery_path_length / self.DRIVE_SPEED
            budget = clamp(nominal * self.RECOVERY_TIME_FACTOR + self.RECOVERY_TIME_MARGIN,
                           self.RECOVERY_MIN_TIMEOUT,
                           self.RECOVERY_MAX_TIMEOUT, )
            self.recovery_deadline = now + budget
        else:
            budget = 0.0

        self.get_logger().info(f"Recovery prepared: waypoints={len(self.recovery_waypoints)} "
                               f"path={recovery_path_length:.2f}m budget={budget:.1f}s")

    # Main controller
    def control_loop(self) -> None:
        """Main control loop"""
        now = time.monotonic()
        dt = max(1e-3, now - self.last_control_time)
        self.last_control_time = now
        self.update_stall_state(now)

        # Target visible, then follow
        if self.confirmed_visible(now):
            self.clear_loss_after_stable_reacquisition(now)
            self.record_target_breadcrumb(now)
            self.follow(dt, now)
            return

        # No Target seen, stand still
        if self.last_target_measurement <= 0.0:
            self.mode = "IDLE"
            self.publish_cmd(0.0, 0.0, dt)
            return

        if self.false_since is None:
            self.false_since = now

        # Target Lost
        if not self.loss_active:
            if now - self.false_since < self.LOSS_CONFIRM_TIME:
                self.mode = "LOSS_PENDING"
                self.publish_cmd(0.0, 0.0, dt)
                return
            self.begin_loss_episode(now)

        # Follow recovery path
        if self.recovery_waypoints:
            if not self.odom_fresh(now):
                self.mode = "RECOVERY_WAIT_ODOM"
                self.publish_cmd(0.0, 0.0, dt)
                return
            if self.recovery_deadline is None or now <= self.recovery_deadline:
                self.recover_along_path(dt, now)
                return
            self.get_logger().warning(
                f"Recovery budget ended with {len(self.recovery_waypoints)} waypoint(s) left."
            )
            self.recovery_waypoints.clear()

        # Start search
        if self.search_started is None:
            self.search_started = now
        self.search(dt, now)

    def follow(self, dt: float, now: float) -> None:
        """Steers the robot to keep the target in the camera frame"""
        self.mode = "FOLLOW"
        raw = float(self.offset)

        # low pass filter to smoothen offset
        self.filtered_offset = self.OFFSET_ALPHA * raw + (1.0 - self.OFFSET_ALPHA) * self.filtered_offset

        # Calculate effective offset with dead zone and edge exponent
        abs_offset = abs(self.filtered_offset)
        tolerance = self.CENTER_TOLERANCE
        if abs_offset <= tolerance:
            effective_offset = 0.0
        else:
            sign = 1.0 if self.filtered_offset > 0.0 else -1.0
            normalized = (abs_offset - tolerance) / max(1e-3, 1.0 - tolerance)
            effective_offset = sign * normalized ** self.EDGE_EXPONENT

        derivative = clamp(
            (effective_offset - self.previous_effective_offset) / dt,
            -self.MAX_OFFSET_DERIV,
            self.MAX_OFFSET_DERIV,
        )
        self.previous_effective_offset = effective_offset
        angular = clamp(
            -(self.ANGULAR_GAIN * effective_offset + self.ANGULAR_D_GAIN * derivative),
            -self.MAX_ANGULAR,
            self.MAX_ANGULAR,
        )

        # height base speed
        if self.bbox_height_px >= self.reverse_height_px:
            self.distance_hold = True
            linear = self.REVERSE_SPEED
        else:
            if self.distance_hold and self.bbox_height_px <= self.hold_exit_height_px:
                self.distance_hold = False
            elif not self.distance_hold and self.bbox_height_px >= self.hold_enter_height_px:
                self.distance_hold = True

            if self.distance_hold:
                linear = 0.0
            elif self.bbox_height_px <= self.slow_height_px:
                linear = self.DRIVE_SPEED
            else:
                fraction = (self.hold_enter_height_px - self.bbox_height_px) / (
                        self.hold_enter_height_px - self.slow_height_px)
                linear = self.DRIVE_SPEED * clamp(fraction, 0.12, 1.0)

        # Lidar side correction
        if linear > 0.0:
            turn_factor = 1.0 - self.TURN_SLOWDOWN * min(1.0, abs(angular) / self.MAX_ANGULAR)
            linear *= clamp(turn_factor, self.MIN_TURN_SPEED_FACTOR, 1.0)

        linear, angular = self.apply_obstacle_avoidance(linear, angular, now)

        # Publish control command with limited speed
        self.publish_cmd(clamp(linear, self.ESCAPE_REVERSE_SPEED, self.DRIVE_SPEED),
                         clamp(angular, -self.MAX_ANGULAR, self.MAX_ANGULAR), dt, )

    def recover_along_path(self, dt: float, now: float) -> None:
        """Follow the saved recover path"""
        # Remove passed waypoints
        while self.recovery_waypoints:
            wx, wy = self.recovery_waypoints[0]
            if math.hypot(wx - self.robot_x, wy - self.robot_y) > self.RECOVERY_WAYPOINT_RADIUS:
                break
            self.recovery_waypoints.popleft()

        # If no waypoint left start search
        if not self.recovery_waypoints:
            if self.search_started is None:
                self.search_started = now
            self.search(dt, now)
            return

        self.mode = "RECOVER_PATH"
        wx, wy = self.recovery_waypoints[0]
        dx = wx - self.robot_x
        dy = wy - self.robot_y
        distance = math.hypot(dx, dy)
        waypoint = (wx, wy)

        if self.recovery_current_waypoint != waypoint:
            self.recovery_current_waypoint = waypoint
            self.recovery_best_distance = distance
            self.recovery_last_progress_time = now
        elif distance <= self.recovery_best_distance - self.RECOVERY_WAYPOINT_MIN_PROGRESS:
            self.recovery_best_distance = distance
            self.recovery_last_progress_time = now
        elif now - self.recovery_last_progress_time >= self.RECOVERY_WAYPOINT_STALL_TIME:
            self.get_logger().warning("Recovery waypoint blocked; skipping it.")
            self.recovery_waypoints.popleft()
            self.recovery_current_waypoint = None
            self.recovery_best_distance = math.inf
            self.recovery_last_progress_time = now
            self.publish_cmd(0.0, 0.0, dt)
            return

        heading = math.atan2(dy, dx)
        error = normalize_angle(heading - self.robot_yaw)
        angular = clamp(
            self.RECOVERY_HEADING_GAIN * error,
            -self.RECOVERY_MAX_ANGULAR,
            self.RECOVERY_MAX_ANGULAR,
        )

        if abs(error) >= self.RECOVERY_TURN_IN_PLACE_ANGLE:
            linear = 0.0
        else:
            heading_factor = max(0.65, math.cos(error))
            distance_factor = clamp(distance / 0.70, self.RECOVERY_MIN_SPEED_FACTOR, 1.0, )
            linear = self.DRIVE_SPEED * heading_factor * distance_factor

        linear, angular = self.apply_obstacle_avoidance(linear, angular, now)
        self.publish_cmd(linear, angular, dt)

    def search(self, dt: float, now: float) -> None:
        """Search strategy"""
        age = now - self.search_started if self.search_started is not None else 0.0

        # Search in the direction the target was last seen
        if age <= self.DIRECTIONAL_SEARCH_TIME:
            self.mode = "SEARCH_DIRECTION"
            if self.expected_search_heading is not None and self.odom_fresh(now):
                error = normalize_angle(self.expected_search_heading - self.robot_yaw)
                angular = (
                    clamp(self.SEARCH_HEADING_GAIN * error, -self.SEARCH_ANGULAR, self.SEARCH_ANGULAR)
                    if abs(error) > 0.20
                    else self.search_direction * 0.65 * self.SEARCH_ANGULAR
                )
            else:
                angular = self.search_direction * self.SEARCH_ANGULAR
            linear, angular = self.apply_obstacle_avoidance(0.0, angular, now)
            self.publish_cmd(linear, angular, dt)
            return

        # Turn in a circle
        if age <= self.DIRECTIONAL_SEARCH_TIME + self.FULL_SEARCH_TIME:
            self.mode = "SEARCH_CIRCLE"
            linear, angular = self.apply_obstacle_avoidance(
                0.0, self.search_direction * self.SEARCH_ANGULAR, now
            )
            self.publish_cmd(linear, angular, dt)
            return

        # Give up and stand still
        self.mode = "LOST_STOP"
        self.publish_stop_immediately()

    # Obstacle avoidance
    def choose_avoid_direction(self) -> float:
        """ Determines in which direction to steer when an obstacle is up ahead"""
        left_clearance = min(self.front_left_range, self.left_range)
        right_clearance = min(self.front_right_range, self.right_range)
        # keep current direction
        if not math.isfinite(left_clearance) and not math.isfinite(right_clearance):
            return self.avoid_direction if self.avoid_direction else 1.0
        # if booth sides are equally blocked keep previous direction
        if abs(left_clearance - right_clearance) < 0.05 and self.avoid_direction:
            return self.avoid_direction
        # Otherwise steer towards free side
        return 1.0 if left_clearance >= right_clearance else -1.0

    def trigger_escape(self, now: float, reason: str, nearest: float) -> None:
        """ Briefly revers and turn to escape"""
        self.avoid_direction = self.choose_avoid_direction()
        self.avoid_until = now + self.AVOID_COMMIT_TIME
        self.escape_until = now + self.ESCAPE_DURATION
        if self.recovery_deadline is not None and self.recovery_waypoints:
            self.recovery_deadline = min(
                self.recovery_deadline + self.ESCAPE_DURATION,
                (self.recovery_started or now) + self.RECOVERY_MAX_TIMEOUT + 5.0,
            )
        self.get_logger().warning(
            f"Escape: {reason}, nearest={nearest:.2f}m, "
            f"turn={'left' if self.avoid_direction > 0 else 'right'}."
        )

    def update_stall_state(self, now: float) -> None:
        """Detects if the robot is stuck"""
        if not self.odom_fresh(now) or self.last_linear < self.STALL_MIN_COMMAND:
            self.stall_anchor_time = 0.0
            return

        if self.stall_anchor_time <= 0.0:
            self.stall_anchor_x = self.robot_x
            self.stall_anchor_y = self.robot_y
            self.stall_anchor_time = now
            return

        if now - self.stall_anchor_time < self.STALL_WINDOW:
            return

        progress = math.hypot(self.robot_x - self.stall_anchor_x, self.robot_y - self.stall_anchor_y, )
        nearest = min(self.front_raw,
                      self.front_left_raw,
                      self.front_right_raw,
                      self.front_range,
                      self.front_left_range,
                      self.front_right_range, )

        if progress < self.STALL_MIN_PROGRESS and nearest < self.FRONT_AVOID_DISTANCE and self.scan_fresh(now):
            self.trigger_escape(now, "forward stall", nearest)

        self.stall_anchor_x = self.robot_x
        self.stall_anchor_y = self.robot_y
        self.stall_anchor_time = now

    def apply_obstacle_avoidance(self, linear: float, angular: float, now: float) -> Tuple[float, float]:
        """Apply local obstacle avoidance while trying to keep person in view"""
        if linear < 0.0:
            return linear, angular

        if not self.scan_fresh(now):
            if now - self.last_scan_warning_time > 3.0:
                self.get_logger().warning(f"No fresh LaserScan on {self.scan_topic}.")
                self.last_scan_warning_time = now
            return linear, angular

        visual_angular = angular
        target_visible = self.confirmed_visible(now)
        close_centered_target = (target_visible
                                 and self.bbox_height_px >= self.hold_enter_height_px
                                 and self.offset is not None
                                 and abs(self.offset) < 0.30)

        # combine lidar readings to find the closest obstacle
        robust_front = min(self.front_range, self.front_left_range, self.front_right_range)
        raw_front = min(self.front_raw, self.front_left_raw, self.front_right_raw)
        nearest_front = min(robust_front, raw_front)
        nearest_any = min(nearest_front, self.left_range, self.right_range)

        # If we are trying to turn but there is no room we might need to escape (drive back a little)
        turn_blocked = (not target_visible
                        and linear <= 0.01
                        and abs(angular) >= self.TURN_ESCAPE_MIN_COMMAND
                        and nearest_any <= self.TURN_ESCAPE_DISTANCE)

        # A centered, close target is handled by bbox-distance control, not by
        # a sideways escape. This avoids driving around the person.
        critical_obstacle = nearest_front <= self.CRITICAL_DISTANCE and not close_centered_target
        if (critical_obstacle or turn_blocked) and now >= self.escape_until:
            self.trigger_escape(now, "critical obstacle" if critical_obstacle else "blocked turn", nearest_any, )

        # If we are escaping override all normal steering
        if now < self.escape_until:
            return self.ESCAPE_REVERSE_SPEED, self.avoid_direction * self.ESCAPE_ANGULAR

        # If the person is already close, do not try to steer around them
        if close_centered_target and linear <= 0.0:
            return linear, visual_angular

        # Apply a small bias to steer away from walls
        side_bias = 0.0
        if self.left_range < self.SIDE_CLEARANCE:
            side_bias -= self.SIDE_AVOID_GAIN * clamp((self.SIDE_CLEARANCE - self.left_range) / self.SIDE_CLEARANCE,
                                                      0.0, 1.0)
        if self.right_range < self.SIDE_CLEARANCE:
            side_bias += self.SIDE_AVOID_GAIN * clamp((self.SIDE_CLEARANCE - self.right_range) / self.SIDE_CLEARANCE,
                                                      0.0, 1.0)
        if target_visible:
            side_bias = clamp(side_bias, -self.VISIBLE_MAX_SIDE_BIAS, self.VISIBLE_MAX_SIDE_BIAS)
        angular += side_bias

        # Check if there is a obstacle directly in front or in the corner
        centre_blocked = self.front_range < self.FRONT_AVOID_DISTANCE
        left_blocked = self.front_left_range < self.CORNER_AVOID_DISTANCE
        right_blocked = self.front_right_range < self.CORNER_AVOID_DISTANCE
        thin_blocked = raw_front < self.THIN_OBSTACLE_DISTANCE
        obstacle_ahead = centre_blocked or left_blocked or right_blocked or thin_blocked

        if not obstacle_ahead:
            return linear, angular

        # Determine the direction to steer around the obstacle
        if now >= self.avoid_until or self.avoid_direction == 0.0:
            if (self.front_left_raw < self.THIN_OBSTACLE_DISTANCE or left_blocked) and not (
                    self.front_right_raw < self.THIN_OBSTACLE_DISTANCE or right_blocked):
                self.avoid_direction = -1.0
            elif (self.front_right_raw < self.THIN_OBSTACLE_DISTANCE or right_blocked) and not (
                    self.front_left_raw < self.THIN_OBSTACLE_DISTANCE or left_blocked):
                self.avoid_direction = 1.0
            else:
                self.avoid_direction = self.choose_avoid_direction()
            self.avoid_until = now + self.AVOID_COMMIT_TIME

        # Slow down if something is in the close by
        if nearest_front <= self.THIN_OBSTACLE_DISTANCE:
            proximity = clamp((self.THIN_OBSTACLE_DISTANCE - nearest_front)
                              / (self.THIN_OBSTACLE_DISTANCE - self.CRITICAL_DISTANCE), 0.0, 1.0, )
            speed_cap = self.MIN_AVOID_FORWARD_SPEED + 0.08 * (1.0 - proximity)
        else:
            proximity = max(
                clamp((self.FRONT_AVOID_DISTANCE - self.front_range) / (
                        self.FRONT_AVOID_DISTANCE - self.CRITICAL_DISTANCE), 0.0, 1.0, ),
                clamp((self.CORNER_AVOID_DISTANCE - min(self.front_left_range, self.front_right_range)) / (
                        self.CORNER_AVOID_DISTANCE - self.CRITICAL_DISTANCE), 0.0, 1.0, ), )
            speed_cap = self.MIN_AVOID_FORWARD_SPEED + max(0.0, linear - self.MIN_AVOID_FORWARD_SPEED) * (
                    1.0 - 0.85 * proximity)

        if linear > 0.0:
            linear = min(linear, speed_cap)

        avoid_turn = self.avoid_direction * self.OBSTACLE_TURN_GAIN * (0.20 + 0.80 * proximity)

        if target_visible:
            # While the target is visible, lidar may slow the robot and may add
            # a small correction only if it agrees with the visual steering.
            # It never turns the camera away from the person.
            limited = clamp(avoid_turn, -self.VISIBLE_MAX_AVOID_TURN, self.VISIBLE_MAX_AVOID_TURN, )
            if abs(visual_angular) >= 0.03:
                if limited * visual_angular > 0.0:
                    angular = visual_angular + limited
                else:
                    angular = visual_angular
                if angular * visual_angular < 0.0:
                    angular = visual_angular
            else:
                # Target is centred: allow only a very small corridor-centering
                # correction, not a committed turn around an obstacle.
                angular = clamp(angular + 0.5 * limited, -self.VISIBLE_MAX_AVOID_TURN, self.VISIBLE_MAX_AVOID_TURN, )
        else:
            # When the target is not visible avoid obstacles more strongly
            angular = 0.60 * visual_angular + 0.60 * avoid_turn

        return linear, angular

    # Publishing and logs
    def publish_cmd(self, linear_target: float, angular_target: float, dt: float) -> None:
        max_dlin = self.MAX_LINEAR_ACCEL * dt
        max_dang = self.MAX_ANGULAR_ACCEL * dt
        linear = self.last_linear + clamp(linear_target - self.last_linear, -max_dlin, max_dlin)
        angular = self.last_angular + clamp(angular_target - self.last_angular, -max_dang, max_dang)
        self.last_linear = linear
        self.last_angular = angular
        self.publish_raw(linear, angular)

    def publish_stop_immediately(self) -> None:
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.publish_raw(0.0, 0.0)

    def publish_raw(self, linear: float, angular: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def status_log(self) -> None:
        if not self.debug:
            return
        now = time.monotonic()
        age = now - self.last_target_measurement if self.last_target_measurement else 999.0
        lost = now - self.lost_since if self.lost_since is not None else 0.0
        rec_left = max(0.0, self.recovery_deadline - now) if self.recovery_deadline else 0.0
        raw = min(self.front_raw, self.front_left_raw, self.front_right_raw)
        self.get_logger().info(
            f"mode={self.mode} visible={self.visible_signal} age={age:.2f}s lost={lost:.2f}s "
            f"offset={self.offset} filt={self.filtered_offset:.2f} "
            f"h={self.bbox_height_px:.1f}/{self.image_height}px "
            f"range_est={self.estimate_target_range():.2f}m "
            f"hold={self.distance_hold} path={len(self.target_history)} "
            f"recovery={len(self.recovery_waypoints)} rec_left={rec_left:.1f}s "
            f"scan=(L{self.left_range:.2f},FL{self.front_left_range:.2f},"
            f"F{self.front_range:.2f},FR{self.front_right_range:.2f},R{self.right_range:.2f},"
            f"raw{raw:.2f}) mask={self.masked_target_rays} "
            f"escape={now < self.escape_until} cmd=({self.last_linear:.2f},{self.last_angular:.2f})"
        )


def main() -> None:
    rclpy.init()
    node = CmdVelFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish the final stop only while the ROS context is still valid.
        if rclpy.ok():
            node.publish_stop_immediately()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
