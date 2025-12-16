#!/usr/bin/env python3
"""
ROS 2 node that streams RealSense RGB-D frames, detects a yellow ball,
and publishes its pose/twist in the Kinova base_link frame.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise RuntimeError(
        "pyrealsense2 is required for ball_tracker_node. "
        "Install the Intel RealSense SDK Python bindings first."
    ) from exc

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R


class BallTrackerNode(Node):
    """
    Continuously estimates a tennis-ball pose/velocity from RealSense frames
    and publishes them to /kinova/ball/init_pose and /kinova/ball/init_twist.
    """

    def __init__(self) -> None:
        super().__init__("ball_tracker_node")

        # -------- Parameters (declare before use) --------
        self.declare_parameter("serial", "")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("fps", 60)
        self.declare_parameter("max_depth_m", 4.5)
        self.declare_parameter("min_radius_px", 3.0)
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("use_manual_extrinsics", False)
        self.declare_parameter("camera_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("camera_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("aruco_marker_id", 7)
        self.declare_parameter("aruco_marker_length_m", 0.15)  # 15 cm marker (same as aruco_ball_2.py)
        self.declare_parameter("calibration_frames", 60)
        # self.declare_parameter("yellow_lower_hsv", [19, 120, 112])  # bee/yellow ball
        # self.declare_parameter("yellow_upper_hsv", [34, 234, 255])   
        self.declare_parameter("yellow_lower_hsv", [28, 69, 75])  # bee/yellow ball
        self.declare_parameter("yellow_upper_hsv", [74, 255, 255])   # bee/yellow ball
        self.declare_parameter("velocity_window", 5)
        self.declare_parameter("show_visualization", True)  # Enable/disable CV2 window
        self.declare_parameter("trail_buffer_size", 64)  # Trail length for visualization

        # -------- Publishers --------
        self.pose_pub = self.create_publisher(PoseStamped, "/kinova/ball/init_pose", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/kinova/ball/init_twist", 10)

        # -------- Camera setup --------
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = self._start_pipeline()
        self.color_intr = self._get_intrinsics()
        self.color_matrix = self._build_intrinsic_matrix(self.color_intr)

        # ArUco setup (compatible with OpenCV 4.6.x)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Calibration state
        self.use_manual = bool(self.get_parameter("use_manual_extrinsics").value)
        self.world_calibrated = self.use_manual
        self.calib_rvecs: list[np.ndarray] = []
        self.calib_tvecs: list[np.ndarray] = []
        self.fixed_R_cm: Optional[np.ndarray] = None
        self.fixed_t_cm: Optional[np.ndarray] = None

        if self.use_manual:
            self.R_bc, self.t_bc = self._load_camera_extrinsics()
            self.get_logger().info("Using manual camera extrinsics")
        else:
            self.get_logger().info("ArUco calibration mode: waiting for marker...")

        # Runtime buffers
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_radius_px = float(self.get_parameter("min_radius_px").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.publish_period = 1.0 / float(max(1e-3, self.get_parameter("publish_rate_hz").value))
        self.yellow_lower = np.array(self.get_parameter("yellow_lower_hsv").value, dtype=np.uint8)
        self.yellow_upper = np.array(self.get_parameter("yellow_upper_hsv").value, dtype=np.uint8)
        self.velocity_window = int(self.get_parameter("velocity_window").value)
        self.samples: Deque[Tuple[rclpy.time.Time, np.ndarray]] = deque(maxlen=self.velocity_window)

        # ArUco params
        self.aruco_marker_id = int(self.get_parameter("aruco_marker_id").value)
        self.aruco_marker_length = float(self.get_parameter("aruco_marker_length_m").value)
        self.calibration_frames = int(self.get_parameter("calibration_frames").value)

        # Visualization params
        self.show_visualization = bool(self.get_parameter("show_visualization").value)
        trail_size = int(self.get_parameter("trail_buffer_size").value)
        self.trail_pts: Deque[Optional[Tuple[int, int]]] = deque(maxlen=trail_size)
        self.fixed_rvec: Optional[np.ndarray] = None  # For drawing world axes
        self.fixed_tvec: Optional[np.ndarray] = None
        self.miss_count = 0  # For trail clearing
        self.MISS_FRAMES_KEEP = 25  # Clear trail after this many missed frames
        
        # Get camera info for display
        dev = self.profile.get_device()
        self.camera_name = dev.get_info(rs.camera_info.name)
        self.camera_serial = dev.get_info(rs.camera_info.serial_number)

        # Log intrinsic matrix for debugging
        self.get_logger().info(f"ArUco K (intrinsic matrix):\n{self.color_matrix}")
        self.get_logger().info(f"ArUco marker ID: {self.aruco_marker_id}, size: {self.aruco_marker_length}m")
        if self.show_visualization:
            self.get_logger().info("Visualization enabled - press 'q' to quit, 'c' to recalibrate")
        
        # Async acquisition loop
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.get_logger().info("RealSense ball tracker started.")

    # ------------------------------------------------------------------ #
    # Camera helpers
    # ------------------------------------------------------------------ #
    def _start_pipeline(self) -> rs.pipeline_profile:
        cfg = rs.config()
        serial = self.get_parameter("serial").value
        if serial:
            cfg.enable_device(serial)
        width = int(self.get_parameter("color_width").value)
        height = int(self.get_parameter("color_height").value)
        fps = int(self.get_parameter("fps").value)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            self.get_logger().error("Unable to start RealSense pipeline. Check the USB connection.")
            raise
        return profile

    def _get_intrinsics(self) -> rs.intrinsics:
        stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        return stream.get_intrinsics()

    def _build_intrinsic_matrix(self, intr: rs.intrinsics) -> np.ndarray:
        """Build 3x3 camera intrinsic matrix from RealSense intrinsics."""
        return np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)  # Must be float32 for OpenCV ArUco

    def _load_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        trans = np.array(self.get_parameter("camera_translation_xyz").value, dtype=np.float64).reshape(3, 1)
        quat_xyzw = np.array(self.get_parameter("camera_quaternion_xyzw").value, dtype=np.float64)
        if quat_xyzw.shape[0] != 4:
            raise ValueError("camera_quaternion_xyzw must have 4 elements [x,y,z,w].")
        rotation = R.from_quat(quat_xyzw).as_matrix()
        return rotation, trans

    # ------------------------------------------------------------------ #
    # ArUco calibration
    # ------------------------------------------------------------------ #
    def _calibrate_with_aruco(self, frame: np.ndarray, display_frame: Optional[np.ndarray] = None) -> bool:
        """Detect ArUco marker and accumulate calibration samples."""
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return False

        for i, m_id in enumerate(ids.flatten()):
            if m_id == self.aruco_marker_id:
                dist_coeffs = np.array(self.color_intr.coeffs, dtype=np.float32).reshape(-1, 1)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], self.aruco_marker_length, self.color_matrix, dist_coeffs
                )
                rvec = rvecs[0, 0, :]
                tvec = tvecs[0, 0, :]
                self.calib_rvecs.append(rvec)
                self.calib_tvecs.append(tvec)
                
                # Draw detected marker for visualization
                if display_frame is not None:
                    cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    cv2.drawFrameAxes(display_frame, self.color_matrix, dist_coeffs, 
                                     rvec, tvec, self.aruco_marker_length * 1.5)
                return True
        return False

    def _finish_calibration(self) -> None:
        """Average collected samples and set fixed transform."""
        if len(self.calib_rvecs) < self.calibration_frames:
            return

        avg_rvec = np.mean(self.calib_rvecs, axis=0)
        avg_tvec = np.mean(self.calib_tvecs, axis=0)

        self.fixed_R_cm, _ = cv2.Rodrigues(avg_rvec)
        self.fixed_t_cm = avg_tvec.reshape(3, 1)
        
        # Store for visualization
        self.fixed_rvec = avg_rvec
        self.fixed_tvec = avg_tvec

        # For base_link transform: marker frame is the base, so camera->marker = camera->base
        self.R_bc = self.fixed_R_cm.T
        self.t_bc = -self.fixed_R_cm.T @ self.fixed_t_cm

        self.world_calibrated = True
        self.get_logger().info(
            f"ArUco calibration complete! Fixed transform: t={self.fixed_t_cm.flatten()}"
        )

    # ------------------------------------------------------------------ #
    # Processing loop
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        last_publish = 0.0
        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                bgr = np.asanyarray(color_frame.get_data())
            except RuntimeError as exc:
                self.get_logger().error(f"RealSense stream error: {exc}")
                continue

            # Create display frame for visualization
            display_frame = bgr.copy() if self.show_visualization else None

            # Calibration phase
            if not self.world_calibrated:
                if self._calibrate_with_aruco(bgr, display_frame):
                    if len(self.calib_rvecs) % 10 == 0:
                        self.get_logger().info(
                            f"Calibrating: {len(self.calib_rvecs)}/{self.calibration_frames} samples"
                        )
                    self._finish_calibration()
                else:
                    # Log when ArUco marker is not detected (every 60 frames to avoid spam)
                    if not hasattr(self, '_no_marker_count'):
                        self._no_marker_count = 0
                    self._no_marker_count += 1
                    if self._no_marker_count % 60 == 1:
                        self.get_logger().info(
                            f"ArUco marker (ID={self.aruco_marker_id}) not detected. "
                            f"Please show the marker to the camera for calibration."
                        )
                
                # Show calibration progress visualization
                if self.show_visualization and display_frame is not None:
                    self._draw_calibration_progress(display_frame)
                    self._show_frame(display_frame)
                continue

            # Rate limiting
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - last_publish < self.publish_period:
                # Still show visualization even when rate limited
                if self.show_visualization:
                    self._draw_tracking_overlay(display_frame, None, None)
                    self._show_frame(display_frame)
                continue
            last_publish = now_sec

            # Ball detection
            detection, ball_info = self._detect_ball_with_info(bgr, depth_frame)
            
            if detection is None:
                self.samples.clear()
                self.miss_count += 1
                if self.miss_count > self.MISS_FRAMES_KEEP:
                    self.trail_pts.appendleft(None)
                if self.show_visualization:
                    self._draw_tracking_overlay(display_frame, None, None)
                    self._show_frame(display_frame)
                continue

            self.miss_count = 0
            point_cam = detection
            point_base = self.R_bc @ point_cam.reshape(3, 1) + self.t_bc
            pos = point_base.reshape(3)
            now = self.get_clock().now()
            vel = self._estimate_velocity(now, pos)
            self._publish(now, pos, vel)
            
            # Visualization
            if self.show_visualization and display_frame is not None:
                self._draw_tracking_overlay(display_frame, ball_info, pos)
                self._show_frame(display_frame)

    def _detect_ball(self, bgr: np.ndarray, depth: rs.depth_frame) -> Optional[np.ndarray]:
        result, _ = self._detect_ball_with_info(bgr, depth)
        return result
    
    def _detect_ball_with_info(self, bgr: np.ndarray, depth: rs.depth_frame) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Detect ball and return both 3D point and visualization info."""
        blurred = cv2.GaussianBlur(bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < self.min_radius_px:
            return None, None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        depth_m = depth.get_distance(cx, cy)
        if depth_m <= 0 or depth_m > self.max_depth_m:
            return None, None

        point_cam = np.array(
            rs.rs2_deproject_pixel_to_point(self.color_intr, [cx, cy], depth_m),
            dtype=np.float64
        )
        
        ball_info = {
            'center': (cx, cy),
            'circle_center': (int(x), int(y)),
            'radius': int(radius),
        }
        return point_cam, ball_info

    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    def _draw_calibration_progress(self, frame: np.ndarray) -> None:
        """Draw calibration progress bar on frame."""
        current_samples = len(self.calib_rvecs)
        progress = current_samples / self.calibration_frames
        
        # Draw loading bar
        bar_x, bar_y, bar_w, bar_h = 160, 240, 320, 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x + 5, bar_y + 5), 
                     (bar_x + 5 + int((bar_w - 10) * progress), bar_y + bar_h - 5), 
                     (0, 255, 0), -1)
        
        status_text = f"CALIBRATING WORLD: {current_samples}/{self.calibration_frames}"
        cv2.putText(frame, status_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Camera info
        cv2.putText(frame, f"{self.camera_name} | S/N: {self.camera_serial}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_tracking_overlay(self, frame: np.ndarray, ball_info: Optional[dict], 
                               pos_world: Optional[np.ndarray]) -> None:
        """Draw tracking visualization on frame."""
        dist_coeffs = np.array(self.color_intr.coeffs, dtype=np.float32).reshape(-1, 1)
        
        # Draw world axes (fixed marker position)
        if self.fixed_rvec is not None and self.fixed_tvec is not None:
            try:
                cv2.drawFrameAxes(frame, self.color_matrix, dist_coeffs,
                                 self.fixed_rvec, self.fixed_tvec, self.aruco_marker_length * 1.5)
                cv2.putText(frame, "WORLD LOCKED", (450, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception:
                pass
        
        # Draw ball detection
        if ball_info is not None:
            cx, cy = ball_info['center']
            circle_x, circle_y = ball_info['circle_center']
            radius = ball_info['radius']
            
            # Draw enclosing circle and center
            cv2.circle(frame, (circle_x, circle_y), radius, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Add to trail
            self.trail_pts.appendleft((cx, cy))
            
            # Draw world position (in meters)
            if pos_world is not None:
                txt = f"World: {pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f} m"
                cv2.putText(frame, txt, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw trail
        for i in range(1, len(self.trail_pts)):
            if self.trail_pts[i - 1] is None or self.trail_pts[i] is None:
                continue
            thickness = int(np.sqrt(self.trail_pts.maxlen / float(i + 1)) * 2.5)
            cv2.line(frame, self.trail_pts[i - 1], self.trail_pts[i], (0, 0, 255), thickness)
        
        # Camera info
        cv2.putText(frame, f"{self.camera_name} | S/N: {self.camera_serial}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _show_frame(self, frame: np.ndarray) -> None:
        """Display frame and handle key presses."""
        cv2.imshow("Ball Tracker (ROS2)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.get_logger().info("Quit requested via visualization window")
            self._stop_evt.set()
            rclpy.shutdown()
        elif key == ord('c'):
            # Force recalibration
            self.world_calibrated = False
            self.calib_rvecs = []
            self.calib_tvecs = []
            self.fixed_rvec = None
            self.fixed_tvec = None
            self.trail_pts.clear()
            self.get_logger().info("Recalibration requested...")

    def _estimate_velocity(self, stamp: rclpy.time.Time, pos: np.ndarray) -> np.ndarray:
        self.samples.append((stamp, pos.copy()))
        if len(self.samples) < 2:
            return np.zeros(3, dtype=np.float64)
        t_prev, p_prev = self.samples[-2]
        dt = (stamp - t_prev).nanoseconds / 1e9
        if dt <= 0:
            return np.zeros(3, dtype=np.float64)
        return (pos - p_prev) / dt

    def _publish(self, stamp: rclpy.time.Time, pos: np.ndarray, vel: np.ndarray) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp.to_msg()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.w = 1.0  # Ball orientation not tracked
        self.pose_pub.publish(pose)

        twist = TwistStamped()
        twist.header = pose.header
        twist.twist.linear.x = float(vel[0])
        twist.twist.linear.y = float(vel[1])
        twist.twist.linear.z = float(vel[2])
        self.twist_pub.publish(twist)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def destroy_node(self) -> bool:
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.show_visualization:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
