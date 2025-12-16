#!/usr/bin/env python3
"""
Ball Prediction Node (Simple Parabola Fitting)
----------------------------------------------
Subscribes:
    /kinova/ball/init_pose   (PoseStamped) - ball position in ArUco/camera frame

Publishes:
    /kinova/goal_pose              (PoseStamped) - predicted catch position in robot frame
    /kinova/ball/robot_frame_pose  (PoseStamped) - ball position transformed to robot frame

Pipeline:
    1. Receive ball pose in ArUco frame
    2. Transform to robot base_link frame
    3. Collect N samples (skip first sample)
    4. Fit parabola: x(t) linear, y(t) linear, z(t) quadratic (gravity)
    5. Predict where ball crosses x = 0.576 plane
    6. Publish goal pose
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped


class BallPredictionNode(Node):

    def __init__(self):
        super().__init__("ball_prediction_node")

        # Subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped, "/kinova/ball/init_pose", self._pose_cb, 10
        )

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, "/kinova/goal_pose", 10)
        self.robot_frame_pub = self.create_publisher(PoseStamped, "/kinova/ball/robot_frame_pose", 10)

        # -----------------------------------------------------
        # ArUco(world) → Robot(base_link) Transform
        # -----------------------------------------------------
        self.t_ar_to_robot = np.array([
            1.591,
            -0.28,
            0.88
        ], dtype=float).reshape(3, 1)

        # Rotation: robot_x = aruco_z, robot_y = aruco_x, robot_z = aruco_y
        self.R_ar_to_robot = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        # -----------------------------------------------------
        # Prediction parameters
        # -----------------------------------------------------
        self.FIXED_X = 0.576      # Catching plane X coordinate
        self.MIN_Z = 0.10         # Minimum Z (safety)
        self.MAX_Z = 0.80         # Maximum Z (reachable)
        self.N_SAMPLES = 5        # Number of samples to collect (excluding first)
        
        # Buffers
        self.pos_buffer = []      # Positions in ROBOT frame
        self.t_buffer = []        # Timestamps
        self.skip_first = True    # Skip very first sample
        
        self.get_logger().info(
            f"Ball Prediction Node started.\n"
            f"  Catching plane: x = {self.FIXED_X}\n"
            f"  Samples needed: {self.N_SAMPLES}"
        )

    # ---------------------------------------------------------
    # Transform ArUco frame → Robot frame
    # ---------------------------------------------------------
    def _transform_to_robot_frame(self, pos_aruco):
        """Transform position from ArUco/camera frame to robot base_link frame."""
        pa = np.array(pos_aruco).reshape(3, 1)
        pr = self.R_ar_to_robot @ pa + self.t_ar_to_robot
        return pr.flatten()

    # ---------------------------------------------------------
    # Publish ball position in robot frame
    # ---------------------------------------------------------
    def _publish_robot_frame_pose(self, pos_robot, stamp):
        """Publish transformed ball position in robot frame."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos_robot[0])
        msg.pose.position.y = float(pos_robot[1])
        msg.pose.position.z = float(pos_robot[2])
        msg.pose.orientation.w = 1.0
        self.robot_frame_pub.publish(msg)

    # ---------------------------------------------------------
    # Pose callback
    # ---------------------------------------------------------
    def _pose_cb(self, msg: PoseStamped):
        # Extract position in ArUco frame
        pos_aruco = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)

        # Transform to robot frame
        pos_robot = self._transform_to_robot_frame(pos_aruco)

        # Publish transformed position in robot frame
        self._publish_robot_frame_pose(pos_robot, msg.header.stamp)

        # Timestamp
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Skip the very first sample (often noisy)
        if self.skip_first:
            self.skip_first = False
            self.get_logger().info(f"Skipping first sample")
            return

        # Add to buffer
        self.pos_buffer.append(pos_robot)
        self.t_buffer.append(t)

        self.get_logger().info(
            f"[{len(self.pos_buffer)}/{self.N_SAMPLES}] Robot frame: "
            f"x={pos_robot[0]:.3f}, y={pos_robot[1]:.3f}, z={pos_robot[2]:.3f}"
        )

        # When we have enough samples, fit and predict
        if len(self.pos_buffer) >= self.N_SAMPLES:
            self._fit_and_predict()
            self._reset_buffers()

    # ---------------------------------------------------------
    # Reset buffers for next trajectory
    # ---------------------------------------------------------
    def _reset_buffers(self):
        self.pos_buffer = []
        self.t_buffer = []
        self.skip_first = True
        self.get_logger().info("Buffers reset for next trajectory.")

    # ---------------------------------------------------------
    # Fit parabola and predict
    # ---------------------------------------------------------
    def _fit_and_predict(self):
        pos = np.array(self.pos_buffer)
        t = np.array(self.t_buffer)

        # Normalize time (start from 0)
        t = t - t[0]

        # ----- Linear fit for x(t) and y(t) -----
        A_lin = np.vstack([t, np.ones_like(t)]).T
        (a_x, b_x), *_ = np.linalg.lstsq(A_lin, pos[:, 0], rcond=None)
        (a_y, b_y), *_ = np.linalg.lstsq(A_lin, pos[:, 1], rcond=None)

        # ----- Quadratic fit for z(t) (gravity affects z) -----
        A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
        (A_z, B_z, C_z), *_ = np.linalg.lstsq(A_quad, pos[:, 2], rcond=None)

        self.get_logger().info(
            f"Fit results:\n"
            f"  x(t) = {a_x:.4f}*t + {b_x:.4f}\n"
            f"  y(t) = {a_y:.4f}*t + {b_y:.4f}\n"
            f"  z(t) = {A_z:.4f}*t² + {B_z:.4f}*t + {C_z:.4f}"
        )

        # ----- Solve for t when x = FIXED_X -----
        if abs(a_x) < 1e-4:
            self.get_logger().warn(f"Ball not moving in X (a_x={a_x:.5f}), cannot predict.")
            return

        t_hit = (self.FIXED_X - b_x) / a_x

        # Basic sanity check
        if t_hit < -0.1:
            self.get_logger().warn(f"t_hit={t_hit:.3f} is negative, ball already passed.")
            return

        # ----- Calculate y, z at t_hit -----
        y_hit = a_y * t_hit + b_y
        z_hit = A_z * t_hit**2 + B_z * t_hit + C_z

        # Clamp z to safe range
        z_hit = np.clip(z_hit, self.MIN_Z, self.MAX_Z)

        self.get_logger().info(
            f"Prediction: t_hit={t_hit:.3f}s, y={y_hit:.3f}, z={z_hit:.3f}"
        )

        # Publish goal
        self._publish_goal(y_hit, z_hit)

    # ---------------------------------------------------------
    # Publish goal pose
    # ---------------------------------------------------------
    def _publish_goal(self, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = self.FIXED_X
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)

        self.get_logger().info(
            f"[GOAL PUBLISHED] x={self.FIXED_X:.3f}, y={y:.3f}, z={z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BallPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
