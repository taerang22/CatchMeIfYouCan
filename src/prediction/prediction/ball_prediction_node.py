#!/usr/bin/env python3
"""
Ball Prediction Node
--------------------
Subscribes:
    /kinova/ball/init_pose   (PoseStamped)
    /kinova/ball/init_twist  (TwistStamped)

Publishes:
    /kinova/goal_pose              (PoseStamped) - predicted catch position
    /kinova/ball/robot_frame_pose  (PoseStamped) - ball position in robot frame
    
Function:
    - Collect N samples of ball position in robot frame
    - Fit parabola: x(t) linear, y(t) linear, z(t) quadratic
    - Predict where ball crosses x = FIXED_X plane
    - Publish goal pose once
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped


class BallPredictionNode(Node):

    def __init__(self):
        super().__init__("ball_prediction_node")

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/kinova/ball/init_pose", self._pose_cb, 10
        )
        self.twist_sub = self.create_subscription(
            TwistStamped, "/kinova/ball/init_twist", self._twist_cb, 10
        )

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, "/kinova/goal_pose", 10)
        self.ball_robot_frame_pub = self.create_publisher(
            PoseStamped, "/kinova/ball/robot_frame_pose", 10
        )

        # Ballistic constants
        self.gravity = -9.81  # m/s^2

        # -----------------------------------------------------
        # ArUco(world) → Robot(base_link) Transform
        # -----------------------------------------------------
        self.t_ar_to_robot = np.array([
            1.591,
            -0.19,
            0.86
        ], dtype=float).reshape(3, 1)

        # Rotation (robot x = aruco z, robot y = aruco x, robot z = aruco y)
        self.R_ar_to_robot = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        # -----------------------------------------------------
        # Prediction parameters
        # -----------------------------------------------------
        self.FIXED_X = 0.576      # Catching plane X coordinate
        self.MIN_Y = -0.4         # Robot workspace Y limits
        self.MAX_Y = 0.4
        self.MIN_Z = 0.10         # Minimum Z (safety)
        self.MAX_Z = 0.75         # Maximum Z (reachable)
        
        self.N_SAMPLES = 3        # Number of samples to collect
        self.RESET_TIMEOUT = 1.0  # Reset buffers after 1s of no data
        
        # Buffers
        self.pos_buffer = []      # Positions in ROBOT frame
        self.t_buffer = []        # Timestamps
        self.last_sample_time = None
        self.goal_published = False  # Only publish once per trajectory
        
        self.get_logger().info(
            f"Ball Prediction Node started.\n"
            f"  Catching plane: x = {self.FIXED_X}\n"
            f"  Workspace: y=[{self.MIN_Y}, {self.MAX_Y}], z=[{self.MIN_Z}, {self.MAX_Z}]\n"
            f"  Samples needed: {self.N_SAMPLES}"
        )

    # ------------------------------------------------------------------ #
    # Transformation
    # ------------------------------------------------------------------ #
    def _transform_to_robot_frame(self, pos_aruco):
        """Transform position from ArUco frame to robot base_link frame."""
        pa = pos_aruco.reshape(3, 1)
        pr = self.R_ar_to_robot @ pa + self.t_ar_to_robot
        return pr.flatten()

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _pose_cb(self, msg: PoseStamped):
        # Extract position in ArUco frame
        pos_aruco = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)

        # Transform to robot frame
        pos_robot = self._transform_to_robot_frame(pos_aruco)

        # Publish transformed position
        self._publish_ball_robot_frame(pos_robot, msg.header.stamp)

        # Get timestamp
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Check for timeout - reset if too long since last sample
        if self.last_sample_time is not None:
            if t - self.last_sample_time > self.RESET_TIMEOUT:
                self._reset_buffers()
        
        self.last_sample_time = t

        # Add to buffer
        self.pos_buffer.append(pos_robot)
        self.t_buffer.append(t)

        self.get_logger().info(
            f"[{len(self.pos_buffer)}/{self.N_SAMPLES}] Robot frame: "
            f"x={pos_robot[0]:.3f}, y={pos_robot[1]:.3f}, z={pos_robot[2]:.3f}"
        )

        # When we have enough samples and haven't published yet, fit and predict
        if len(self.pos_buffer) >= self.N_SAMPLES and not self.goal_published:
            self._fit_and_predict()

    def _twist_cb(self, msg: TwistStamped):
        # Velocity is used for backup prediction if needed
        # Currently not used - parabola fitting is more accurate
        pass

    # ------------------------------------------------------------------ #
    # Publish ball position in robot frame
    # ------------------------------------------------------------------ #
    def _publish_ball_robot_frame(self, pos_robot, stamp):
        """Publish ball position transformed to robot base_link frame."""
        ball_msg = PoseStamped()
        ball_msg.header.stamp = stamp
        ball_msg.header.frame_id = "base_link"
        
        ball_msg.pose.position.x = float(pos_robot[0])
        ball_msg.pose.position.y = float(pos_robot[1])
        ball_msg.pose.position.z = float(pos_robot[2])
        ball_msg.pose.orientation.w = 1.0
        
        self.ball_robot_frame_pub.publish(ball_msg)

    # ------------------------------------------------------------------ #
    # Reset buffers
    # ------------------------------------------------------------------ #
    def _reset_buffers(self):
        self.pos_buffer = []
        self.t_buffer = []
        self.goal_published = False
        self.get_logger().info("Buffers reset for next trajectory.")

    # ------------------------------------------------------------------ #
    # Fit parabola and predict
    # ------------------------------------------------------------------ #
    def _fit_and_predict(self):
        pos = np.array(self.pos_buffer)
        t = np.array(self.t_buffer)

        # Normalize time (start from 0)
        t = t - t[0]

        # ----- Linear fit for x(t) and y(t) -----
        A_lin = np.vstack([t, np.ones_like(t)]).T
        (vx, x0), *_ = np.linalg.lstsq(A_lin, pos[:, 0], rcond=None)
        (vy, y0), *_ = np.linalg.lstsq(A_lin, pos[:, 1], rcond=None)

        # ----- Quadratic fit for z(t) (gravity affects z) -----
        A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
        (az, vz, z0), *_ = np.linalg.lstsq(A_quad, pos[:, 2], rcond=None)

        self.get_logger().info(
            f"Fit results:\n"
            f"  x(t) = {vx:.3f}*t + {x0:.3f}  (vx={vx:.2f} m/s)\n"
            f"  y(t) = {vy:.3f}*t + {y0:.3f}  (vy={vy:.2f} m/s)\n"
            f"  z(t) = {az:.3f}*t² + {vz:.3f}*t + {z0:.3f}"
        )

        # ----- Check if ball is moving toward robot -----
        if vx > -0.1:  # Ball should have negative vx (approaching robot)
            self.get_logger().warn(
                f"Ball not approaching robot (vx={vx:.3f}). Skipping prediction."
            )
            self._reset_buffers()
            return

        # ----- Solve for t when x = FIXED_X -----
        # x(t) = vx*t + x0 = FIXED_X
        # t_hit = (FIXED_X - x0) / vx
        t_hit = (self.FIXED_X - x0) / vx

        if t_hit < 0:
            self.get_logger().warn(f"t_hit={t_hit:.3f}s is negative, ball already passed.")
            self._reset_buffers()
            return

        if t_hit > 2.0:
            self.get_logger().warn(f"t_hit={t_hit:.3f}s too far in future (>2s).")
            self._reset_buffers()
            return

        # ----- Calculate y, z at t_hit -----
        y_hit = vy * t_hit + y0
        z_hit = az * t_hit**2 + vz * t_hit + z0

        self.get_logger().info(
            f"Raw prediction: t_hit={t_hit:.3f}s, y={y_hit:.3f}, z={z_hit:.3f}"
        )

        # ----- Clamp to workspace -----
        y_clamped = np.clip(y_hit, self.MIN_Y, self.MAX_Y)
        z_clamped = np.clip(z_hit, self.MIN_Z, self.MAX_Z)

        if y_clamped != y_hit or z_clamped != z_hit:
            self.get_logger().warn(
                f"Clamped to workspace: y={y_hit:.3f}→{y_clamped:.3f}, z={z_hit:.3f}→{z_clamped:.3f}"
            )

        # ----- Publish goal -----
        self._publish_goal(y_clamped, z_clamped, t_hit)
        self.goal_published = True

    # ------------------------------------------------------------------ #
    # Publish goal pose
    # ------------------------------------------------------------------ #
    def _publish_goal(self, y, z, t_hit):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "base_link"
        
        goal.pose.position.x = self.FIXED_X
        goal.pose.position.y = float(y)
        goal.pose.position.z = float(z)
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"{'='*50}\n"
            f"[GOAL PUBLISHED]\n"
            f"  Position: x={self.FIXED_X:.3f}, y={y:.3f}, z={z:.3f}\n"
            f"  Time to catch: {t_hit:.3f}s\n"
            f"{'='*50}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BallPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
