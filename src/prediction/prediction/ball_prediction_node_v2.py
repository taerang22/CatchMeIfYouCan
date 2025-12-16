#!/usr/bin/env python3
"""
Ball Prediction Node v2 (Ultra-Light, Physics-Based)
----------------------------------------------------
업데이트 사항:
    - 회귀 제거 (z는 중력 모델 기반 역추정)
    - x,y는 평균 속도 기반
    - sliding window = 6 samples
    - outlier rejection 강화
    - running time 극단적으로 작음 (O(1))
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped


class BallPredictionNode(Node):

    def __init__(self):
        super().__init__("ball_prediction_node_v2")

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/kinova/ball/init_pose", self._pose_cb, 10
        )

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, "/kinova/goal_pose", 10)
        self.ball_robot_frame_pub = self.create_publisher(
            PoseStamped, "/kinova/ball/robot_frame_pose", 10
        )

        # ---- Constants ----
        self.g = -9.81   # Gravity
        self.FIXED_X = 0.576

        self.MIN_Y = -0.4
        self.MAX_Y = 0.4
        self.MIN_Z = 0.10
        self.MAX_Z = 0.75

        self.WINDOW = 6   # Sliding window size
        self.RESET_TIMEOUT = 1.0

        # ---- Transform: ArUco -> Robot ----
        self.t_ar_to_robot = np.array([[1.591], [-0.19], [0.86]])
        self.R_ar_to_robot = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], float)

        # Buffers
        self.pos_buffer = []
        self.t_buffer = []
        self.last_time = None
        self.goal_sent = False

        self.get_logger().info("Ball Prediction v2 started (Ultra-light mode).")

    # ============================================================
    # Transform
    # ============================================================
    def transform_to_robot(self, p):
        """ArUco → Robot frame"""
        pr = self.R_ar_to_robot @ p.reshape(3, 1) + self.t_ar_to_robot
        return pr.flatten()

    # ============================================================
    # Pose callback
    # ============================================================
    def _pose_cb(self, msg):

        # Extract position in ArUco
        p_ar = np.array([msg.pose.position.x,
                         msg.pose.position.y,
                         msg.pose.position.z], float)

        # Transform → robot frame
        p_rb = self.transform_to_robot(p_ar)
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        # Publish robot frame pose
        self.publish_robot_frame(p_rb, stamp)

        # Reset timeout
        if self.last_time is not None and t - self.last_time > self.RESET_TIMEOUT:
            self.reset_buffers()

        self.last_time = t

        # Add to sliding window buffer
        self.pos_buffer.append(p_rb)
        self.t_buffer.append(t)

        if len(self.pos_buffer) > self.WINDOW:
            self.pos_buffer.pop(0)
            self.t_buffer.pop(0)

        # Not enough samples yet
        if len(self.pos_buffer) < self.WINDOW or self.goal_sent:
            return

        # Try prediction
        self.predict()

    # ============================================================
    # Publish transformed ball
    # ============================================================
    def publish_robot_frame(self, p, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.w = 1.0
        self.ball_robot_frame_pub.publish(msg)

    # ============================================================
    # Reset
    # ============================================================
    def reset_buffers(self):
        self.pos_buffer = []
        self.t_buffer = []
        self.goal_sent = False
        self.get_logger().info("Buffers reset.")

    # ============================================================
    # Prediction logic
    # ============================================================
    def predict(self):
        pos = np.array(self.pos_buffer)
        t = np.array(self.t_buffer)
        t0 = t[0]
        t = t - t0   # Normalize

        # -------------------------------------------------------
        # Outlier rejection: x must be decreasing
        # -------------------------------------------------------
        if pos[-1, 0] > pos[0, 0]:
            return  # ignore, ball not approaching

        # -------------------------------------------------------
        # Compute x,y velocities
        # v = Δx / Δt
        # -------------------------------------------------------
        dt = t[-1] - t[0]
        if dt < 1e-6:
            return

        vx = (pos[-1, 0] - pos[0, 0]) / dt
        vy = (pos[-1, 1] - pos[0, 1]) / dt

        if vx > -0.05:
            return

        x0 = pos[0, 0]
        y0 = pos[0, 1]

        # -------------------------------------------------------
        # Solve for vz from z(t) = z0 + vz t + 0.5 g t²
        # Use first two samples for stability
        # -------------------------------------------------------
        z0 = pos[0, 2]
        z1 = pos[1, 2]
        t1 = t[1]

        vz = (z1 - z0 - 0.5 * self.g * t1 * t1) / t1

        # -------------------------------------------------------
        # Solve time to reach x = FIXED_X
        # FIXED_X = x0 + vx * t_hit
        # -------------------------------------------------------
        t_hit = (self.FIXED_X - x0) / vx

        if t_hit < 0 or t_hit > 2.0:
            return

        # -------------------------------------------------------
        # Compute predicted y,z
        # -------------------------------------------------------
        y_hit = y0 + vy * t_hit
        z_hit = z0 + vz * t_hit + 0.5 * self.g * t_hit * t_hit

        # Clamp workspace
        y_hit = float(np.clip(y_hit, self.MIN_Y, self.MAX_Y))
        z_hit = float(np.clip(z_hit, self.MIN_Z, self.MAX_Z))

        # -------------------------------------------------------
        # Publish goal
        # -------------------------------------------------------
        self.publish_goal(y_hit, z_hit, t_hit)
        self.goal_sent = True

    # ============================================================
    # Publish goal
    # ============================================================
    def publish_goal(self, y, z, t_hit):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(self.FIXED_X)
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)

        self.get_logger().info(
            f"[GOAL] x={self.FIXED_X:.3f}, y={y:.3f}, z={z:.3f}, t_hit={t_hit:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BallPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
