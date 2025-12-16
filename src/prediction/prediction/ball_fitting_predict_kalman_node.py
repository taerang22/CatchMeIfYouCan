#!/usr/bin/env python3
"""
Ball Prediction Node (Parabola fitting + Kalman Filter + Anomaly Filtering)
--------------------------------------------------------------------------
Subscribes:
    /kinova/ball/init_pose (PoseStamped)

Publishes:
    /kinova/goal_pose (PoseStamped)

Pipeline:
    raw pose → KF smoothing → anomaly rejection → buffer → parabola fitting → predict → publish
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped


# -----------------------------
# Simple 1D Kalman Filter
# -----------------------------
class SimpleKalman1D:
    def __init__(self, Q=0.001, R=0.02):
        self.x = None
        self.P = 1.0
        self.Q = Q
        self.R = R

    def update(self, z):
        if self.x is None:
            self.x = z
            return z

        # prediction
        x_pred = self.x
        P_pred = self.P + self.Q

        # update
        K = P_pred / (P_pred + self.R)

        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


# -----------------------------
# Main node
# -----------------------------
class BallPredictionParabolaKalmanNode(Node):

    def __init__(self):
        super().__init__("ball_prediction_parabola_kf_node")

        # input: ball measurements
        self.pose_sub = self.create_subscription(
            PoseStamped, "/kinova/ball/init_pose", self._pose_cb, 10
        )

        # output: predicted hit position
        self.goal_pub = self.create_publisher(PoseStamped, "/kinova/goal_pose", 10)

        # parabola fitting buffers
        self.N = 10
        self.pos_buffer = []
        self.t_buffer = []

        # catching plane x coordinate
        self.FIXED_X = 0.576

        # Kalman filters for axes
        self.kfx = SimpleKalman1D(Q=0.001, R=0.02)
        self.kfy = SimpleKalman1D(Q=0.001, R=0.02)
        self.kfz = SimpleKalman1D(Q=0.001, R=0.05)

        self.get_logger().info("Ball Prediction Node started.")


    # ----------------------------------------
    # Reset KF + buffers after a prediction
    # ----------------------------------------
    def _reset_buffers(self):
        self.pos_buffer = []
        self.t_buffer = []

        # reset filters as well
        self.kfx = SimpleKalman1D(Q=0.001, R=0.02)
        self.kfy = SimpleKalman1D(Q=0.001, R=0.02)
        self.kfz = SimpleKalman1D(Q=0.001, R=0.05)

        self.get_logger().info("Buffers + KF reset.")


    # ----------------------------------------
    # Outlier check
    # ----------------------------------------
    def _is_anomaly(self, new_pos):
        if len(self.pos_buffer) == 0:
            return False

        last = self.pos_buffer[-1]

        dx = abs(new_pos[0] - last[0])
        dy = abs(new_pos[1] - last[1])
        dz = abs(new_pos[2] - last[2])

        XY_JUMP = 0.80
        Z_JUMP = 1.20

        if dx > XY_JUMP or dy > XY_JUMP or dz > Z_JUMP:
            return True

        return False


    # ----------------------------------------
    # Pose callback
    # ----------------------------------------
    def _pose_cb(self, msg):

        pos_raw = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # timestamp
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # KF smoothing
        pos_filt = np.array([
            self.kfx.update(pos_raw[0]),
            self.kfy.update(pos_raw[1]),
            self.kfz.update(pos_raw[2]),
        ])

        # anomaly rejection
        if self._is_anomaly(pos_filt):
            self.get_logger().warn(f"Rejected outlier: {pos_filt}")
            return

        # push to buffer
        self.pos_buffer.append(pos_filt)
        self.t_buffer.append(t)

        self.get_logger().info(
            f"[{len(self.pos_buffer)}/10] x={pos_filt[0]:.3f}, y={pos_filt[1]:.3f}, z={pos_filt[2]:.3f}"
        )

        if len(self.pos_buffer) == self.N:
            self._fit_and_predict()


    # ----------------------------------------
    # Fit parabola and predict hit point
    # ----------------------------------------
    def _fit_and_predict(self):
        pos = np.array(self.pos_buffer)
        t = np.array(self.t_buffer)

        t = t - t[0]  # normalize time

        # ----- linear fits: x(t), y(t) -----
        A_lin = np.vstack([t, np.ones_like(t)]).T
        (a_x, b_x), *_ = np.linalg.lstsq(A_lin, pos[:, 0], rcond=None)
        (a_y_lin, b_y_lin), *_ = np.linalg.lstsq(A_lin, pos[:, 1], rcond=None)

        # ----- quadratic fit: z(t) -----
        A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
        (A_z, B_z, C_z), *_ = np.linalg.lstsq(A_quad, pos[:, 2], rcond=None)

        # -------------- t_hit 계산 --------------
        if abs(a_x) < 1e-3:
            self.get_logger().warn(f"x velocity too small a_x={a_x:.5f}")
            self._reset_buffers()
            return

        t_hit = (self.FIXED_X - b_x) / a_x

        # sanity check: expected time range
        t_min = t[0]
        t_max = t[-1]

        if not (t_min - 0.05 <= t_hit <= t_max + 0.3):
            self.get_logger().warn(f"t_hit {t_hit:.3f} out of valid range")
            self._reset_buffers()
            return

        # ---------- y,z at hit ----------
        y_hit = a_y_lin * t_hit + b_y_lin
        z_hit = A_z * t_hit**2 + B_z * t_hit + C_z

        # position sanity
        if abs(y_hit) > 2.0 or not (0.0 < z_hit < 2.5):
            self.get_logger().warn(f"Unrealistic hit point y={y_hit:.3f}, z={z_hit:.3f}")
            self._reset_buffers()
            return

        z_hit = max(0.10, z_hit)

        self.get_logger().info(
            f"[HIT] t={t_hit:.3f}, y={y_hit:.3f}, z={z_hit:.3f}"
        )

        self._publish_goal(y_hit, z_hit)
        self._reset_buffers()


    # ----------------------------------------
    # Publish goal pose
    # ----------------------------------------
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
            f"[GOAL SENT] x={self.FIXED_X:.3f}, y={y:.3f}, z={z:.3f}"
        )


# ----------------------------------------
# MAIN
# ----------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = BallPredictionParabolaKalmanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
