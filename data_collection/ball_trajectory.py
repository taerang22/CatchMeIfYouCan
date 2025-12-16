#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import pickle
from pathlib import Path
import datetime

class BallTrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('ball_trajectory_recorder')

        # --- Parameters ---
        self.declare_parameter('ball_topic', '/vicon/ball/ball/pose')
        self.ball_topic = self.get_parameter('ball_topic').value

        # --- Data storage ---
        self.positions = []
        self.timestamps = []

        # --- Subscriber ---
        self.sub = self.create_subscription(
            PoseStamped,
            self.ball_topic,
            self.ball_callback,
            10
        )

        self.get_logger().info(f"Subscribed to {self.ball_topic}")

    def ball_callback(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.positions.append(pos)
        self.timestamps.append(self.get_clock().now().nanoseconds * 1e-9)
        self.get_logger().info(f"Ball position: {pos}")

    def destroy_node(self):
    # --- Save trajectory when shutting down ---
        if self.positions:
            save_dir = Path.cwd() / 'ball_traj_data'   # Save inside Tae/ball_traj_data
            save_dir.mkdir(parents=True, exist_ok=True)  # Make folder if it doesn't exist
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as pickle
            out_path = save_dir / f'ball_trajectory_{ts}.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'positions': np.array(self.positions),
                    'timestamps': np.array(self.timestamps),
                }, f)

            self.get_logger().info(f"Trajectory saved to {out_path}")

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BallTrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
