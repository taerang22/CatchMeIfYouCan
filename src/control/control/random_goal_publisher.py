#!/usr/bin/env python3
"""
Virtual node that publishes a random goal pose to /kinova/goal_pose.
- Fixed x = 0.576 (catch plane)
- Random y and z within robot workspace
- Waits for subscribers before publishing

Usage:
    ros2 run control random_goal_publisher
    ros2 run control random_goal_publisher --ros-args -p y_min:=-0.3 -p y_max:=0.3 -p z_min:=0.2 -p z_max:=0.6
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import random


class RandomGoalPublisher(Node):
    def __init__(self):
        super().__init__("random_goal_publisher")

        # Parameters for random ranges
        self.declare_parameter("x_fixed", 0.576)
        self.declare_parameter("y_min", -0.35)
        self.declare_parameter("y_max", 0.35)
        self.declare_parameter("z_min", 0.15)
        self.declare_parameter("z_max", 0.65)

        # Get parameter values
        self.x_fixed = self.get_parameter("x_fixed").value
        self.y_min = self.get_parameter("y_min").value
        self.y_max = self.get_parameter("y_max").value
        self.z_min = self.get_parameter("z_min").value
        self.z_max = self.get_parameter("z_max").value

        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, "/kinova/goal_pose", 10)

        # State tracking
        self._published = False
        self._waiting_logged = False

        # Timer to check for subscribers and publish
        self.create_timer(0.1, self._wait_and_publish)

        self.get_logger().info(
            f"RandomGoalPublisher ready.\n"
            f"  x = {self.x_fixed} (fixed)\n"
            f"  y ∈ [{self.y_min}, {self.y_max}]\n"
            f"  z ∈ [{self.z_min}, {self.z_max}]\n"
            f"  Waiting for subscribers on /kinova/goal_pose..."
        )

    def _wait_and_publish(self):
        """Wait for at least one subscriber before publishing."""
        if self._published:
            return

        # Check if there are any subscribers
        num_subs = self.goal_pub.get_subscription_count()
        
        if num_subs == 0:
            if not self._waiting_logged:
                self.get_logger().info("Waiting for control_node to subscribe...")
                self._waiting_logged = True
            return
        
        # We have subscribers! Generate and publish
        self.get_logger().info(f"Found {num_subs} subscriber(s). Publishing goal...")
        
        # Generate random y and z
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)

        # Create goal message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = self.x_fixed
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0  # Identity quaternion

        # Publish
        self.goal_pub.publish(msg)
        self._published = True

        self.get_logger().info(
            f"[PUBLISHED] Random Goal Pose:\n"
            f"  x = {self.x_fixed:.4f}\n"
            f"  y = {y:.4f}\n"
            f"  z = {z:.4f}"
        )

        # Shutdown after a short delay
        self.get_logger().info("Goal published successfully! Shutting down in 1 second...")
        self.create_timer(1.0, self._shutdown)

    def _shutdown(self):
        self.get_logger().info("Shutting down...")
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = RandomGoalPublisher()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

