#!/usr/bin/env python3
"""
Kinova MPC Controller Node

Subscribes to /kinova/joint_vel_cmd (std_msgs/Float64MultiArray) from /mpc_catch_node
and sends joint velocity commands to the Kinova robot using the Kortex API.

Uses HIGH-LEVEL SERVOING (SINGLE_LEVEL_SERVOING) for joint velocity control.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Hardcoded robot connection info
ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"

# Conversion factor
RAD2DEG = 180.0 / np.pi

# Import KinovaHelper from the same package
from .kinova_helper import KinovaHelper
try:
    from kortex_api.Exceptions.KServerException import KServerException
except Exception:
    KServerException = Exception


class KinovaMPCController(Node):
    """
    Subscribes to /kinova/joint_vel_cmd (Float64MultiArray) and sends joint velocity
    commands to the Kinova robot via send_joint_speeds().
    
    Uses HIGH-LEVEL SERVOING (SINGLE_LEVEL_SERVOING).
    """

    def __init__(self):
        super().__init__("kinova_mpc_controller")

        # ====== Parameters ======
        self.declare_parameter('robot_ip', ROBOT_IP)
        self.declare_parameter('username', USERNAME)
        self.declare_parameter('password', PASSWORD)
        self.declare_parameter('joint_cmd_topic', '/kinova/joint_vel_cmd')

        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        username = self.get_parameter('username').get_parameter_value().string_value
        password = self.get_parameter('password').get_parameter_value().string_value
        joint_cmd_topic = self.get_parameter('joint_cmd_topic').get_parameter_value().string_value

        # ====== Connect to Kinova ======
        self.get_logger().info(f"Connecting to Kinova at {robot_ip}...")
        self.helper = KinovaHelper(ip=robot_ip, username=username, password=password)
        self.get_logger().info(f"Connected to Kinova at {robot_ip}")

        # ====== Ensure SINGLE_LEVEL_SERVOING mode (high-level) ======
        self.get_logger().info("Setting SINGLE_LEVEL_SERVOING mode (high-level)...")
        self.helper.set_servoing_mode(self.helper.SINGLE_LEVEL_SERVOING)

        # ====== Subscriber ======
        self.sub_joint_cmd = self.create_subscription(
            Float64MultiArray,
            joint_cmd_topic,
            self.joint_cmd_callback,
            10
        )

        self.get_logger().info(f"Kinova MPC Controller started. Subscribing to [{joint_cmd_topic}]")

        # Clean shutdown
        rclpy.get_default_context().on_shutdown(self._on_shutdown)

    def joint_cmd_callback(self, msg: Float64MultiArray):
        """
        Callback for /kinova/joint_vel_cmd.
        Receives joint velocities (rad/s) from MPC and converts to deg/s for the robot.
        """
        # Convert rad/s to deg/s
        velocities_deg = [v * RAD2DEG for v in msg.data]

        try:
            self.helper.send_joint_speeds(velocities_deg)
        except KServerException as e:
            self.get_logger().error(f"Joint velocity command error: {e}")

    def _on_shutdown(self):
        """Clean shutdown: stop robot and close connection."""
        try:
            self.get_logger().info("Shutting down Kinova MPC Controller...")
            
            # Send zero velocities to stop the robot
            n_joints = 7  # Kinova Gen3 has 7 joints
            self.helper.send_joint_speeds([0.0] * n_joints)
            
            # Close connection
            if hasattr(self.helper, "close"):
                self.helper.close()
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = KinovaMPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
