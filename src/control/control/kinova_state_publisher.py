import rclpy
import os
import sys
import time
import socket
import subprocess
import math
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

# Automatically add kortex_api path if running in a conda env
site_packages_path = os.path.expanduser('~/miniconda3/envs/kinova/lib/python3.10/site-packages')
if site_packages_path not in sys.path:
    sys.path.append(site_packages_path)

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Session_pb2, Common_pb2
from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport as TransportClientTcp
from kortex_api.SessionManager import SessionManager


def ping_robot(ip_address):
    """Check if robot responds to ICMP ping."""
    while True:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', ip_address],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            print(f"[PING] Robot at {ip_address} is reachable.")
            break
        else:
            print(f"[PING] Robot at {ip_address} not reachable. Retrying...")
            time.sleep(3)


def is_tcp_port_open(ip, port, timeout=1):
    """Check if TCP port is open."""
    try:
        sock = socket.create_connection((ip, port), timeout=timeout)
        sock.close()
        return True
    except socket.error:
        return False


def wait_for_tcp(ip, port=10000, timeout=1):
    """Wait until TCP port is open."""
    while True:
        if is_tcp_port_open(ip, port, timeout):
            print(f"[TCP] Port {port} on {ip} is open.")
            break
        else:
            print(f"[TCP] Port {port} not open on {ip}, retrying...")
            time.sleep(3)


class KinovaStatePublisher(Node):
    """
    Publishes Kinova state:
        - /kinova/eef_pose : geometry_msgs/Pose
            * position [m]
            * orientation quaternion (from XYZ Euler in deg)
        - /kinova/gripper_pos : std_msgs/Float64
            * finger position [0.0 (open) ~ 1.0 (closed)]
        - /kinova/joint_states : sensor_msgs/JointState
            * name: joint names
            * position: joint positions [rad]
            * velocity: joint velocities [rad/s]
            * effort: joint torques [Nm]
        - /kinova/eef/vel : geometry_msgs/Twist
            * linear  [m/s]
            * angular [rad/s]
    """

    # Kinova Gen3 joint names
    JOINT_NAMES = [
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7"
    ]

    def __init__(self):
        super().__init__('kinova_state_publisher_node')

        # Publishers
        self.eef_pose_pub = self.create_publisher(Pose, '/kinova/eef_pose', 10)
        self.gripper_position_pub = self.create_publisher(Float64, '/kinova/gripper_pos', 10)
        self.joint_states_pub = self.create_publisher(JointState, '/kinova/joint_states', 10)
        self.eef_vel_pub = self.create_publisher(Twist, '/kinova/eef/vel', 10)

        self.get_logger().info(
            "[KINOVA] Publishers created: /kinova/eef_pose, /kinova/gripper_pos, "
            "/kinova/joint_states, /kinova/eef/vel"
        )

        # Robot connection credentials
        ip = "192.168.1.10"
        username = "admin"
        password = "admin"

        # Check robot is reachable
        ping_robot(ip)
        wait_for_tcp(ip, 10000)

        # Establish API connection
        self.transport = TransportClientTcp()
        self.router = RouterClient(self.transport, self.error_callback)
        self.transport.connect(ip, 10000)

        # Create session
        self.session_manager = SessionManager(self.router)
        self.session_info = Session_pb2.CreateSessionInfo()
        self.session_info.username = username
        self.session_info.password = password
        self.session_info.session_inactivity_timeout = 60000
        self.session_info.connection_inactivity_timeout = 2000

        print("Creating session...")
        self.session_manager.CreateSession(self.session_info)
        print("Session created.")

        # Create Base and BaseCyclic clients
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

        # Timer callbacks (100 Hz)
        self.create_timer(0.01, self.publish_kinematics)       # pose + velocity
        self.create_timer(0.01, self.publish_gripper_position) # gripper

    def error_callback(self, err):
        self.get_logger().error(f"{err.toString()}")

    def publish_gripper_position(self):
        """Publish gripper finger position as 0.0 ~ 1.0."""
        try:
            request = Base_pb2.GripperRequest()
            request.mode = Base_pb2.GRIPPER_POSITION
            feedback = self.base.GetMeasuredGripperMovement(request)

            if feedback.finger:
                pos = feedback.finger[0].value
            else:
                pos = 0.0

            gripper_msg = Float64()
            gripper_msg.data = pos
            self.gripper_position_pub.publish(gripper_msg)

        except KServerException as ex:
            self.get_logger().error(f"{ex}")

    def publish_kinematics(self):
        """
        Use BaseCyclic.RefreshFeedback() to get:
            - EEF pose (x, y, z, theta_x/y/z)
            - Joint velocities
            - EEF twist (linear & angular velocity)
        """
        try:
            # Read-only cyclic feedback (works in high-level servoing)
            feedback = self.base_cyclic.RefreshFeedback()
            bf = feedback.base  # BaseFeedback shortcut

            # ---------- End-effector pose ----------
            eef_msg = Pose()
            eef_msg.position.x = bf.tool_pose_x
            eef_msg.position.y = bf.tool_pose_y
            eef_msg.position.z = bf.tool_pose_z

            quat = R.from_euler(
                'xyz',
                [bf.tool_pose_theta_x, bf.tool_pose_theta_y, bf.tool_pose_theta_z],
                degrees=True
            ).as_quat()
            eef_msg.orientation.x = quat[0]
            eef_msg.orientation.y = quat[1]
            eef_msg.orientation.z = quat[2]
            eef_msg.orientation.w = quat[3]
            self.eef_pose_pub.publish(eef_msg)

            # ---------- Joint states (name, position, velocity, effort) ----------
            js_msg = JointState()
            js_msg.header.stamp = self.get_clock().now().to_msg()
            js_msg.header.frame_id = "base_link"
            js_msg.name = self.JOINT_NAMES
            
            # Position [rad], velocity [rad/s], effort [Nm]
            js_msg.position = [act.position * math.pi / 180.0 for act in feedback.actuators]
            js_msg.velocity = [act.velocity * math.pi / 180.0 for act in feedback.actuators]
            js_msg.effort = [act.torque for act in feedback.actuators]
            
            self.joint_states_pub.publish(js_msg)

            # ---------- End-effector velocity (Twist) ----------
            vel_msg = Twist()
            # Linear velocity [m/s]
            vel_msg.linear.x = bf.tool_twist_linear_x
            vel_msg.linear.y = bf.tool_twist_linear_y
            vel_msg.linear.z = bf.tool_twist_linear_z

            # Angular velocity [rad/s]
            vel_msg.angular.x = bf.tool_twist_angular_x
            vel_msg.angular.y = bf.tool_twist_angular_y
            vel_msg.angular.z = bf.tool_twist_angular_z

            self.eef_vel_pub.publish(vel_msg)

        except KServerException as ex:
            self.get_logger().error(f"{ex}")

    def destroy_node(self):
        """Gracefully close session and transport when shutting down."""
        try:
            self.session_manager.CloseSession()
            self.router.SetActivationStatus(False)
            self.transport.disconnect()
            print("Session closed and transport disconnected.")
        except Exception as e:
            print("[CLEANUP ERROR]", e)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinovaStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted via Keyboard")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()