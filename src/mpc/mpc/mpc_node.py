#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC-based Cartesian catch controller for Kinova Gen3 (joint-velocity version).

- 공 예측: /kinova/goal_pose (PoseStamped, p_hit + t_hit)
- 현재 EEF 상태: /kinova/eef_pose, /kinova/eef/vel
- 현재 조인트 상태: joint_state_topic (JointState)

출력:
- joint_cmd_topic : std_msgs/Float64MultiArray (joint velocity command)
- (옵션) eef_twist_topic : geometry_msgs/Twist (디버그용)

모든 튜닝 파라미터/토픽 이름은 config/mpc_config.yaml에서 관리.
"""

import os
from pathlib import Path

import numpy as np
# Fix for numpy compatibility with transforms3d (np.float removed in numpy 1.24+)
if not hasattr(np, "float"):
    np.float = float

import yaml
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Twist
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import transforms3d.quaternions as tq 

# Drake imports for kinematics
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context

# Adjust this import to match your actual package structure
from mpc.cartesian_mpc import CartesianCatchMPC


def ros_time_to_sec(t: Time) -> float:
    """Convert ROS Time to float seconds."""
    return float(t.sec) + float(t.nanosec) * 1e-9


class MPCCatchNode(Node):
    def __init__(self):
        super().__init__("mpc_catch_node")

        # ------------------------------------------------------
        # 1) Load YAML config
        # ------------------------------------------------------
        # config_file 파라미터 (기본: "mpc_config.yaml")
        self.declare_parameter("config_file", "mpc_config.yaml")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value

        # 현재 파일 기준으로 config 디렉토리 안에 있는 yaml을 읽도록 설정
        current_dir = Path(__file__).resolve().parent
        config_path = current_dir / "config" / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.get_logger().info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ------------------------------------------------------
        # 2) Extract config sections
        # ------------------------------------------------------
        cfg_control = cfg.get("control", {})
        cfg_mpc = cfg.get("mpc", {})
        cfg_topics = cfg.get("topics", {})
        cfg_robot = cfg.get("robot", {})
        cfg_debug = cfg.get("debug", {})

        # Control
        self.control_dt = float(cfg_control.get("control_dt", 0.01))

        # MPC parameters
        a_max = float(cfg_mpc.get("a_max", 3.0))
        dt_nominal = float(cfg_mpc.get("dt_nominal", 0.05))
        N_max = int(cfg_mpc.get("N_max", 40))
        w_terminal = float(cfg_mpc.get("w_terminal", 200.0))
        w_control = float(cfg_mpc.get("w_control", 1.0))
        min_T = float(cfg_mpc.get("min_T", 0.25))
        max_T = float(cfg_mpc.get("max_T", 1.0))

        # Topics
        self.topic_eef_pose = cfg_topics.get("eef_pose", "/kinova/eef_pose")
        self.topic_eef_vel = cfg_topics.get("eef_vel", "/kinova/eef/vel")
        self.topic_goal_pose = cfg_topics.get("goal_pose", "/kinova/goal_pose")
        self.topic_joint_state = cfg_topics.get("joint_state", "/kinova/joint_states")
        self.topic_joint_cmd = cfg_topics.get("joint_cmd", "/kinova/joint_vel_cmd")
        self.topic_eef_twist = cfg_topics.get("eef_twist_cmd", "/kinova/eef_twist_cmd")

        # Robot / IK
        urdf_path_raw = cfg_robot.get("urdf_path", "")
        # Resolve URDF path relative to the config directory
        if urdf_path_raw and not os.path.isabs(urdf_path_raw):
            self.urdf_path = str(current_dir / urdf_path_raw)
        else:
            self.urdf_path = str(urdf_path_raw)
        self.joint_names = list(cfg_robot.get("joint_names", []))
        self.lam = float(cfg_robot.get("jacobian_damping_lambda", 1e-4))
        self.v_max = float(cfg_robot.get("v_max", 0.8))
        self.ang_kp = float(cfg_robot.get("ang_kp", 4.0))
        self.ang_w_max = float(cfg_robot.get("ang_w_max", 2.0))

        # Debug / behavior
        self.debug = bool(cfg_debug.get("enable", True))
        self.debug_every = int(cfg_debug.get("debug_every", 20))
        self.publish_cartesian_twist = bool(cfg_debug.get("publish_cartesian_twist", False))

        # ------------------------------------------------------
        # 3) MPC object
        # ------------------------------------------------------
        self.mpc = CartesianCatchMPC(
            dt_sim=self.control_dt,
            N_max=N_max,
            a_max=a_max,
            w_terminal=w_terminal,
            w_control=w_control,
            dt_nominal=dt_nominal,
            min_T=min_T,
            max_T=max_T,
        )

        # ------------------------------------------------------
        # 4) States & flags
        # ------------------------------------------------------
        self.have_eef_pose = False
        self.have_eef_vel = False
        self.have_goal = False
        self.have_joint_state = False

        # EEF state
        self.p_eef = np.zeros(3)
        self.v_eef = np.zeros(3)
        self.q_eef = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]

        # Ball target
        self.p_hit = np.zeros(3)
        self.goal_active = False  # goal_pose를 받았는지 여부 (발행 제어용)
        
        # 거리/속도 기반 정지 조건
        self.distance_threshold = 0.05  # 5cm 이내 거리에서 정지
        self.velocity_threshold = 0.05  # 5cm/s 이하 속도에서 정지

        # Joint state
        self.q = None  # np array (n_joints,)
        self.joint_name_to_idx = {}

        # Debug counter
        self.debug_counter = 0

        # ------------------------------------------------------
        # 5) Drake Kinematics setup
        # ------------------------------------------------------
        self.plant = MultibodyPlant(time_step=0.0) # time_step=0.0 for kinematics only
        self.parser = Parser(self.plant)
        
        if not self.urdf_path or not os.path.exists(self.urdf_path):
            raise FileNotFoundError(
                f"URDF file not found at path specified in config: {self.urdf_path}"
            )
        
        self.get_logger().info(f"Loading URDF from {self.urdf_path}")
        self.parser.AddModels(self.urdf_path)
        self.plant.Finalize()

        # Create a context for calculations
        self.plant_context: Context = self.plant.CreateDefaultContext()

        # Get the end-effector frame. URDF의 end-effector link 이름을 정확히 기입해야 합니다.
        # 예시: "end_effector_link". 실제 URDF 파일의 링크 이름으로 변경하세요.
        eef_link_name = "end_effector_link" 
        self.eef_frame = self.plant.GetFrameByName(eef_link_name)
        self.world_frame = self.plant.world_frame()
        self.robot_model_instance = self.plant.GetModelInstanceByName("gen3")

        # ------------------------------------------------------
        # 5) ROS interfaces
        # ------------------------------------------------------
        # Subscribers
        self.sub_eef_pose = self.create_subscription(
            Pose,
            self.topic_eef_pose,
            self.cb_eef_pose,
            10,
        )
        self.sub_eef_vel = self.create_subscription(
            Twist,
            self.topic_eef_vel,
            self.cb_eef_vel,
            10,
        )
        self.sub_goal = self.create_subscription(
            PoseStamped,
            self.topic_goal_pose,
            self.cb_goal_pose,
            10,
        )
        self.sub_joint_state = self.create_subscription(
            JointState,
            self.topic_joint_state,
            self.cb_joint_state,
            10,
        )

        # Publishers
        self.pub_joint_cmd = self.create_publisher(
            Float64MultiArray,
            self.topic_joint_cmd,
            10,
        )
        self.pub_eef_cmd = self.create_publisher(  # optional (debug)
            Twist,
            self.topic_eef_twist,
            10,
        )

        # Timer
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # Log basic info
        self.get_logger().info(
            f"MPC Catch Node started. YAML config: {config_path}"
        )
        self.get_logger().info(
            f"Publishing joint velocities to [{self.topic_joint_cmd}]"
        )
        if self.publish_cartesian_twist:
            self.get_logger().info(
                f"Also publishing EEF Twist to [{self.topic_eef_twist}]"
            )
        if self.debug:
            self.get_logger().info(
                f"Debug enabled, debug_every = {self.debug_every} steps"
            )

    # ==================== Callbacks ====================

    def cb_eef_pose(self, msg: Pose):
        self.p_eef[:] = [msg.position.x, msg.position.y, msg.position.z]
        self.q_eef[:] = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        self.have_eef_pose = True

    def cb_eef_vel(self, msg: Twist):
        self.v_eef[:] = [msg.linear.x, msg.linear.y, msg.linear.z]
        self.have_eef_vel = True

    def cb_goal_pose(self, msg: PoseStamped):
        self.p_hit[:] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        self.goal_active = True  # 새 goal 수신 → 제어 활성화
        self.have_goal = True

        if self.debug:
            self.get_logger().info(
                f"[GOAL] 새로운 goal 수신! p_hit = {self.p_hit}"
            )

    def cb_joint_state(self, msg: JointState):
        """
        joint_names가 비어있으면 첫 콜백에서 msg.name 전체를 joint_names로 사용.
        그렇지 않으면 config의 joint_names 순서대로 q를 채움.
        """
        if self.q is None:
            if self.joint_names:
                self.joint_name_to_idx = {
                    name: i for i, name in enumerate(self.joint_names)
                }
                self.q = np.zeros(len(self.joint_names), dtype=float)
            else:
                # joint_names가 config에 없으면 JointState.name 전체 사용
                self.joint_names = list(msg.name)
                self.joint_name_to_idx = {
                    name: i for i, name in enumerate(self.joint_names)
                }
                self.q = np.zeros(len(self.joint_names), dtype=float)
                self.get_logger().warn(
                    "Config 'robot.joint_names' empty. Using JointState.name as joint set."
                )

        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        for name, idx in self.joint_name_to_idx.items():
            if name in name_to_pos:
                self.q[idx] = name_to_pos[name]

        self.have_joint_state = True

    # ==================== Orientation helpers ====================

    def compute_desired_orientation(self) -> np.ndarray:
        """
        EEF local z-axis가 p_hit 방향을 보도록 하는 목표 quaternion q_des 계산.
        반환: [x, y, z, w]
        """
        dir_vec = self.p_hit - self.p_eef
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return self.q_eef.copy()

        z_des = dir_vec / norm
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_des, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])

        x_des = np.cross(up, z_des)
        x_des /= np.linalg.norm(x_des)
        y_des = np.cross(z_des, x_des)

        R = np.column_stack((x_des, y_des, z_des))
        q_wxyz = tq.mat2quat(R)
        q_des = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        return q_des

    def compute_angular_velocity(self, q_des: np.ndarray) -> np.ndarray:
        """
        현재 q_eef에서 q_des로 회전하기 위한 각속도 w_cmd 계산.
        """
        q_curr_wxyz = np.array(
            [self.q_eef[3], self.q_eef[0], self.q_eef[1], self.q_eef[2]]
        )
        q_des_wxyz = np.array(
            [q_des[3], q_des[0], q_des[1], q_des[2]]
        )

        q_curr_inv = tq.qconjugate(q_curr_wxyz)
        q_err = tq.qmult(q_des_wxyz, q_curr_inv)

        angle = 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
        if angle < 1e-4:
            return np.zeros(3)

        axis = q_err[1:4] / np.sin(angle / 2.0)
        w_cmd = self.ang_kp * angle * axis

        norm_w = np.linalg.norm(w_cmd)
        if norm_w > self.ang_w_max:
            w_cmd *= self.ang_w_max / (norm_w + 1e-8)

        return w_cmd

    # ==================== Jacobian-based IK ====================

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the 6 x n Jacobian J(q) such that [v; w] = J(q) * qdot.
        This uses the pydrake library for high-performance computation.
        """
        # Set the joint positions in the Drake context
        self.plant.SetPositions(self.plant_context, self.robot_model_instance, q)

        # Calculate the spatial velocity Jacobian
        # J_v_WE = J_v_W_E(q)
        J = self.plant.CalcJacobianSpatialVelocity(self.plant_context, 
                                                   with_respect_to=self.plant.GetVelocitiesFrame(self.robot_model_instance), 
                                                   frame_E=self.eef_frame, 
                                                   p_Eo=np.zeros(3), # Jacobian at the frame origin
                                                   frame_W=self.world_frame, 
                                                   frame_v=self.world_frame)
        return J

    def compute_qdot_from_twist(self, v_des: np.ndarray, w_cmd: np.ndarray):
        """
        원하는 Cartesian twist [v_des; w_cmd]로부터 damped least-squares IK로 qdot 계산.

            qdot = J^T (J J^T + λ I)^(-1) * v6
        """
        if self.q is None or not self.have_joint_state:
            if self.debug:
                self.get_logger().warn(
                    "Joint state not available yet; skipping joint command."
                )
            return None

        v6 = np.concatenate([v_des.reshape(3,), w_cmd.reshape(3,)], axis=0)
        if not np.all(np.isfinite(v6)):
            if self.debug:
                self.get_logger().warn("[IK] v6 has NaN/Inf; skipping.")
            return None

        try:
            J = self.compute_jacobian(self.q)  # (6, n)
        except NotImplementedError as e:
            self.get_logger().error(str(e))
            return None

        if J.shape[0] != 6:
            self.get_logger().error(f"Jacobian must be 6 x n, got {J.shape}.")
            return None

        JJt = J @ J.T
        A = JJt + self.lam * np.eye(6)

        cond_A = np.linalg.cond(A)
        if cond_A > 1e12:
            if self.debug:
                self.get_logger().warn(
                    f"[IK] A ill-conditioned (cond={cond_A:.2e}); skipping."
                )
            return None

        qdot = J.T @ np.linalg.solve(A, v6)
        return qdot

    # ==================== Main control loop ====================

    def control_loop(self):
        if not (self.have_eef_pose and self.have_eef_vel):
            return

        # goal이 없으면 제어하지 않음
        if not self.have_goal:
            return

        # --- 거리/속도 기반 정지 조건 확인 ---
        dist_to_target = np.linalg.norm(self.p_hit - self.p_eef)
        vel_magnitude = np.linalg.norm(self.v_eef)
        
        # 목표에 충분히 가까워졌으면 zero command 발행 후 goal 비활성화
        if dist_to_target < self.distance_threshold or vel_magnitude < self.velocity_threshold:
            if self.pub_joint_cmd is not None and self.q is not None:
                zero_msg = Float64MultiArray()
                zero_msg.data = [0.0] * len(self.joint_names)
                self.pub_joint_cmd.publish(zero_msg)

            if self.publish_cartesian_twist:
                stop = Twist()
                self.pub_eef_cmd.publish(stop)

            if self.debug:
                self.get_logger().info(
                    f"[CTRL] Goal reached! dist={dist_to_target:.6f}m, "
                    f"vel={vel_magnitude:.6f}m/s → 공 잡음 완료."
                )
            
            self.goal_active = False
            self.debug_counter += 1
            return

        # --- 1) MPC: Cartesian acceleration (t_hit 제거) ---
        a_cmd, _ = self.mpc.solve(
            p_hit=self.p_hit,
            p_eef=self.p_eef,
            v_eef=self.v_eef,
            p_ref=None,
        )

        # --- 2) Integrate to get desired linear velocity ---
        v_des = self.v_eef + a_cmd * self.control_dt
        # Saturate the desired velocity based on its norm
        v_norm = np.linalg.norm(v_des)
        if v_norm > self.v_max:
            v_des = v_des * (self.v_max / v_norm)

        # --- 3) Desired orientation ---
        q_des = self.compute_desired_orientation()

        # --- 4) Angular velocity to track q_des ---
        w_cmd = self.compute_angular_velocity(q_des)

        # --- 5) IK: Cartesian twist → joint velocity ---
        qdot_cmd = self.compute_qdot_from_twist(v_des, w_cmd)

        # Debug print
        if self.debug:
            if (self.debug_counter % max(self.debug_every, 1) == 0):
                self.get_logger().info(
                    "===== MPC DEBUG =====\n"
                    f"dist_to_target = {dist_to_target:.6f} m\n"
                    f"vel_magnitude  = {vel_magnitude:.6f} m/s\n"
                    f"goal_active    = {self.goal_active}\n"
                    f"p_eef          = {self.p_eef}\n"
                    f"v_eef          = {self.v_eef}\n"
                    f"p_hit          = {self.p_hit}\n"
                    f"a_cmd          = {a_cmd}\n"
                    f"v_des          = {v_des}\n"
                    f"w_cmd          = {w_cmd}\n"
                    f"qdot_cmd       = {qdot_cmd if qdot_cmd is not None else 'None'}\n"
                    "======================"
                )
            self.debug_counter += 1

        # --- 6) Publish joint velocity command ---
        if qdot_cmd is not None:
            msg = Float64MultiArray()
            msg.data = qdot_cmd.tolist()
            self.pub_joint_cmd.publish(msg)

        # Optional: Publish EEF Twist for logging/debugging
        if self.publish_cartesian_twist:
            cmd_twist = Twist()
            cmd_twist.linear.x = float(v_des[0])
            cmd_twist.linear.y = float(v_des[1])
            cmd_twist.linear.z = float(v_des[2])
            cmd_twist.angular.x = float(w_cmd[0])
            cmd_twist.angular.y = float(w_cmd[1])
            cmd_twist.angular.z = float(w_cmd[2])
            self.pub_eef_cmd.publish(cmd_twist)


def main(args=None):
    rclpy.init(args=args)
    node = MPCCatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
