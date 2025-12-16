import time, math, threading
from contextlib import contextmanager
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from .transform_utils import *

# ───────────── Kortex transports ─────────────
from kortex_api.TCPTransport  import TCPTransport  as TransportClientTcp
from kortex_api.UDPTransport  import UDPTransport  as TransportClientUdp
from kortex_api.RouterClient  import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.Exceptions.KServerException import KServerException

# ───────────── RPC stubs ─────────────
from kortex_api.autogen.client_stubs.BaseClientRpc          import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc    import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc  import VisionConfigClient   # ← ADD

# ───────────── protobuf messages ─────────────
from kortex_api.autogen.messages import (
    Base_pb2,
    BaseCyclic_pb2,
    Session_pb2,
    VisionConfig_pb2,                 # ← ADD
    DeviceConfig_pb2,                 # ← ADD (vision id search용)
)

# ───────────── Vision resolution table ─────────────
RES_TABLE = {                         # ← ADD
    VisionConfig_pb2.RESOLUTION_320x240:   (320, 240),
    VisionConfig_pb2.RESOLUTION_424x240:   (424, 240),
    VisionConfig_pb2.RESOLUTION_480x270:   (480, 270),
    VisionConfig_pb2.RESOLUTION_640x480:   (640, 480),
    VisionConfig_pb2.RESOLUTION_1280x720:  (1280, 720),
    VisionConfig_pb2.RESOLUTION_1920x1080: (1920, 1080),
}

# ───────────── Servo mode enum ─────────────
UNSPECIFIED_SERVOING_MODE = 0
SINGLE_LEVEL_SERVOING     = 2
LOW_LEVEL_SERVOING        = 3
BYPASS_SERVOING           = 4

# ───────────── Safety limits & helpers ─────────────
MAX_SPEED_LARGE_DEG = 40.0   # joints 1-4  (표 기준 여유값)
MAX_SPEED_SMALL_DEG = 30.0   # joints 5-7
DEG2RAD             = math.pi / 180.0      # ← ADD


class KinovaHelper:
    UNSPECIFIED_SERVOING_MODE = UNSPECIFIED_SERVOING_MODE
    SINGLE_LEVEL_SERVOING     = SINGLE_LEVEL_SERVOING
    LOW_LEVEL_SERVOING        = LOW_LEVEL_SERVOING
    BYPASS_SERVOING           = BYPASS_SERVOING

    def __init__(self, ip="192.168.1.10", username="admin", password="admin",
                 vision_only: bool = False):
        self.ip, self.username, self.password = ip, username, password
        self._vision_only = vision_only

        # -- TCP connection (general commands) ------------------------------
        self.transport = TransportClientTcp()
        self.router    = RouterClient(self.transport, self._error_callback)
        self.transport.connect(ip, 10000)

        self.session_manager = SessionManager(self.router)
        self.session_info = Session_pb2.CreateSessionInfo()
        self.session_info.username  = username
        self.session_info.password  = password
        self.session_info.session_inactivity_timeout    = 60000
        self.session_info.connection_inactivity_timeout = 2000
        self.session_manager.CreateSession(self.session_info)

        self._cmd            = BaseCyclic_pb2.Command()
        self._refresh_thread = None
        self._running        = False

        # -- Clients -------------------------------------------------------
        self.device_manager = DeviceManagerClient(self.router)
        self.vision_cfg     = VisionConfigClient(self.router)                  

        if not vision_only:
            self.base = BaseClient(self.router)

            self.transport_rt = TransportClientUdp()
            self.router_rt    = RouterClient(self.transport_rt, self._error_callback)
            self.transport_rt.connect(ip, 10001)

            self.session_manager_rt = SessionManager(self.router_rt)
            self.session_manager_rt.CreateSession(self.session_info)

            self.base_cyclic   = BaseCyclicClient(self.router_rt)
            self.monitor_thread = None
        else:
            self.base = None
            self.base_cyclic = None
            self.transport_rt = None
            self.monitor_thread = None
            self.servoing_mode_okay = True

    @contextmanager
    def session(self):
        """Usage: `with KinovaHelper(...) as h:`"""
        try:
            yield self
        finally:
            self.close()

    def close(self):
        """Close sessions and disconnect routers/transports."""
        try:
            self.session_manager.CloseSession()
            if hasattr(self, "session_manager_rt"):
                self.session_manager_rt.CloseSession()
        except KServerException:
            pass
        finally:
            self.transport.disconnect()
            if self.transport_rt:
                self.transport_rt.disconnect()

    def get_intrinsics(self, sensor: int, resolution: int | None = None):
        vid = self._vision_id()

        if resolution is not None:
            try:
                prof = VisionConfig_pb2.IntrinsicProfileIdentifier(
                    sensor=sensor, resolution=resolution)
                ip = self.vision_cfg.GetIntrinsicParametersProfile(prof, vid)
                w, h = getattr(ip, "width", RES_TABLE[resolution][0]), \
                       getattr(ip, "height", RES_TABLE[resolution][1])
                return ip, int(w), int(h)
            except Exception:
                pass

        try:
            sid = VisionConfig_pb2.SensorIdentifier(); sid.sensor = sensor
            ip  = self.vision_cfg.GetIntrinsicParameters(sid, vid)
            res = getattr(ip, "resolution",
                          VisionConfig_pb2.RESOLUTION_480x270)
            w, h = RES_TABLE.get(res, (480, 270))
            return ip, w, h
        except Exception as e:
            raise RuntimeError(f"[KinovaHelper] cannot fetch intrinsics: {e}")

    def get_rgb_intrinsics(self):
        return self.get_intrinsics(VisionConfig_pb2.SENSOR_COLOR,
                                   VisionConfig_pb2.RESOLUTION_1280x720)

    def get_depth_intrinsics(self):
        return self.get_intrinsics(VisionConfig_pb2.SENSOR_DEPTH,
                                   VisionConfig_pb2.RESOLUTION_480x270)

    def get_extrinsic_sensor(self, sensor: int):
        vid = self._vision_id()

        try:
            ss = VisionConfig_pb2.SensorSettings(); ss.sensor = sensor
            self.vision_cfg.SetSensorSettings(ss, vid)
            time.sleep(0.05)
        except Exception:
            pass

        ext = self.vision_cfg.GetExtrinsicParameters(vid)
        R = np.array([[ext.rotation.row1.column1, ext.rotation.row1.column2, ext.rotation.row1.column3],
                      [ext.rotation.row2.column1, ext.rotation.row2.column2, ext.rotation.row2.column3],
                      [ext.rotation.row3.column1, ext.rotation.row3.column2, ext.rotation.row3.column3]],
                     dtype=np.float32)
        t = np.array([ext.translation.t_x, ext.translation.t_y, ext.translation.t_z],
                     dtype=np.float32)
        T = np.eye(4, dtype=np.float32);  T[:3, :3] = R;  T[:3, 3] = t
        return T

    _T_DEPTH2COLOR_MANUAL = np.array([[1, 0, 0, -0.02706],
                                      [0, 1, 0, -0.00997],
                                      [0, 0, 1, -0.004706],
                                      [0, 0, 0, 1      ]], dtype=np.float32)

    def get_depth2color(self, prefer_firmware=True, use_flip=True):
        flip = np.diag([1, -1, 1, 1], k=0).astype(np.float32) if use_flip else np.eye(4)
        if prefer_firmware:
            try:
                Td = self.get_extrinsic_sensor(VisionConfig_pb2.SENSOR_DEPTH)
                Tc = self.get_extrinsic_sensor(VisionConfig_pb2.SENSOR_COLOR)
                if not np.allclose(Td, Tc, atol=1e-5):
                    return flip @ (Tc @ np.linalg.inv(Td))
            except Exception as e:
                print("[KinovaHelper] firmware extrinsic failed:", e)

        print("[KinovaHelper] using MANUAL Depth→Color transform")
        return flip @ self._T_DEPTH2COLOR_MANUAL.copy()

    def _vision_id(self):
        if hasattr(self, "_vid"):
            return self._vid
        handles = [h for h in self.device_manager.ReadAllDevices().device_handle
                   if h.device_type == DeviceConfig_pb2.VISION]
        if not handles:
            raise RuntimeError("No vision module found")
        self._vid = handles[0].device_identifier
        return self._vid

    def _servoing_monitor_loop(self):
        while not self._vision_only:
            try:
                mode_info = self.base.GetServoingMode()
                self.servoing_mode_okay = \
                    (mode_info.servoing_mode == Base_pb2.SINGLE_LEVEL_SERVOING)
                if not self.servoing_mode_okay:
                    print("Servoing mode lost. Recovering...")
                    self._recover_control()
            except KServerException as e:
                if "SESSION_NOT_IN_CONTROL" in str(e):
                    print("Session lost. Recovering...")
                    self._recover_control()
                else:
                    print("Error:", e)
            time.sleep(0.1)   # 10 Hz check

    def set_servoing_mode(self, mode: int):
        """Set the base servo-mode (SINGLE_LEVEL, LOW_LEVEL, etc.)."""
        info = Base_pb2.ServoingModeInformation()
        info.servoing_mode = mode
        try:
            self.base.SetServoingMode(info)
            print(f"[KinovaHelper] Servoing mode set → {mode}")
            time.sleep(0.3)  # short delay for stability
        except KServerException as e:
            print("[KinovaHelper] set_servoing_mode error:", e)

    def set_servo_mode(self, mode: int, wait_ok: bool = False):  # ← backward-compat alias
        """
        Old API wrapper. Internally calls set_servoing_mode().
        If wait_ok=True, poll until the mode is confirmed by the controller.
        """
        self.set_servoing_mode(mode)          # reuse new name
        if wait_ok:
            for _ in range(50):               # max ~5 s
                cur = self.get_servoing_mode()
                if cur == mode:
                    return
                time.sleep(0.1)

    def get_servoing_mode(self) -> int | None:
        """Return the current servo-mode enum value (or None on error)."""
        try:
            return self.base.GetServoingMode().servoing_mode
        except KServerException as e:
            print("[KinovaHelper] get_servoing_mode error:", e)
            return None

    def _recover_control(self):
        try:
            self.session_manager.CloseSession()
            self.session_manager.CreateSession(self.session_info)
            sm = Base_pb2.ServoingModeInformation()
            sm.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(sm)
            self.servoing_mode_okay = True
            print("Re-established control.")
        except KServerException as e:
            print("Recovery Failed:", e)
            self.servoing_mode_okay = False

    def ensure_control_and_servoing_mode(self):

        if self._vision_only:
            raise RuntimeError("vision_only=True")
        if not getattr(self, "servoing_mode_okay", True):
            self._recover_control()

    def _error_callback(self, err):
        try:
            print("[KinovaHelper-ERR]", err.toString())
        except Exception:
            print("[KinovaHelper-ERR]", err)

    def get_camera2eef_homogenous_matrix(self):
        dev_mgr = DeviceManagerClient(self.router)
        vis_cfg = VisionConfigClient(self.router)
        vid = next(h.device_identifier
                   for h in dev_mgr.ReadAllDevices().device_handle
                   if h.device_type == DeviceConfig_pb2.VISION)

        # Intrinsics
        prof = VisionConfig_pb2.IntrinsicProfileIdentifier(
            sensor     = VisionConfig_pb2.SENSOR_COLOR,
            resolution = VisionConfig_pb2.RESOLUTION_1280x720)
        intr = vis_cfg.GetIntrinsicParametersProfile(prof, vid)
        width, height = RES_TABLE[prof.resolution]

        # Extrinsics
        ext = vis_cfg.GetExtrinsicParameters(vid)

        R = np.array([
            [ext.rotation.row1.column1, ext.rotation.row1.column2, ext.rotation.row1.column3],
            [ext.rotation.row2.column1, ext.rotation.row2.column2, ext.rotation.row2.column3],
            [ext.rotation.row3.column1, ext.rotation.row3.column2, ext.rotation.row3.column3],
        ])

        t = np.array([ext.translation.t_x, ext.translation.t_y, ext.translation.t_z]).reshape(3, 1)
        
        T = np.eye(4); T[:3,:3]=R; T[:3,3:]=t
        
        return T
    
    def get_eef_homogenous_matrix(self):
        pose = self.base.GetMeasuredCartesianPose()
        # Position in meters
        x = pose.x
        y = pose.y
        z = pose.z

        roll = pose.theta_x * np.pi / 180.0
        pitch = pose.theta_y * np.pi / 180.0
        yaw = pose.theta_z * np.pi / 180.0

        translation = np.array([x, y, z])
        rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        
        # Create 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        print(T)
        
        return T
    
    def send_joint_speed(self, joint_idx: int, vel_deg: float):
        """
        Set velocity for a single joint in LOW_LEVEL_SERVOING mode.
        """
        fbk = self.base_cyclic.RefreshFeedback()

        # 1) Clamp velocity to joint limit
        limit = MAX_SPEED_LARGE_DEG if joint_idx < 4 else MAX_SPEED_SMALL_DEG
        vel_deg = max(-limit, min(vel_deg, limit))

        # 2) Build command
        self._cmd.Clear()
        for i in range(len(fbk.actuators)):
            ac = self._cmd.actuators.add()
            if i == joint_idx:
                ac.position = float('nan')                # ignore position
                ac.velocity = vel_deg * DEG2RAD           # deg/s → rad/s
            else:
                ac.position = fbk.actuators[i].position
                ac.velocity = 0.0
        self._cmd.frame_id = fbk.frame_id + 1

        # 3) Send command
        self.base_cyclic.Refresh(self._cmd)

    def send_joint_speeds(self, velocities_deg: list):
        """
        Set velocities for all joints using HIGH-LEVEL SERVOING (SINGLE_LEVEL_SERVOING).
        This is simpler and does NOT require LOW_LEVEL_SERVOING mode.
        
        Args:
            velocities_deg: List of joint velocities in deg/s (length = number of joints, typically 7)
        """
        joint_speeds = Base_pb2.JointSpeeds()
        
        for i, vel in enumerate(velocities_deg):
            # Clamp velocity to joint limit
            limit = MAX_SPEED_LARGE_DEG if i < 4 else MAX_SPEED_SMALL_DEG
            vel_clamped = max(-limit, min(vel, limit))
            
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = vel_clamped  # deg/s
        
        try:
            self.base.SendJointSpeedsCommand(joint_speeds)
        except KServerException as e:
            print(f"[KinovaHelper] send_joint_speeds error: {e}")

    def _refresh_loop(self, period=0.01):

        backoff = 0.15
        while self._running:
            try:
                if self.get_servoing_mode() != self.LOW_LEVEL_SERVOING:
                    self.set_servoing_mode(self.LOW_LEVEL_SERVOING)
                    time.sleep(backoff)
                    continue

                self.base_cyclic.Refresh(self._cmd)  
            except KServerException as ex:
                if "WRONG_SERVOING_MODE" in str(ex):
                    self.set_servoing_mode(self.LOW_LEVEL_SERVOING)
                else:
                    print(f"Refresh error: {ex}")
            time.sleep(period)


    def start_refresh_thread(self):
        if self._running:
            return            
        self._running = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def stop_refresh_thread(self):
        self._running = False
        if self._refresh_thread:
            self._refresh_thread.join()

    def get_joint_feedback(self, joint_idx: int):
        """Return BaseCyclic feedback for the given joint index."""
        fbk = self.base_cyclic.RefreshFeedback()
        return fbk.actuators[joint_idx]
    def clear_faults(self):
        self.base.ClearFaults()
        time.sleep(0.5)  # Allow some time for the command to take effect

    def get_current_pose(self):
        data = self.base.GetMeasuredCartesianPose()
        return np.array([data.x, data.y, data.z, data.theta_x, data.theta_y, data.theta_z])
        
    def get_gripper_position(self):
        """
        Reads the current position of the gripper finger (finger 0).
        Returns:
            float: Finger position (0.0 to 1.0), or 0.0 if no finger data available.
        """
        try:
            request = Base_pb2.GripperRequest()
            request.mode = Base_pb2.GRIPPER_POSITION
            feedback = self.base.GetMeasuredGripperMovement(request)

            if feedback.finger:
                position = feedback.finger[0].value
                # print(f"Gripper finger position: {position:.3f}")
                return position
            else:
                print("No finger feedback found, returning 0.0")
                return 0.0

        except KServerException as e:
            print("Error:", e)
            return 0.0


    def move_joint_angles(self, joint_angles_list):
        """
        Absolute joint
        WARNING: Do not use this.
        """
        raise DeprecationWarning("Safety Warning. Do not use this. This will just directly move to that angle without any planning.")
        
        self.ensure_control_and_servoing_mode()

        if len(joint_angles_list) != 7:
            raise ValueError("Expected 7")

        constrained_joint_angles = Base_pb2.ConstrainedJointAngles()
        for i, angle in enumerate(joint_angles_list):
            joint = constrained_joint_angles.joint_angles.joint_angles.add()
            joint.joint_identifier = i
            joint.value = angle

        try:
            self.base.PlayJointTrajectory(constrained_joint_angles)
            print("Joint trajectory sent.")
        except KServerException as e:
            print("Error:", e)

    def move_eef_pose(
            self,
            x, y, z,
            theta_x, theta_y, theta_z,
            transl_speed=0.5,      # m/s
            orient_speed=30.0       # deg/s
        ):
            """
            Absolute pose move with speed constraint.
            - x, y, z: position in meters
            - theta_x, theta_y, theta_z: rotation vectors in degrees
            - transl_speed: linear speed in m/s
            - orient_speed: angular speed in deg/s

            Speed limits (Kinova Gen3, from user guide):
            - Maximum Cartesian translation speed (TCP): 0.50 m/s (50 cm/s)
            -> Do NOT command values above 0.5 m/s.
            - Joint speed limits (general):
                * Large actuators (joints 1–4): ~79.6 deg/s (≈ 1.39 rad/s)
                * Small actuators (joints 5–7): ~69.9 deg/s (≈ 1.22 rad/s)
            Cartesian speeds near the max may be further clamped by these joint limits
            and by safety limits configured on the robot.

            Recommended ranges for experiments:
            - transl_speed 0.10–0.30 m/s : safe/slow motions for initial testing
            - transl_speed 0.30–0.40 m/s : moderate speed
            - transl_speed up to 0.50 m/s: near hardware limit; use only in a clear,
            safe workspace with no humans nearby.

            Note:
            - CartesianSpeed.orientation is specified in deg/s but is not fully used
            by the firmware yet; orientation motion is still constrained mainly by
            the joint speed limits and safety settings.
            """
            constrained_pose = Base_pb2.ConstrainedPose()
            pose = constrained_pose.target_pose

            # Target pose
            pose.x = x
            pose.y = y
            pose.z = z
            pose.theta_x = theta_x
            pose.theta_y = theta_y
            pose.theta_z = theta_z

            # ----- Velocity constraint setting -----
            speed = constrained_pose.constraint.speed      # CartesianSpeed
            speed.translation = float(transl_speed)        # [m/s]
            speed.orientation = float(orient_speed)        # [deg/s]

            try:
                self.base.PlayCartesianTrajectory(constrained_pose)
            except KServerException as e:
                print("Error:", e)

    def move_eef_waypoint(self, waypoints_array, duration_per_waypoint=None):
        """
        Move end-effector to a desired pose using ExecuteWaypointTrajectory.
        - x, y, z: position in meters
        - theta_x, theta_y, theta_z: rotation vectors in degrees
        """
        # Create a Waypoint List
        waypoint_list = Base_pb2.WaypointList()
        waypoint_list.use_optimal_blending = True
        if duration_per_waypoint is not None:
            waypoint_list.duration = duration_per_waypoint * waypoints_array.shape[0]  

        for idx, wp in enumerate(waypoints_array):
            # print(wp.size)
            # print(wp.shape, wp)
            waypoint = waypoint_list.waypoints.add()
            waypoint.name = f"waypoint_{idx}"
            waypoint.cartesian_waypoint.pose.x = wp[0]
            waypoint.cartesian_waypoint.pose.y = wp[1]
            waypoint.cartesian_waypoint.pose.z = wp[2]
            waypoint.cartesian_waypoint.pose.theta_x = wp[3]
            waypoint.cartesian_waypoint.pose.theta_y = wp[4]
            waypoint.cartesian_waypoint.pose.theta_z = wp[5]
            waypoint.cartesian_waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            waypoint.cartesian_waypoint.maximum_linear_velocity = 0.5
            waypoint.cartesian_waypoint.maximum_angular_velocity = 20
            if idx == waypoints_array.shape[0] - 1:
                # Last waypoint, set blending radius to 0.0
                waypoint.cartesian_waypoint.blending_radius = 0.0
            else:
                waypoint.cartesian_waypoint.blending_radius = 0.2

        # Validate
        # result = self.base.ValidateWaypointList(waypoint_list)
        # # print(result)
        # if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        #     print("Trajectory validation error:")
        #     for error in result.trajectory_error_report.trajectory_error_elements:
        #         print(error)
        #         print(f"- Waypoint Index: {error.waypoint_index}")
        #         print(f"  Error Value   : {error.error_value}")
        #     print("Validation failed.")
        #     return

        try:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                lambda notif, e=e: e.set() if notif.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT] else None,
                Base_pb2.NotificationOptions())

            self.base.ExecuteWaypointTrajectory(waypoint_list)

            finished = e.wait(timeout=None)
            self.base.Unsubscribe(notification_handle)

            if finished:
                # print("Waypoint trajectory execution completed.")
                return
            else:
                print("Timeout during waypoint trajectory execution.")

        except KServerException as ex:
            print("Execution Error:", ex)



    def move_eef_twist_to_target(self, target_x, target_y, target_z, max_linear_vel=0.5):
        """
        Send a single twist command toward target position.
        Call this repeatedly (e.g., at 50-100 Hz) for velocity control.
        OPTIMIZED FOR MAXIMUM SPEED - minimal slowdown zone.
        
        Args:
            target_x, target_y, target_z: target position in meters
            max_linear_vel: maximum linear velocity in m/s (default 0.5 = 50 cm/s MAX)
        
        Returns:
            distance: remaining distance to target (meters)
        """
        # Get current position
        current_pose = self.get_current_pose()
        current_pos = np.array([current_pose[0], current_pose[1], current_pose[2]])
        target = np.array([target_x, target_y, target_z])
        
        # Calculate error
        error = target - current_pos
        distance = np.linalg.norm(error)
        
        if distance < 0.005:  # Less than 5mm - stop
            self.twist_cmd([0, 0, 0, 0, 0, 0])
            return distance
        
        # Calculate velocity direction (unit vector)
        direction = error / distance
        
        # AGGRESSIVE speed profile - full speed until very close
        if distance > 0.04:  # More than 4cm away - FULL SPEED
            speed = max_linear_vel
        elif distance > 0.02:  # 2-4cm - slight slowdown (80%)
            speed = max_linear_vel * 0.8
        else:  # Last 2cm - proportional slowdown
            speed = max(0.10, distance * 4.0)  # Min 10 cm/s, aggressive
        
        # Velocity command
        vel = direction * speed
        self.twist_cmd([vel[0], vel[1], vel[2], 0, 0, 0])
        
        return distance
    
    def stop_twist(self):
        """Stop all twist motion."""
        self.twist_cmd([0, 0, 0, 0, 0, 0])

    def move_eef_waypoint_single(self, x, y, z, theta_x, theta_y, theta_z):
        """For every single input"""
        # Build a single waypoint
        waypoint = Base_pb2.CartesianWaypoint()
        waypoint.pose.x = x
        waypoint.pose.y = y
        waypoint.pose.z = z
        waypoint.pose.theta_x = theta_x
        waypoint.pose.theta_y = theta_y
        waypoint.pose.theta_z = theta_z
        waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        waypoint.blending_radius = 0.0  # No blending for single waypoint

        # Put into waypoint list
        waypoints = Base_pb2.WaypointList()
        waypoints.use_optimal_blending = False
        waypoint_entry = waypoints.waypoints.add()
        waypoint_entry.name = "eef_waypoint"
        waypoint_entry.cartesian_waypoint.CopyFrom(waypoint)

        # Validate trajectory
        # result = self.base.ValidateWaypointList(waypoints)
        # if len(result.trajectory_error_report.trajectory_error_elements) != 0:
        #     print("Trajectory validation error:")
        #     result.trajectory_error_report.PrintDebugString()
        #     return

        try:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                lambda notif, e=e: e.set() if notif.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT] else None,
                Base_pb2.NotificationOptions())

            self.base.ExecuteWaypointTrajectory(waypoints)

            finished = e.wait(timeout=None)
            self.base.Unsubscribe(notification_handle)

            if finished:
                print("Waypoint trajectory execution completed.")
            else:
                print("Timeout during waypoint trajectory execution.")

        except KServerException as ex:
            print("Error during ExecuteWaypointTrajectory:", ex)

    
    def move_joint_waypoints_via_ik(self,
                        waypoints_array,
                        blending_radius: float = 0.0,
                        use_optimal: bool = False,
                        position_threshold: float = 0.05,
                        timeout: float = 60.0):

        # 1) Filtering
        raw = Base_pb2.WaypointList()
        raw.use_optimal_blending = False
        prev_wp = None
        for idx, wp in enumerate(waypoints_array):
            wp = [float(v) for v in wp]
            if prev_wp is not None:
                dx = wp[0] - prev_wp[0]
                dy = wp[1] - prev_wp[1]
                dz = wp[2] - prev_wp[2]
                if (dx*dx + dy*dy + dz*dz)**0.5 < position_threshold:
                    continue

            w = raw.waypoints.add()
            w.name = f"wp_{idx}"
            cw = w.cartesian_waypoint
            cw.pose.x            = wp[0]
            cw.pose.y            = wp[1]
            cw.pose.z            = wp[2]
            cw.pose.theta_x      = wp[3]
            cw.pose.theta_y      = wp[4]
            cw.pose.theta_z      = wp[5]
            cw.reference_frame   = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            cw.blending_radius   = blending_radius

            prev_wp = wp

        if not raw.waypoints:
            return

        # 2) Validate
        report = self.base.ValidateWaypointList(raw)
        errs = report.trajectory_error_report.trajectory_error_elements
        if errs:
            for e in errs:
                print(f"  idx={e.waypoint_index}  err={e.error_value:.6f}")
            return

        # 3) RAW or OPTIMAL
        to_run = report.optimal_waypoint_list if use_optimal else raw
        lbl    = "optimal" if use_optimal else "raw"
        print(f"Executing {lbl} list with {len(to_run.waypoints)} waypoints")

        # 4) Wait til end
        done = threading.Event()
        def _cb(ntf, ev=done):
            if ntf.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                ev.set()

        handle = self.base.OnNotificationActionTopic(
            _cb,
            Base_pb2.NotificationOptions()
        )

        try:
            self.base.ExecuteWaypointTrajectory(to_run)
        except KServerException as ex:
            self.base.Unsubscribe(handle)
            return

        if done.wait(timeout):
            print("completed.")
        self.base.Unsubscribe(handle)

    def gripper_control(self, position, mode="position"):
        """ 0 and 1"""
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1

        if mode == "position":
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            finger.value = position
            print(f"Goal position: {position:.2f}")
        try:
            self.base.SendGripperCommand(gripper_command)
        except KServerException as e:
            print("Error:", e)

    def go_home(self):
        try:
            action_type = Base_pb2.RequestedActionType()
            action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
            action_list = self.base.ReadAllActions(action_type)
            home_action = next((a for a in action_list.action_list if a.name == "Home"), None)

            if home_action:
                self.base.ExecuteAction(home_action)
                print("Go to Home")
            else:
                print("'Home' not found.")
        except KServerException as e:
            print("Error:", e)

    def go_retract(self):
        try:
            action_type = Base_pb2.RequestedActionType()
            action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
            action_list = self.base.ReadAllActions(action_type)
            retract_action = next((a for a in action_list.action_list if a.name == "Retract"), None)

            if retract_action:
                self.base.ExecuteAction(retract_action)
                print("Go to Retract")
            else:
                print("'Retract' not found.")
        except KServerException as e:
            print("Error:", e)
    
    def start_keepalive(self, hz: int = 50):
        self.start_refresh_thread()

    def stop_keepalive(self):
        self.stop_refresh_thread()
    

    def twist_cmd(self, vel):
        """
        Args:
            [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        """
        twist_command = Base_pb2.TwistCommand()
        twist_command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist_command.duration = 0  

        twist = twist_command.twist
        twist.linear_x  = vel[0]
        twist.linear_y  = vel[1]
        twist.linear_z  = vel[2]
        twist.angular_x = vel[3]
        twist.angular_y = vel[4]
        twist.angular_z = vel[5]

        try:
            self.base.SendTwistCommand(twist_command)
            # print("Twist command sent:", vel)
        except KServerException as e:
            print("Error sending twist command:", e)

    def move_eef_twist(self, target_pose):
        pos_p_gain = 0.5
        pos_i_gain = 0.1
        rot_p_gain = 18.0
        rot_i_gain = 2.0

        pos_error = -np.inf * np.ones(3,)
        rot_error = -np.inf * np.ones(3,)
        int_pos_error = np.zeros(3,)
        int_rot_error = np.zeros(3,)
        last_t = time.time()
        
        while np.linalg.norm(pos_error) > 0.01 or np.linalg.norm(rot_error) > 0.01:
            dt = time.time() - last_t
            curr_pose = self.get_current_pose()
            print("Current Pose:", curr_pose)
            
            pos_error, rot_error = get_error(curr_pose, target_pose)
            print("Pos Error:", pos_error)
            print("Rot Error:", rot_error)
            int_pos_error += np.clip(pos_error * dt, -0.1, 0.1)
            int_rot_error += np.clip(rot_error * dt, -0.3, 0.3)
            print("Int Pos Error:", int_pos_error)
            print("Int Rot Error:", int_rot_error)

            velocity = np.zeros(6,)
            velocity[:3] =  - pos_p_gain * pos_error -  pos_i_gain * int_pos_error
            velocity[3:] = - rot_p_gain * rot_error - rot_i_gain * int_pos_error
            self.twist_cmd(velocity)
            
            last_t = time.time()
        
        self.twist_cmd(np.zeros(6,))  # Stop the robot