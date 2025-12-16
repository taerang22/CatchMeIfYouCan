#!/usr/bin/env python3
import threading
import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

# Hardcoded robot connection info
ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"

# Import your helper from the same package directory
from .kinova_helper import KinovaHelper
try:
    from kortex_api.Exceptions.KServerException import KServerException
except Exception:
    KServerException = Exception


class KinovaControllerNode(Node):
    """
    Subscribes to /kinova/goal_pose (geometry_msgs/PoseStamped).
    Uses DIRECT VELOCITY CONTROL for maximum speed (50 cm/s).
    Returns to HOME after robot reaches goal AND no new goal received.
    """

    # Tolerances/behavior
    POSITION_TOL_M = 0.015      # 1.5 cm tolerance for "reached" check (tighter)
    ANGLE_TOL_DEG = 2.0         # per-axis
    
    # Speed settings (Kinova Gen3 max: 0.50 m/s translation)
    TRANSL_SPEED = 0.50         # m/s - MAXIMUM SPEED (50 cm/s)
    
    # Home return timeout (starts AFTER robot reaches goal)
    HOME_RETURN_TIMEOUT_S = 3.0  # seconds to wait before returning home
    
    # Velocity control rate - higher = smoother & faster response
    CONTROL_RATE_HZ = 100.0     # 100 Hz velocity control loop

    def __init__(self):
        super().__init__("kinova_controller")

        # Connect
        self.helper = KinovaHelper(ip=ROBOT_IP, username=USERNAME, password=PASSWORD)
        self.get_logger().info(f"Connected to Kinova at {ROBOT_IP}")
        
        # IMPORTANT: Stop any residual motion from previous sessions
        try:
            self.helper.stop_twist()
            self.get_logger().info("Stopped any residual motion")
        except Exception:
            pass

        # Latest goal (updated by callback)
        self.latest_goal = None
        self.goal_lock = threading.Lock()
        
        # Track goal state - all initialized to safe "not moving" state
        self.target_position = None      # [x, y, z] of current goal
        self.moving_to_goal = False      # True while robot is moving to goal
        self.reached_goal_time = None    # Time when robot reached goal
        self.returning_home = False
        
        # Timing for execution measurement
        self.move_start_time = None      # Time when robot started moving
        self.start_position = None       # Position when movement started

        # Subscribe to goal pose - will receive ALL messages
        self.sub_goal = self.create_subscription(
            PoseStamped, 
            "/kinova/goal_pose", 
            self._goal_cb, 
            10
        )
        
        # HIGH-FREQUENCY timer for velocity control (50 Hz)
        self.control_timer = self.create_timer(1.0 / self.CONTROL_RATE_HZ, self._velocity_control_loop)
        
        # Lower frequency timer for home return check (5 Hz)
        self.home_timer = self.create_timer(0.2, self._check_home_return)
        
        self.get_logger().info(
            f"Kinova Controller started (VELOCITY CONTROL MODE).\n"
            f"  Max Speed: {self.TRANSL_SPEED} m/s ({self.TRANSL_SPEED*100:.0f} cm/s)\n"
            f"  Control rate: {self.CONTROL_RATE_HZ} Hz\n"
            f"  Position tolerance: {self.POSITION_TOL_M*100:.1f} cm\n"
            f"  Home return timeout: {self.HOME_RETURN_TIMEOUT_S}s (after reaching goal)\n"
            f"  Waiting for /kinova/goal_pose..."
        )

        # Clean shutdown
        rclpy.get_default_context().on_shutdown(self._on_shutdown)

    # ---------- Callback - receives EVERY goal ----------
    def _goal_cb(self, msg: PoseStamped):
        """Callback for every goal_pose message - prints and processes."""
        
        # Cancel any pending home return
        self.returning_home = False
        self.reached_goal_time = None
        
        # Extract goal position
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal_z = msg.pose.position.z
        
        # Print received goal pose
        self.get_logger().info(
            f"[RECEIVED] Goal Pose: x={goal_x:.4f}, y={goal_y:.4f}, z={goal_z:.4f}"
        )
        
        # Get current robot pose
        try:
            current_pose = self.helper.get_current_pose()
        except Exception as e:
            self.get_logger().error(f"Failed to get current pose: {e}")
            return

        # Store start position and time for execution timing
        self.start_position = np.array([current_pose[0], current_pose[1], current_pose[2]])
        start_distance = np.linalg.norm(np.array([goal_x, goal_y, goal_z]) - self.start_position)
        
        self.get_logger().info(
            f"[VELOCITY CONTROL] Moving at {self.TRANSL_SPEED*100:.0f} cm/s\n"
            f"  From: ({self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f})\n"
            f"  To:   ({goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f})\n"
            f"  Distance: {start_distance*100:.1f} cm"
        )

        # Store target position - velocity control loop will handle movement
        self.target_position = np.array([goal_x, goal_y, goal_z])
        self.moving_to_goal = True
        self.move_start_time = time.time()
    
    # ---------- Velocity Control Loop (100 Hz) ----------
    def _velocity_control_loop(self):
        """High-frequency timer callback for direct velocity control."""
        
        # Safety checks - only move if we have a valid goal
        if not self.moving_to_goal:
            return
        
        if self.target_position is None:
            # No target - stop and reset state
            self.get_logger().warn("[SAFETY] moving_to_goal=True but target_position=None! Stopping.")
            try:
                self.helper.stop_twist()
            except Exception:
                pass
            self.moving_to_goal = False
            return
        
        if self.returning_home:
            # Stop twist while returning home
            try:
                self.helper.stop_twist()
            except Exception:
                pass
            return
        
        # Timeout safety - if we've been moving for too long (>10s), something is wrong
        if self.move_start_time is not None:
            move_duration = time.time() - self.move_start_time
            if move_duration > 10.0:
                self.get_logger().warn(f"[SAFETY] Moving for {move_duration:.1f}s - timeout! Stopping.")
                try:
                    self.helper.stop_twist()
                except Exception:
                    pass
                self.moving_to_goal = False
                self.target_position = None
                return
        
        try:
            # Send velocity command toward target
            distance = self.helper.move_eef_twist_to_target(
                self.target_position[0],
                self.target_position[1],
                self.target_position[2],
                max_linear_vel=self.TRANSL_SPEED
            )
            
            # Check if we've reached the target
            if distance < self.POSITION_TOL_M:
                # Stop and mark as reached
                self.helper.stop_twist()
                self.moving_to_goal = False
                self.reached_goal_time = time.time()
                
                # Calculate execution time
                if self.move_start_time is not None:
                    execution_time = self.reached_goal_time - self.move_start_time
                    
                    current_pose = self.helper.get_current_pose()
                    current_pos = np.array([current_pose[0], current_pose[1], current_pose[2]])
                    
                    if self.start_position is not None:
                        travel_distance = np.linalg.norm(current_pos - self.start_position)
                        avg_speed = travel_distance / execution_time if execution_time > 0 else 0
                    else:
                        travel_distance = 0
                        avg_speed = 0
                    
                    self.get_logger().info(
                        f"\n{'='*50}\n"
                        f"[REACHED] Robot reached goal position!\n"
                        f"{'='*50}\n"
                        f"  Final distance to target: {distance*100:.1f} cm\n"
                        f"  Travel distance: {travel_distance*100:.1f} cm\n"
                        f"  ⏱️  EXECUTION TIME: {execution_time:.3f} seconds\n"
                        f"  Average speed: {avg_speed*100:.1f} cm/s ({avg_speed:.3f} m/s)\n"
                        f"{'='*50}\n"
                        f"  Will return to HOME in {self.HOME_RETURN_TIMEOUT_S}s if no new goal..."
                    )
                
                # Reset timing
                self.move_start_time = None
                self.start_position = None
                
        except Exception as e:
            self.get_logger().warn(f"Velocity control error: {e}")
            self.helper.stop_twist()
    
    # ---------- Check home return ----------
    def _check_home_return(self):
        """Timer callback to check if we should return home."""
        
        if self.returning_home:
            return
        
        # SAFETY: If not moving but twist might still be active, stop it
        if not self.moving_to_goal and self.target_position is None:
            # Periodically ensure robot is stopped when idle
            pass  # Don't spam stop commands, just check state
        
        # If we've reached the goal and no new goal came, check timeout
        if self.reached_goal_time is not None:
            elapsed = time.time() - self.reached_goal_time
            
            if elapsed > self.HOME_RETURN_TIMEOUT_S:
                self.returning_home = True
                self.reached_goal_time = None
                
                # Clear ALL movement state
                self.target_position = None
                self.moving_to_goal = False
                self.move_start_time = None
                self.start_position = None
                
                # Explicitly stop any twist motion before going home
                try:
                    self.helper.stop_twist()
                except Exception:
                    pass
                
                self.get_logger().info(
                    f"[HOME] No new goal for {self.HOME_RETURN_TIMEOUT_S}s after reaching target. "
                    f"Returning to HOME position..."
                )
                try:
                    self.helper.go_home()
                except Exception as e:
                    self.get_logger().error(f"Failed to go home: {e}")
                finally:
                    self.returning_home = False

    # ---------- Helpers ----------
    def _quat_to_rpy_xyz_deg(self, qx, qy, qz, qw):
        """Convert quaternion to roll-pitch-yaw in degrees."""
        r = R.from_quat([qx, qy, qz, qw])
        rpy_deg = np.degrees(r.as_euler("xyz", degrees=False))
        return rpy_deg.tolist()

    def _on_shutdown(self):
        try:
            self.get_logger().info("Shutting down Kinova session…")
            # Stop any ongoing motion
            if hasattr(self.helper, "stop_twist"):
                self.helper.stop_twist()
            if hasattr(self.helper, "stop_keepalive"):
                self.helper.stop_keepalive()
            if hasattr(self.helper, "close"):
                self.helper.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = KinovaControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
