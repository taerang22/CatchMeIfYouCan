#!/usr/bin/env python3
"""
Visualize ball trajectory prediction from rosbag.
Compares actual trajectory, fitted parabola, and predicted intersection.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sqlite3
import struct

# Try to import ROS 2, but make it optional
ROS2_AVAILABLE = False
ROSBAG2_AVAILABLE = False

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS 2 not available. Trying rosbag2_py...")
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        ROSBAG2_AVAILABLE = True
    except ImportError:
        print("Warning: rosbag2_py not available. Will try manual parsing.")


class BallPredictionVisualizer:
    """Replicates the prediction logic from ball_fitting_predict_node.py"""
    
    def __init__(self):
        # Same parameters as the node
        self.g = 9.81
        
        # ArUco → Robot base_link transform
        x_offset = 0.07
        self.t_ar_to_robot = np.array([1.45 + x_offset, -0.31, 0.89], float).reshape(3, 1)
        self.R_ar_to_robot = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], float)
        
        # Target planes
        gripper_offset = 0.15
        self.x_plane_calc = 0.576 - gripper_offset  # 0.426
        self.x_plane_pub = 0.576
        self.min_z = 0.10
    
    def transform_to_robot_frame(self, pos_aruco):
        """Transform from ArUco frame to robot frame"""
        pa = pos_aruco.reshape(3, 1)
        pr = self.R_ar_to_robot @ pa + self.t_ar_to_robot
        return pr.flatten()
    
    def predict_from_parabola(self, pos_buffer, time_buffer):
        """
        Predict intersection using parabola fitting.
        Returns: (predicted_point, t_hit, fitted_params)
        """
        pos = np.array(pos_buffer)  # (N, 3)
        N = len(pos)
        
        # Normalize time to [0, 1, 2, ..., N-1] for fitting
        t = np.arange(N, dtype=float)
        
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        
        # Linear fit for x(t), y(t)
        A_lin = np.vstack([t, np.ones_like(t)]).T
        a_x, b_x = np.linalg.lstsq(A_lin, x, rcond=None)[0]
        a_y, b_y = np.linalg.lstsq(A_lin, y, rcond=None)[0]
        
        # Quadratic fit for z(t) (parabola)
        A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
        A_z, B_z, C_z = np.linalg.lstsq(A_quad, z, rcond=None)[0]
        
        # Solve x-plane intersection
        if abs(a_x) < 1e-6:
            return None, None, None
        
        t_hit = (self.x_plane_calc - b_x) / a_x
        if t_hit <= 0:
            return None, None, None
        
        # Evaluate parabola at t_hit
        y_hit = a_y * t_hit + b_y
        z_hit = A_z * t_hit**2 + B_z * t_hit + C_z
        
        if z_hit < self.min_z:
            z_hit = self.min_z
        
        p_hit = np.array([self.x_plane_pub, y_hit, z_hit])
        
        fitted_params = {
            'x': (a_x, b_x),
            'y': (a_y, b_y),
            'z': (A_z, B_z, C_z),
            't_hit': t_hit
        }
        
        return p_hit, t_hit, fitted_params
    
    def generate_parabola(self, fitted_params, t_range):
        """Generate parabola points for visualization"""
        a_x, b_x = fitted_params['x']
        a_y, b_y = fitted_params['y']
        A_z, B_z, C_z = fitted_params['z']
        
        x_parabola = a_x * t_range + b_x
        y_parabola = a_y * t_range + b_y
        z_parabola = A_z * t_range**2 + B_z * t_range + C_z
        
        return np.column_stack([x_parabola, y_parabola, z_parabola])


def parse_cdr_pose(data):
    """Manually parse PoseStamped from CDR format"""
    # CDR format: header (stamp + frame_id) + pose (position + orientation)
    # This is a simplified parser - may need adjustment based on actual CDR encoding
    offset = 0
    
    # Skip header (stamp: sec + nanosec = 8 bytes, frame_id string)
    # For simplicity, we'll just extract the pose position
    # Position: x, y, z (each 8 bytes for double)
    try:
        # Try to find position data (doubles are 8 bytes)
        # CDR encoding: alignment to 8 bytes, then x, y, z
        # This is a heuristic approach
        pos_bytes = data[-24:]  # Last 24 bytes likely contain position
        x, y, z = struct.unpack('<ddd', pos_bytes)
        return np.array([x, y, z])
    except:
        # Fallback: try different byte positions
        for i in range(0, len(data) - 24, 8):
            try:
                x, y, z = struct.unpack('<ddd', data[i:i+24])
                # Sanity check: reasonable ball position
                if -10 < x < 10 and -10 < y < 10 and -5 < z < 5:
                    return np.array([x, y, z])
            except:
                continue
    return None


def read_csv(csv_path):
    """Read ball pose data from CSV"""
    import csv
    poses = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different CSV formats
            if 'timestamp' in row:
                # Simple format: timestamp, x, y, z
                timestamp = int(row['timestamp'])
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
            elif 'bag_timestamp_ns' in row:
                # ROS bag format: timestamp_sec, timestamp_nanosec, position_x, etc.
                timestamp = int(row['bag_timestamp_ns'])
                x = float(row['position_x'])
                y = float(row['position_y'])
                z = float(row['position_z'])
            elif 'timestamp_sec' in row:
                # Alternative ROS format
                timestamp_sec = int(row['timestamp_sec'])
                timestamp_nsec = int(row.get('timestamp_nanosec', 0))
                timestamp = timestamp_sec * 1e9 + timestamp_nsec
                x = float(row['position_x'])
                y = float(row['position_y'])
                z = float(row['position_z'])
            else:
                # Try to find x, y, z columns
                x_key = [k for k in row.keys() if 'x' in k.lower() and 'position' in k.lower()][0] if any('x' in k.lower() for k in row.keys()) else 'x'
                y_key = [k for k in row.keys() if 'y' in k.lower() and 'position' in k.lower()][0] if any('y' in k.lower() for k in row.keys()) else 'y'
                z_key = [k for k in row.keys() if 'z' in k.lower() and 'position' in k.lower()][0] if any('z' in k.lower() for k in row.keys()) else 'z'
                timestamp = int(row.get('timestamp', 0)) if 'timestamp' in row else len(poses)
                x = float(row[x_key])
                y = float(row[y_key])
                z = float(row[z_key])
            
            poses.append({
                'timestamp': timestamp,
                'position': np.array([x, y, z])
            })
    
    return poses, []


def read_rosbag(rosbag_path):
    """Read ball pose data from rosbag"""
    bag_path = Path(rosbag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Rosbag not found: {rosbag_path}")
    
    poses = []
    twists = []
    
    if ROSBAG2_AVAILABLE:
        # Use rosbag2_py API
        storage_options = StorageOptions(uri=str(bag_path), storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == "/kinova/ball/init_pose":
                try:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    poses.append({
                        'timestamp': timestamp,
                        'position': np.array([
                            msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z
                        ])
                    })
                except Exception as e:
                    print(f"Warning: Failed to deserialize pose: {e}")
            elif topic == "/kinova/ball/init_twist":
                try:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    twists.append({
                        'timestamp': timestamp,
                        'velocity': np.array([
                            msg.twist.linear.x,
                            msg.twist.linear.y,
                            msg.twist.linear.z
                        ])
                    })
                except Exception as e:
                    print(f"Warning: Failed to deserialize twist: {e}")
    
    elif ROS2_AVAILABLE:
        # Use ROS 2 deserialization with SQLite
        db_file = bag_path / f"{bag_path.name}_0.db3"
        if not db_file.exists():
            db_files = list(bag_path.glob("*.db3"))
            if not db_files:
                raise FileNotFoundError(f"No .db3 file found in {bag_path}")
            db_file = db_files[0]
        
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
        rows = cursor.fetchall()
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        for topic_id, timestamp, data in rows:
            topic_name, topic_type = topics[topic_id]
            
            if topic_name == "/kinova/ball/init_pose":
                try:
                    msg_type = get_message(topic_type)
                    msg = deserialize_message(data, msg_type)
                    poses.append({
                        'timestamp': timestamp,
                        'position': np.array([
                            msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z
                        ])
                    })
                except Exception as e:
                    print(f"Warning: Failed to deserialize pose: {e}")
            elif topic_name == "/kinova/ball/init_twist":
                try:
                    msg_type = get_message(topic_type)
                    msg = deserialize_message(data, msg_type)
                    twists.append({
                        'timestamp': timestamp,
                        'velocity': np.array([
                            msg.twist.linear.x,
                            msg.twist.linear.y,
                            msg.twist.linear.z
                        ])
                    })
                except Exception as e:
                    print(f"Warning: Failed to deserialize twist: {e}")
        
        conn.close()
    else:
        # Manual parsing fallback
        print("Using manual CDR parsing (may be less accurate)...")
        db_file = bag_path / f"{bag_path.name}_0.db3"
        if not db_file.exists():
            db_files = list(bag_path.glob("*.db3"))
            if not db_files:
                raise FileNotFoundError(f"No .db3 file found in {bag_path}")
            db_file = db_files[0]
        
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
        rows = cursor.fetchall()
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        for topic_id, timestamp, data in rows:
            topic_name, topic_type = topics[topic_id]
            
            if topic_name == "/kinova/ball/init_pose":
                pos = parse_cdr_pose(data)
                if pos is not None:
                    poses.append({
                        'timestamp': timestamp,
                        'position': pos
                    })
        
        conn.close()
    
    if len(poses) == 0:
        print("\n" + "="*70)
        print("ERROR: Could not parse rosbag messages.")
        print("="*70)
        print("Please source ROS 2 environment first:")
        print("  source /opt/ros/humble/setup.bash  # or your ROS 2 install path")
        print("  # Then activate conda: conda activate mujoco")
        print("  # Then run this script again")
        print("="*70)
        raise ValueError("No valid pose messages found in rosbag.")
    
    return poses, twists


def find_actual_intersection(positions, x_plane):
    """Find where the actual trajectory crosses the x-plane"""
    if len(positions) < 2:
        return None
    
    pos_array = np.array([p['position'] for p in positions])
    x_coords = pos_array[:, 0]
    
    # Find where x crosses the plane
    for i in range(len(x_coords) - 1):
        x1, x2 = x_coords[i], x_coords[i+1]
        if (x1 <= x_plane <= x2) or (x2 <= x_plane <= x1):
            # Linear interpolation
            t_frac = (x_plane - x1) / (x2 - x1) if abs(x2 - x1) > 1e-6 else 0.5
            p1 = pos_array[i]
            p2 = pos_array[i+1]
            actual_point = p1 + t_frac * (p2 - p1)
            return actual_point
    
    return None


def visualize(rosbag_path_or_csv, apply_transform=False):
    """
    Main visualization function
    
    Args:
        rosbag_path_or_csv: Path to CSV or rosbag
        apply_transform: If True, apply ArUco→base_link transform (as ball_fitting_predict_node.py does)
                        If False, assume data is already in base_link frame
    """
    path = Path(rosbag_path_or_csv)
    
    # Check if it's a CSV file
    if path.suffix == '.csv':
        print(f"Reading CSV: {path}")
        poses, twists = read_csv(path)
    else:
        print(f"Reading rosbag: {path}")
        poses, twists = read_rosbag(path)
    
    if len(poses) < 5:
        print(f"Warning: Only {len(poses)} poses found. Need at least 5 for prediction.")
        return
    
    print(f"Found {len(poses)} poses and {len(twists)} twists")
    
    # Initialize visualizer
    viz = BallPredictionVisualizer()
    
    # IMPORTANT: ball_fitting_predict_node.py receives /kinova/ball/init_pose
    # which is published by ball_tracker_node. ball_tracker_node already transforms
    # from camera frame to base_link using ArUco calibration.
    # However, ball_fitting_predict_node.py assumes the data is in ArUco frame
    # and applies ANOTHER transformation. This seems like a double transformation issue.
    #
    # For visualization, we need to match what ball_fitting_predict_node.py does:
    # It applies R_ar_to_robot and t_ar_to_robot transformation.
    # But the actual data might already be in a different base_link frame.
    #
    # Let's check: if CSV Z values are reasonable for base_link (z >= 0.4m minimum),
    # then data might already be in base_link. Otherwise, transformation is needed.
    
    path = Path(rosbag_path_or_csv)
    positions_robot = []
    
    # Check if data looks like it's already in base_link
    pos_check = np.array([p['position'] for p in poses])
    z_min = pos_check[:, 2].min()
    
    print(f"\nData analysis:")
    print(f"  CSV frame_id: {poses[0] if poses else 'N/A'}")
    print(f"  Z range: {pos_check[:, 2].min():.3f} to {pos_check[:, 2].max():.3f}")
    
    # ball_fitting_predict_node.py ALWAYS applies transformation
    # So we must do the same to match its behavior
    if apply_transform:
        print(f"\nApplying transformation (as ball_fitting_predict_node.py does):")
        print(f"  R_ar_to_robot =")
        print(f"    {viz.R_ar_to_robot}")
        print(f"  t_ar_to_robot = {viz.t_ar_to_robot.flatten()}")
        print(f"  Formula: pos_base_link = R_ar_to_robot @ pos_aruco + t_ar_to_robot")
        
        for i, pose in enumerate(poses):
            # ball_fitting_predict_node.py treats input as ArUco frame
            pos_aruco = pose['position']
            pos_robot = viz.transform_to_robot_frame(pos_aruco)
            
            if i < 3:
                print(f"\n  Point {i}:")
                print(f"    Input (ArUco?):  [{pos_aruco[0]:.6f}, {pos_aruco[1]:.6f}, {pos_aruco[2]:.6f}]")
                print(f"    Output (base_link): [{pos_robot[0]:.6f}, {pos_robot[1]:.6f}, {pos_robot[2]:.6f}]")
            
            positions_robot.append({
                'timestamp': pose['timestamp'],
                'position': pos_robot
            })
    else:
        print(f"\nUsing data as-is (no transformation)")
        for pose in poses:
            positions_robot.append({
                'timestamp': pose['timestamp'],
                'position': pose['position']
            })
    
    # Sort by timestamp
    positions_robot.sort(key=lambda x: x['timestamp'])
    
    # Extract position arrays
    pos_array = np.array([p['position'] for p in positions_robot])
    timestamps = np.array([p['timestamp'] for p in positions_robot])
    
    # Normalize time to start from t=0 (relative time)
    if len(timestamps) > 0:
        t_start = timestamps[0]
        time_normalized = (timestamps - t_start) * 1e-9  # Convert nanoseconds to seconds
    else:
        time_normalized = np.arange(len(pos_array))
    
    # Use first 5 points for prediction (as the node does)
    pos_buffer = pos_array[:5].tolist()
    time_buffer = timestamps[:5].tolist()
    
    # Run prediction
    predicted_point, t_hit, fitted_params = viz.predict_from_parabola(pos_buffer, time_buffer)
    
    if predicted_point is None:
        print("Prediction failed!")
        return
    
    print(f"\nPrediction Results:")
    print(f"  Predicted intersection: [{predicted_point[0]:.3f}, {predicted_point[1]:.3f}, {predicted_point[2]:.3f}]")
    print(f"  Time to hit: {t_hit:.3f} (normalized)")
    print(f"\nFitted Parameters:")
    print(f"  x(t) = {fitted_params['x'][0]:.3f}*t + {fitted_params['x'][1]:.3f}")
    print(f"  y(t) = {fitted_params['y'][0]:.3f}*t + {fitted_params['y'][1]:.3f}")
    print(f"  z(t) = {fitted_params['z'][0]:.3f}*t² + {fitted_params['z'][1]:.3f}*t + {fitted_params['z'][2]:.3f}")
    
    # Find actual intersection
    actual_intersection = find_actual_intersection(positions_robot, viz.x_plane_pub)
    
    if actual_intersection is not None:
        print(f"\nActual intersection: [{actual_intersection[0]:.3f}, {actual_intersection[1]:.3f}, {actual_intersection[2]:.3f}]")
        error = np.linalg.norm(predicted_point - actual_intersection)
        print(f"  Prediction error: {error:.3f} m")
    
    # Create visualization - Split into 2 windows (base_link frame, t=0부터)
    # Color map for time progression
    colors = plt.cm.viridis(time_normalized / time_normalized.max() if len(time_normalized) > 1 else [0])
    
    # Window 1: 3D + X-Y projection
    fig1 = plt.figure(figsize=(16, 8))
    
    # 3D plot
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    scatter_3d = ax1.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 
                             c=time_normalized, cmap='viridis', s=100, 
                             label='Actual Trajectory', edgecolors='black', linewidths=0.5)
    ax1.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 
             '--', color='gray', linewidth=1, alpha=0.5)
    
    # Add time labels for all points
    for i in range(len(pos_array)):
        ax1.text(pos_array[i, 0], pos_array[i, 1], pos_array[i, 2], 
                f' t={time_normalized[i]:.3f}s', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
    
    ax1.set_xlabel('X (m) - base_link', fontsize=10)
    ax1.set_ylabel('Y (m) - base_link', fontsize=10)
    ax1.set_zlabel('Z (m) - base_link', fontsize=10)
    ax1.set_title('3D Actual Trajectory (base_link frame, t=0부터)', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(scatter_3d, ax=ax1, label='Time (s)', shrink=0.8)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # X-Y projection
    ax2 = fig1.add_subplot(1, 2, 2)
    scatter2 = ax2.scatter(pos_array[:, 0], pos_array[:, 1], c=time_normalized, 
                           cmap='viridis', s=100, edgecolors='black', linewidths=0.5,
                           label='Actual Trajectory')
    ax2.plot(pos_array[:, 0], pos_array[:, 1], '--', color='gray', linewidth=1, alpha=0.5)
    # Add time labels for all points
    for i in range(len(pos_array)):
        ax2.annotate(f't={time_normalized[i]:.3f}s', 
                    (pos_array[i, 0], pos_array[i, 1]),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
    ax2.set_xlabel('X (m) - base_link', fontsize=11)
    ax2.set_ylabel('Y (m) - base_link', fontsize=11)
    ax2.set_title('X-Y Projection (base_link frame)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Time (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Window 2: X-Z + Y-Z projections
    fig2 = plt.figure(figsize=(16, 8))
    
    # X-Z projection (showing parabola)
    ax3 = fig2.add_subplot(1, 2, 1)
    scatter3 = ax3.scatter(pos_array[:, 0], pos_array[:, 2], c=time_normalized, 
                           cmap='viridis', s=100, edgecolors='black', linewidths=0.5,
                           label='Actual Trajectory')
    ax3.plot(pos_array[:, 0], pos_array[:, 2], '--', color='gray', linewidth=1, alpha=0.5)
    # Add time labels for all points
    for i in range(len(pos_array)):
        ax3.annotate(f't={time_normalized[i]:.3f}s', 
                    (pos_array[i, 0], pos_array[i, 2]),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
    ax3.set_xlabel('X (m) - base_link', fontsize=11)
    ax3.set_ylabel('Z (m) - base_link', fontsize=11)
    ax3.set_title('X-Z Projection (base_link frame)', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3, label='Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Y-Z projection
    ax4 = fig2.add_subplot(1, 2, 2)
    scatter4 = ax4.scatter(pos_array[:, 1], pos_array[:, 2], c=time_normalized, 
                           cmap='viridis', s=100, edgecolors='black', linewidths=0.5,
                           label='Actual Trajectory')
    ax4.plot(pos_array[:, 1], pos_array[:, 2], '--', color='gray', linewidth=1, alpha=0.5)
    # Add time labels for all points
    for i in range(len(pos_array)):
        ax4.annotate(f't={time_normalized[i]:.3f}s', 
                    (pos_array[i, 1], pos_array[i, 2]),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
    ax4.set_xlabel('Y (m) - base_link', fontsize=11)
    ax4.set_ylabel('Z (m) - base_link', fontsize=11)
    ax4.set_title('Y-Z Projection (base_link frame)', fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(scatter4, ax=ax4, label='Time (s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figures
    input_path = Path(rosbag_path_or_csv)
    output_path1 = input_path.parent / f"{input_path.stem}_visualization_1.png"
    output_path2 = input_path.parent / f"{input_path.stem}_visualization_2.png"
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to:")
    print(f"  Window 1: {output_path1}")
    print(f"  Window 2: {output_path2}")
    
    plt.show()


if __name__ == "__main__":
    import sys
    
    # Default path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Try to find CSV first, then rosbag
        base_path = Path(__file__).parent.parent.parent / "rosbag" / "ball_tracking_data_3"
        csv_path = base_path.parent / f"{base_path.name}_pose.csv"
        if csv_path.exists():
            input_path = csv_path
        else:
            input_path = base_path
    
    # Check if input is CSV (no ROS 2 needed) or rosbag (needs ROS 2)
    path = Path(input_path)
    if path.suffix == '.csv':
        # CSV file - no ROS 2 needed
        visualize(input_path)
    else:
        # Rosbag - needs ROS 2
        if ROS2_AVAILABLE and not ROSBAG2_AVAILABLE:
            rclpy.init()
            try:
                visualize(input_path)
            finally:
                rclpy.shutdown()
        else:
            visualize(input_path)

