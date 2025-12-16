#!/usr/bin/env python3
"""
Visualize Ball Trajectory in Robot Base Frame
----------------------------------------------
Reads CSV file (ArUco frame) → Transforms to Robot frame → 3D Plot

Usage:
    python visualize_trajectory_robot_frame.py ball_tracking_data_16/kinova_ball_init_pose.csv
    python visualize_trajectory_robot_frame.py ball_tracking_data_16  # auto-finds CSV
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


# ============================================================
# ArUco → Robot Base Transform (from ball_prediction_node.py)
# ============================================================
x_offset = 0.07
T_AR_TO_ROBOT = np.array([
    1.45 + x_offset,  # +152 cm in robot X
    -0.31,            # -31 cm in robot Y
    0.89              # +89 cm in robot Z
], dtype=float).reshape(3, 1)

# Rotation: robot_x = aruco_z, robot_y = aruco_x, robot_z = aruco_y
R_AR_TO_ROBOT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)


def transform_aruco_to_robot(pos_aruco: np.ndarray) -> np.ndarray:
    """Transform position from ArUco frame to Robot base_link frame."""
    pa = np.array(pos_aruco).reshape(3, 1)
    pr = R_AR_TO_ROBOT @ pa + T_AR_TO_ROBOT
    return pr.flatten()


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def main():
    # Parse argument
    if len(sys.argv) < 2:
        print("Usage: python visualize_trajectory_robot_frame.py <csv_file_or_folder>")
        print("Example: python visualize_trajectory_robot_frame.py ball_tracking_data_22")
        sys.exit(1)

    path = Path(sys.argv[1])
    
    # Find CSV file
    if path.is_dir():
        csv_files = list(path.glob("*init_pose*.csv"))
        if not csv_files:
            print(f"No *init_pose*.csv found in {path}")
            sys.exit(1)
        csv_path = csv_files[0]
    else:
        csv_path = path
    
    print(f"Loading: {csv_path}")
    df = load_csv(csv_path)
    
    # Extract ArUco frame positions
    pos_aruco = df[['position_x', 'position_y', 'position_z']].values
    
    # Transform to robot frame
    pos_robot = np.array([transform_aruco_to_robot(p) for p in pos_aruco])
    
    # Calculate timestamps (relative to start)
    t_sec = df['timestamp_sec'].values
    t_nsec = df['timestamp_nanosec'].values
    timestamps = t_sec + t_nsec * 1e-9
    timestamps = timestamps - timestamps[0]  # Relative to start
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"Trajectory Statistics ({len(pos_robot)} samples)")
    print(f"{'='*60}")
    print(f"\nArUco Frame (original):")
    print(f"  X: {pos_aruco[:, 0].min():.3f} to {pos_aruco[:, 0].max():.3f} m")
    print(f"  Y: {pos_aruco[:, 1].min():.3f} to {pos_aruco[:, 1].max():.3f} m")
    print(f"  Z: {pos_aruco[:, 2].min():.3f} to {pos_aruco[:, 2].max():.3f} m")
    
    print(f"\nRobot Base Frame (transformed):")
    print(f"  X: {pos_robot[:, 0].min():.3f} to {pos_robot[:, 0].max():.3f} m")
    print(f"  Y: {pos_robot[:, 1].min():.3f} to {pos_robot[:, 1].max():.3f} m")
    print(f"  Z: {pos_robot[:, 2].min():.3f} to {pos_robot[:, 2].max():.3f} m")
    
    print(f"\nTime span: {timestamps[-1]:.3f} seconds")
    print(f"Average rate: {len(timestamps)/timestamps[-1]:.1f} Hz")
    
    # ============================================================
    # 3D Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 6))
    
    # ----- Plot 1: ArUco Frame (Original) -----
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(pos_aruco[:, 0], pos_aruco[:, 1], pos_aruco[:, 2], 
                           c=timestamps, cmap='viridis', s=30, alpha=0.8)
    ax1.plot(pos_aruco[:, 0], pos_aruco[:, 1], pos_aruco[:, 2], 
             'b-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax1.scatter(*pos_aruco[0], color='green', s=100, marker='o', label='Start')
    ax1.scatter(*pos_aruco[-1], color='red', s=100, marker='x', label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('ArUco Frame (Original Data)')
    ax1.legend()
    
    # ----- Plot 2: Robot Frame (Transformed) -----
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(pos_robot[:, 0], pos_robot[:, 1], pos_robot[:, 2], 
                           c=timestamps, cmap='plasma', s=30, alpha=0.8)
    ax2.plot(pos_robot[:, 0], pos_robot[:, 1], pos_robot[:, 2], 
             'r-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax2.scatter(*pos_robot[0], color='green', s=100, marker='o', label='Start')
    ax2.scatter(*pos_robot[-1], color='red', s=100, marker='x', label='End')
    
    # Draw robot base (origin)
    ax2.scatter(0, 0, 0, color='black', s=200, marker='^', label='Robot Base')
    
    # Draw catching plane x = 0.576
    FIXED_X = 0.576
    y_range = np.linspace(-0.5, 0.5, 10)
    z_range = np.linspace(0, 0.8, 10)
    Y_plane, Z_plane = np.meshgrid(y_range, z_range)
    X_plane = np.full_like(Y_plane, FIXED_X)
    ax2.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='cyan')
    ax2.text(FIXED_X, 0, 0.9, f'Catch Plane\nx={FIXED_X}', fontsize=8)
    
    ax2.set_xlabel('X (m) - toward robot')
    ax2.set_ylabel('Y (m) - left/right')
    ax2.set_zlabel('Z (m) - height')
    ax2.set_title('Robot Base Frame (Transformed)')
    ax2.legend()
    
    # Set reasonable limits for robot frame
    ax2.set_xlim(0, 2.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(-0.5, 1.5)
    
    # ----- Plot 3: 2D Top View (Robot Frame) -----
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(pos_robot[:, 0], pos_robot[:, 1], 
                           c=timestamps, cmap='plasma', s=30, alpha=0.8)
    ax3.plot(pos_robot[:, 0], pos_robot[:, 1], 'r-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax3.scatter(pos_robot[0, 0], pos_robot[0, 1], color='green', s=100, marker='o', label='Start')
    ax3.scatter(pos_robot[-1, 0], pos_robot[-1, 1], color='red', s=100, marker='x', label='End')
    
    # Robot base
    ax3.scatter(0, 0, color='black', s=200, marker='^', label='Robot Base')
    
    # Catching line
    ax3.axvline(x=FIXED_X, color='cyan', linestyle='--', alpha=0.5, label=f'Catch x={FIXED_X}')
    
    ax3.set_xlabel('X (m) - toward robot')
    ax3.set_ylabel('Y (m) - left/right')
    ax3.set_title('Top View (Robot Frame)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(0, 2.0)
    ax3.set_ylim(-1.0, 1.0)
    
    plt.colorbar(scatter3, ax=ax3, label='Time (s)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.parent / f"{csv_path.stem}_robot_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    plt.show()
    
    # ============================================================
    # Export transformed data
    # ============================================================
    df_robot = df.copy()
    df_robot['robot_x'] = pos_robot[:, 0]
    df_robot['robot_y'] = pos_robot[:, 1]
    df_robot['robot_z'] = pos_robot[:, 2]
    
    output_csv = csv_path.parent / f"{csv_path.stem}_robot_frame.csv"
    df_robot.to_csv(output_csv, index=False)
    print(f"Saved transformed CSV to: {output_csv}")


if __name__ == "__main__":
    main()

