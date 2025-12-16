#!/usr/bin/env python3
"""
Visualize data 16: Time series plots for X, Y, Z coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from visualize_prediction
sys.path.insert(0, str(Path(__file__).parent))
from visualize_prediction import read_csv


def visualize_time_series(csv_path):
    """Visualize time series for X, Y, Z coordinates"""
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return
    
    print(f"Reading {csv_file.name}...")
    
    # Read CSV
    poses, _ = read_csv(csv_file)
    
    if len(poses) < 2:
        print(f"Error: Only {len(poses)} poses found")
        return
    
    # Use original data (no transformation)
    positions = []
    timestamps = []
    for pose in poses:
        positions.append(pose['position'])
        timestamps.append(pose['timestamp'])
    
    # Sort by timestamp
    sorted_data = sorted(zip(timestamps, positions), key=lambda x: x[0])
    timestamps = [t for t, _ in sorted_data]
    positions = [p for _, p in sorted_data]
    
    pos_array = np.array(positions)
    timestamps = np.array(timestamps)
    
    # Normalize time to start from t=0
    if len(timestamps) > 0:
        t_start = timestamps[0]
        time_normalized = (timestamps - t_start) * 1e-9  # Convert nanoseconds to seconds
    else:
        time_normalized = np.arange(len(pos_array))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time vs X
    ax1 = axes[0]
    ax1.plot(time_normalized, pos_array[:, 0], 'o-', color='blue', 
             linewidth=2, markersize=8, label='X position')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X (m)', fontsize=11)
    ax1.set_title('Data 16 - X Position vs Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Time vs Y
    ax2 = axes[1]
    ax2.plot(time_normalized, pos_array[:, 1], 'o-', color='green', 
             linewidth=2, markersize=8, label='Y position')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('Data 16 - Y Position vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Time vs Z
    ax3 = axes[2]
    ax3.plot(time_normalized, pos_array[:, 2], 'o-', color='red', 
             linewidth=2, markersize=8, label='Z position')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Z (m)', fontsize=11)
    ax3.set_title('Data 16 - Z Position vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_file.parent.parent / "data_16_time_series.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent / "rosbag"
    csv_path = base_dir / "ball_tracking_data_16" / "kinova_ball_init_pose.csv"
    
    visualize_time_series(csv_path)

