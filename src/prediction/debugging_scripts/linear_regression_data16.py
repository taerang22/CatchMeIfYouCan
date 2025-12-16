#!/usr/bin/env python3
"""
Linear regression for data 16: X(t) and Z(t) from 0 to 0.6 seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from visualize_prediction
sys.path.insert(0, str(Path(__file__).parent))
from visualize_prediction import read_csv


def linear_regression_0_06s(csv_path):
    """Perform linear regression for X(t) and Z(t) from 0 to 0.6 seconds"""
    
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
    
    # Filter data from 0 to 0.6 seconds
    mask = (time_normalized >= 0) & (time_normalized <= 0.6)
    t_filtered = time_normalized[mask]
    x_filtered = pos_array[mask, 0]
    z_filtered = pos_array[mask, 2]
    
    print(f"\nData points in range [0, 0.6]s: {len(t_filtered)}")
    if len(t_filtered) < 2:
        print("Error: Need at least 2 points for linear regression")
        return
    
    # Linear regression for X(t): X = a_x * t + b_x
    A_x = np.vstack([t_filtered, np.ones_like(t_filtered)]).T
    a_x, b_x = np.linalg.lstsq(A_x, x_filtered, rcond=None)[0]
    
    # Linear regression for Z(t): Z = a_z * t + b_z
    A_z = np.vstack([t_filtered, np.ones_like(t_filtered)]).T
    a_z, b_z = np.linalg.lstsq(A_z, z_filtered, rcond=None)[0]
    
    # Calculate R-squared
    x_pred = a_x * t_filtered + b_x
    z_pred = a_z * t_filtered + b_z
    
    x_ss_res = np.sum((x_filtered - x_pred) ** 2)
    x_ss_tot = np.sum((x_filtered - np.mean(x_filtered)) ** 2)
    x_r2 = 1 - (x_ss_res / x_ss_tot) if x_ss_tot > 0 else 0
    
    z_ss_res = np.sum((z_filtered - z_pred) ** 2)
    z_ss_tot = np.sum((z_filtered - np.mean(z_filtered)) ** 2)
    z_r2 = 1 - (z_ss_res / z_ss_tot) if z_ss_tot > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("Linear Regression Results (t: 0 ~ 0.6 seconds)")
    print("="*70)
    print(f"\nX(t) = a_x * t + b_x")
    print(f"  a_x (slope):     {a_x:.6f} m/s")
    print(f"  b_x (intercept): {b_x:.6f} m")
    print(f"  R²:              {x_r2:.6f}")
    print(f"  Equation:        X(t) = {a_x:.6f} * t + {b_x:.6f}")
    
    print(f"\nZ(t) = a_z * t + b_z")
    print(f"  a_z (slope):     {a_z:.6f} m/s")
    print(f"  b_z (intercept): {b_z:.6f} m")
    print(f"  R²:              {z_r2:.6f}")
    print(f"  Equation:        Z(t) = {a_z:.6f} * t + {b_z:.6f}")
    print("="*70)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # X vs Time
    ax1 = axes[0]
    ax1.scatter(t_filtered, x_filtered, color='blue', s=100, 
               edgecolors='black', linewidths=0.5, label='Data points', zorder=3)
    t_line = np.linspace(0, 0.6, 100)
    x_line = a_x * t_line + b_x
    ax1.plot(t_line, x_line, 'r--', linewidth=2, 
            label=f'Linear fit: X(t) = {a_x:.3f}t + {b_x:.3f}')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X (m)', fontsize=11)
    ax1.set_title(f'X(t) Linear Regression (0~0.6s, R²={x_r2:.4f})', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.05, 0.65)
    
    # Z vs Time
    ax2 = axes[1]
    ax2.scatter(t_filtered, z_filtered, color='red', s=100, 
               edgecolors='black', linewidths=0.5, label='Data points', zorder=3)
    z_line = a_z * t_line + b_z
    ax2.plot(t_line, z_line, 'b--', linewidth=2, 
            label=f'Linear fit: Z(t) = {a_z:.3f}t + {b_z:.3f}')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Z (m)', fontsize=11)
    ax2.set_title(f'Z(t) Linear Regression (0~0.6s, R²={z_r2:.4f})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.05, 0.65)
    
    plt.tight_layout()
    
    # Save figure
    data_num = csv_file.parent.name.split('_')[-1]
    output_path = csv_file.parent.parent / f"data_{data_num}_linear_regression.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent / "rosbag"
    
    # Process both data 14 and 16
    for data_num in [14, 16]:
        csv_path = base_dir / f"ball_tracking_data_{data_num}" / "kinova_ball_init_pose.csv"
        if csv_path.exists():
            print(f"\n{'='*70}")
            print(f"Processing Data {data_num}")
            print(f"{'='*70}")
            linear_regression_0_06s(csv_path)
        else:
            print(f"Warning: {csv_path} not found")

