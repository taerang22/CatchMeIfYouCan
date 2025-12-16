#!/usr/bin/env python3
"""
Compare predicted parabola with actual trajectory for data 16 (0~0.68s).
Uses the same prediction method as ball_fitting_predict_node.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from visualize_prediction
sys.path.insert(0, str(Path(__file__).parent))
from visualize_prediction import read_csv


def predict_from_parabola(pos_buffer, time_buffer):
    """
    Predict using parabola fitting:
    - X(t), Z(t): Linear (straight line)
    - Y(t): Parabola (quadratic)
    """
    pos = np.array(pos_buffer)  # shape (5, 3)
    t = np.arange(len(pos), dtype=float)  # [0, 1, 2, 3, 4]
    
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    
    # Linear fits for x(t), z(t) - straight lines
    A_lin = np.vstack([t, np.ones_like(t)]).T
    a_x, b_x = np.linalg.lstsq(A_lin, x, rcond=None)[0]
    a_z, b_z = np.linalg.lstsq(A_lin, z, rcond=None)[0]
    
    # Quadratic fit for y(t) - parabola
    A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
    A_y, B_y, C_y = np.linalg.lstsq(A_quad, y, rcond=None)[0]
    
    return {
        'x': (a_x, b_x),
        'y': (A_y, B_y, C_y),
        'z': (a_z, b_z)
    }


def compare_prediction_actual(csv_path, max_time=0.68, n_points=5):
    """Compare predicted parabola with actual trajectory"""
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return
    
    print(f"Reading {csv_file.name}...")
    
    # Read CSV
    poses, _ = read_csv(csv_file)
    
    if len(poses) < n_points:
        print(f"Error: Need at least {n_points} poses, found {len(poses)}")
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
        time_normalized = (timestamps - t_start) * 1e-9
    else:
        time_normalized = np.arange(len(pos_array))
    
    # Filter data from 0 to max_time seconds
    mask = (time_normalized >= 0) & (time_normalized <= max_time)
    t_filtered = time_normalized[mask]
    pos_filtered = pos_array[mask]
    
    print(f"\nData points in range [0, {max_time}]s: {len(t_filtered)}")
    
    if len(t_filtered) < n_points:
        print(f"Error: Need at least {n_points} points, found {len(t_filtered)}")
        return
    
    # Use first n_points for prediction
    pos_buffer = pos_filtered[:n_points].tolist()
    time_buffer = t_filtered[:n_points].tolist()
    
    print(f"\nUsing first {n_points} points for prediction:")
    for i in range(n_points):
        print(f"  Point {i+1}: t={time_buffer[i]:.3f}s, "
              f"X={pos_buffer[i][0]:.3f}, Y={pos_buffer[i][1]:.3f}, Z={pos_buffer[i][2]:.3f}")
    
    # Perform prediction using parabola fitting
    fitted_params = predict_from_parabola(pos_buffer, time_buffer)
    
    a_x, b_x = fitted_params['x']
    A_y, B_y, C_y = fitted_params['y']
    a_z, b_z = fitted_params['z']
    
    print(f"\nPredicted Parameters (from first {n_points} points):")
    print(f"  x(t) = {a_x:.6f} * t + {b_x:.6f}  (linear)")
    print(f"  y(t) = {A_y:.6f} * t² + {B_y:.6f} * t + {C_y:.6f}  (parabola)")
    print(f"  z(t) = {a_z:.6f} * t + {b_z:.6f}  (linear)")
    
    # Generate predicted trajectory for all time points
    # Normalize time to [0, 1, 2, ..., n_points-1] for the first n_points
    # Then extend to all filtered time points
    t_normalized = (t_filtered - t_filtered[0]) / (t_filtered[n_points-1] - t_filtered[0]) * (n_points - 1)
    
    # Predicted values
    x_pred = a_x * t_normalized + b_x
    y_pred = A_y * t_normalized**2 + B_y * t_normalized + C_y
    z_pred = a_z * t_normalized + b_z
    
    # Fit actual Y trajectory using ALL filtered data points
    # Use the SAME time normalization scale for fair comparison
    y_actual_data = pos_filtered[:, 1]
    
    # Fit parabola to actual Y data using the same normalized time scale
    A_quad_actual = np.vstack([t_normalized**2, t_normalized, np.ones_like(t_normalized)]).T
    A_y_actual, B_y_actual, C_y_actual = np.linalg.lstsq(A_quad_actual, y_actual_data, rcond=None)[0]
    
    print(f"\nActual Y Parameters (from all {len(t_filtered)} points, same time scale):")
    print(f"  y(t) = {A_y_actual:.6f} * t² + {B_y_actual:.6f} * t + {C_y_actual:.6f}  (parabola)")
    
    print(f"\nY Equation Comparison (normalized time scale):")
    print(f"  Predicted: y(t) = {A_y:.6f} * t² + {B_y:.6f} * t + {C_y:.6f}")
    print(f"  Actual:    y(t) = {A_y_actual:.6f} * t² + {B_y_actual:.6f} * t + {C_y_actual:.6f}")
    print(f"  Difference in coefficients:")
    print(f"    A (t²): {abs(A_y - A_y_actual):.6f} ({abs((A_y - A_y_actual)/A_y_actual*100) if A_y_actual != 0 else 0:.2f}%)")
    print(f"    B (t):  {abs(B_y - B_y_actual):.6f} ({abs((B_y - B_y_actual)/B_y_actual*100) if B_y_actual != 0 else 0:.2f}%)")
    print(f"    C:      {abs(C_y - C_y_actual):.6f} ({abs((C_y - C_y_actual)/C_y_actual*100) if C_y_actual != 0 else 0:.2f}%)")
    
    # Actual Y using fitted equation (for comparison)
    y_actual_fitted = A_y_actual * t_normalized**2 + B_y_actual * t_normalized + C_y_actual
    
    # Actual values
    x_actual = pos_filtered[:, 0]
    y_actual = pos_filtered[:, 1]
    z_actual = pos_filtered[:, 2]
    
    # Calculate errors
    x_error = np.abs(x_actual - x_pred)
    y_error = np.abs(y_actual - y_pred)
    z_error = np.abs(z_actual - z_pred)
    
    print(f"\nPrediction Errors (mean absolute error):")
    print(f"  X: {np.mean(x_error):.6f} m")
    print(f"  Y: {np.mean(y_error):.6f} m")
    print(f"  Z: {np.mean(z_error):.6f} m")
    
    # Create visualization: 3 subplots for X-t, Y-t, Z-t
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # X vs Time
    ax1 = axes[0]
    ax1.plot(t_filtered, x_actual, 'o-', color='blue', linewidth=2, 
             markersize=8, label='Actual X', alpha=0.8)
    ax1.plot(t_filtered, x_pred, '--', color='red', linewidth=2, 
             label=f'Predicted X (linear)', alpha=0.8)
    ax1.fill_between(t_filtered, x_actual - x_error, x_actual + x_error, 
                    alpha=0.2, color='red', label='Error')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X (m)', fontsize=11)
    ax1.set_title(f'X(t) - Actual vs Predicted (0~{max_time}s)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.02, max_time + 0.02)
    
    # Y vs Time (parabola)
    ax2 = axes[1]
    ax2.plot(t_filtered, y_actual, 'o-', color='green', linewidth=2, 
             markersize=8, label='Actual Y (data points)', alpha=0.8)
    ax2.plot(t_filtered, y_actual_fitted, '-', color='blue', linewidth=2, 
             label=f'Actual Y fitted: {A_y_actual:.3f}t² + {B_y_actual:.3f}t + {C_y_actual:.3f}', alpha=0.8)
    ax2.plot(t_filtered, y_pred, '--', color='red', linewidth=2, 
             label=f'Predicted Y: {A_y:.3f}t² + {B_y:.3f}t + {C_y:.3f}', alpha=0.8)
    ax2.fill_between(t_filtered, y_actual - y_error, y_actual + y_error, 
                    alpha=0.2, color='red', label='Error')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title(f'Y(t) - Actual vs Predicted Parabola (0~{max_time}s, using {n_points} points)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-0.02, max_time + 0.02)
    
    # Z vs Time (linear)
    ax3 = axes[2]
    ax3.plot(t_filtered, z_actual, 'o-', color='purple', linewidth=2, 
             markersize=8, label='Actual Z', alpha=0.8)
    ax3.plot(t_filtered, z_pred, '--', color='red', linewidth=2, 
             label=f'Predicted Z (linear)', alpha=0.8)
    ax3.fill_between(t_filtered, z_actual - z_error, z_actual + z_error, 
                    alpha=0.2, color='red', label='Error')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Z (m)', fontsize=11)
    ax3.set_title(f'Z(t) - Actual vs Predicted (0~{max_time}s)', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-0.02, max_time + 0.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_file.parent.parent / f"data_16_prediction_comparison_{n_points}points.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return {
        'predicted': (A_y, B_y, C_y),
        'actual': (A_y_actual, B_y_actual, C_y_actual)
    }


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent / "rosbag"
    csv_path = base_dir / "ball_tracking_data_16" / "kinova_ball_init_pose.csv"
    
    print("="*70)
    print("Analysis with first 5 points")
    print("="*70)
    result_5 = compare_prediction_actual(csv_path, max_time=0.68, n_points=5)
    
    print("\n\n" + "="*70)
    print("Analysis with first 10 points")
    print("="*70)
    result_10 = compare_prediction_actual(csv_path, max_time=0.68, n_points=10)

