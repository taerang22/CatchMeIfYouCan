#!/usr/bin/env python3
"""
Verify parabola fitting using the first 10 trajectory points.
Fits:
    x(t) = a_x t + b_x
    y(t) = A_y t^2 + B_y t + C_y
    z(t) = a_z t + b_z
Compares predicted vs. actual, computes errors, and visualizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys


# --------------------------------------------------------
# CSV Loader (Matches Your Data Format)
# --------------------------------------------------------
def read_csv_default(csv_path):
    """
    Reads CSV with fields:
    timestamp_sec, timestamp_nanosec, position_x, position_y, position_z, ...
    """
    positions = []
    timestamps = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x = float(row["position_x"])
            y = float(row["position_y"])
            z = float(row["position_z"])

            t = float(row["timestamp_sec"]) + float(row["timestamp_nanosec"]) * 1e-9

            positions.append([x, y, z])
            timestamps.append(t)

    return np.array(positions), np.array(timestamps)


# --------------------------------------------------------
# Fit Parabola from first 10 samples
# --------------------------------------------------------
def fit_parabola(pos_buffer, t_buffer):
    pos = np.array(pos_buffer)
    t = np.array(t_buffer)

    # Normalize time to start from 0
    t = t - t[0]

    # Linear fit: X(t), Z(t)
    A_lin = np.vstack([t, np.ones_like(t)]).T
    a_x, b_x = np.linalg.lstsq(A_lin, pos[:, 0], rcond=None)[0]
    a_z, b_z = np.linalg.lstsq(A_lin, pos[:, 2], rcond=None)[0]

    # Quadratic fit: Y(t)
    A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
    A_y, B_y, C_y = np.linalg.lstsq(A_quad, pos[:, 1], rcond=None)[0]

    return (a_x, b_x), (A_y, B_y, C_y), (a_z, b_z)


# --------------------------------------------------------
# Main Evaluation Function
# --------------------------------------------------------
def evaluate_fit(csv_path, n_points=10, max_time=None, save_plot=True):
    positions, timestamps = read_csv_default(csv_path)

    # Normalize time
    t = timestamps - timestamps[0]

    # Optional filtering by max_time
    if max_time is not None:
        mask = (t >= 0) & (t <= max_time)
        t = t[mask]
        positions = positions[mask]

    print(f"Loaded {len(positions)} total points.")

    if len(positions) < n_points:
        print(f"ERROR: Only found {len(positions)} points, but need {n_points}.")
        sys.exit(1)

    # First N points for fitting
    pos_buf = positions[:n_points]
    t_buf = t[:n_points]

    print("\nUsing first 10 points:")
    for i in range(n_points):
        print(f"  t={t_buf[i]:.4f}, x={pos_buf[i,0]:.4f}, y={pos_buf[i,1]:.4f}, z={pos_buf[i,2]:.4f}")

    # Fit models
    (a_x, b_x), (A_y, B_y, C_y), (a_z, b_z) = fit_parabola(pos_buf, t_buf)

    print("\nFitted Models:")
    print(f"  x(t) = {a_x:.6f} t + {b_x:.6f}")
    print(f"  y(t) = {A_y:.6f} t^2 + {B_y:.6f} t + {C_y:.6f}")
    print(f"  z(t) = {a_z:.6f} t + {b_z:.6f}")

    # Predict full trajectory
    t_full = t - t_buf[0]  # normalize

    x_pred = a_x * t_full + b_x
    y_pred = A_y * t_full**2 + B_y * t_full + C_y
    z_pred = a_z * t_full + b_z

    # Actual
    x_true = positions[:, 0]
    y_true = positions[:, 1]
    z_true = positions[:, 2]

    # Errors
    x_err = np.abs(x_true - x_pred)
    y_err = np.abs(y_true - y_pred)
    z_err = np.abs(z_true - z_pred)

    print("\nMean Absolute Errors:")
    print(f"  X error = {np.mean(x_err):.6f}")
    print(f"  Y error = {np.mean(y_err):.6f}")
    print(f"  Z error = {np.mean(z_err):.6f}")

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, x_true, "bo-", label="Actual X", alpha=0.7)
    axes[0].plot(t, x_pred, "r--", label="Predicted X")
    axes[0].set_title("X(t): Actual vs Predicted")
    axes[0].set_ylabel("X (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, y_true, "go-", label="Actual Y", alpha=0.7)
    axes[1].plot(t, y_pred, "r--", label="Predicted Y")
    axes[1].set_title("Y(t): Actual vs Predicted (Quadratic)")
    axes[1].set_ylabel("Y (m)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, z_true, "mo-", label="Actual Z", alpha=0.7)
    axes[2].plot(t, z_pred, "r--", label="Predicted Z")
    axes[2].set_title("Z(t): Actual vs Predicted")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Z (m)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_plot:
        out_path = Path(csv_path).with_suffix("").__str__() + "_fit10.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to: {out_path}")

    plt.show()

    return {
        "fit": {
            "x": (a_x, b_x),
            "y": (A_y, B_y, C_y),
            "z": (a_z, b_z)
        },
        "errors": {
            "x": float(np.mean(x_err)),
            "y": float(np.mean(y_err)),
            "z": float(np.mean(z_err)),
        }
    }


# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == "__main__":

    # Example usage: choose dataset number here
    DATASET = 13

    base = Path(__file__).parent
    csv_path = base / f"ball_tracking_data_{DATASET}" / "kinova_ball_init_pose.csv"

    print("Using CSV:", csv_path)

    evaluate_fit(csv_path, n_points=10, max_time=0.7)
