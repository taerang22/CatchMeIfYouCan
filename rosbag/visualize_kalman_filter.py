#!/usr/bin/env python3
"""
Visualize Kalman Filter smoothing on ball trajectory
----------------------------------------------------
Reads CSV with:
    timestamp_sec, timestamp_nanosec, position_x, position_y, position_z

Applies 1D Kalman Filter on x, y, z and plots:
    - Raw vs Smoothed position
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


# ------------------------------------------------------------
# Minimal 1D Kalman Filter
# ------------------------------------------------------------
class SimpleKalman1D:
    def __init__(self, Q=0.001, R=0.02):
        self.x = None
        self.P = 1.0
        self.Q = Q
        self.R = R

    def update(self, z):
        if self.x is None:
            self.x = z
            return z

        # Prediction
        x_pred = self.x
        P_pred = self.P + self.Q

        # Kalman gain
        K = P_pred / (P_pred + self.R)

        # Update
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


# ------------------------------------------------------------
# CSV Reader (Matching Your Data Format)
# ------------------------------------------------------------
def load_csv(csv_path):
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

    positions = np.array(positions)
    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]  # normalize

    return positions, timestamps


# ------------------------------------------------------------
# Apply KF on entire CSV trajectory
# ------------------------------------------------------------
def apply_kalman_filter(positions):
    kfx = SimpleKalman1D(Q=0.001, R=0.02)
    kfy = SimpleKalman1D(Q=0.001, R=0.02)
    kfz = SimpleKalman1D(Q=0.001, R=0.05)

    smoothed = []

    for x, y, z in positions:
        xs = kfx.update(x)
        ys = kfy.update(y)
        zs = kfz.update(z)
        smoothed.append([xs, ys, zs])

    return np.array(smoothed)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def visualize_kf(csv_path):
    positions, t = load_csv(csv_path)
    smoothed = apply_kalman_filter(positions)

    x_raw, y_raw, z_raw = positions.T
    x_kf, y_kf, z_kf = smoothed.T

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # X(t)
    axes[0].plot(t, x_raw, "bo-", markersize=3, alpha=0.6, label="Raw X")
    axes[0].plot(t, x_kf, "r-", linewidth=2, label="KF Smoothed X")
    axes[0].set_title("X(t) — Raw vs Kalman Filter")
    axes[0].set_ylabel("X [m]")
    axes[0].grid(True)
    axes[0].legend()

    # Y(t)
    axes[1].plot(t, y_raw, "go-", markersize=3, alpha=0.6, label="Raw Y")
    axes[1].plot(t, y_kf, "r-", linewidth=2, label="KF Smoothed Y")
    axes[1].set_title("Y(t) — Raw vs Kalman Filter")
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True)
    axes[1].legend()

    # Z(t) — 가장 중요한 plot (튀는 데이터 확인)
    axes[2].plot(t, z_raw, "mo-", markersize=3, alpha=0.6, label="Raw Z (Noisy)")
    axes[2].plot(t, z_kf, "r-", linewidth=2, label="KF Smoothed Z")
    axes[2].set_title("Z(t) — Raw vs Kalman Filter (Noise Suppression)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Z [m]")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()

    out_path = Path(csv_path).with_suffix("").__str__() + "_kf_vs_raw.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization to: {out_path}\n")

    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    DATASET = 13
    base = Path(__file__).parent
    csv_path = base / f"ball_tracking_data_{DATASET}" / "kinova_ball_init_pose.csv"

    print(f"Visualizing Kalman smoothing for:\n{csv_path}\n")
    visualize_kf(csv_path)
