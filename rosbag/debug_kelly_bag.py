#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# Load CSV files
# ---------------------------------------------
init_df = pd.read_csv("ball_tracking_data_20/kinova_ball_init_pose.csv")
goal_df = pd.read_csv("ball_tracking_data_20/kinova_goal_pose.csv")

# ---------------------------------------------
# Extract initial pose points (already in base frame!)
# ---------------------------------------------
init_points = init_df[["position_x", "position_y", "position_z"]].values

# time for color
times = init_df["timestamp_sec"] + init_df["timestamp_nanosec"] * 1e-9
times_norm = (times - times.min()) / (times.max() - times.min() + 1e-9)

# ---------------------------------------------
# Extract goal pose points (base frame)
# ---------------------------------------------
goal_points = goal_df[["position_x", "position_y", "position_z"]].values

# ---------------------------------------------
# Plotting
# ---------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# initial trajectory
sc = ax.scatter(
    init_points[:, 0], init_points[:, 1], init_points[:, 2],
    c=times_norm, cmap="viridis", s=60, label="Initial Pose (base_link)"
)

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Normalized time")

# goal pose points
ax.scatter(
    goal_points[:, 0], goal_points[:, 1], goal_points[:, 2],
    c="red", marker="x", s=80, label="Goal Pose"
)

ax.set_xlabel("X (base)")
ax.set_ylabel("Y (base)")
ax.set_zlabel("Z (base)")
ax.set_title("Base Frame Initial Pose Trajectory vs Goal Pose")
ax.legend()

plt.tight_layout()
plt.show()
