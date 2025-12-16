#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "ball_tracking_data_3/kinova_ball_init_pose.csv"

# Load rosbag CSV
df = pd.read_csv(CSV_PATH)
pos_list = df[["position_x", "position_y", "position_z"]].to_numpy()

# Node parameters
gripper_offset = 0.15
x_plane_calc = 0.576 - gripper_offset
x_plane_pub  = 0.576
min_z = 0.10
update_threshold = 0.05  # same as prediction node

# Buffers & state
pos_buffer = []
goal_results = []        # offline goal poses
prev_goal_y = None
prev_goal_z = None


def fit_and_predict(pos5):
    """pos5: shape (5,3) → return predicted goal pose tuple or None"""
    t = np.arange(5, dtype=float)

    x = pos5[:,0]
    y = pos5[:,1]
    z = pos5[:,2]

    # Fit x, y linearly
    A_lin = np.vstack([t, np.ones_like(t)]).T
    a_x, b_x = np.linalg.lstsq(A_lin, x, rcond=None)[0]
    a_y, b_y = np.linalg.lstsq(A_lin, y, rcond=None)[0]

    # Fit z quadratically
    A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
    A_z, B_z, C_z = np.linalg.lstsq(A_quad, z, rcond=None)[0]

    # Solve intersection with plane
    if abs(a_x) < 1e-6:
        return None

    t_hit = (x_plane_calc - b_x) / a_x
    if t_hit <= 0:
        return None

    y_hit = a_y * t_hit + b_y
    z_hit = A_z * (t_hit**2) + B_z * t_hit + C_z

    if z_hit < min_z:
        z_hit = min_z

    return (x_plane_pub, y_hit, z_hit)


# -------------------------
# REPLAY ROSBAG MESSAGE SEQUENCE
# -------------------------
for p in pos_list:
    pos_buffer.append(p)

    if len(pos_buffer) > 6:
        pos_buffer.pop(0)

    if len(pos_buffer) == 6:
        pos5 = pos_buffer[1:]   # skip 1st, use last 5
        goal = fit_and_predict(pos5)

        if goal is None:
            continue

        gx, gy, gz = goal

        # Same update rule as prediction node
        if prev_goal_y is not None:
            if (abs(gy - prev_goal_y) < update_threshold and
                abs(gz - prev_goal_z) < update_threshold):
                continue

        # Update accepted goal
        prev_goal_y = gy
        prev_goal_z = gz
        goal_results.append((gx, gy, gz))


goal_results = np.array(goal_results)

print("Offline predicted goal poses:")
print(goal_results)
print("Count:", len(goal_results))


# 3D VISUALIZATION
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")

if len(goal_results) > 0:
    ax.scatter(goal_results[:,0], goal_results[:,1], goal_results[:,2],
               c='red', s=50, label="Offline Predicted Goals")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Offline Goal Pose Prediction From Rosbag CSV")
ax.legend()

plt.show()
