#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "ball_tracking_data_22/kinova_ball_init_pose.csv"


# ============================================================
# Load CSV
# ============================================================
df = pd.read_csv(CSV_PATH)

# Extract columns explicitly
x = df["position_x"].to_numpy()
y = df["position_y"].to_numpy()
z = df["position_z"].to_numpy()

# construct time (sec + nanosec)
t = df["timestamp_sec"] + df["timestamp_nanosec"] * 1e-9
t = t - t.iloc[0]   # normalize (start = 0)


# ============================================================
# 1) TIME vs X,Y,Z
# ============================================================
fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t, x, "-o", markersize=3)
axs[0].set_ylabel("X (m)")
axs[0].grid()

axs[1].plot(t, y, "-o", markersize=3)
axs[1].set_ylabel("Y (m)")
axs[1].grid()

axs[2].plot(t, z, "-o", markersize=3)
axs[2].set_ylabel("Z (m)")
axs[2].set_xlabel("Time (s)")
axs[2].grid()

fig1.suptitle("Time vs X / Y / Z")
plt.tight_layout()
plt.show()


# ============================================================
# 2) 3D Trajectory colored by time
# ============================================================
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x, y, z,
    c=t,
    cmap="viridis",
    marker="o",
    s=15
)

# start & end points highlighted
ax.scatter(x[0], y[0], z[0], c="red", s=80, label="Start")
ax.scatter(x[-1], y[-1], z[-1], c="black", s=80, label="End")

fig2.colorbar(scatter, ax=ax, label="Time (s)")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3D Ball Trajectory (colored by time)")
ax.legend()

plt.show()


# ============================================================
# 3) 2D Plane Projections (XY, XZ, YZ)
# ============================================================
fig3, axarr = plt.subplots(1, 3, figsize=(14, 4))

axarr[0].plot(x, y, "-o", markersize=3)
axarr[0].set_xlabel("X")
axarr[0].set_ylabel("Y")
axarr[0].set_title("XY Projection")
axarr[0].grid()

axarr[1].plot(x, z, "-o", markersize=3)
axarr[1].set_xlabel("X")
axarr[1].set_ylabel("Z")
axarr[1].set_title("XZ Projection")
axarr[1].grid()

axarr[2].plot(y, z, "-o", markersize=3)
axarr[2].set_xlabel("Y")
axarr[2].set_ylabel("Z")
axarr[2].set_title("YZ Projection")
axarr[2].grid()

plt.tight_layout()
plt.show()
