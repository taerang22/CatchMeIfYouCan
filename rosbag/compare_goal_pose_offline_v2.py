import pandas as pd
import numpy as np

CSV_PATH = "ball_tracking_data_22/kinova_ball_init_pose.csv"

# v2 algorithm parameters
WINDOW = 6
FIXED_X = 0.576
GRAVITY = -9.81
MIN_Y, MAX_Y = -0.4, 0.4
MIN_Z, MAX_Z = 0.10, 0.75

# outlier threshold
Z_JUMP_THRESHOLD = 0.40  # 40 cm jump → reject as outlier


def simulate_v2_goal(pos_window, t_window):
    """
    Given window of last N positions and timestamps,
    compute v2-style goal pose.
    """

    pos = np.array(pos_window)
    t = np.array(t_window)

    # -----------------------------
    # Normalize time
    # -----------------------------
    t = t - t[0]
    dt = t[-1] - t[0]
    if dt <= 1e-6:
        return None

    # --------------------------------------
    # Outlier rejection: large z jumps
    # --------------------------------------
    dz = np.diff(pos[:, 2])
    if np.any(np.abs(dz) > Z_JUMP_THRESHOLD):
        return None

    # --------------------------------------
    # Compute short-term velocities
    # --------------------------------------
    vx = (pos[-1, 0] - pos[0, 0]) / dt
    vy = (pos[-1, 1] - pos[0, 1]) / dt

    x0 = pos[0, 0]
    y0 = pos[0, 1]
    z0 = pos[0, 2]

    # --------------------------------------
    # Solve vz using ballistic model
    # z(t1) = z0 + vz*t1 + 0.5*g*t1²
    # --------------------------------------
    t1 = t[1]
    if t1 > 1e-6:
        z1 = pos[1, 2]
        vz = (z1 - z0 - 0.5 * GRAVITY * t1 * t1) / t1
    else:
        return None

    # --------------------------------------
    # Solve time-to-hit for x = FIXED_X
    # FIXED_X = x0 + vx * t_hit
    # --------------------------------------
    if abs(vx) < 1e-6:
        return None

    t_hit = (FIXED_X - x0) / vx

    if t_hit < 0 or t_hit > 2.0:
        return None

    # --------------------------------------
    # Predict y,z at t_hit
    # --------------------------------------
    y_hit = y0 + vy * t_hit

    z_hit = z0 + vz * t_hit + 0.5 * GRAVITY * t_hit * t_hit

    # clamp workspace
    y_hit = float(np.clip(y_hit, MIN_Y, MAX_Y))
    z_hit = float(np.clip(z_hit, MIN_Z, MAX_Z))

    return np.array([FIXED_X, y_hit, z_hit])


# ========================================================
# Run Offline Simulation
# ========================================================

df = pd.read_csv(CSV_PATH)

positions = df[["position_x", "position_y", "position_z"]].to_numpy()
timestamps = df["timestamp_sec"].to_numpy() + df["timestamp_nanosec"].to_numpy() * 1e-9

pos_window = []
t_window = []
final_goal = None

for p, t in zip(positions, timestamps):
    pos_window.append(p)
    t_window.append(t)

    # keep sliding window
    if len(pos_window) > WINDOW:
        pos_window.pop(0)
        t_window.pop(0)

    # need full window
    if len(pos_window) < WINDOW:
        continue

    goal = simulate_v2_goal(pos_window, t_window)

    if goal is not None:
        final_goal = goal

# ========================================================
# Print result
# ========================================================
print("\n===========================")
print("  FINAL v2 GOAL POSE (CSV) ")
print("===========================")

if final_goal is None:
    print("No valid goal computed.")
else:
    print(f"x={final_goal[0]:.4f}, y={final_goal[1]:.4f}, z={final_goal[2]:.4f}")
