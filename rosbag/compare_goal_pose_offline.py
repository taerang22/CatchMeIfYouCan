import pandas as pd
import numpy as np

CSV_PATH = "ball_tracking_data_22/kinova_ball_init_pose.csv"

# Node parameters
GRIPPER_OFFSET = 0.15
X_PLANE_CALC = 0.576 - GRIPPER_OFFSET      # for solving
X_PLANE_PUB  = 0.576                       # final publish x
MIN_Z = 0.10
UPDATE_THRESHOLD = 0.05                    # y/z difference threshold to update


def compute_goal_from_five(pos5):
    """Compute goal pose from 5 samples (pos5 = shape (5,3))."""
    t = np.arange(5, dtype=float)

    x = pos5[:,0]
    y = pos5[:,1]
    z = pos5[:,2]

    # Linear fit for x,y
    A_lin = np.vstack([t, np.ones_like(t)]).T
    a_x, b_x = np.linalg.lstsq(A_lin, x, rcond=None)[0]
    a_y, b_y = np.linalg.lstsq(A_lin, y, rcond=None)[0]

    # Quadratic fit for z
    A_quad = np.vstack([t**2, t, np.ones_like(t)]).T
    A_z, B_z, C_z = np.linalg.lstsq(A_quad, z, rcond=None)[0]

    # Solve intersection
    if abs(a_x) < 1e-6:
        return None

    t_hit = (X_PLANE_CALC - b_x) / a_x
    if t_hit <= 0:
        return None

    y_hit = a_y * t_hit + b_y
    z_hit = A_z*t_hit**2 + B_z*t_hit + C_z
    z_hit = max(z_hit, MIN_Z)

    return np.array([X_PLANE_PUB, y_hit, z_hit])


# ------------------------
# Load CSV
# ------------------------
df = pd.read_csv(CSV_PATH)
pos_list = df[["position_x", "position_y", "position_z"]].to_numpy()

buffer = []       # store last 6 samples (1 skip + 5 fit)
current_goal = None


# ------------------------
# Replay CSV like the node
# ------------------------
for p in pos_list:
    buffer.append(p)

    # keep max 6 samples
    if len(buffer) > 6:
        buffer.pop(0)

    # Need exactly 6 samples → skip first & use last 5
    if len(buffer) == 6:
        pos5 = np.array(buffer[1:])  # skip buffer[0]

        predicted = compute_goal_from_five(pos5)
        if predicted is None:
            continue

        # if no previous goal, accept first
        if current_goal is None:
            current_goal = predicted
            print(f"[INIT GOAL] {current_goal}")
            continue

        # compare difference in y,z
        dy = abs(predicted[1] - current_goal[1])
        dz = abs(predicted[2] - current_goal[2])

        if dy > UPDATE_THRESHOLD or dz > UPDATE_THRESHOLD:
            print(f"[GOAL UPDATED] old={current_goal}, new={predicted}")
            current_goal = predicted

# ------------------------
# Final Goal Pose
# ------------------------
print("\n==========================")
print(" FINAL OFFLINE GOAL POSE ")
print("==========================")
print(f"x={current_goal[0]:.4f}, y={current_goal[1]:.4f}, z={current_goal[2]:.4f}")
