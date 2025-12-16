#!/usr/bin/env python3
import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# CDR DECODERS
# ================================================================

def read_cdr_string(cdr, offset):
    (strlen,) = struct.unpack_from("<I", cdr, offset)
    offset += 4

    s = cdr[offset : offset + strlen].decode("utf-8", errors="ignore")
    offset += strlen

    # 4-byte alignment
    pad = (4 - (offset % 4)) % 4
    offset += pad

    return s, offset


def deserialize_pose_stamped(cdr: bytes):
    offset = 4  # IMPORTANT: skip CDR encapsulation header

    # --- header.stamp ---
    sec, nanosec = struct.unpack_from("<iI", cdr, offset)
    offset += 8

    # --- frame ID ---
    frame_id, offset = read_cdr_string(cdr, offset)

    # ---- pose.position.x,y,z ----
    x, y, z = struct.unpack_from("<ddd", cdr, offset)
    offset += 24

    # orientation is skipped
    return x, y, z


# ================================================================
# ROSBAG READER
# ================================================================

def load_pose_messages(db3_path, topic_name):
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Topic not found: {topic_name}")
    topic_id = row[0]

    cursor.execute("SELECT data FROM messages WHERE topic_id=?", (topic_id,))
    rows = cursor.fetchall()

    positions = []

    for (raw,) in rows:
        try:
            x, y, z = deserialize_pose_stamped(raw)
            positions.append([x, y, z])
        except Exception as e:
            print("Parse error:", e)

    positions = np.array(positions, float)
    print(f"Loaded {positions.shape[0]} PoseStamped msgs.")
    return positions


# ================================================================
# PLOTTER
# ================================================================

def plot_positions(positions):
    if len(positions) == 0:
        print("No positions to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:,0],
        positions[:,1],
        positions[:,2],
        marker='o',
        markersize=3,
        linestyle='-'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ball Trajectory from rosbag (.db3) WITHOUT ROS2")

    plt.show()


# ================================================================
# MAIN ENTRY
# ================================================================

if __name__ == "__main__":
    bag = "ball_tracking_data_2/ball_tracking_data_2_0.db3"
    topic = "/kinova/ball/init_pose"

    positions = load_pose_messages(bag, topic)
    plot_positions(positions)
