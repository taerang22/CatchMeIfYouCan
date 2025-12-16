#!/usr/bin/env python3
"""
Simple ROS2 bag to CSV converter using sqlite3 directly.
Works without needing ROS2 environment.

Usage:
    python bag_to_csv_simple.py <bag_folder>
    python bag_to_csv_simple.py ball_tracking_data
    python bag_to_csv_simple.py --all
"""

import sys
import os
import sqlite3
import csv
import struct
from pathlib import Path


def parse_pose_stamped(data):
    """Parse geometry_msgs/PoseStamped from CDR serialization."""
    try:
        offset = 4  # Skip CDR header (00 01 00 00)
        
        # Header - stamp (sec: int32, nanosec: uint32)
        stamp_sec = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        stamp_nanosec = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # frame_id string (uint32 length + chars including null)
        str_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if str_len > 0:
            frame_id = data[offset:offset+str_len-1].decode('utf-8', errors='ignore')
            offset += str_len
        else:
            frame_id = ''
        
        # Align to 4 bytes (ROS2 CDR alignment after string)
        remainder = offset % 4
        if remainder != 0:
            offset += (4 - remainder)
        
        # Pose - Position (3 doubles: x, y, z)
        pos_x = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        pos_y = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        pos_z = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        
        # Pose - Orientation quaternion (4 doubles: x, y, z, w)
        ori_x = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        ori_y = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        ori_z = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        ori_w = struct.unpack_from('<d', data, offset)[0]
        
        return {
            'timestamp_sec': stamp_sec,
            'timestamp_nanosec': stamp_nanosec,
            'frame_id': frame_id,
            'position_x': pos_x,
            'position_y': pos_y,
            'position_z': pos_z,
            'orientation_x': ori_x,
            'orientation_y': ori_y,
            'orientation_z': ori_z,
            'orientation_w': ori_w,
        }
    except Exception as e:
        return None


def parse_twist_stamped(data):
    """Parse geometry_msgs/TwistStamped from CDR serialization."""
    try:
        offset = 4  # Skip CDR header
        
        # Header - stamp
        stamp_sec = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        stamp_nanosec = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # frame_id string
        str_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if str_len > 0:
            frame_id = data[offset:offset+str_len-1].decode('utf-8', errors='ignore')
            offset += str_len
        else:
            frame_id = ''
        
        # Align to 4 bytes
        remainder = offset % 4
        if remainder != 0:
            offset += (4 - remainder)
        
        # Twist - Linear velocity (3 doubles: x, y, z)
        lin_x = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        lin_y = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        lin_z = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        
        # Twist - Angular velocity (3 doubles: x, y, z)
        ang_x = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        ang_y = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        ang_z = struct.unpack_from('<d', data, offset)[0]
        
        return {
            'timestamp_sec': stamp_sec,
            'timestamp_nanosec': stamp_nanosec,
            'frame_id': frame_id,
            'linear_x': lin_x,
            'linear_y': lin_y,
            'linear_z': lin_z,
            'angular_x': ang_x,
            'angular_y': ang_y,
            'angular_z': ang_z,
        }
    except Exception as e:
        return None


def parse_float64_multiarray(data, timestamp_ns):
    """Parse std_msgs/Float64MultiArray from CDR serialization."""
    try:
        offset = 4  # Skip CDR header
        
        # MultiArrayLayout - skip dim array (usually empty for simple arrays)
        dim_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Skip dimensions if any (each has label string, size, stride)
        for _ in range(dim_len):
            str_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + str_len
            # Align
            remainder = offset % 4
            if remainder != 0:
                offset += (4 - remainder)
            offset += 8  # size (uint32) + stride (uint32)
        
        # data_offset (uint32)
        data_offset_val = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Align to 8 bytes for double array
        remainder = offset % 8
        if remainder != 0:
            offset += (8 - remainder)
        
        # Array length (uint32)
        arr_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Align to 8 bytes for doubles
        remainder = offset % 8
        if remainder != 0:
            offset += (8 - remainder)
        
        # Read array of doubles
        values = []
        for i in range(arr_len):
            val = struct.unpack_from('<d', data, offset)[0]
            values.append(val)
            offset += 8
        
        # Create result with joint columns
        result = {'bag_timestamp_ns': timestamp_ns}
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        for i, val in enumerate(values):
            if i < len(joint_names):
                result[joint_names[i]] = val
            else:
                result[f'value_{i}'] = val
        
        return result
    except Exception as e:
        return None


def bag_to_csv(bag_path: str):
    """Convert a ROS2 bag to CSV files."""
    bag_path = Path(bag_path).resolve()
    
    # Find the .db3 file
    db_files = list(bag_path.glob('*.db3'))
    if not db_files:
        print(f"No .db3 file found in {bag_path}")
        return
    
    db_file = db_files[0]
    print(f"\n{'='*60}")
    print(f"Converting: {bag_path.name}")
    print(f"{'='*60}")
    
    # Connect to sqlite database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Get topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: {'name': row[1], 'type': row[2]} for row in cursor.fetchall()}
    
    print("Topics:")
    for tid, info in topics.items():
        cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (tid,))
        count = cursor.fetchone()[0]
        print(f"  {info['name']}: {count} messages")
    
    # Process each topic
    for topic_id, topic_info in topics.items():
        topic_name = topic_info['name']
        topic_type = topic_info['type']
        
        # Select parser based on type
        is_multiarray = False
        if 'PoseStamped' in topic_type:
            parser = parse_pose_stamped
        elif 'TwistStamped' in topic_type:
            parser = parse_twist_stamped
        elif 'Float64MultiArray' in topic_type:
            parser = parse_float64_multiarray
            is_multiarray = True
        else:
            continue
        
        # Get messages
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,)
        )
        
        # Create CSV file
        safe_name = topic_name.replace('/', '_').strip('_')
        csv_path = bag_path / f"{safe_name}.csv"
        
        rows = []
        for timestamp, data in cursor.fetchall():
            if is_multiarray:
                parsed = parser(data, timestamp)
            else:
                parsed = parser(data)
                if parsed:
                    parsed['bag_timestamp_ns'] = timestamp
            if parsed:
                rows.append(parsed)
        
        if rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  ✓ {safe_name}.csv ({len(rows)} rows)")
    
    conn.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable bags:")
        for item in Path('.').iterdir():
            if item.is_dir() and (item / 'metadata.yaml').exists():
                print(f"  - {item.name}")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        for item in sorted(Path('.').iterdir()):
            if item.is_dir() and (item / 'metadata.yaml').exists():
                try:
                    bag_to_csv(str(item))
                except Exception as e:
                    print(f"Error converting {item}: {e}")
    else:
        bag_to_csv(sys.argv[1])
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
