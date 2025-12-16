#!/usr/bin/env python3
"""
Convert ROS2 bag files to CSV format.

Usage:
    python bag_to_csv.py <bag_folder>
    python bag_to_csv.py ball_tracking_data
    python bag_to_csv.py --all  # Convert all bags in current directory
"""

import sys
import os
import csv
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def get_rosbag_options(path, storage_id='sqlite3'):
    """Get storage and converter options for rosbag."""
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    return storage_options, converter_options


def bag_to_csv(bag_path: str, output_dir: str = None):
    """
    Convert a ROS2 bag to CSV files (one per topic).
    
    Args:
        bag_path: Path to the bag folder
        output_dir: Output directory for CSV files (default: same as bag)
    """
    bag_path = str(Path(bag_path).resolve())
    
    if output_dir is None:
        output_dir = bag_path
    
    print(f"\n{'='*60}")
    print(f"Converting: {bag_path}")
    print(f"{'='*60}")
    
    # Open the bag
    storage_options, converter_options = get_rosbag_options(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic info
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    print(f"Topics found:")
    for topic_name, topic_type in type_map.items():
        print(f"  - {topic_name} [{topic_type}]")
    
    # Prepare CSV writers for each topic
    csv_files = {}
    csv_writers = {}
    msg_counts = {}
    
    for topic_name in type_map.keys():
        # Create safe filename
        safe_name = topic_name.replace('/', '_').strip('_')
        csv_path = os.path.join(output_dir, f"{safe_name}.csv")
        
        csv_files[topic_name] = open(csv_path, 'w', newline='')
        csv_writers[topic_name] = None  # Will create after seeing first message
        msg_counts[topic_name] = 0
    
    # Read messages
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        
        if topic_name not in type_map:
            continue
        
        # Deserialize message
        msg_type = get_message(type_map[topic_name])
        msg = deserialize_message(data, msg_type)
        
        # Convert message to dict
        row = {'timestamp_ns': timestamp}
        row.update(msg_to_dict(msg))
        
        # Create CSV writer with headers on first message
        if csv_writers[topic_name] is None:
            csv_writers[topic_name] = csv.DictWriter(
                csv_files[topic_name], 
                fieldnames=row.keys()
            )
            csv_writers[topic_name].writeheader()
        
        csv_writers[topic_name].writerow(row)
        msg_counts[topic_name] += 1
    
    # Close files and print summary
    print(f"\nOutput files:")
    for topic_name, f in csv_files.items():
        f.close()
        safe_name = topic_name.replace('/', '_').strip('_')
        csv_path = os.path.join(output_dir, f"{safe_name}.csv")
        print(f"  - {csv_path} ({msg_counts[topic_name]} messages)")
    
    return msg_counts


def msg_to_dict(msg, prefix=''):
    """Recursively convert a ROS message to a flat dictionary."""
    result = {}
    
    # Get all slots (fields) of the message
    if hasattr(msg, '__slots__'):
        for slot in msg.__slots__:
            attr_name = slot.lstrip('_')
            value = getattr(msg, attr_name)
            full_key = f"{prefix}{attr_name}" if prefix else attr_name
            
            # Handle nested messages
            if hasattr(value, '__slots__'):
                result.update(msg_to_dict(value, f"{full_key}_"))
            # Handle arrays/lists
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and hasattr(value[0], '__slots__'):
                    for i, item in enumerate(value):
                        result.update(msg_to_dict(item, f"{full_key}_{i}_"))
                else:
                    result[full_key] = str(value)
            else:
                result[full_key] = value
    else:
        result[prefix.rstrip('_')] = msg
    
    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable bags in current directory:")
        for item in Path('.').iterdir():
            if item.is_dir() and (item / 'metadata.yaml').exists():
                print(f"  - {item.name}")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        # Convert all bags in current directory
        for item in Path('.').iterdir():
            if item.is_dir() and (item / 'metadata.yaml').exists():
                try:
                    bag_to_csv(str(item))
                except Exception as e:
                    print(f"Error converting {item}: {e}")
    else:
        # Convert specified bag
        bag_path = sys.argv[1]
        if not os.path.exists(bag_path):
            print(f"Error: {bag_path} does not exist")
            sys.exit(1)
        
        bag_to_csv(bag_path)
    
    print("\n✅ Conversion complete!")


if __name__ == '__main__':
    main()

