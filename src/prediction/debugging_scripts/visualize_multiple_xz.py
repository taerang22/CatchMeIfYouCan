#!/usr/bin/env python3
"""
Visualize multiple ball trajectories on XZ plane (parabola view).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from visualize_prediction
sys.path.insert(0, str(Path(__file__).parent))
from visualize_prediction import read_csv, BallPredictionVisualizer


def visualize_multiple_xz(csv_paths, output_path=None):
    """Visualize multiple trajectories on XZ plane - each in separate window"""
    
    # Filter valid paths
    valid_paths = []
    for csv_path in csv_paths:
        csv_file = Path(csv_path)
        if csv_file.exists():
            valid_paths.append(csv_file)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    if len(valid_paths) == 0:
        print("Error: No valid CSV files found!")
        return
    
    # Create separate figures for each dataset
    figures = []
    
    for idx, csv_file in enumerate(valid_paths):
        print(f"\nProcessing {csv_file.name} ({csv_file.parent.name})...")
        
        # Read CSV
        poses, _ = read_csv(csv_file)
        
        if len(poses) < 2:
            print(f"  Warning: Only {len(poses)} poses found, skipping...")
            continue
        
        # Use original CSV data (before any transformation)
        # This is the raw data from /kinova/ball/init_pose topic
        positions_robot = []
        for pose in poses:
            positions_robot.append({
                'timestamp': pose['timestamp'],
                'position': pose['position']  # Original CSV data, no transform
            })
        
        # Sort by timestamp
        positions_robot.sort(key=lambda x: x['timestamp'])
        
        # Extract position arrays
        pos_array = np.array([p['position'] for p in positions_robot])
        timestamps = np.array([p['timestamp'] for p in positions_robot])
        
        # Normalize time
        if len(timestamps) > 0:
            t_start = timestamps[0]
            time_normalized = (timestamps - t_start) * 1e-9
        else:
            time_normalized = np.arange(len(pos_array))
        
        # Create separate figure for each dataset
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        figures.append(fig)
        
        # Plot on ZY plane
        data_num = csv_file.parent.name.split('_')[-1]
        scatter = ax.scatter(pos_array[:, 2], pos_array[:, 1], 
                            c=time_normalized, cmap='viridis', s=100, 
                            edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.plot(pos_array[:, 2], pos_array[:, 1], 
               '--', color='gray', linewidth=1.5, alpha=0.6)
        
        # Add labels for all points
        # For data 13: show sequence number only (no time)
        # For other data: show time
        is_data_13 = (data_num == '13')
        for i in range(len(pos_array)):
            if is_data_13:
                # Data 13: show sequence number only
                label = f'#{i+1}'
            else:
                # Other data: show time
                label = f't={time_normalized[i]:.3f}s'
            
            ax.annotate(label, 
                       (pos_array[i, 2], pos_array[i, 1]),
                       xytext=(3, 3), textcoords='offset points', 
                       fontsize=8 if is_data_13 else 7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
        
        ax.set_xlabel('Z (m) - original frame', fontsize=11)
        ax.set_ylabel('Y (m) - original frame', fontsize=11)
        ax.set_title(f'Data {data_num} - ZY Projection (original, before transform)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Time (s)')
        
        plt.tight_layout()
        
        # Save individual figure
    if output_path is None:
        save_path = Path(valid_paths[0]).parent.parent / f"data_{data_num}_zy.png"
    else:
        save_path = Path(output_path).parent / f"data_{data_num}_zy.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    # Save combined figure (all in one)
    if output_path is None:
        combined_path = Path(valid_paths[0]).parent.parent / "multiple_trajectories_zy_separate.png"
    else:
        combined_path = Path(output_path)
    
    # Create one combined figure for saving
    n_plots = len(valid_paths)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_plots == 1:
        axes_combined = [axes_combined]
    elif n_rows == 1:
        axes_combined = axes_combined if isinstance(axes_combined, list) else [axes_combined]
    else:
        axes_combined = axes_combined.flatten()
    
    # Re-plot for combined figure
    for idx, csv_file in enumerate(valid_paths):
        poses, _ = read_csv(csv_file)
        if len(poses) < 2:
            continue
        
        positions_robot = []
        for pose in poses:
            positions_robot.append({
                'timestamp': pose['timestamp'],
                'position': pose['position']
            })
        positions_robot.sort(key=lambda x: x['timestamp'])
        pos_array = np.array([p['position'] for p in positions_robot])
        timestamps = np.array([p['timestamp'] for p in positions_robot])
        if len(timestamps) > 0:
            t_start = timestamps[0]
            time_normalized = (timestamps - t_start) * 1e-9
        else:
            time_normalized = np.arange(len(pos_array))
        
        ax = axes_combined[idx]
        data_num = csv_file.parent.name.split('_')[-1]
        scatter = ax.scatter(pos_array[:, 2], pos_array[:, 1], 
                            c=time_normalized, cmap='viridis', s=80, 
                            edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.plot(pos_array[:, 2], pos_array[:, 1], 
               '--', color='gray', linewidth=1.5, alpha=0.6)
        ax.set_xlabel('Z (m) - original frame', fontsize=9)
        ax.set_ylabel('Y (m) - original frame', fontsize=9)
        ax.set_title(f'Data {data_num} (original)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter, ax=ax, label='Time (s)')
    
    for idx in range(len(valid_paths), len(axes_combined)):
        axes_combined[idx].axis('off')
    
    plt.tight_layout()
    fig_combined.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined visualization saved to: {combined_path}")
    
    # Show all individual windows
    plt.show()


if __name__ == "__main__":
    # Default: ball_tracking_data_9, 13, 14, 15, 16
    base_dir = Path(__file__).parent.parent.parent / "rosbag"
    
    csv_files = []
    for i in [9, 13, 14, 15, 16]:
        csv_path = base_dir / f"ball_tracking_data_{i}" / "kinova_ball_init_pose.csv"
        if csv_path.exists():
            csv_files.append(csv_path)
        else:
            print(f"Warning: {csv_path} not found")
    
    if len(csv_files) == 0:
        print("Error: No CSV files found!")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name} ({f.parent.name})")
    
    visualize_multiple_xz(csv_files)

