# Data Collection

Record ball trajectories using Vicon motion capture.

## Scripts

### Record Trajectory
```bash
python3 ball_trajectory.py
# Press Ctrl+C to save
```

Subscribes to `/vicon/ball/ball/pose` and saves to `ball_traj_data/*.pkl`.

### Visualize Trajectory
```bash
python3 visualize_trajectory.py
```

Edit `file_to_load` in script to select which `.pkl` to visualize.

## Data Format

```python
{
    'positions': np.array[[x, y, z], ...],   # (N, 3) meters
    'timestamps': np.array[t1, t2, ...]      # (N,) seconds
}
```

## Prerequisites

1. Start Vicon bridge: `ros2 launch vicon_bridge all_segments_launch.py`
2. Verify: `ros2 topic echo /vicon/ball/ball/pose`
