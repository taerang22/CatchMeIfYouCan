# Perception Package

Detects and tracks a yellow ball using Intel RealSense RGB-D camera.

## Node

```bash
ros2 run perception ball_tracker_node
```

## Topics

**Publishes:**
- `/kinova/ball/init_pose` (`PoseStamped`) — Ball position in robot frame
- `/kinova/ball/init_twist` (`TwistStamped`) — Ball velocity

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | `60` | Camera frame rate |
| `yellow_lower_hsv` | `[21, 63, 137]` | HSV lower threshold |
| `yellow_upper_hsv` | `[67, 202, 255]` | HSV upper threshold |
| `use_manual_extrinsics` | `false` | Use manual camera transform |
| `aruco_marker_id` | `7` | ArUco marker for calibration |

## Calibration

**ArUco (default):** Place 7x7 ArUco marker (ID=7, 20cm) at robot base. Node auto-calibrates on startup.

**Manual:** Set `use_manual_extrinsics:=true` and provide `camera_translation_xyz`, `camera_quaternion_xyzw`.

## Dependencies

`pyrealsense2`, `opencv-python`, `numpy`, `scipy`
