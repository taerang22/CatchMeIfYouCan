# MPC Package

Model Predictive Control for time-optimal ball catching using Drake.

## Node

```bash
ros2 run mpc mpc_node
```

## Topics

**Subscribes:**
- `/kinova/eef_pose` (`Pose`) — End-effector pose
- `/kinova/eef/vel` (`Twist`) — End-effector velocity
- `/kinova/goal_pose` (`PoseStamped`) — Target catch pose + time
- `/kinova/joint_states` (`JointState`) — Joint positions

**Publishes:**
- `/kinova/joint_vel_cmd` (`Float64MultiArray`) — Joint velocities (rad/s)

## Configuration

Edit `mpc/config/mpc_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_dt` | `0.01` s | Control loop period |
| `a_max` | `3.0` m/s² | Max acceleration |
| `v_max` | `0.8` m/s | Max velocity |
| `w_terminal` | `200.0` | Terminal cost weight |
| `w_control` | `1.0` | Control effort weight |

## Algorithm

1. **Cartesian MPC:** Time-varying LQR computes acceleration command
2. **Orientation:** Points end-effector toward catch target
3. **IK:** Damped least-squares converts twist → joint velocities

## Dependencies

`drake`, `transforms3d`, `numpy`, `pyyaml`
