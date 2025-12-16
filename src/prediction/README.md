# Prediction Package

Predicts where and when a ball will cross the catch plane using ballistic physics.

## Node

```bash
ros2 run prediction ball_prediction_node
```

## Topics

**Subscribes:**
- `/kinova/ball/init_pose` (`PoseStamped`) — Ball position
- `/kinova/ball/init_twist` (`TwistStamped`) — Ball velocity

**Publishes:**
- `/kinova/goal_pose` (`PoseStamped`) — Catch position + time (`header.stamp` = t_hit)

## Algorithm

Uses projectile motion under gravity (g = 9.81 m/s²) to predict intersection with vertical catch plane at `x = 0.45m`.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catch_plane_x` | `0.45` m | X-coordinate of catch plane |
| `min_z` | `0.10` m | Minimum safe Z height |

## Dependencies

`numpy`, `ball_predictor` (external module)
