# Control Package

Kinova Gen3 robot interface via Kortex API.

## Nodes

### 1. State Publisher
```bash
ros2 run control kinova_state_publisher
```

**Publishes (100 Hz):**
- `/kinova/eef_pose` (`Pose`) — End-effector pose
- `/kinova/eef/vel` (`Twist`) — End-effector velocity
- `/kinova/joint_states` (`JointState`) — Joint states
- `/kinova/gripper_pos` (`Float64`) — Gripper position

### 2. MPC Controller
```bash
ros2 run control kinova_mpc_controller
```

**Subscribes:**
- `/kinova/joint_vel_cmd` (`Float64MultiArray`) — Joint velocities (rad/s)

### 3. Pose Controller
```bash
ros2 run control kinova_controller_node
```

**Subscribes:**
- `/kinova/goal_pose` (`PoseStamped`) — Target pose (blocking execution)

## Robot Connection

| Parameter | Value |
|-----------|-------|
| IP | `192.168.1.10` |
| Username | `admin` |
| Password | `admin` |

## Safety Limits

- Max Cartesian speed: **0.5 m/s**
- Joints 1-4: **40 deg/s**
- Joints 5-7: **30 deg/s**

## Dependencies

`kortex-api`, `protobuf==3.20.3`, `scipy`, `numpy`
