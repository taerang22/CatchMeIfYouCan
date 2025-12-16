# 🤖 Kinova Gen3 Ball-Catching Robot System

A **ROS2-based robotic system** that enables a **Kinova Gen3 robotic arm** to track and catch a flying ball in real-time using computer vision, trajectory prediction, and Model Predictive Control (MPC).

---

## 📐 System Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│     Perception      │────▶│     Prediction       │────▶│        MPC          │────▶│      Control        │
│  (Ball Tracking)    │     │ (Ballistic Physics)  │     │ (Motion Planning)   │     │   (Robot Driver)    │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘     └─────────────────────┘
     RealSense D435            BallPredictor              Drake LQR/MPC             Kortex API
     OpenCV + HSV              Catch Plane               Jacobian IK              Joint Velocities
```

---

## 📦 Package Descriptions

### 1. `perception` — Ball Detection & Tracking

**Location:** `src/perception/`

Detects and tracks a yellow ball using an Intel RealSense RGB-D camera.

| Node | Description |
|------|-------------|
| `ball_tracker_node` | Main perception node |

**Features:**
- Real-time yellow ball detection using HSV color filtering
- 3D position estimation from RGB-D depth alignment
- Velocity estimation via finite differencing
- ArUco marker-based camera-to-robot calibration (or manual extrinsics)
- Publishes ball pose and velocity in robot `base_link` frame

**Subscriptions:** *(none — uses RealSense SDK directly)*

**Publications:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/ball/init_pose` | `geometry_msgs/PoseStamped` | Ball position in robot frame |
| `/kinova/ball/init_twist` | `geometry_msgs/TwistStamped` | Ball velocity in robot frame |

**Parameters:**
- `serial` — RealSense camera serial number
- `fps` — Frame rate (default: 60)
- `yellow_lower_hsv`, `yellow_upper_hsv` — HSV thresholds for ball detection
- `aruco_marker_id`, `aruco_marker_length_m` — ArUco calibration settings
- `use_manual_extrinsics` — Use predefined camera transform instead of ArUco

---

### 2. `prediction` — Ball Trajectory Prediction

**Location:** `src/prediction/`

Predicts where and when the ball will cross a vertical "catch plane" using ballistic physics.

| Node | Description |
|------|-------------|
| `ball_prediction_node` | Ballistic trajectory predictor |

**Features:**
- Ballistic motion model with gravity (9.81 m/s²)
- Predicts intersection with vertical catch plane (x = constant)
- Computes time-to-hit (`t_hit`) for time-optimal control
- Safety constraints (minimum Z height)

**Subscriptions:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/ball/init_pose` | `geometry_msgs/PoseStamped` | Ball position |
| `/kinova/ball/init_twist` | `geometry_msgs/TwistStamped` | Ball velocity |

**Publications:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/goal_pose` | `geometry_msgs/PoseStamped` | Target catch position + time (`header.stamp` = `t_hit`) |

**Parameters:**
- `catch_plane_x` — X-coordinate of catch plane (default: 0.45 m)
- `min_z` — Minimum safe Z height (default: 0.10 m)

---

### 3. `mpc` — Model Predictive Control

**Location:** `src/mpc/`

Computes optimal end-effector trajectories using time-varying LQR and converts them to joint velocities.

| Node | Description |
|------|-------------|
| `mpc_node` | Cartesian MPC controller with Jacobian-based IK |

**Features:**
- **Cartesian MPC:** Time-varying LQR using Drake's `LinearQuadraticRegulator`
- **Kinematics:** Drake MultibodyPlant for accurate Jacobian computation
- **IK:** Damped least-squares inverse kinematics (J^T (JJ^T + λI)^(-1))
- **Orientation Control:** Points end-effector toward catch target
- **Safety:** Velocity saturation and singularity handling

**Subscriptions:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/eef_pose` | `geometry_msgs/Pose` | Current end-effector pose |
| `/kinova/eef/vel` | `geometry_msgs/Twist` | Current end-effector velocity |
| `/kinova/goal_pose` | `geometry_msgs/PoseStamped` | Target catch position from prediction |
| `/kinova/joint_states` | `sensor_msgs/JointState` | Current joint positions |

**Publications:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/joint_vel_cmd` | `std_msgs/Float64MultiArray` | Joint velocity commands (rad/s) |
| `/kinova/eef_twist_cmd` | `geometry_msgs/Twist` | (Debug) Cartesian twist command |

**Configuration:** `src/mpc/mpc/config/mpc_config.yaml`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `control_dt` | Control loop period | 0.01 s |
| `a_max` | Max Cartesian acceleration | 3.0 m/s² |
| `v_max` | Max Cartesian velocity | 0.8 m/s |
| `w_terminal` | Terminal cost weight | 200.0 |
| `w_control` | Control effort weight | 1.0 |

---

### 4. `control` — Robot Interface & State Publisher

**Location:** `src/control/`

Interfaces with the Kinova Gen3 robot via the Kortex API.

| Node | Description |
|------|-------------|
| `kinova_state_publisher` | Publishes robot state at 100 Hz |
| `kinova_mpc_controller` | Executes joint velocity commands from MPC |
| `kinova_controller_node` | Executes goal poses (high-level, blocking) |

#### `kinova_state_publisher`

Publishes real-time robot state from the Kinova Kortex API.

**Publications:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/eef_pose` | `geometry_msgs/Pose` | End-effector pose (position + quaternion) |
| `/kinova/eef/vel` | `geometry_msgs/Twist` | End-effector velocity |
| `/kinova/joint_states` | `sensor_msgs/JointState` | Joint positions, velocities, torques |
| `/kinova/gripper_pos` | `std_msgs/Float64` | Gripper position (0.0–1.0) |

#### `kinova_mpc_controller`

Receives joint velocity commands from MPC and sends them to the robot.

**Subscriptions:**
| Topic | Type | Description |
|-------|------|-------------|
| `/kinova/joint_vel_cmd` | `std_msgs/Float64MultiArray` | Joint velocities (rad/s) |

#### `kinova_helper.py`

Low-level Kortex API wrapper providing:
- Session management (TCP/UDP)
- Cartesian pose control (`move_eef_pose`)
- Joint velocity control (`send_joint_speeds`)
- Gripper control
- Home/Retract actions
- Camera intrinsics/extrinsics access

---

## 📊 Data Collection

**Location:** `data_collection/`

Scripts for recording ball trajectories using Vicon motion capture.

| Script | Description |
|--------|-------------|
| `ball_trajectory.py` | Records ball positions from `/vicon/ball/ball/pose` and saves to `.pkl` |
| `visualize_trajectory.py` | 3D visualization of recorded trajectories |

---

## 🔄 ROS2 Topic Flow

```
                    ┌──────────────────────────┐
                    │    RealSense Camera      │
                    └────────────┬─────────────┘
                                 │ (RGB-D frames)
                                 ▼
                    ┌──────────────────────────┐
                    │   ball_tracker_node      │
                    │      (perception)        │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
    /kinova/ball/init_pose              /kinova/ball/init_twist
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  ball_prediction_node    │
                    │      (prediction)        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                        /kinova/goal_pose
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      mpc_catch_node      │◀──── /kinova/eef_pose
                    │         (mpc)            │◀──── /kinova/eef/vel
                    └────────────┬─────────────┘◀──── /kinova/joint_states
                                 │
                                 ▼
                      /kinova/joint_vel_cmd
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  kinova_mpc_controller   │
                    │       (control)          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Kinova Gen3 Robot     │
                    └──────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Build the workspace

```bash
cd ~/C106
colcon build
source install/setup.bash
```

### 2. Launch nodes (in separate terminals)

```bash
# Terminal 1: State publisher
ros2 run control kinova_state_publisher

# Terminal 2: Perception (ball tracking)
ros2 run perception ball_tracker_node

# Terminal 3: Prediction
ros2 run prediction ball_prediction_node

# Terminal 4: MPC controller
ros2 run mpc mpc_node

# Terminal 5: Robot velocity controller
ros2 run control kinova_mpc_controller
```

---

## 🔧 Dependencies

See `requirements.txt` for Python dependencies. Key packages:

| Package | Purpose |
|---------|---------|
| `rclpy` | ROS2 Python client |
| `numpy`, `scipy` | Numerical computing |
| `opencv-python` | Computer vision |
| `pyrealsense2` | Intel RealSense SDK |
| `drake` | Kinematics & MPC |
| `kortex-api` | Kinova robot control |
| `transforms3d` | 3D transformations |

---

## 📁 Project Structure

```
C106/
├── src/
│   ├── perception/          # Ball tracking (RealSense + OpenCV)
│   │   ├── README.md        # ← Package documentation
│   │   └── perception/
│   │       └── ball_tracker_node.py
│   ├── prediction/          # Ballistic trajectory prediction
│   │   ├── README.md        # ← Package documentation
│   │   └── prediction/
│   │       └── ball_prediction_node.py
│   ├── mpc/                 # Model Predictive Control
│   │   ├── README.md        # ← Package documentation
│   │   └── mpc/
│   │       ├── mpc_node.py
│   │       ├── cartesian_mpc.py
│   │       ├── config/mpc_config.yaml
│   │       └── urdf/kinovaGen3.urdf
│   └── control/             # Kinova robot interface
│       ├── README.md        # ← Package documentation
│       └── control/
│           ├── kinova_state_publisher.py
│           ├── kinova_mpc_controller.py
│           ├── kinova_controller_node.py
│           └── kinova_helper.py
├── data_collection/         # Vicon trajectory recording
│   └── README.md            # ← Data collection docs
├── build/                   # colcon build artifacts
├── install/                 # colcon install artifacts
├── requirements.txt         # Python dependencies
└── README.md                # ← This file
```

---

## 📚 Package Documentation

Each package has its own detailed README:

| Package | README | Description |
|---------|--------|-------------|
| **perception** | [`src/perception/README.md`](src/perception/README.md) | Ball detection, HSV tuning, ArUco calibration |
| **prediction** | [`src/prediction/README.md`](src/prediction/README.md) | Ballistic physics, catch plane intersection |
| **mpc** | [`src/mpc/README.md`](src/mpc/README.md) | LQR tuning, Jacobian IK, Drake integration |
| **control** | [`src/control/README.md`](src/control/README.md) | Kortex API, joint limits, safety |
| **data_collection** | [`data_collection/README.md`](data_collection/README.md) | Vicon recording, trajectory visualization |

---

## 📝 License

This project is developed for research purposes at ICONLAB.
