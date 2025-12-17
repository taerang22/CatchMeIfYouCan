## Hardware
- Kinova Gen3 robotic arm
- Intel RealSense D435 RGB-D camera
- Custom CAD-ed catapult launcher

(사진 + 다이어그램)

## Software
The system was implemented using ROS 2 and consists of multiple nodes:
- ball_tracker_node
- ball_prediction_node
- mpc_catch_node

(ROS graph / flow chart)

## System Workflow
1. The RGB-D camera detects the ball and estimates its 3D position and velocity.
2. A prediction node estimates where the ball will intersect a predefined catching plane.
3. The MPC controller generates a desired Cartesian motion.
4. Jacobian-based inverse kinematics converts the motion into joint velocities.
