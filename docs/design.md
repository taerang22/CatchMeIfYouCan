## Design Criteria
Our system must:
- Detect and localize a fast-moving ball in real time
- Predict the interception point with low latency
- Generate smooth, stable catching motions
- Operate robustly under perception noise

## System Design
(시스템 다이어그램 이미지)

The system is divided into three modules: Perception, Prediction, and Control.

## Design Choices and Trade-offs
- A single RGB-D camera was used instead of stereo vision to reduce system complexity.
- A simple polynomial trajectory fit was chosen over a full physics-based model to ensure
  real-time performance.
- Model Predictive Control (MPC) was used to balance responsiveness and stability.

## Impact on Real-World Criteria
These design choices improve real-time robustness and efficiency,
but limit performance under extreme ball trajectories or severe occlusions.
