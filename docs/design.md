# Design

## Design Criteria
The system must detect, predict, and intercept a fast-moving object in real time
while maintaining stable and smooth robot motion.

## System Architecture
(시스템 다이어그램 이미지 삽입)

## Design Choices and Trade-offs
- Single RGB-D camera vs multi-camera setup
- Simple curve fitting vs physics-based prediction
- MPC for stability vs reactive control for speed

## Impact on Real-World Use
These choices improve robustness and real-time performance,
but limit handling of extreme trajectories.
