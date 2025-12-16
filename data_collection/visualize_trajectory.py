import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

####
# Specify your pickle file
file_to_load = Path('ball_traj_data') / 'ball_trajectory_20251030_194010.pkl'

# Load the pickle file
with open(file_to_load, 'rb') as f:
    data = pickle.load(f)

# Extract positions
positions = data['positions']  # This assumes your pickle stores positions under 'positions'

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', markersize=2)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
plt.title(f'3D Ball Trajectory: {file_to_load}')
plt.show()
