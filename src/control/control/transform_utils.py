import numpy as np
from scipy.spatial.transform import Rotation as R

def get_error(curr_pose, target_pose):
    """
    Calculate the error between the current pose and target pose.
    Returns a 3dim array of position error and 3 dim array of angular error.
    """
    curr_R = R.from_euler('xyz', curr_pose[3:], degrees=True).as_matrix()
    target_R = R.from_euler('xyz', target_pose[3:], degrees=True).as_matrix()
    
    pos_error = curr_pose[:3] - target_pose[:3]
    
    r1 = curr_R[:, 0]; r2 = curr_R[:, 1]; r3 = curr_R[:, 2]
    rd1 = target_R[:, 0]; rd2 = target_R[:, 1]; rd3 = target_R[:, 2]
    rot_error = - np.cross(r1, rd1) - np.cross(r2, rd2) - np.cross(r3, rd3)

    return pos_error, rot_error