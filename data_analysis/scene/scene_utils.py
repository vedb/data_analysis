import numpy as np
import cv2

def undistort_unproject_pts(pts_uv, camera_matrix, dist_coefs):
    """
    This function converts a set of 2D image coordinates to vectors in pinhole camera space.
    Hereby the intrinsics of the camera are taken into account.
    UV is converted to normalized image space (think frustum with image plane at z=1) then undistored
    adding a z_coordinate of 1 yield vectors pointing from 0,0,0 to the undistored image pixel.
    @return: ndarray with shape=(n, 3)

    """
    #     pts_uv = np.array(pts_uv)
    num_pts = pts_uv.size / 2
    #     print(type(np.float32(pts_uv)))
    # pts_uv.shape = (num_pts, 1, 2)
    pts_uv = cv2.fisheye.undistortPoints(np.float32(pts_uv), camera_matrix, dist_coefs)
    pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))
    #     pts_3d = np.dot(np.linalg.pinv(camera_matrix), np.squeeze(pts_3d).T).T
    #     pts_3d.shape = (num_pts,3)
