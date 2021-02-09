from pupil_recording_interface.externals.circle_detector import find_pupil_circle_marker
import numpy as np
import cv2
import  copy

def _opencv_ellipse_to_dict(ellipse_dict):
    data = {}
    data["ellipse"] = {
        "center": (ellipse_dict.ellipse.center[0], ellipse_dict.ellipse.center[1]),
        "axes": (
            ellipse_dict["ellipses"].minor_radius * 2.0,
            ellipse_dict.ellipse.major_radius * 2.0,
        ),
        "angle": ellipse_dict.ellipse.angle * 180.0 / np.pi - 90.0,
    }
    data["diameter"] = max(data["ellipse"]["axes"])
    data["location"] = data["ellipse"]["center"]
    data["confidence"] = ellipse_dict.confidence


def find_concentric_circles(video_data, timestamps, scale=1.0, progress_bar=None):
    """Assumes uint8 RGB image input"""
    if progress_bar is None:
        progress_bar = lambda x: x
    n_frames, vdim, hdim = video_data.shape[:3]
    out = []
    for frame in progress_bar(range(n_frames)):
        x = video_data[frame]
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        world_circles = find_pupil_circle_marker(gray, scale)
        output = np.array([np.nan, np.nan])
        for iw, w in enumerate(world_circles):
            ellipses = w["ellipses"]
            for e in ellipses:
                this_circle = {}
                ellipse_centers = np.array([e[0] for e in ellipses])
                this_circle["location"] = ellipse_centers.mean(0).tolist()
                this_circle["norm_pos"] = this_circle["location"] / np.array(
                    [hdim, vdim]
                )
                this_circle["norm_pos"] = this_circle["norm_pos"].tolist()
                ellipse_radii = np.array([e[1] for e in ellipses])
                this_circle["size"] = ellipse_radii.max(0)
                this_circle["timestamp"] = timestamps[frame]
            out.append(this_circle)
    return out


def find_checkerboard(
    video_data, timestamps=None, checkerboard_size=(6, 8), scale=None, progress_bar=None
):
    """Use opencv to detect checkerboard pattern"""
    if progress_bar is None:
        progress_bar = lambda x: x

    rows, cols = checkerboard_size
    n_frames, vdim, hdim = video_data.shape[:3]
    times = []  # frame timestamps for detected keypoints
    locations = []  # 2d points in image plane.
    norm_pos = []  # 2d normalized points in image plane.
    mean_locations = []  # Mean 2d points in image plane
    mean_norm_pos = []  # Mean 2d normalized points in image plane
    reference_dict = {}
    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    n_iter = min([len(timestamps), len(video_data)])
    for frame_time, frame in progress_bar(zip(timestamps, video_data), total=n_iter):
        if np.ndim(video_data) == 4:
            # Color image; remove color
            # TODO: add option for BGR image?
            # color_frame = copy.deepcopy(cv2.resize(frame, None, fx=scale, fy=scale))
            # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            vdim, hdim = frame.shape[:2]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(frame, checkerboard_size, None)
        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
            times.append(frame_time)
            corners = np.squeeze(corners2)
            # Draw and display the corners
            # frame = cv2.drawChessboardCorners(color_frame, (6, 8), corners, ret)
            # corners[:, 0] = corners[:, 0] * (1 / scale)
            corners[:, 1] = corners[:, 1] * (1 / scale)
            locations.append(corners)
            marker_position = np.mean(corners, axis=0)
            mean_locations.append(marker_position)
            corners[:, 0] = corners[:, 0] * (scale/hdim)
            corners[:, 1] = corners[:, 1] * (scale/vdim)
            norm_pos.append(corners)
            marker_position = np.mean(corners, axis=0)
            mean_norm_pos.append(marker_position)

    #     cv2.imshow('Frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    # # print('\nDone!')
    # cv2.destroyAllWindows()
    # # print("timestamp shape: ", np.asarray(times).shape)
    # print("locations shape: ", np.asarray(locations).shape)
    # print("mean locations shape: ", np.asarray(mean_locations).shape)
    # print("norm pos shape: ", np.asarray(norm_pos).shape)
    # print("mean norm pos shape: ", np.asarray(mean_norm_pos).shape)
    # print("loc: ", mean_locations)
    # print("norm: ", mean_norm_pos)
    reference_dict['location'] = np.asarray(locations)
    reference_dict['norm_pos'] = np.asarray(norm_pos)
    reference_dict['mean_location'] = np.asarray(mean_locations)
    reference_dict['mean_norm_pos'] = np.asarray(mean_norm_pos)
    reference_dict['timestamp'] = np.asarray(times)
    return [reference_dict]#(np.asarray(x) for x in [times, locations, mean_locations, norm_pos, mean_norm_pos])
