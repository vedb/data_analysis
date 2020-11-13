import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2


def plot_gaze_accuracy(self, markerPosition, gazeDataFrame, gazeIndex):
    """"""
    horizontal_pixels = 1280
    vertical_pixels = 1024
    horizontal_FOV = 92.5
    vertical_FOV = 70.8

    ratio_x = horizontal_FOV / horizontal_pixels
    ratio_y = vertical_FOV / vertical_pixels

    gaze_norm_x = gazeDataFrame.iloc[gazeIndex].norm_pos_x.values
    gaze_norm_y = gazeDataFrame.iloc[gazeIndex].norm_pos_y.values

    gaze_pixel_x = gaze_norm_x * horizontal_pixels
    gaze_pixel_y = gaze_norm_y * vertical_pixels

    print("gazeX shape = ", gaze_pixel_x.shape)
    print("gazeY shape = ", gaze_pixel_y.shape)
    # print(np.array([gaze_pixel_x, gaze_pixel_y]).shape)
    gaze_homogeneous = cv2.convertPointsToHomogeneous(
        np.array([gaze_pixel_x, gaze_pixel_y]).T
    )
    gaze_homogeneous = np.squeeze(gaze_homogeneous)

    gaze_homogeneous[:, 0] = pixels_to_angle_x(gaze_homogeneous[:, 0])
    gaze_homogeneous[:, 1] = pixels_to_angle_y(gaze_homogeneous[:, 1])

    # This is important because the gaze values should be inverted in y direction
    gaze_homogeneous[:, 1] = -gaze_homogeneous[:, 1]

    print("gaze homogeneous shape =", gaze_homogeneous.shape)

    # print('gaze homogeneous =',gaze_homogeneous[0:5,:])

    marker_homogeneous = cv2.convertPointsToHomogeneous(markerPosition)
    marker_homogeneous = np.squeeze(marker_homogeneous)

    marker_homogeneous[:, 0] = pixels_to_angle_x(marker_homogeneous[:, 0])
    marker_homogeneous[:, 1] = pixels_to_angle_y(marker_homogeneous[:, 1])
    print("marker homogeneous shape =", marker_homogeneous.shape)
    # print('marker homogeneous =',marker_homogeneous[0:5,:])

    rmse_x = rmse(marker_homogeneous[:, 0], gaze_homogeneous[:, 0])
    rmse_y = rmse(marker_homogeneous[:, 1], gaze_homogeneous[:, 1])
    print("RMSE_az = ", rmse_x)
    print("RMSE_el = ", rmse_y)

    azimuthRange = 45
    elevationRange = 45
    fig = plt.figure(figsize=(10, 10))
    plt.plot(
        marker_homogeneous[:, 0],
        marker_homogeneous[:, 1],
        "or",
        markersize=8,
        alpha=0.6,
        label="marker",
    )
    plt.plot(
        gaze_homogeneous[:, 0],
        gaze_homogeneous[:, 1],
        "+b",
        markersize=8,
        alpha=0.6,
        label="gaze",
    )
    plt.title("Marker Vs. Gaze Positions (Raw)", fontsize=18)
    plt.legend(fontsize=12)
    plt.text(-40, 40, ("RMSE_az = %.2f" % (rmse_x)), fontsize=14)
    plt.text(-40, 35, ("RMSE_el = %.2f" % (rmse_y)), fontsize=14)
    # plt.text(-40,30, ('Distance = %d [inch] %d [cm]'%(depth_inch[depthIndex], depth_cm[depthIndex])), fontsize = 14)
    plt.xlabel("azimuth (degree)", fontsize=14)
    plt.ylabel("elevation (degree)", fontsize=14)
    plt.xticks(np.arange(-azimuthRange, azimuthRange + 1, 5), fontsize=14)
    plt.yticks(np.arange(-elevationRange, elevationRange + 1, 5), fontsize=14)
    plt.xlim((-azimuthRange, elevationRange))
    plt.ylim((-azimuthRange, elevationRange))
    plt.grid(True)

    # plt.savefig(dataPath + '/offline_data/gaze_accuracy_'+str(start_seconds)+'_'+ str(end_seconds)+'.png', dpi = 200 )
    plt.show()


def plot_calibration(point_mapper,):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(eye_video[0])
    # Alpha of 0.75
    cols_2d[:, 3] = 0.75
    confidence_thresh = 0.8

    pupil_keep = np.all(measured_pos > 0, axis=1) & (
        pupil_confidence > confidence_thresh
    )  # 97% of data for this data set

    measured_pos_good = measured_pos[pupil_keep, :]

    dot_h = ax[0].plot(
        measured_pos_good[::4, 0], measured_pos_good[::4, 1], "r.", alpha=0.03
    )
    grid_h = ax[0].scatter(
        eye_grid_x.flatten() * 192, eye_grid_y.flatten() * 192, c=cols_2d, s=5
    )
    for dh in dot_h:
        dh.zorder = 1
    grid_h.zorder = 2
    ax[1].imshow(world_vid[0])
    dot_h2 = ax[1].plot(
        gaze_pos[::4, 0] * vhdim, gaze_pos[::4, 1] * vvdim, "r.", alpha=0.03
    )
    grid_h2 = ax[1].scatter(
        imgrid[:, 0] * vhdim, imgrid[:, 1] * vvdim, c=cols_2d, s=20, marker="."
    )
    for dh in dot_h2:
        dh.zorder = 1
    grid_h2.zorder = 2

    # ax[1].set_ylim([vvdim, 0])
    # ax[1].set_xlim([0, vhdim])
