import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2
import scipy.interpolate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gaze_accuracy(reference_pos, gaze_pos, confidence):
    """"""
    horizontal_pixels = 2048
    vertical_pixels = 1536
    horizontal_FOV = 110
    vertical_FOV = 90

    ratio_x = horizontal_FOV / horizontal_pixels
    ratio_y = vertical_FOV / vertical_pixels

    gaze_norm_x = gaze_pos[:,0]
    gaze_norm_y = gaze_pos[:,1]

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


def plot_gaze_accuracy_heatmap(marker, gaze_pos, confidence, file_name, reference_type="calibration"):
    """"""
    horizontal_pixels = 2048
    vertical_pixels = 1536
    horizontal_FOV = 110
    vertical_FOV = 90
    sns.set()

    ratio_x = horizontal_FOV / horizontal_pixels
    ratio_y = vertical_FOV / vertical_pixels

    if confidence:
        threshold = 0.6
        valid_index = np.argwhere(np.asarray(confidence) > threshold)

        gaze_norm_x = gaze_pos[valid_index, 0]
        gaze_norm_y = gaze_pos[valid_index, 1]
        marker_norm_x = marker[valid_index, 0]
        marker_norm_y = marker[valid_index, 1]
    else:
        threshold = None
        gaze_norm_x = gaze_pos[:, 0]
        gaze_norm_y = gaze_pos[:, 1]
        marker_norm_x = marker[:, 0]
        marker_norm_y = marker[:, 1]


    gaze_pixel_x = gaze_norm_x * horizontal_pixels
    gaze_pixel_y = gaze_norm_y * vertical_pixels

    if reference_type == 'Calibration':
        marker_pixel_x = marker_norm_x # horizontal_pixels
        marker_pixel_y = marker_norm_y # vertical_pixels
    else:
        marker_pixel_x = marker_norm_x * 2# horizontal_pixels
        marker_pixel_y = marker_norm_y * 4# vertical_pixels

    gaze_pixel_x = gaze_pixel_x * (110/2048) - 55
    gaze_pixel_y = gaze_pixel_y * (90 / 1536) - 45

    marker_pixel_x = marker_pixel_x *  (110/2048) - 55
    marker_pixel_y = marker_pixel_y *  (90/1536) - 45

    print("gaze shape = ", gaze_pixel_x.shape, gaze_pixel_y.shape)
    print("marker shape = ", marker_pixel_x.shape, marker_pixel_y.shape)

    x = gaze_pixel_x
    y = gaze_pixel_y
    xy = np.column_stack([x.flat, y.flat])  # Create a (N, 2) array of (x, y) pairs.

    colors = np.power(np.power(marker_pixel_x - gaze_pixel_x, 2) + np.power(marker_pixel_y - gaze_pixel_y, 2), 0.5)
    z = colors

    azimuthRange = (-60,60)#(0, 2048)
    elevationRange = (-45,45)# (0, 1536)
    np.random.seed(0)

    # plt.scatter(x, y)
    # plt.savefig('scatterplot.png', dpi=300)

    # plt.tricontourf(x, y, z)
    # plt.savefig('tricontourf.png', dpi=300)

    # Interpolate and generate heatmap:
    #grid_x, grid_y = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
    # grid_x, grid_y = np.mgrid[0:10:1000j, 0:10:1000j]
    for method in ['linear']:  # , 'nearest','cubic'] :
        plt.figure(figsize=(10, 8))
        #grid_z = scipy.interpolate.griddata(xy, z, (grid_x, grid_y), method='linear')
        # [pcolormesh with missing values?](https://stackoverflow.com/a/31687006/395857)
        import numpy.ma as ma
        #plt.pcolormesh(grid_x, grid_y, grid_z, cmap='YlOrRd', vmin=0, vmax=100)# ma.masked_invalid(grid_z)
        #plt.plot(marker_pixel_x, marker_pixel_y, 'or', markersize=6, alpha=0.8, label='marker')
        # plt.plot(gaze_pixel_x, gaze_pixel_y, 'xc', markersize=6, alpha=0.4, label='gaze')

        plt.scatter(marker_pixel_x, marker_pixel_y, edgecolors='face', c=colors, s=50, cmap='YlOrRd', alpha=0.9, vmin=0, vmax=15)
        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels(['0','1','2','>3'])
        cbar.set_label('Error (pixels)', fontsize=12, rotation=90)

        plt.title('Gaze Accuracy [{0}] C>{1}'.format(reference_type, threshold))
        plt.xlim(azimuthRange)
        plt.ylim(elevationRange)
        plt.legend(fontsize=10)
        # plt.colorbar()
        plt.grid(True)
        plt.xlabel('azimuth (pixels)', fontsize=14)
        plt.ylabel('elevation (pixels)', fontsize=14)
        plt.axes().set_aspect('equal')
        plt.savefig(file_name, dpi=200)
        #plt.show()
        plt.close()


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

def plot_pupil_condifence_BP(right_pupil, left_pupil, sessions):
    return True