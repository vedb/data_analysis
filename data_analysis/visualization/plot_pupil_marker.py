"""
Created on Tue Jun 16 10:22:18 2020

Usage: python plot_pupil_marker.py 

        Parameters
        ----------


@author: KamranBinaee
"""
import sys
import os
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


pupil_dataFrame = pd.read_csv('detected_pupil_positions.csv')
marker_dataFrame = pd.read_csv('detected_marker_positions.csv')
gaze_dataFrame = pd.read_csv('EyeTrackingData/gaze_positions.csv')

gaze_pixel_x = pupil_dataFrame.pupil_x.values
# Because the y coordinate is upside  down in opencv compared to matplotlib
gaze_pixel_y = 400 - pupil_dataFrame.pupil_y.values

marker_pixel_x = marker_dataFrame.marker_x.values
# Because the y coordinate is upside  down in opencv compared to matplotlib
marker_pixel_y = 1024 - marker_dataFrame.marker_y.values

print('pupilX shape = ', gaze_pixel_x.shape)
print('pupilY shape = ',gaze_pixel_y.shape)

print('markerX shape = ', marker_pixel_x.shape)
print('markerY shape = ',marker_pixel_y.shape)


fig = plt.figure(figsize = (22,10))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.suptitle('Pupil vs. Marker Position', fontsize = 24)
ax1.plot(gaze_pixel_x, gaze_pixel_y, 'xr', markersize = 8, alpha = 0.6, label = 'pupil')
plt.grid(True)
ax2.plot(marker_pixel_x, marker_pixel_y, 'ob', markersize = 8, alpha = 0.6, label = 'marker')
#plt.title('Marker Vs. pupil Positions (Raw)', fontsize = 18)
ax1.legend(fontsize = 18)
ax2.legend(fontsize = 18)
ax1.set_xlabel('X (pixels)', fontsize = 16)
ax1.set_ylabel('Y (pixels)', fontsize = 16)
ax1.set_xlim(0,400)
ax1.set_ylim(0,400)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)

ax2.set_xlabel('X (pixels)', fontsize = 16)
ax2.set_ylabel('Y (pixels)', fontsize = 16)
ax2.set_xlim(0,1280)
ax2.set_ylim(0,1024)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)

ax1.grid(True)
ax2.grid(True)
plt.savefig('Pupil_Vs_Marker.png', dpi = 200 )
plt.show()

plt.close()

x = gaze_dataFrame.norm_pos_x.values[2000:8200*4]*1280.0
y = gaze_dataFrame.norm_pos_y.values[2000:8200*4]*1024.0

print(len(x))
print(x.mean(), x.std(), x.max(), x.min())
print(y.mean(), y.std(), y.max(), y.min())
fig = plt.figure(figsize = (10,8))
plt.hist2d(x, y, range=np.array([(0, 1280), (0, 1024)]), bins=(50,50), cmap=plt.cm.coolwarm) #bins=[np.arange(0,1280,20), np.arange(0,1024,20)])# , norm=colors.LogNorm()
plt.colorbar()
plt.title('2D Histogram of Gaze Positions', fontsize = 18)
plt.xlim(0,1280)
plt.ylim(0,1024)
plt.savefig('gaze_histogram.png', dpi = 200 )
plt.show()
plt.close()