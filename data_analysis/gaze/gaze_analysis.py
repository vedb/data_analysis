import yaml
import sys
import os
import cv2

# Todo: This should not be here, most probably goes into vm_tools
def calibrate_gaze(cyclopeanPOR_XY, truePOR_XY, method = cv2.RANSAC, threshold = 5, plottingFlag = False):

    result = cv2.findHomography(cyclopeanPOR_XY, truePOR_XY, method = method , ransacReprojThreshold = threshold)
    #print(result[0])
    #print('size', len(result[1]),'H=', result[1])
    totalFrameNumber = truePOR_XY.shape[0]
    arrayOfOnes = np.ones((totalFrameNumber,1), dtype = float)

    homogrophy = result[0]
    print('H=', homogrophy, '\n')
    #print('Res', result[1])
    cyclopeanPOR_XY = np.hstack((cyclopeanPOR_XY, arrayOfOnes))
    truePOR_XY = np.hstack((truePOR_XY, arrayOfOnes))
    projectedPOR_XY = np.zeros((totalFrameNumber,3))
    
    for i in range(totalFrameNumber):
        projectedPOR_XY[i,:] = np.dot(homogrophy, cyclopeanPOR_XY[i,:])
        #print cyclopeanPOR_XY[i,:]
    
    #projectedPOR_XY[:, 0], projectedPOR_XY[:, 1] = metricToPixels(projectedPOR_XY[:, 0], projectedPOR_XY[:, 1])
    #cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1] = metricToPixels(cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1])
    #truePOR_XY[:, 0], truePOR_XY[:, 1] = metricToPixels(truePOR_XY[:, 0], truePOR_XY[:, 1])
    data = projectedPOR_XY
    frameCount = range(len(cyclopeanPOR_XY))

    if( plottingFlag == True ):
        xmin = 550#min(cyclopeanPOR_XY[frameCount,0])
        xmax = 1350#max(cyclopeanPOR_XY[frameCount,0])
        ymin = 250#min(cyclopeanPOR_XY[frameCount,1])
        ymax = 800#max(cyclopeanPOR_XY[frameCount,1])
        #print xmin, xmax, ymin, ymax
        fig1 = plt.figure(figsize = (10,8))
        plt.plot(data[frameCount,0], data[frameCount,1], 'bx', label='Calibrated POR')
        plt.plot(cyclopeanPOR_XY[frameCount,0], cyclopeanPOR_XY[frameCount,1], 'g2', label='Uncalibrated POR')
        plt.plot(truePOR_XY[frameCount,0], truePOR_XY[frameCount,1], 'r8', label='Ground Truth POR')
        #l1, = plt.plot([],[])
        
        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)
        plt.xlabel('X')
        plt.ylabel('Y')
        if ( method == cv2.RANSAC):
            methodTitle = ' RANSAC '
        elif( method == cv2.LMEDS ):
            methodTitle = ' Least Median '
        elif( method == 0 ):
            methodTitle = ' Homography '
        plt.title('Calibration Result using'+ methodTitle+'\nWith System Calibration ')
        plt.grid(True)
        #plt.axis('equal')
        #line_ani = animation.FuncAnimation(fig1, update_line1, frames = 11448, fargs=(sessionData, l1), interval=14, blit=True)
        legend = plt.legend(loc=[0.8,0.92], shadow=True, fontsize='small')# 'upper center'
        plt.show()

    print ('MSE_after = ', findResidualError(projectedPOR_XY, truePOR_XY))
    print ('MSE_before = ', findResidualError(cyclopeanPOR_XY, truePOR_XY))
    return homogrophy

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def pixels_to_angle_x(array):
    return (array - horizontal_pixels/2) * ratio_x

def pixels_to_angle_y(array):
    return (array - vertical_pixels/2) * ratio_y
