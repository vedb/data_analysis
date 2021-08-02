from glob import glob
import cv2
import os
import sys
import yaml
import matplotlib as mpl
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from ellipses import LSqEllipse  # The code is pulled from https://github.com/bdhammel/least-squares-ellipse-fitting
import time
# This annotation script is written by Kamran Binaee and Kaylie Cappurro inspired by DeepVog repo by Taha Emre



def _get_annotation_path_from_image_path(image_file_name, path, eye_part):
    # print('get txt file name: ', path + image_file_name + eye_part + '.txt')
    return path + image_file_name + eye_part + '.txt'


def fit_pupil(image_path, saving_directory, curr_image_number, plot=False, write_annotation=False, eye_part='pupil'):

    # Mouse enumeration for development use only
    # Todo: Remove this after we're done with the design
    '''
    BACK = 8
    FORWARD = 9
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    '''
    upper_color = 'purple'
    lower_color = 'green'
    if 'pupil' in eye_part:
        point_color = 'yellow'
        fill_color = 'orange'
    elif 'iris' in eye_part:
        point_color = 'blue'
        fill_color = 'cyan'
    elif 'upper' in eye_part:
        point_color = 'purple'
        fill_color = 'grey'
    elif 'lower' in eye_part:
        point_color = 'green'
        fill_color = 'white'


    base = os.path.basename(image_path)
    image_file_name = os.path.splitext(base)[0]
    result = 'success'
    while True:
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 15))
        img = imread(image_path)
        ax.set_title('Annotating {} for ID:{}\n File Name:{}'.format(eye_part.replace('_',''),curr_image_number, os.path.basename(image_path)))
        ax.imshow(img, cmap='gray')
        ax.set_xlim(-20, 420)
        ax.set_ylim(-20, 420)

        if 'upper' in eye_part or 'lower' in eye_part:
            if 'upper' in eye_part:
                annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory,
                                                                                '_lower')
                my_color = lower_color
            elif 'lower' in eye_part:
                annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory,
                                                                                '_upper')
                my_color = upper_color
            if os.path.exists(annotated_text_file_name):
                with open(annotated_text_file_name.replace(".txt", "_points.txt")) as f:
                    w, h = [x for x in next(f).split()]  # read first line
                    array = []
                    for line in f:  # read rest of lines
                        array.append([np.float(x) for x in line.split(',')])
                previous_x = [np.float(x[0]) for x in array]
                previous_y = [np.float(x[1]) for x in array]
                ax.plot(previous_x, previous_y, c=my_color, marker='x')

        key_points = plt.ginput(-1, mouse_pop=2, mouse_stop=3,
                                timeout=-1)  # If negative, accumulate clicks until the input is terminated manually.
        points_x = [x[0] for x in key_points]
        points_y = [x[1] for x in key_points]
        if not key_points:
            plt.close()
            result = 'proceed'
            break
        if 'pupil' in eye_part or 'iris' in eye_part:
            fitted = LSqEllipse()
            fitted.fit([points_x, points_y])
            center_coord, width, height, angle = fitted.parameters()
            axes = np.array([width, height])
            angle = np.rad2deg(angle)
        elif 'upper' in eye_part or 'lower' in eye_part:
            poly = np.poly1d(np.polyfit(points_x, points_y, 4))
            print("\npoly calculated:", poly)
            print('\n')

        if write_annotation:
            annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory,
                                                                            eye_part)
            with open(annotated_text_file_name, 'w+') as f:
                # if all([c <= 50 for c in center_coord]):
                #     points_str = '-1:-1'
                # else:
                if 'pupil' in eye_part or 'iris' in eye_part:
                    points_str = '{}, {}'.format(center_coord[0], center_coord[1])
                    f.write(points_str)
                elif 'upper' in eye_part or 'lower' in eye_part:
                    f.write('{}, {}\n'.format(min(points_x), points_y[np.argmin(points_x)]))
                    print("The left most eyelid point: {:.2f} {:.2f}".format(min(points_x), points_y[np.argmin(points_x)]))
                    f.write('{}, {}\n'.format(max(points_x), points_y[np.argmax(points_x)]))
                    print("The right most eyelid point: {:.2f} {:.2f}".format(max(points_x), points_y[np.argmax(points_x)]))

            with open(annotated_text_file_name.replace(".txt","_points.txt"), 'w+') as f:  # For detecting selected
                for point in key_points:
                    f.write('{}, {}\n'.format(point[0], point[1]))

        if plot:
            all_x = [x[0] for x in key_points]
            all_y = [x[1] for x in key_points]
            plt.scatter(x=all_x, y=all_y, c=point_color, marker='x')

            if 'pupil' in eye_part or 'iris' in eye_part:
                ell = mpl.patches.Ellipse(xy=center_coord, width=axes[0] * 2,
                                          height=axes[1] * 2, angle=angle, fill=True, color=fill_color, alpha=0.4)
                ax.add_artist(ell)
            elif 'upper' in eye_part or 'lower' in eye_part:
                for my_x in np.arange(min(points_x), max(points_x), 1):
                    my_y = poly(my_x)
                    plt.plot(my_x, my_y, c=point_color, marker='o')

            output_image_file = saving_directory + image_file_name + eye_part + "_ellipse.png"
            fig.savefig(output_image_file)
            print("saved: ", os.path.basename(output_image_file))
            print('\n')
            plt.show()
            confirmation_point = plt.ginput(1, timeout=-1, mouse_add=3, mouse_stop=3)
            plt.close()
            if len(confirmation_point) == 0:
                break
        # Hacky way to read the q press to quit the loop otherwise the QT doesn't let go of the thread
        # TODO: gracefully stop the tool
        time.sleep(.01)
        answer = input("")
        print('q pressed!!', answer)
        if answer == 'q':
            print("\n\nQuiting the Annotation tool!")
            result = 'quit'
            # plt.ioff()
            # plt.close()
            # sys.exit(0)
            break
    return result


def annotate(image_directory, saving_directory, eye_part='pupil'):
    images_paths = sorted(glob(os.path.join(image_directory, '*png')))
    # imag_paths = sorted(glob(os.path.join(base_dir, '*jpg')))
    annotation_paths = glob(os.path.join(saving_directory, '*txt'))
    i = 0
    for image_path in images_paths:
        print("Running Annotation for: {} ID: {}/{}".format(os.path.basename(image_path), i, len(images_paths)))
        base = os.path.basename(image_path)
        image_file_name = os.path.splitext(base)[0]
        annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory, eye_part)
        if annotated_text_file_name in annotation_paths:
            print("Found the existing txt file for: ", os.path.basename(annotated_text_file_name))
        else:
            result = fit_pupil(image_path=image_path, saving_directory=saving_directory, curr_image_number=i, plot=True, write_annotation=True, eye_part=eye_part)
            if result == 'quit':
                print("\n\nQuit!!\n\n")
                break
        i = i + 1


def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath, "r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict


if __name__ == '__main__':
    plt = mpl.pyplot
    fig = plt.figure()
    # mpl.rcParams["savefig.directory"] = os.chdir(
    #     os.path.dirname('/home/kamran/Downloads/eye_image_annotation_results/'))

    # File Path for the yaml file
    parameters_fpath = os.getcwd() + "/annotation_parameters.yaml"
    param_dict = parse_pipeline_parameters(parameters_fpath)
    image_directory = param_dict['directory']['image_directory']
    saving_directory = param_dict['directory']['saving_directory']
    eye_part = param_dict['annotation']['eye_part']
    print(param_dict)
    print(eye_part)
    if 'pupil' in eye_part:
        eye_part = 'pupil'
    elif 'iris' in eye_part:
        eye_part = 'iris'
    elif 'upper' in eye_part:
        eye_part = 'upper'
    elif 'lower' in eye_part:
        eye_part = 'lower'
    else:
        raise ValueError("Wrong Eye Part for Annotation!!!")
    annotate(image_directory=image_directory, saving_directory=saving_directory, eye_part='_' + eye_part)
    sys.exit(0)
