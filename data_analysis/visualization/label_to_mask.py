import copy
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# This annotation script is written by Kamran Binaee and Kaylie Cappurro inspired by DeepVog repo written by Taha Emre

def create_fits(text_files, image_path, count):
    pupil_points = []
    iris_points = []
    upper_points = []
    lower_points = []
    tags = ['pupil', 'iris', 'upper', 'lower']
    colors = ['yellow', 'cyan', 'green', 'purple']

    my_dpi = 200
    fig, ax = plt.subplots(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)#figsize=(15, 15)
    # fig = Figure()
    canvas = FigureCanvas(fig)
    # ax = fig.gca()
    # ax.set_title('Showing the labels for {}'.format(os.path.basename(text_files[0])))
    # img = imread(image_path)
    img = np.zeros((400, 400))
    ax.imshow(img, cmap='gray')
    # ax.set_xlim(-20, 420)
    # ax.set_ylim(-20, 420)

    i = 0
    for txt_file in text_files:
        if os.path.exists(txt_file):
            print("Reading txt file: {}".format(os.path.basename(txt_file)))
            with open(txt_file) as f:
                w, h = [x for x in next(f).split()]  # read first line
                array = []
                for line in f:  # read rest of lines
                    array.append([np.float(x) for x in line.split(',')])
            _x = [np.float(x[0]) for x in array]
            _y = [np.float(x[1]) for x in array]
            # ax.sc(_x, previous_y, c=colors[i], marker='x', label=tags[i])
            # plt.scatter(x=_x, y=_y, c=colors[i], marker='x', label=tags[i])
            if 'pupil' in txt_file:
                pupil_points = np.array([_x, _y])
                # print(pupil_points.shape)
            elif 'iris' in txt_file:
                iris_points = np.array([_x, _y])
                # print(iris_points.shape)
            elif 'upper' in txt_file:
                upper_points = np.array([_x, _y])
                upper_poly = np.poly1d(np.polyfit(_x, _y, 4))
            elif 'lower' in txt_file:
                lower_points = np.array([_x, _y])
                lower_poly = np.poly1d(np.polyfit(_x, _y, 4))
        i = i + 1

    fitted = LSqEllipse()
    fitted.fit([iris_points[0,:], iris_points[1,:]])
    center_coord, width, height, angle = fitted.parameters()
    axes = np.array([width, height])
    angle = np.rad2deg(angle)
    iris_ellipse = mpl.patches.Ellipse(xy=center_coord, width=axes[0] * 2,
                                       height=axes[1] * 2, angle=angle, fill=True, color=[0,1,1], alpha=1)
    ax.add_artist(iris_ellipse)

    fitted = LSqEllipse()
    fitted.fit([pupil_points[0,:], pupil_points[1,:]])
    center_coord, width, height, angle = fitted.parameters()
    axes = np.array([width, height])
    angle = np.rad2deg(angle)
    pupil_ellipse = mpl.patches.Ellipse(xy=center_coord, width=axes[0] * 2,
                                        height=axes[1] * 2, angle=angle, fill=True, color=[1,0,1], alpha=1)
    ax.add_artist(pupil_ellipse)
    ax.axis('off')
    # canvas.draw()  # draw the canvas, cache the renderer

    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # print(image_from_plot.shape)
    # print(np.unique(image_from_plot[:,:]))

    a = copy.deepcopy(image_from_plot)
    a.setflags(write=1)
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    current_label = -1
    previous_label = -1

    if '_R' in image_path:
        dummy_0 = lower_poly
        dummy_1 = upper_poly
        lower_poly = dummy_1
        upper_poly = dummy_0
    for q in range(0, 400):
        current_label = 0
        for p in range(0, 400):
            if q > upper_poly(p) and q > lower_poly(p):
                a[q,p,:] = [128,128,128]
                # print('skin top')
            elif q <= upper_poly(p) and q >= lower_poly(p):
                # print(image_from_plot[p,q,:])
                if (image_from_plot[q,p,1]>0):
                    # print("iris")
                    a[q, p, :] = [255, 0, 0]
                elif (image_from_plot[q,p,0]>0):
                    # print("pupil")
                    a[q, p, :] = [0, 0, 255]
                else:
                    a[q,p,:] = [255,255,0]
                    # print('sclera')
            # elif q > upper_poly(p) and q <= lower_poly(p):
            #     a[q, p, :] = [0, 0, 255]
            elif q < upper_poly(p) and q < lower_poly(p):
                a[q,p,:] = [255,255,255]
                # print('skin bottom')

    print(np.unique(a.reshape(400*400, 3), axis=0))
    ax.imshow(a)
    ax.axis('off')
    # canvas.draw()  # draw the canvas, cache the renderer

    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    base = os.path.basename(image_path)
    image_file_name = os.path.splitext(base)[0]
    label_file_name = saving_directory + image_file_name + '_label.png'
    cv2.imwrite(label_file_name, a)
    input_image = cv2.imread(image_path)
    final_frame = np.concatenate((input_image, a), axis=1)
    label_file_name = saving_directory + image_file_name + '_label_&_input.png'
    cv2.imwrite(label_file_name, final_frame)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(final_frame)
    # cv2.imshow("Input Vs. Label", final_frame)
    # plt.savefig(str(count)+'_test.png')
    # plt.legend()
    # plt.show()

def label_to_mask(image_directory, saving_directory, param_dict):
    images_paths = sorted(glob(os.path.join(image_directory, '*png')))
    annotation_paths = glob(os.path.join(saving_directory, '*txt'))
    i = 0
    for image_path in images_paths:
        print("Running Mask Generation for: {} ID: {}/{}".format(os.path.basename(image_path), i, len(images_paths)))
        base = os.path.basename(image_path)
        image_file_name = os.path.splitext(base)[0]
        files_missing = []
        pupil_points_file_name = saving_directory + image_file_name + '_pupil_points.txt'
        iris_points_file_name = saving_directory + image_file_name + '_iris_points.txt'
        upper_points_file_name = saving_directory + image_file_name + '_upper_points.txt'
        lower_points_file_name = saving_directory + image_file_name + '_lower_points.txt'
        text_files = [pupil_points_file_name, iris_points_file_name, upper_points_file_name, lower_points_file_name]
        for txt_file in text_files:
            if not os.path.exists(txt_file):
                files_missing.append(os.path.basename(txt_file))
                print("Annotation File {} not found!".format(txt_file))
        if len(files_missing) > 0:
            print("One file missing!!", files_missing)
            continue
        i = i + 1

        result = create_fits(text_files, image_path, i)


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
    label_to_mask(image_directory=image_directory, saving_directory=saving_directory, param_dict=param_dict)
    sys.exit(0)
