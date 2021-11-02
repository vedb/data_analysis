import os
import yaml
import cv2
import numpy as np
from glob import glob
import sys
import copy
from os import listdir
from os.path import isfile, join


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_img


def augment_dataset(new_dataset_directory, param_dict):

    dataset_directory = param_dict['directory']['ved_dataset_directory']

    image_path = dataset_directory + "/images/"
    array_path = dataset_directory + "/labels/"
    mask_path = dataset_directory + "/mask/"

    image_files = [f for f in sorted(listdir(image_path)) if isfile(join(image_path, f))]
    array_files = [f for f in sorted(listdir(array_path)) if isfile(join(array_path, f))]
    mask_files = [f for f in sorted(listdir(mask_path)) if isfile(join(mask_path, f))]

    gamma_range = [0, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 2.7, 3.0]
    beta_list = [0, 10, 20, 30, 40]

    scale = 1
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org_1 = (int(400 * scale), int(30 * scale))
    org_2 = (int(450 * scale), int(30 * scale))
    # fontScale
    fontScale = 1 * scale
    # Blue color in BGR
    color = (250, 250, 0)
    # Line thickness of 2 px
    thickness = 2
    # Window name in which image is displayed
    window_name = 'Image'
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Blue color in BGR
    font_color = (255, 10, 40)

    try:
        for i in range(len(image_files)):
            image_file = image_files[i]
            array_file = array_files[i]
            mask_file = mask_files[i]
            img = cv2.imread(image_path + image_file)
            my_array = np.load(array_path + array_file)
            my_mask = cv2.imread(mask_path + mask_file)
            input_image = copy.deepcopy(img)
            for gamma in gamma_range:
                for beta in beta_list:
                    new_image = np.zeros(input_image.shape, input_image.dtype)

                    if gamma > 0:
                        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
                        contrast_image = cv2.LUT(np.array(input_image), table)
                    else:
                        contrast_image = input_image
                    if beta > 0:
                        new_image = increase_brightness(contrast_image.astype(np.uint8), value=beta)
                    else:
                        new_image = contrast_image.astype(np.uint8)

                    a = np.concatenate((img.astype(np.uint8), contrast_image.astype(np.uint8)), axis=1)
                    final_frame = np.concatenate((a, new_image.astype(np.uint8)), axis=1)
                    my_string = "gamma = {} beta = {}".format(gamma, beta)
                    final_frame = cv2.putText(final_frame, my_string,
                                              org_1, font,
                                              fontScale, font_color, thickness, cv2.LINE_AA)
                    cv2.imshow("Input Vs. Contrast Stretched & Brightness Shift", final_frame)
                    my_string = "_{}_{}.png".format(gamma, beta)
                    image_file_name = new_dataset_directory + "/images/" + image_file.replace(".png", my_string)
                    cv2.imwrite(image_file_name, new_image.astype(np.uint8))

                    my_string = "_{}_{}_array.npy".format(gamma, beta)
                    array_file_name = new_dataset_directory + "/labels/" + array_file.replace("_array.npy", my_string)
                    np.save(array_file_name, my_array)

                    my_string = "_{}_{}_mask.png".format(gamma, beta)
                    mask_file_name = new_dataset_directory + "/mask/" + mask_file.replace("_mask.png", my_string)
                    cv2.imwrite(mask_file_name, my_mask.astype(np.uint8))

                    while cv2.waitKey(50) & 0xFF == ord("q"):
                        raise StopIteration
    except StopIteration:
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath, "r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict


if __name__ == '__main__':

    # File Path for the yaml file
    parameters_fpath = os.getcwd() + "/annotation_parameters.yaml"
    param_dict = parse_pipeline_parameters(parameters_fpath)

    new_dataset_directory = "/hdd01/eye_images/VED_Dataset_Augmented/"
    print(param_dict)
    print("New Dataset Directory:", new_dataset_directory)
    augment_dataset(new_dataset_directory=new_dataset_directory,
                    param_dict=param_dict)
    sys.exit(0)
