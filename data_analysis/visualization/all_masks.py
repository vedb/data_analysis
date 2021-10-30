import os
import yaml
import cv2
import numpy as np
from glob import glob
import sys
import copy


def create_label(raw_image_path, label_image_path, param_dict):

    saving_directory_mask = label_image_path.replace("all_labels", "all_masks")
    saving_directory_array = label_image_path.replace("all_labels", "all_arrays")
    raw_image = cv2.imread(raw_image_path)
    label_image = cv2.imread(label_image_path)
    final_mask = np.zeros(label_image.shape, dtype=np.uint8)
    mask_color = np.zeros(label_image.shape, dtype=np.uint8)
    mask_array = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
    image_width = raw_image.shape[0]
    image_height = raw_image.shape[1]
    # print("pixel value: ", label_image[10][30][:])
    for i in range(image_width):
        for j in range(image_height):
            if np.array_equal(label_image[i][j][:], [128, 128, 128]) or np.array_equal(label_image[i][j][:], [0, 0, 0]) or np.array_equal(label_image[i][j][:], [255, 255, 255]):
                mask_color[i][j] = [0, 0, 0]
                mask_array[i][j] = 0
            elif np.array_equal(label_image[i][j][:], [255, 255, 0]):
                mask_color[i][j] = [75, 75, 75]
                mask_array[i][j] = 1
            elif np.array_equal(label_image[i][j][:], [255, 0, 0]):
                mask_color[i][j] = [150, 150, 150]
                mask_array[i][j] = 2
            elif np.array_equal(label_image[i][j][:], [0, 0, 255]):
                mask_color[i][j] = [225, 225, 225]
                mask_array[i][j] = 3
    final_mask = mask_color
    colors = np.unique(label_image.reshape(400*400, 3), axis=0)
    tag = "{} regions:\n{}".format(len(colors), colors)
    # print(tag)
    frame = np.concatenate((raw_image, label_image), axis=1)
    font = cv2.FONT_HERSHEY_PLAIN
    frame = cv2.putText(frame, str(len(colors))+" regions", (450, 50), font, 1.5, [250,50,0], 2, cv2.LINE_AA)
    # frame = np.concatenate((frame, final_mask), axis=1)
    cv2.imshow("Input & Label", frame)
    cv2.imshow("Mask", final_mask.astype(np.uint8))
    base = os.path.basename(saving_directory_mask)
    mask_file_name = os.path.splitext(base)[0].replace("_label", "_mask")
    # print("mask: ", os.path.dirname(os.path.realpath(saving_directory_mask)), mask_file_name)
    file_name = os.path.dirname(os.path.realpath(saving_directory_mask)) + "/" + mask_file_name + ".png"
    # print(file_name)
    cv2.imwrite(file_name, final_mask.astype(np.uint8))
    base = os.path.basename(saving_directory_array)
    mask_file_name = os.path.splitext(base)[0].replace("_label", "_array")
    # print("array: ", mask_file_name, os.path.dirname(os.path.realpath(saving_directory_array)), mask_file_name)
    file_name = os.path.dirname(os.path.realpath(saving_directory_array)) + "/" + mask_file_name + ".npy"
    np.save(file_name, mask_array)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    # cv2.destroyAllWindows()
    return True


def all_labels(image_directory, label_directory, param_dict):
    result = False
    images_paths = sorted(glob(os.path.join(image_directory, '*png')))
    i = 0
    for image_path in images_paths:
        print("Running Label Generation for: {} ID: {}/{}".format(os.path.basename(image_path), i, len(images_paths)))
        base = os.path.basename(image_path)
        image_file_name = os.path.splitext(base)[0]
        files_missing = []
        label_image_file_name = label_directory + "/" + image_file_name + "_label.png"
        input_label_image_file_name = label_directory + "/" + image_file_name + "_label_&_input.png"
        image_files = [label_image_file_name, input_label_image_file_name]
        for image_file in image_files:
            if not os.path.exists(image_file):
                files_missing.append(os.path.basename(image_file))
                print("Image File {} not found!".format(image_file))
        if len(files_missing) > 0:
            print("One or more file missing!!", files_missing)
            continue
        i = i + 1

        result = create_label(raw_image_path=image_path,
                              label_image_path=label_image_file_name,
                              param_dict=param_dict)
    cv2.destroyAllWindows()
    return result


def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath, "r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict


if __name__ == '__main__':

    # File Path for the yaml file
    parameters_fpath = os.getcwd() + "/annotation_parameters.yaml"
    param_dict = parse_pipeline_parameters(parameters_fpath)
    image_directory = param_dict['directory']['image_directory']
    parent_label_directory = param_dict['directory']['label_directory']
    print(param_dict)
    parent_directory = "/hdd01/eye_images/Final_eye_images/Bates_labels/all_labels/"
    directory_contents = ["/hdd01/eye_images/2021_07_05_13_09_00/",
                          "/hdd01/eye_images/2021_07_13_11_34_04/",
                          "/hdd01/eye_images/2021_07_13_14_32_23/",
                          "/hdd01/eye_images/2021_07_13_14_37_32/",
                          "/hdd01/eye_images/2021_07_13_14_43_36/",
                          "/hdd01/eye_images/2021_07_14_17_00_32/",
                          "/hdd01/eye_images/2021_07_16_12_29_06/",
                          "/hdd01/eye_images/2021_07_16_13_00_43/",
                          "/hdd01/eye_images/2021_07_21_13_47_43/",
                          "/hdd01/eye_images/2021_07_22_14_04_09/",
                          "/hdd01/eye_images/2021_07_22_14_49_47/",
                          "/hdd01/eye_images/2021_07_22_15_25_59/"
                          ]
    for image_directory in directory_contents:
        label_directory = parent_label_directory + os.path.basename(os.path.normpath(image_directory))
        print("Label Directory:", label_directory)
        all_labels(image_directory=image_directory,
                   label_directory=label_directory,
                   param_dict=param_dict)
    sys.exit(0)
