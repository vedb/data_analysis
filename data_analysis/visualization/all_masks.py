import os
import yaml
import cv2
import numpy as np
from glob import glob
import sys
import copy

def create_label(raw_image_path, label_image_path):
    raw_image = cv2.imread(raw_image_path)
    label_image = cv2.imread(label_image_path)
    final_mask = np.zeros(label_image.shape, dtype=np.float64)
    c = np.zeros(label_image.shape, dtype=np.float64)
    # final_mask = copy.deepcopy(label_image)
    # final_mask.setflags(write=1)
    # final_mask[label_image == [0, 0, 0]] = 0 # [0, 0, 0]
    # idx = np.where(label_image[:,:,0] == 128 and label_image[:,:,1] == 128 and label_image[:,:,2] == 128)
    idx = np.where(label_image == [128, 128, 128])
    final_mask = np.where(label_image[:,:] == [128, 128, 128], final_mask, [0,0,0])
    # c = np.tile([0.,0.,0.],[idx.shape[0],idx.shape[1],1])
    # print("Hi")
    # final_mask[idx[0]][:,idx[1]] = c[idx[0]][:,idx[1]]
    # idx = np.where(label_image[:,:,0] == 255 and label_image[:,:,1] == 255 and label_image[:,:,2] == 255)
    idx = np.where(label_image == [255, 255, 255])
    final_mask = np.where(label_image[:,:] == [255, 255, 255], final_mask, [0,0,0])
    # c = np.tile([0.,0.,0.],[idx.shape[0],idx.shape[1],1])
    # print("Hooo")
    # final_mask[idx[0]][:,idx[1]] = c[idx[0]][:,idx[1]]
    # idx = np.where(label_image[:,:,0] == 255 and label_image[:,:,1] == 255 and label_image[:,:,2] == 0)
    idx = np.where(label_image == [255, 255, 0])
    final_mask = np.where(label_image[:,:] == [255, 255, 0], final_mask, [50,50,50])
    # c = np.tile([50.,50.,50.],[idx.shape[0],idx.shape[1],1])
    # final_mask[idx[0]][:,idx[1]] = c[idx[0]][:,idx[1]] + 50
    # idx = np.where(label_image[:,:,0] == 255 and label_image[:,:,1] == 0 and label_image[:,:,2] == 0)
    idx = np.where(label_image == [255, 0, 0])
    final_mask = np.where(label_image[:,:] == [255, 0, 0], final_mask, [100,100,100])
    # c = np.tile([100., 100., 100.],[idx.shape[0],idx.shape[1],1])
    # final_mask[idx[0]][:,idx[1]] = c[idx[0]][:,idx[1]] + 100
    # idx = np.where(label_image[:,:,0] == 0 and label_image[:,:,1] == 0 and label_image[:,:,2] == 255)
    idx = np.where(label_image == [0, 0, 255])
    final_mask = np.where(label_image[:,:] == [0, 0, 255], final_mask, [200,200,200])
    # c = np.tile([200., 200., 200.],[idx.shape[0],idx.shape[1],1])
    # final_mask[idx[0]][:,idx[1]] = c[idx[0]][:,idx[1]] + 200
    colors = np.unique(label_image.reshape(400*400, 3), axis=0)
    tag = "{} regions:\n{}".format(len(colors), colors)
    print(tag)
    frame = np.concatenate((raw_image, label_image), axis=1)
    font = cv2.FONT_HERSHEY_PLAIN
    frame = cv2.putText(frame, str(len(colors))+" regions", (450, 50), font, 1.5, [250,50,0], 2, cv2.LINE_AA)
    # frame = np.concatenate((frame, final_mask), axis=1)
    cv2.imshow("Input & Label", frame)
    cv2.imshow("Mask", final_mask.astype(np.uint8))
    base = os.path.basename(raw_image_path)
    mask_file_name = os.path.splitext(base)[0]
    file_name = saving_directory + mask_file_name + "_mask.png"
    print(file_name)
    cv2.imwrite(file_name, final_mask.astype(np.uint8))
    if cv2.waitKey(4000) & 0xFF == ord("q"):
        return True
    # cv2.destroyAllWindows()
    return True


def all_labels(image_directory,
               saving_directory,
               label_directory,
               param_dict):
    images_paths = sorted(glob(os.path.join(image_directory, '*png')))
    # annotation_paths = glob(os.path.join(saving_directory, '*txt'))
    i = 0
    for image_path in images_paths[0:20]:
        print("Running Label Generation for: {} ID: {}/{}".format(os.path.basename(image_path), i, len(images_paths)))
        base = os.path.basename(image_path)
        image_file_name = os.path.splitext(base)[0]
        files_missing = []
        label_image_file_name = label_directory + image_file_name + "_label.png"
        input_label_image_file_name = label_directory + image_file_name + "_label___input.png"
        image_files = [label_image_file_name, input_label_image_file_name]
        for image_file in image_files:
            if not os.path.exists(image_file):
                files_missing.append(os.path.basename(image_file))
                print("Image File {} not found!".format(txt_file))
        if len(files_missing) > 0:
            print("One or more file missing!!", files_missing)
            continue
        i = i + 1

        result = create_label(image_path, label_image_file_name)
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
    saving_directory = param_dict['directory']['saving_directory']
    label_directory = param_dict['directory']['label_directory']
    print(param_dict)
    all_labels(image_directory=image_directory, saving_directory=saving_directory, label_directory=label_directory,
               param_dict=param_dict)
    sys.exit(0)
