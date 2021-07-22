
# This annotation script is written by Kamran Binaee and Kaylie Cappurro inspired by DeepVog repo by Taha Emre



def _get_annotation_path_from_image_path(image_file_name, path, eye_part):
    print('looking for: ', path + image_file_name + eye_part + '.txt')
    return path + image_file_name + eye_part + '.txt'


def fit_pupil(image_path, saving_directory, curr_image_number, plot=False, write_annotation=False, eye_part='pupil'):
    save_directory = "/home/kamran/Downloads/eye_image_annotation_results/"

    if 'pupil' in eye_part:
        point_color = 'yellow'
        fill_color = 'orange'
    else:
        point_color = 'blue'
        fill_color = 'cyan'

    image_file_name = os.path.basename(image_path)

    while True:
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 15))
        img = imread(image_path)
        ax.set_title('({}): {}'.format(curr_image_number, os.path.basename(image_path)))
        ax.imshow(img, cmap='gray')

        key_points = plt.ginput(-1, mouse_pop=2, mouse_stop=3,
                                timeout=-1)  # If negative, accumulate clicks until the input is terminated manually.

        if not key_points:
            if write_annotation:
                annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory,
                                                                                eye_part)
                with open(annotated_text_file_name, 'w+') as f:
                    f.write("closed_eye")

                with open(annotated_text_file_name.replace(".txt","_points.txt"),
                          'w+') as f:  # For detecting selected
                    f.write("closed_eye")
            plt.close()
            break

        fitted = LSqEllipse()
        fitted.fit([[x[0] for x in key_points], [x[1] for x in key_points]])
        center_coord, width, height, angle = fitted.parameters()
        axes = np.array([width, height])
        angle = np.rad2deg(angle)

        if write_annotation:
            annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory,
                                                                            eye_part)
            with open(annotated_text_file_name, 'w+') as f:
                if all([c <= 50 for c in center_coord]):
                    points_str = '-1:-1'
                else:
                    points_str = '{}:{}'.format(center_coord[0], center_coord[1])
                f.write(points_str)

            with open(annotated_text_file_name.replace(".txt","_points.txt"), 'w+') as f:  # For detecting selected
                for point in key_points:
                    f.write('{}:{}\n'.format(point[0], point[1]))

        if plot:
            # ax.annotate('pred center', xy=center_coord, xycoords='data',
            #             xytext=(0.2, 0.2), textcoords='figure fraction',
            #             arrowprops=dict(arrowstyle="->"), color='y')
            # plt.scatter(x=center_coord[0], y=center_coord[1], c='red', marker='x')
            all_x = [x[0] for x in key_points]
            all_y = [x[1] for x in key_points]
            plt.scatter(x=all_x, y=all_y, c=point_color, marker='x')

            ell = mpl.patches.Ellipse(xy=center_coord, width=axes[0] * 2,
                                      height=axes[1] * 2, angle=angle, fill=True, color=fill_color, alpha=0.4)

            ax.add_artist(ell)
            # i = 0
            # while os.path.exists("%d_ellipse.png" % i):
            #     i += 1
            # fig.savefig('%d_ellipse.png' % i)
            output_image_file = os.path.dirname(image_path) + "/Results/" + \
                                os.path.splitext(os.path.basename(image_path))[0] + eye_part + "_ellipse.png"
            output_image_file = saving_directory + image_file_name + eye_part + "_ellipse.png"
            fig.savefig(output_image_file)
            print("saved: ", output_image_file)
            plt.show()
            confirmation_point = plt.ginput(1, timeout=-1, mouse_add=3, mouse_stop=3)
            plt.close()
            if len(confirmation_point) == 0:
                break


def annotate(image_directory, saving_directory, eye_part='pupil'):
    images_paths = sorted(glob(os.path.join(image_directory, '*png')))
    # imag_paths = sorted(glob(os.path.join(base_dir, '*jpg')))
    annotation_paths = glob(os.path.join(saving_directory, '*txt'))
    i = 1
    for image_path in images_paths:
        print("Running Annotation for: ", image_path)
        image_file_name = os.path.basename(image_path)
        annotated_text_file_name = _get_annotation_path_from_image_path(image_file_name, saving_directory, eye_part)
        if annotated_text_file_name in annotation_paths:
            print("Found the existing txt file for: ", annotated_text_file_name)
            continue
        else:
            fit_pupil(image_path, saving_directory=saving_directory, curr_image_number=i, plot=True, write_annotation=True, eye_part=eye_part)
            i += 1

    # for imag_path in imag_paths:
    #     if _get_annotation_path_from_image_path(imag_path, eye_part) in annotation_paths:
    #         continue
    #     else:
    #         fit_pupil(imag_path, curr_image_number=i, plot=True, write_annotation=True, eye_part=eye_part)
    #         i += 1


def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath, "r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict


if __name__ == '__main__':
    from glob import glob
    import cv2
    import os
    import sys
    import yaml
    import matplotlib as mpl
    import numpy as np
    from skimage.io import imread

    from ellipses import LSqEllipse  # The code is pulled from https://github.com/bdhammel/least-squares-ellipse-fitting

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
    # if len(sys.argv) > 2:
    #     annotate(base_dir=sys.argv[1], eye_part='_' + sys.argv[2])
    # else:
    #     annotate()