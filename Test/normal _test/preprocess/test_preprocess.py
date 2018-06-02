import cv2
import os
import scipy.misc as scm
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from test_preprocess_config import Valid_FLAGS


test_data_file = Valid_FLAGS.test_data_file


def open_img(name, color='RGB'):
    """ Open an image
    Args:
        name	: Name of the sample
        color	: Color Mode (RGB/BGR/GRAY)
    """
    img = cv2.imread(os.path.join(Valid_FLAGS.test_img_directory, name))
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif color == 'BGR':
        return img
    elif color == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')


def read_test_data():
    """
    To read labels in csv
    """
    test_table = []  # The names of images being trained
    label_file = pd.read_csv(test_data_file)
    print('READING LABELS OF TRAIN DATA')
    for i in range(label_file.shape[0]):
        name = str(label_file.at[i, 'image_id'])
        test_table.append(name)
    print('LABEL READING FINISHED')
    return test_table


def main(argv):
    # make processed img save dir
    suffix_path = os.path.join('processed_b',
                               'Images')
    img_save_dir_blouse = os.path.join(suffix_path,
                                       'blouse')
    img_save_dir_dress = os.path.join(suffix_path,
                                      'dress')
    img_save_dir_outwear = os.path.join(suffix_path,
                                        'outwear')
    img_save_dir_skirt = os.path.join(suffix_path,
                                      'skirt')
    img_save_dir_trousers = os.path.join(suffix_path,
                                         'trousers')
    os.system('mkdir -p {}'.format(img_save_dir_blouse))
    os.system('mkdir -p {}'.format(img_save_dir_dress))
    os.system('mkdir -p {}'.format(img_save_dir_outwear))
    os.system('mkdir -p {}'.format(img_save_dir_skirt))
    os.system('mkdir -p {}'.format(img_save_dir_trousers))

    # read data
    test_set = read_test_data()
    test_num = len(test_set)
    print('test_num:', test_num)

    # process img
    test_iter = 0
    while test_iter < test_num:
        name = test_set[test_iter]
        img = cv2.imread(os.path.join(Valid_FLAGS.test_img_directory, name))
        img_shape = img.shape
        img_x = img_shape[0]
        img_y = img_shape[1]

        img_512 = np.zeros((512, 512, 3), dtype= np.float32)
        x_padding = (512 - img_x) // 2
        y_padding = (512 - img_y) // 2
        img_512[x_padding:x_padding+img_x, y_padding:y_padding+img_y, :] = img[:, :, :]

        name_split = name.split('/')
        dress_type = name_split[1]
        img_root_name = name_split[2]
        img_save_path = os.path.join(suffix_path, dress_type)
        filename = os.path.join(img_save_path, img_root_name)
        cv2.imwrite(filename=filename, img=img_512)

        test_iter = test_iter + 1

    print('Test Data Process Ends\n')


if __name__ == '__main__':
    tf.app.run()
