import numpy as np
import cv2
import time
import math
import sys
import os
import imageio
import tensorflow as tf
import configparser
import importlib
import pandas as pd
import scipy.misc as scm
import matplotlib.pyplot as plt
import csv
from skimage import transform
import scipy.misc as scm
from PIL import Image, ImageEnhance, ImageFilter
import random

from models.nets import cpm_body
from config import FLAGS
from test_config import Valid_FLAGS


test_data_file = Valid_FLAGS.test_data_file
cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
total_joints_coord_set = []
total_joints_coord_set.append(['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                         'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                         'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                         'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                         'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'])
aug_num = 10


def read_valid_data():
    """
    To read labels in csv
    """
    test_table = []  # The names of images being trained
    label_file = pd.read_csv(test_data_file)
    print('READING VALID DATA')
    for i in range(label_file.shape[0]):
        name = str(label_file.at[i, 'image_id'])
        test_table.append(name)
    print('VALID DATA READING FINISHED\n')
    return test_table


def read_IOU_data():
    IOU_dict = {}       # The labels of images
    label_file = pd.read_csv(Valid_FLAGS.IOU_file)
    print('READING IOU DATA')
    print('Total num:', label_file.shape[0])

    for i in range(label_file.shape[0]):
        name = str(label_file.at[i, 'image_id'])
        dress_type = str(label_file.at[i, 'image_category'])
        yita = label_file.at(i, 'yita')
        IOU_dict[name] = {'dress_type': dress_type, 'yita': yita}

    print('IOU DATA READING FINISHED\n')
    return IOU_dict


def read_exist_data():
    exist_dict = {}
    label_file = pd.read_csv(Valid_FLAGS.exist_results_file)
    print('READING EXIST RESULTS DATA')
    print('Total num:', label_file.shape[0])

    for i in range(label_file.shape[0]):
        name = str(label_file.at[i, 'image_id'])
        dress_type = str(label_file.at[i, 'image_category'])
        exist_dict[name] = {'dress_type': dress_type}

        for joint_name in Valid_FLAGS.total_joints_list:
            exist_dict[name][joint_name] = str(label_file.at[i, joint_name])

    print('EXIST DATA READING FINISHED')
    return exist_dict


def open_img(name, color='RGB'):
    """ Open an image
    Args:
        name	: Name of the sample
        color	: Color Mode (RGB/BGR/GRAY)
    """
    img = cv2.imread(os.path.join(Valid_FLAGS.processed_testimg_directory, name))
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


def make_gaussian(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  # 把一行数变成一列数
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def rotate(image, angle, center=None, scale=1.0):
    """
    a implemetation of rotation using cv2
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def rotate_augment(img, max_rotation=45):
    """ # TODO : IMPLEMENT DATA AUGMENTATION
    """
    r_angle = 0

    if random.choice([0, 1]):
        r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        # img = transform.rotate(img, r_angle, preserve_range=True)
        img = rotate(img, r_angle)

    return img, r_angle


# input image size=512,out image size=512,input joints size = 512
def size_augment(img, min_compress_ratio=0.7, max_compress_ratio=1.35):
    compress_ratio = np.random.uniform(min_compress_ratio, max_compress_ratio)
    size = compress_ratio * img.shape[0]
    size = int(round(size))

    # img resize
    resized_img = cv2.resize(img, (size, size))
    resized_img_shape = resized_img.shape

    if compress_ratio <= 1.0:
        # resized img padding to 512
        img_x = resized_img_shape[0]
        img_y = resized_img_shape[1]
        img2 = np.zeros((512, 512, 3), dtype=np.float32)
        img_x_padding = (512 - img_x) // 2
        img_y_padding = (512 - img_y) // 2
        img2[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = resized_img[:, :, :]
        aug_img = img2

    else:
        img_x = resized_img_shape[0]
        img_y = resized_img_shape[1]
        img2 = np.zeros((512, 512, 3), dtype=np.float32)
        img_x_padding = (img_x - 512) // 2
        img_y_padding = (img_y - 512) // 2
        img2[:, :, :] = resized_img[img_x_padding:img_x_padding + 512, img_y_padding:img_y_padding + 512, :]
        aug_img = img2

    return aug_img, compress_ratio


def color_augment(img):
    image = Image.fromarray(img)
    # image.show()
    # 亮度增强
    if random.choice([0, 1]):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.choice([0.6,0.8,1.2,1.4])
        image = enh_bri.enhance(brightness)
        # image.show()

    # 色度增强
    if random.choice([0, 1]):
        enh_col = ImageEnhance.Color(image)
        color = random.choice([0.6,0.8,1.2,1.4])
        image = enh_col.enhance(color)
        # image.show()

    # 对比度增强
    if random.choice([0, 1]):
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.choice([0.6,0.8,1.2,1.4])
        image = enh_con.enhance(contrast)
        # image.show()

    # 锐度增强
    if random.choice([0, 1]):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.choice([0.6,0.8,1.2,1.4])
        image = enh_sha.enhance(sharpness)
        # image.show()

    # mo hu
    if random.choice([0, 1]):
        image = image.filter(ImageFilter.BLUR)

    img = np.asarray(image)
    return img


def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Create dirs for saving models and logs
    suffix_path = 'r2_with_IOU'

    valid_log_save_dir = os.path.join(suffix_path,
                                      'models',
                                      'logs',
                                      'test')
    result_img_save_dir_blouse = os.path.join(suffix_path,
                                              'Images',
                                              'blouse')
    result_img_save_dir_dress = os.path.join(suffix_path,
                                             'Images',
                                             'dress')
    result_img_save_dir_outwear = os.path.join(suffix_path,
                                               'Images',
                                               'outwear')
    result_img_save_dir_skirt = os.path.join(suffix_path,
                                             'Images',
                                             'skirt')
    result_img_save_dir_trousers = os.path.join(suffix_path,
                                                'Images',
                                                'trousers')
    os.system('mkdir -p {}'.format(valid_log_save_dir))
    os.system('mkdir -p {}'.format(result_img_save_dir_blouse))
    os.system('mkdir -p {}'.format(result_img_save_dir_dress))
    os.system('mkdir -p {}'.format(result_img_save_dir_outwear))
    os.system('mkdir -p {}'.format(result_img_save_dir_skirt))
    os.system('mkdir -p {}'.format(result_img_save_dir_trousers))

    """
        read data
    """
    valid_set = read_valid_data()
    IOU_dict = read_IOU_data()
    existed_dict = read_exist_data()

    """
        Build graph
    """
    model = cpm_model.CPM_Model(total_num=FLAGS.total_num,
                                input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                batch_size=aug_num,
                                stages=FLAGS.cpm_stages,
                                num_joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=True)
    print('=====Model Build=====\n')

    """ 
        Validation
    """
    with tf.Session() as sess:
        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        """
            Create session and restore weights
        """
        # Restore pretrained weights
        if Valid_FLAGS.model_load_dir != '':
            if Valid_FLAGS.model_load_dir.endswith('.pkl'):
                model.load_weights_from_file(Valid_FLAGS.model_load_dir, sess, finetune=True)
                # Check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))
            else:
                checkpoint = tf.train.get_checkpoint_state(Valid_FLAGS.model_load_dir)
                # 获取最新保存的模型检查点文件
                ckpt = checkpoint.model_checkpoint_path
                saver.restore(sess, ckpt)
                # check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

        valid_img = np.zeros((aug_num, 512, 512, 3), dtype=np.float32)
        valid_centermap = np.zeros((aug_num, 512, 512, 1), dtype=np.float32)
        valid_num = len(valid_set)
        print('test sample num:', valid_num)

        valid_iter = 0
        while valid_iter < valid_num:
            # 读关键点信息
            name = valid_set[valid_iter]
            # print(name)
            if 0 < IOU_dict[name]['yita'] <= 0.1:
                # 读图片
                img = open_img(name)

                # get dress type to determine the extract index
                name_split = name.split('/')
                dress_type = name_split[1]
                if dress_type == 'blouse':
                    index = Valid_FLAGS.blouse_index
                elif dress_type == 'dress':
                    index = Valid_FLAGS.dress_index
                elif dress_type == 'outwear':
                    index = Valid_FLAGS.outwear_index
                elif dress_type == 'skirt':
                    index = Valid_FLAGS.skirt_index
                else:
                    index = Valid_FLAGS.trousers_index

                valid_img[0] = img
                center_map = make_gaussian(height=512, width=512, sigma=150, center=None)
                center_map = np.asarray(center_map)
                center_map = center_map.reshape(512, 512, 1)
                valid_centermap[0] = center_map

                compress_ratio = np.ones((10), dtype=np.float32)
                r_angle = np.zeros((10), dtype=np.float32)

                aug_needed_heatmaps = []
                cnt = 1
                while cnt < aug_num:
                    # test img aug
                    # img2 = color_augment(img)
                    img2, compress_ratio[cnt] = size_augment(img, min_compress_ratio=0.7, max_compress_ratio=2)
                    img2, r_angle[cnt] = rotate_augment(img2)

                    valid_img[cnt] = img2
                    valid_centermap[cnt] = center_map

                    cnt += 1

                """ 
                    修改重点：DataGenerator应用的地方
                """
                # Read one batch data
                batch_x_np, batch_centermap = valid_img, valid_centermap
                # print(batch_x_np.shape,batch_gt_heatmap_np.shape, batch_centermap.shape)

                if Valid_FLAGS.normalize_img:
                    # Normalize images
                    batch_x_np = batch_x_np / 255.0 - 0.5
                else:
                    batch_x_np -= 128.0

                '''
                # Generate heatmaps from joints
                batch_gt_heatmap_np = cpm_utils.make_heatmaps_from_joints(FLAGS.input_size,
                                                                          FLAGS.heatmap_size,
                                                                          FLAGS.joint_gaussian_variance,
                                                                          batch_joints_np)
                '''

                stage_heatmap_np = sess.run([model.stage_heatmap[FLAGS.cpm_stages - 1]
                                             ], feed_dict={model.input_images: batch_x_np,
                                                           model.cmap_placeholder: batch_centermap,
                                                          })

                for batch in range(aug_num):
                    demo_stage_heatmap = stage_heatmap_np[-1][batch, :, :, 0:FLAGS.num_of_joints].reshape(
                        FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints)
                    demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))

                    # reverse rotate aug
                    demo_stage_heatmap= transform.rotate(demo_stage_heatmap, -r_angle[batch], preserve_range=True)

                    # reverse size aug
                    reverse_ratio = 1.0 / compress_ratio[batch]
                    size = reverse_ratio * demo_stage_heatmap.shape[0]
                    size = int(round(size))

                    if reverse_ratio <= 1.0:
                        # img resize
                        resized_demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (size, size))

                        # resized img padding to 512
                        resized_demo_stage_heatmap_shape = resized_demo_stage_heatmap.shape
                        hm_x = resized_demo_stage_heatmap_shape[0]
                        hm_y = resized_demo_stage_heatmap_shape[1]
                        hm2 = np.zeros((512, 512, Valid_FLAGS.total_num_joints), dtype=np.float32)
                        hm_x_padding = (512 - hm_x) // 2
                        hm_y_padding = (512 - hm_y) // 2
                        hm2[hm_x_padding:hm_x_padding + hm_x,
                                       hm_y_padding:hm_y_padding + hm_y, :] = resized_demo_stage_heatmap[:,:,:]
                    else:
                        # img resize
                        resized_demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (size, size))

                        # resized img padding to 512
                        resized_demo_stage_heatmap_shape = resized_demo_stage_heatmap.shape
                        hm_x = resized_demo_stage_heatmap_shape[0]
                        hm_y = resized_demo_stage_heatmap_shape[1]
                        hm2 = np.zeros((512, 512, Valid_FLAGS.total_num_joints), dtype=np.float32)
                        hm_x_padding = (hm_x - 512) // 2
                        hm_y_padding = (hm_y - 512) // 2
                        hm2[:, :, :] = resized_demo_stage_heatmap[hm_x_padding:hm_x_padding + 512,
                                       hm_y_padding:hm_y_padding + 512, :]

                    aug_needed_heatmaps.append(hm2)

                    # Draw intermediate results
                    if Valid_FLAGS.if_show:
                        if FLAGS.color_channel == 'GRAY':
                            demo_img = np.repeat(batch_x_np[batch], 3, axis=2)
                            if FLAGS.normalize_img:
                                demo_img += 0.5
                            else:
                                demo_img += 128.0
                                demo_img /= 255.0

                        elif FLAGS.color_channel == 'RGB':
                            if FLAGS.normalize_img:
                                demo_img = batch_x_np[batch] + 0.5
                            else:
                                demo_img += 128.0
                                demo_img /= 255.0
                        else:
                            raise ValueError('Non support image type.')

                        # get heatmap of each stage
                        demo_stage_heatmaps = []
                        for stage in range(FLAGS.cpm_stages):
                            demo_stage_heatmap = np.amax(hm2, axis=2)
                            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                            demo_stage_heatmaps.append(demo_stage_heatmap)

                        # extract the heatmap that a certain type of cloth need
                        needed_stage_heatmaps = []
                        needed_stage_heatmap = np.zeros((FLAGS.input_size, FLAGS.input_size, FLAGS.num_of_joints), dtype=np.float32)
                        for item in range(len(index)):
                            needed_stage_heatmap[:, :, index[item]] = hm2[:, :, index[item]]
                        needed_stage_heatmap = np.amax(needed_stage_heatmap, axis=2)
                        needed_stage_heatmap = np.reshape(needed_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                        needed_stage_heatmap = np.repeat(needed_stage_heatmap, 3, axis=2)
                        needed_stage_heatmaps.append(needed_stage_heatmap)

                        # draw heatmap
                        if FLAGS.normalize_img:
                            blend_img = 0.5 * demo_img + 0.5 * demo_stage_heatmaps[0]
                        else:
                            blend_img = 0.5 * demo_img / 255.0 + 0.5 * demo_stage_heatmaps[0]

                        upper_img = np.concatenate(
                            (demo_stage_heatmaps[0], blend_img, demo_img),
                            axis=1)

                        if FLAGS.normalize_img:
                            blend_img_needed = 0.5 * demo_img + 0.5 * needed_stage_heatmaps[0]
                        else:
                            blend_img_needed = 0.5 * demo_img / 255.0 + 0.5 * needed_stage_heatmaps[0]

                        lower_img = np.concatenate(
                            (needed_stage_heatmaps[0], blend_img_needed, demo_img),
                            axis=1)
                        demo_img = np.concatenate((upper_img, lower_img), axis=0)
                        cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
                        cv2.waitKey(1000)

                joint_coord_extraction(name, dress_type, aug_needed_heatmaps, existed_dict)
            valid_iter = valid_iter + 1
            print('valid_iter', valid_iter)

        total_joints_coord_set2 = transform_dict2list(valid_set, existed_dict)
        write2csv(total_joints_coord_set2)
        print('=====Test Ends=====\n')


def print_current_training_stats(global_step, cur_lr, stage_losses, total_loss, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGS.training_iters,
                                                                                 cur_lr, time_elapsed)
    losses = ' | '.join(
        ['S{} loss: {:>7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in range(FLAGS.cpm_stages)])
    losses += ' | Total loss: {}'.format(total_loss)
    print(stats)
    print(losses + '\n')


def joint_coord_extraction(img_name, dress_type, aug_needed_heatmaps, existed_dict):
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 3))
    '''
    last_heatmap = np.zeros((FLAGS.input_size, FLAGS.input_size, FLAGS.num_of_joints), dtype=np.float32)
    for i in range(aug_num):
        for joint_num in range(FLAGS.num_of_joints):
            last_heatmap[:, :, joint_num] += aug_needed_heatmaps[i][:, :, joint_num]
    '''
    last_heatmap = sum(aug_needed_heatmaps)
    # hm = np.amax(last_heatmap, axis=2)
    # plt.imshow(hm)
    # plt.show()

    for joint_num in range(FLAGS.num_of_joints):
        last_heatmap[:, :, joint_num] *= (255 / np.max(last_heatmap[:, :, joint_num]))
        if np.min(last_heatmap[:, :, joint_num]) > -50:
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (FLAGS.input_size, FLAGS.input_size))
            joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1], 1]
            # print(joint_coord_set[joint_num, :])
            # plt.imshow(last_heatmap[:, :, joint_num])
            # plt.show()
        else:
            joint_coord_set[joint_num, :] = [256, 256, 0]
            # print(joint_coord_set[joint_num, :])

    # joint_coord_set[:, 0:2] *= 2.0
    raw_img = cv2.imread(os.path.join(Valid_FLAGS.test_img_directory, img_name))
    raw_img_shape = raw_img.shape
    raw_img_x = raw_img_shape[0]
    raw_img_y = raw_img_shape[1]
    x_padding = (512 - raw_img_x) // 2
    y_padding = (512 - raw_img_y) // 2
    for joint_num in range(FLAGS.num_of_joints):
        joint_coord_set[joint_num, 0:2] -= [x_padding, y_padding]

    temp_list = [img_name, dress_type]
    for i in range(24):
        temp_list.append('-1_-1_-1')

    if dress_type == 'blouse':
        index = Valid_FLAGS.blouse_index
    elif dress_type == 'dress':
        index = Valid_FLAGS.dress_index
    elif dress_type == 'outwear':
        index = Valid_FLAGS.outwear_index
    elif dress_type == 'skirt':
        index = Valid_FLAGS.skirt_index
    else:
        index = Valid_FLAGS.trousers_index

    for item in range(len(index)):
        temp_list[index[item]+2] = str(int(joint_coord_set[index[item], 1])) + '_' + \
                                   str(int(joint_coord_set[index[item], 0])) + '_' + \
                                   str(int(joint_coord_set[index[item], 2]))

    for i in range(len(Valid_FLAGS.total_joints_list)):
        existed_dict[img_name][Valid_FLAGS.total_joints_list[i]] = temp_list[i+2]

    return existed_dict


def transform_dict2list(valid_set, existed_dict):
    total_joints_coord_set2 = []
    total_joints_coord_set2.append(
        ['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
         'shoulder_right',
         'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
         'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
         'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
         'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'])

    item = 0
    valid_num = len(valid_set)
    while item < valid_num:
        name = valid_set[item]

        name_split = name.split('/')
        dress_type = name_split[1]

        temp_list = [name, dress_type]

        for i in range(24):
            temp_list.append('-1_-1_-1')

        for i in range(24):
            temp_list[i+2] = existed_dict[name][Valid_FLAGS.total_joints_list[i]]

        total_joints_coord_set2.append(temp_list)

        item += 1
    return  total_joints_coord_set2


def write2csv(total_joints_coord_set2):
    with open('r2_with_IOU.csv', 'w') as single_csv:
        writer = csv.writer(single_csv, lineterminator='\n')
        writer.writerows(total_joints_coord_set2)


if __name__ == '__main__':
    tf.app.run()
