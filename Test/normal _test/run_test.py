import numpy as np
import cv2
import time
import math
import sys
import os
# import imageio
import tensorflow as tf
import configparser
import importlib
import pandas as pd
import scipy.misc as scm
import matplotlib.pyplot as plt
import csv

from models.nets import cpm_body
from config import FLAGS
from test_config import Valid_FLAGS


test_data_file = Valid_FLAGS.test_data_file
cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
total_joints_coord_set = []
total_joints_coord_set.append(Valid_FLAGS.total_joints_list)


def read_valid_data():
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


def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Create dirs for saving models and logs
    suffix_path = 'test_b_results_logs_and_imgs_add_occlusion_and_fpn_22w'
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

    """
        Build graph
    """
    model = cpm_model.CPM_Model(
                                input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                batch_size=1,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=True)
    model.build_loss3(optimizer='Adam')
    print('=====Model Build=====\n')

    merged_summary = tf.summary.merge_all()

    """ 
        Validation
    """
    with tf.Session() as sess:
        # Create tensorboard
        valid_writer = tf.summary.FileWriter(valid_log_save_dir, sess.graph)
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

        valid_img = np.zeros((1, 512, 512, 3), dtype=np.float32)
        valid_centermap = np.zeros((1, 512, 512, 1), dtype=np.float32)
        valid_num = len(valid_set)
        print('test sample num:', valid_num)

        valid_iter = 0
        while valid_iter < valid_num:
            # 读关键点信息
            name = valid_set[valid_iter]
            # print(name)

            # 读图片
            img = open_img(name)
            # img= scm.imresize(img, (256, 256))
            valid_img[0] = img

            # 生成centermap
            center_map = make_gaussian(height=512, width=512, sigma=150, center=None)
            center_map = np.asarray(center_map)
            valid_centermap[0] = center_map.reshape(512, 512, 1)

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
            stage_losses_np, total_loss_np, summaries,  \
            stage_heatmap_np= sess.run([model.stage_loss,
                                        model.total_loss,
                                        merged_summary,
                                        model.stage_heatmap
                                        ], feed_dict={model.input_images: batch_x_np,
                                                      model.cmap_placeholder: batch_centermap,
                                                      model.gt_hmap_placeholder: np.zeros((1,
                                                                                           FLAGS.heatmap_size,
                                                                                           FLAGS.heatmap_size,
                                                                                           Valid_FLAGS.total_num_joints),
                                                                                           dtype=np.float32),
                                                      model.train_weights_placeholder: np.zeros((1, Valid_FLAGS.total_num_joints),
                                                                                                dtype=np.float32)
                                                      })
            valid_writer.add_summary(summaries)

            # Draw intermediate results
            if FLAGS.color_channel == 'GRAY':
                demo_img = np.repeat(batch_x_np[0], 3, axis=2)
                if FLAGS.normalize_img:
                    demo_img += 0.5
                else:
                    demo_img += 128.0
                    demo_img /= 255.0

            elif FLAGS.color_channel == 'RGB':
                if FLAGS.normalize_img:
                    demo_img = batch_x_np[0] + 0.5
                else:
                    demo_img += 128.0
                    demo_img /= 255.0
            else:
                raise ValueError('Non support image type.')

            demo_stage_heatmaps = []

            # get heatmap of each stage
            for stage in range(FLAGS.cpm_stages):
                demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                    (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
                demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
                demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                demo_stage_heatmaps.append(demo_stage_heatmap)

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

            # extract the heatmap that a certain type of cloth need
            needed_stage_heatmaps = []
            for stage in range(FLAGS.cpm_stages):
                needed_stage_heatmap = np.zeros((FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints), dtype=np.float32)
                for item in range(len(index)):
                    stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                    needed_stage_heatmap[:, :, index[item]] = stage_heatmap[:, :, index[item]]
                needed_stage_heatmap = cv2.resize(needed_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
                needed_stage_heatmap = np.amax(needed_stage_heatmap, axis=2)
                needed_stage_heatmap = np.reshape(needed_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                needed_stage_heatmap = np.repeat(needed_stage_heatmap, 3, axis=2)
                needed_stage_heatmaps.append(needed_stage_heatmap)

            # draw heatmap
            if FLAGS.normalize_img:
                blend_img = 0.5 * demo_img + 0.5 * demo_stage_heatmaps[FLAGS.cpm_stages - 1]
            else:
                blend_img = 0.5 * demo_img / 255.0 + 0.5 * demo_stage_heatmaps[FLAGS.cpm_stages - 1]

            upper_img = np.concatenate(
                (demo_stage_heatmaps[FLAGS.cpm_stages - 1], blend_img, demo_img),
                axis=1)

            if FLAGS.normalize_img:
                blend_img_needed = 0.5 * demo_img + 0.5 * needed_stage_heatmaps[FLAGS.cpm_stages - 1]
            else:
                blend_img_needed = 0.5 * demo_img / 255.0 + 0.5 * needed_stage_heatmaps[FLAGS.cpm_stages - 1]

            lower_img = np.concatenate(
                (needed_stage_heatmaps[FLAGS.cpm_stages - 1], blend_img_needed, demo_img),
                axis=1)
            demo_img = np.concatenate((upper_img, lower_img), axis=0)
            # cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
            # cv2.waitKey(1000)
            '''
            # save img results
            img_root_name = name_split[2]
            result_img_save_path = os.path.join(suffix_path,
                                                'Images',
                                                dress_type)
            # print(name_str,name_spliter, name_split)
            filename = os.path.join(result_img_save_path, img_root_name)
            cv2.imwrite(filename, (demo_img * 255).astype(np.uint8))
            '''
            joint_coord_extraction(name, dress_type, stage_heatmap_np, total_joints_coord_set)
            valid_iter = valid_iter + 1
            print('valid_iter', valid_iter)

        print('=====Test Ends=====\n')
        write2csv(total_joints_coord_set)


def print_current_training_stats(global_step, cur_lr, stage_losses, total_loss, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGS.training_iters,
                                                                                 cur_lr, time_elapsed)
    losses = ' | '.join(
        ['S{} loss: {:>7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in range(FLAGS.cpm_stages)])
    losses += ' | Total loss: {}'.format(total_loss)
    print(stats)
    print(losses + '\n')


def joint_coord_extraction(img_name, dress_type, stage_heatmap_np, total_joints_coord_set):
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 3))
    last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))

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
        temp_list[index[item]] = str(int(joint_coord_set[index[item], 1])) + '_' + \
                                 str(int(joint_coord_set[index[item], 0])) + '_' + \
                                 str(int(joint_coord_set[index[item], 2]))

    total_joints_coord_set.append(temp_list)

    return total_joints_coord_set


def write2csv(total_joints_coord_set):
    with open('single_model_test_joints_b_add_occlusion_and_fpn_22w.csv', 'w') as single_csv:
        writer = csv.writer(single_csv, lineterminator='\n')
        writer.writerows(total_joints_coord_set)


if __name__ == '__main__':
    tf.app.run()
