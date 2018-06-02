import numpy as np
import pandas as pd
import cv2
import os
import csv

total_joints_list = ['neckline_left','neckline_right','center_front','shoulder_left','shoulder_right',
                    'armpit_left','armpit_right','waistline_left', 'waistline_right','cuff_left_in',
                    'cuff_left_out','cuff_right_in','cuff_right_out', 'top_hem_left','top_hem_right',
                    'waistband_left','waistband_right','hemline_left', 'hemline_right','crotch',
                    'bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']
blouse_index = [0,1,3,4,2,
                5,6,13,14,9,
                10,11,12]
dress_index = [0,1,3,4,2,
               5,6,7,8,9,
               10,11,12,17,18]
outwear_index = [0, 1, 3, 4, 5,
                 6, 7, 8, 9, 10,
                 11, 12, 13, 14]
skirt_index = [15, 16, 17, 18]
trousers_index = [15, 16, 19, 20, 21,
                  22, 23]


def read_data(data_file):
    """
    To read labels in csv
    """
    data_dict = {}  # The labels of images
    label_file = pd.read_csv(data_file)
    print('READING LABELS OF TRAIN DATA')
    print('Total num:', label_file.shape[0])

    for i in range(label_file.shape[0]):
        name = str(label_file.at[i, 'image_id'])
        dress_type = str(label_file.at[i, 'image_category'])
        joints = []
        weight = []
        for joint_name in total_joints_list:
            joint_value = []
            value = str(label_file.at[i, joint_name])
            value = value.split('_')
            # print(value)
            joint_value.append(int(value[0]))
            joint_value.append(int(value[1]))
            joints.append(joint_value)
            if value[2] != '1':
                if value[2] == '0':
                    weight.append(0)
                else:
                    weight.append(-1)
            else:
                weight.append(1)

        joints = np.reshape(joints, (-1, 2))
        data_dict[name] = {'dress_type': dress_type, 'joints': joints, 'weights': weight}
    print('TRAIN LABEL READING FINISHED\n')
    return data_dict


def open_img(img_dir, name, color='RGB'):
    """ Open an image
    Args:
        name	: Name of the sample
        color	: Color Mode (RGB/BGR/GRAY)
    """
    img = cv2.imread(os.path.join(img_dir, name))
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


def process_train_img_to_center512(img, joints):
    img_shape = img.shape
    img_x = img_shape[0]
    img_y = img_shape[1]
    img512 = np.zeros((512, 512, 3), dtype=np.float32)
    img_x_padding = (512 - img_x) // 2
    img_y_padding = (512 - img_y) // 2
    img512[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = img[:, :, :]

    joints512 = np.copy(joints)
    joints512[:, 1] = joints[:, 1]+img_x_padding
    joints512[:, 0] = joints[:, 0]+img_y_padding
    return img512, joints512


def crop_data_new(height, width):
    """ Automatically returns a padding vector
        Args:
            height		: Original Height
            width		: Original Width
            crop_box    : the center point and final point of crop box
    """
    padding = [[0, 0], [0, 0], [0, 0]]
    crop_box = [width//2, width//2, width-2, width-2]
    if width == height:
        pass
    elif width > height:
        pad_size = (width - height) // 2
        padding[0][0] = padding[0][1] = pad_size
    else:
        pad_size = (height - width) // 2
        padding[1][0] = padding[1][1] = pad_size
        crop_box[2] = crop_box[3] = height-2
        crop_box[1] = crop_box[0] = height//2
    return padding, crop_box


def make_gaussian(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def generate_hm(height, width, index_list, joints, maxlength, weight):
    """ Generate a full Heap Map for every joints in an array
    Args:
        height			: Wanted Height for the Heat Map
        width			: Wanted Width for the Heat Map
        joints			: Array of Joints
        maxlength		: Length of the Bounding Box
    """
    num_joints = joints.shape[0]
    hm = np.zeros((height, width, num_joints), dtype=np.float32)

    for i in range(len(index_list)):
        if not (np.array_equal(joints[index_list[i]], [-1, -1])) and weight[index_list[i]] != -1:
            s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
            hm[:, :, index_list[i]] = make_gaussian(height, width, sigma=s,
                                                          center=(joints[index_list[i], 0], joints[index_list[i], 1]))
        else:
            hm[:, :, index_list[i]] = np.zeros((height, width))
    return hm


def relative_joints(box, padding, joints, to_size=64):
    """ Convert Absolute joint coordinates to crop box relative joint coordinates
    (Used to compute Heat Maps)
    Args:
    box			: Bounding Box
        padding	: Padding Added to the original Image
        to_size	: Heat Map wanted Size
    """
    new_j = np.copy(joints)
    max_l = max(box[2], box[3])
    new_j = new_j + [padding[1][0], padding[0][0]]
    new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
    new_j = new_j * to_size / (max_l + 0.0000001)
    return new_j.astype(np.int32)


def write2csv(total_joints_coord_set):
    with open(path_to_save, 'w') as single_csv:
        writer = csv.writer(single_csv, lineterminator='\n')
        writer.writerows(total_joints_coord_set)


PRED_FILE1 = './single_model/round_2/single_model_r2_add_occlusion_add_fpn_add_aug2_total_dataset_32w_no_add1.csv'
PRED_FILE2 = './single_model/round_2/single_model_r2_add_occlusion_add_fpn_add_aug2_total_dataset_43w_no_add1.csv'

img_dir = '/home/xiaozhi/Documents/FashionAI_data/raw_data/round2_test_a/'
path_to_save = 'ensembled_r2_3.csv'

total_joints_coord_set = []
total_joints_coord_set.append(['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                         'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                         'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                         'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                         'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'])

data_dict1 = read_data(PRED_FILE1)
data_dict2 = read_data(PRED_FILE2)
iter = 1

for name in data_dict1.keys():
    # 读关键点信息
    joints1 = data_dict1[name]['joints']
    joints2 = data_dict2[name]['joints']
    weight1 = np.asarray(data_dict1[name]['weights'])
    weight2 = np.asarray(data_dict2[name]['weights'])
    dress_type = data_dict1[name]['dress_type']

    if dress_type == 'blouse':
        index = blouse_index
    elif dress_type == 'dress':
        index = dress_index
    elif dress_type == 'outwear':
        index = outwear_index
    elif dress_type == 'skirt':
        index = skirt_index
    else:
        index = trousers_index

    img = open_img(img_dir, name)

    # process img to 512
    _, joints1 = process_train_img_to_center512(img, joints1)
    img, joints2 = process_train_img_to_center512(img, joints2)

    # crop box
    padd, cbox = crop_data_new(img.shape[0], img.shape[1])

    new_j1 = relative_joints(cbox, padd, joints1, to_size=512)
    new_j2 = relative_joints(cbox, padd, joints2, to_size=512)

    hm1 = generate_hm(512, 512, index, new_j1, 512, weight1)
    hm2 = generate_hm(512, 512, index, new_j2, 512, weight2)

    ensemble_hm = hm1 + hm2

    joint_coord_set = np.zeros((24, 3))

    for joint_num in range(24):
        ensemble_hm[:, :, joint_num] *= (255 / np.max(ensemble_hm[:, :, joint_num]))
        if np.min(ensemble_hm[:, :, joint_num]) > -50:
            joint_coord = np.unravel_index(np.argmax(ensemble_hm[:, :, joint_num]),
                                           (512, 512))
            joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1], 1]
        else:
            joint_coord_set[joint_num, :] = [256, 256, 0]

    raw_img = cv2.imread(os.path.join(img_dir, name))
    raw_img_shape = raw_img.shape
    raw_img_x = raw_img_shape[0]
    raw_img_y = raw_img_shape[1]
    x_padding = (512 - raw_img_x) // 2
    y_padding = (512 - raw_img_y) // 2

    for joint_num in range(24):
        joint_coord_set[joint_num, 0:2] -= [x_padding, y_padding]
        # joint_coord_set[joint_num, 0:2] += [1, 1]

    temp_list = [name, dress_type]
    for i in range(24):
        temp_list.append('-1_-1_-1')

    for item in range(len(index)):
        temp_list[index[item]+2] = str(int(joint_coord_set[index[item], 1])) + '_' + \
                                   str(int(joint_coord_set[index[item], 0])) + '_' + \
                                   str(int(joint_coord_set[index[item], 2]))

    total_joints_coord_set.append(temp_list)
    print('Step:', iter)
    iter += 1

    if iter % 100 == 0:
        write2csv(total_joints_coord_set)

write2csv(total_joints_coord_set)
