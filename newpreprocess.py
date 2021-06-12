import os
import nibabel as nib
from PIL import Image
from torchvision import transforms
import numpy as np
import tqdm
from skimage import measure
import random

import argparse


def process(args):
    box_size = args.box_size
    if args.set == "train":
        target_list = args.train_label_list
        data_list = args.train_list
        root_data_path = args.train_data_path
        root_label_path = args.train_label_path
    elif args.set == "val":
        target_list = args.valid_label_list
        data_list = args.valid_list
        root_data_path = args.valid_data_path
        root_label_path = args.valid_label_path
    else:
        data_list, target_list, root_data_path, root_label_path = None, None, None, None
        print("preprocess only for train and valid data")
        exit(1)
    # data_save_path = process_path + set + "_data"
    # label_save_path = process_path + set + "_label"
    args.label_save_path = os.path.join(args.process_path, args.set + '_label')
    args.data_save_path = os.path.join(args.process_path, args.set + '_data')

    if not os.path.exists(args.data_save_path):
        os.makedirs(args.data_save_path)
    if not os.path.exists(args.label_save_path):
        os.makedirs(args.label_save_path)

    for i in tqdm.tqdm(range(len(data_list))):
        data_file = data_list[i]
        label_file = target_list[i]
        data = nib.load(os.path.join(root_data_path, data_file)).get_data()
        label = nib.load(os.path.join(root_label_path, label_file)).get_data()
        property = measure.regionprops(label)

        for idx in range(len(property)):
            center = property[idx].centroid
            if int(center[0]) + box_size / 2 >= data.shape[0]:
                dim_1_low = data.shape[0] - box_size
                dim_1_high = data.shape[0]
            elif int(center[0]) - box_size / 2 <= 0:
                dim_1_low = 0
                dim_1_high = box_size
            else:
                dim_1_low = int(center[0]) - int(box_size / 2)
                dim_1_high = int(center[0]) + int(box_size / 2)

            if int(center[1]) + box_size / 2 >= data.shape[1]:
                dim_2_low = data.shape[1] - box_size
                dim_2_high = data.shape[1]
            elif int(center[1]) - box_size / 2 <= 0:
                dim_2_idx = np.arange(0, box_size)
                dim_2_low = 0
                dim_2_high = box_size
            else:
                dim_2_low = int(center[1]) - int(box_size / 2)
                dim_2_high = int(center[1]) + int(box_size / 2)

            if int(center[2]) + box_size / 2 >= data.shape[2]:
                dim_3_low = data.shape[2] - box_size
                dim_3_high = data.shape[2]
            elif int(center[2]) - box_size / 2 <= 0:
                dim_3_low = 0
                dim_3_high = box_size
            else:
                dim_3_low = int(center[2]) - int(box_size / 2)
                dim_3_high = int(center[2]) + int(box_size / 2)

            box = data[dim_1_low:dim_1_high, dim_2_low:dim_2_high, dim_3_low:dim_3_high].astype(np.int16)
            target_box = label[dim_1_low:dim_1_high, dim_2_low:dim_2_high, dim_3_low:dim_3_high].astype(np.int16)
            precessed_data_file = os.path.join(args.data_save_path,
                                               data_file.split('-')[0] + '-' + str(idx + 1) + "-image")
            precessed_label_file = os.path.join(args.label_save_path,
                                                label_file.split('-')[0] + '-' + str(idx + 1) + "-label")

            np.save(precessed_data_file, box.reshape(1, *(box.shape)))
            np.save(precessed_label_file, target_box.reshape(1, *(box.shape)))

        if len(property)==0:
            L = 20
        else:
            L = len(property)
        for idx in range(L):
            dim_1_low = random.randint(0, data.shape[0] - box_size)
            dim_1_high = dim_1_low + box_size
            dim_2_low = random.randint(0, data.shape[1] - box_size)
            dim_2_high = dim_2_low + box_size
            dim_3_low = random.randint(0, data.shape[2] - box_size)
            dim_3_high = dim_3_low + box_size
            box = data[dim_1_low:dim_1_high, dim_2_low:dim_2_high, dim_3_low:dim_3_high].astype(np.int16)
            target_box = label[dim_1_low:dim_1_high, dim_2_low:dim_2_high, dim_3_low:dim_3_high].astype(np.int16)
            precessed_data_file = os.path.join(args.data_save_path,
                                               data_file.split('-')[0] + '-' + str(idx + 1 + len(property)) + "-image")
            precessed_label_file = os.path.join(args.label_save_path,
                                                label_file.split('-')[0] + '-' + str(
                                                    idx + 1 + len(property)) + "-label")
            np.save(precessed_data_file, box.reshape(1, *(box.shape)))
            np.save(precessed_label_file, target_box.reshape(1, *(box.shape)))


if __name__ == '__main__':
    # 这个脚本把train和val的nii.gz读出来，按label分块，存到dataset/fixed_data里面去
    # 存储shape是(1，64，64，64)
    parser = argparse.ArgumentParser(description='data preprocessing argument')
    parser.add_argument("--set", default='val', help='train or val')
    parser.add_argument('--box_size', default=64, type=int, help='size of the bounding box')
    args = parser.parse_args()

    args.root_path = os.path.join(os.getcwd(), 'dataset')
    args.process_path = os.path.join(args.root_path, 'fixed_data')
    args.train_data_path = os.path.join(args.root_path, 'raw_data', "train_data")
    args.valid_data_path = os.path.join(args.root_path, 'raw_data', 'val_data')
    # args.test_data_path = os.path.join(args.root_path, 'raw_data', 'test_data')
    args.train_label_path = os.path.join(args.root_path, 'raw_data', 'train_label')
    args.valid_label_path = os.path.join(args.root_path, 'raw_data', 'val_label')

    args.train_list = list(os.listdir(args.train_data_path))
    args.train_list.sort()
    # args.test_list = list(os.listdir(args.test_data_path))
    # args.test_list.sort()
    args.valid_list = list(os.listdir(args.valid_data_path))
    args.valid_list.sort()
    args.train_label_list = list(os.listdir(args.train_label_path))
    args.train_label_list.sort()
    args.valid_label_list = list(os.listdir(args.valid_label_path))
    args.valid_label_list.sort()

    process(args)
