# -*- coding: utf-8 -*-

import os
import nibabel as nib
import math
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from config import args
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor as PPE
import itertools

root_path = os.path.join(os.getcwd(), 'dataset')
process_path = os.path.join(root_path, 'fixed_data')
train_data_path = os.path.join(process_path, "train_data")
valid_data_path = os.path.join(process_path, 'val_data')
test_data_path = os.path.join(root_path, 'raw_data', 'test_data')
train_label_path = os.path.join(process_path, 'train_label')
valid_label_path = os.path.join(process_path, 'val_label')
valid_test_like_path = os.path.join(root_path, 'raw_data', 'val_data')
# 这些都是原始的nii文件所在目录
"""
root_path = "../ribfrac-images/"
process_path = root_path + "processed/"
train_data_path = process_path + "train_data/"
valid_data_path = root_path + "valid_data/"
test_data_path = root_path + "test_data/"
train_label_path = process_path + "train_label/"
valid_label_path = root_path + "ribfrac-valid-labels/"

上边是我的目录
只有train使用64*64*64的方块，validation和test都取原始数据，注意文件目录
"""

train_list = list(os.listdir(train_data_path))
train_list.sort()
test_list = list(os.listdir(test_data_path))
test_list.sort()
valid_list = list(os.listdir(valid_data_path))
valid_list.sort()
train_label_list = list(os.listdir(train_label_path))
train_label_list.sort()
valid_label_list = list(os.listdir(valid_label_path))
valid_label_list.sort()
valid_test_like_list = os.listdir(valid_test_like_path)
valid_test_like_list.sort()

for i in train_label_list:
    if i.endswith('csv'):
        train_label_list.remove(i)
for i in valid_label_list:
    if i.endswith('csv'):
        valid_label_list.remove(i)


class RibDataSet(Dataset):
    def __init__(self, args, set="train"):
        self.clip_upper = args.upper
        self.clip_lower = args.lower
        self.clip_factor = max(self.clip_upper, self.clip_lower)
        self.transform = torch.from_numpy
        self.label_transform = torch.from_numpy

        self.data_list = train_list
        self.label_list = train_label_list
        self.root_data_path = train_data_path
        self.root_label_path = train_label_path

        self.set = set

    def __getitem__(self, index):
        data_file = os.path.join(self.root_data_path, self.data_list[index])
        img = np.load(data_file, allow_pickle=True)
        # print(img.shape)

        img[img >= self.clip_upper] = self.clip_upper
        img[img <= self.clip_lower] = self.clip_lower
        img = np.array(img, dtype=float)
        img /= self.clip_factor

        if self.transform is not None:
            img = self.transform(img)
        label_file = os.path.join(self.root_label_path, self.label_list[index])
        target = np.load(label_file)
        target[target != 0] = 1
        if self.label_transform is not None:
            target = self.label_transform(target)

            return img, target

    def __len__(self):
        return len(self.data_list)


class RibTest(Dataset):
    def __init__(self, set="val"):
        self.clip_upper = args.upper
        self.clip_lower = args.lower
        self.clip_factor = max(self.clip_upper, self.clip_lower)
        self.transform = torch.from_numpy
        self.label_transform = torch.from_numpy
        if set == "val":
            # self.label_list = valid_label_list
            # self.root_label_path = valid_label_path
            self.data_list = valid_test_like_list
            self.root_data_path = valid_test_like_path
        else:
            self.data_list = test_list
            self.root_data_path = test_data_path
        self.set = set

    def __getitem__(self, index):
        data_file = os.path.join(self.root_data_path, self.data_list[index])
        img = nib.load(data_file).get_data()
        img[img >= self.clip_upper] = self.clip_upper
        img[img <= self.clip_lower] = self.clip_lower
        img = np.array(img, dtype=float)
        img /= self.clip_factor

        # if self.set == "val":
            # label_file = os.path.join(self.root_label_path, self.label_list[index])
            # target = nib.load(label_file).get_data()
            # target[target != 0] = 1
            # target = self.label_transform(target)
        # else:
            # img_id = self.data_list[index].split('c')[1].split('-')[0]
        img_id = self.data_list[index].split('.')[0]

        img_list = []
        block_size = 64
        l_iter = math.ceil(img.shape[0] / (block_size / 2)) - 1
        w_iter = math.ceil(img.shape[1] / (block_size / 2)) - 1
        d_iter = math.ceil(img.shape[2] / (block_size / 2)) - 1
        exe_func = partial(self.multipro, block_size=block_size, img=img)
        # with PPE() as executor:
        #     img_list = executor.map(exe_func, itertools.product(range(l_iter), range(w_iter), range(d_iter)))
        for tp in itertools.product(range(l_iter), range(w_iter), range(d_iter)):
            img_list.append(exe_func(tp))
        img_list = list(img_list)
        # if self.set == "val":
        #     return img_list, target
        # else:
        return img_list, img_id

    def __len__(self):
        return len(self.data_list)

    def multipro(self, idx_tuple, block_size, img):
        # print(idx_tuple)
        l, w, d = idx_tuple
        block_dict = dict()
        l_high = int(min((l + 2) * block_size / 2, img.shape[0]))
        l_low = int(l_high - block_size)
        w_high = int(min((w + 2) * block_size / 2, img.shape[1]))
        w_low = int(w_high - block_size)
        d_high = int(min((d + 2) * block_size / 2, img.shape[2]))
        d_low = int(d_high - block_size)

        bbox = (l_low, l_high, w_low, w_high, d_low, d_high)
        block_dict["bbox"] = bbox
        array = img[l_low:l_high, w_low:w_high, d_low:d_high].astype(np.float)[np.newaxis, :]
        array = self.transform(array)
        block_dict["data"] = array
        # img_list.append(block_dict)
        return block_dict


if __name__ == '__main__':
    test_dataset = RibTest(set='val')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    for idx, (img, img_id_or_label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        print("look img and id")
        print('------')
