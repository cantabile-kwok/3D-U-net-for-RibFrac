# -*- coding: utf-8 -*-

import os
import nibabel as nib
import math
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
# from config import args
from tqdm import tqdm
from dataset.transforms import RandomFlip_LR, RandomFlip_UD

root_path = os.path.join(os.getcwd(), 'dataset')
process_path = os.path.join(root_path, 'fixed_data')
train_data_path = os.path.join(process_path, "train_data")
valid_data_path = os.path.join(process_path, 'val_data')
test_data_path = os.path.join(root_path, 'raw_data', 'test_data')
if not os.path.exists(test_data_path):
    print('warning: making ', test_data_path)
    os.mkdir(test_data_path)
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
        self.flip1 = RandomFlip_UD(prob=0.3)
        self.flip2 = RandomFlip_LR(prob=0.3)

        if set=='train':
            self.data_list = train_list
            self.label_list = train_label_list
            self.root_data_path = train_data_path
            self.root_label_path = train_label_path
        elif set == "val":
            self.data_list = valid_list
            self.label_list = valid_label_list
            self.root_data_path = valid_data_path
            self.root_label_path = valid_label_path
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
        target[target != 0] = 1  # 二值化
        if self.label_transform is not None:
            target = self.label_transform(target)

        # img, target = self.flip1(self.flip2(img, target))
        img, target = self.flip1(img, target)
        img, target = self.flip2(img, target)

        return img, target

    def __len__(self):
        return len(self.data_list)


class GenerateHelper:
    def __init__(self, l_iter, w_iter, d_iter, block_size, transform, img):
        self.l_iter = l_iter
        self.w_iter = w_iter
        self.d_iter = d_iter
        self.block_size = block_size
        self.transform = transform
        self.img = img

    @property
    def gen(self):
        for l in range(self.l_iter):
            for w in range(self.w_iter):
                for d in range(self.d_iter):
                    block_dict = dict()
                    l_high = int(min((l + 2) * self.block_size / 2, self.img.shape[0]))
                    l_low = int(l_high - self.block_size)
                    w_high = int(min((w + 2) * self.block_size / 2, self.img.shape[1]))
                    w_low = int(w_high - self.block_size)
                    d_high = int(min((d + 2) * self.block_size / 2, self.img.shape[2]))
                    d_low = int(d_high - self.block_size)

                    bbox = (l_low, l_high, w_low, w_high, d_low, d_high)
                    block_dict["bbox"] = bbox
                    array = self.img[l_low:l_high, w_low:w_high, d_low:d_high].astype(np.float)[np.newaxis, :]
                    array = self.transform(array)
                    block_dict["data"] = array
                    # img_list.append(block_dict)
                    yield block_dict


class RibTest(Dataset):
    def __init__(self,args, set="val"):
        self.clip_upper = args.upper
        self.clip_lower = args.lower
        self.clip_factor = max(self.clip_upper, self.clip_lower)
        self.transform = torch.from_numpy
        self.label_transform = torch.from_numpy
        self.block_size = args.block_size

        if set == "val":
            self.data_list = valid_test_like_list
            self.root_data_path = valid_test_like_path
        elif set == 'test':
            self.data_list = test_list
            self.root_data_path = test_data_path
        else:
            print('only val and test set are supported for RibTest')
            exit(1)
        self.set = set

    def __getitem__(self, index):
        data_file = os.path.join(self.root_data_path, self.data_list[index])
        img = nib.load(data_file).get_data()
        img[img >= self.clip_upper] = self.clip_upper
        img[img <= self.clip_lower] = self.clip_lower
        img = np.array(img, dtype=float)
        img /= self.clip_factor
        block_size = self.block_size
        img_id = self.data_list[index].split('-')[0]

        # img_list = []

        l_iter = math.ceil(img.shape[0] / (block_size / 2)) - 1
        w_iter = math.ceil(img.shape[1] / (block_size / 2)) - 1
        d_iter = math.ceil(img.shape[2] / (block_size / 2)) - 1

        self.generate_helper = GenerateHelper(l_iter, w_iter, d_iter, self.block_size, self.transform, img)
        return self.generate(l_iter, w_iter, d_iter, img), img_id, img.shape

    # @property
    def generate(self, l_iter, w_iter, d_iter, img):
        for l in range(l_iter):
            for w in range(w_iter):
                for d in range(d_iter):
                    block_dict = dict()
                    l_high = int(min((l + 2) * self.block_size / 2, img.shape[0]))
                    l_low = int(l_high - self.block_size)
                    w_high = int(min((w + 2) * self.block_size / 2, img.shape[1]))
                    w_low = int(w_high - self.block_size)
                    d_high = int(min((d + 2) * self.block_size / 2, img.shape[2]))
                    d_low = int(d_high - self.block_size)

                    bbox = (l_low, l_high, w_low, w_high, d_low, d_high)
                    block_dict["bbox"] = bbox
                    array = img[l_low:l_high, w_low:w_high, d_low:d_high].astype(np.float)[np.newaxis, :]
                    array = self.transform(array)
                    block_dict["data"] = array
                    # img_list.append(block_dict)
                    yield block_dict

    def __len__(self):
        return len(self.data_list)


# class tmp:
#     def __init__(self):
#         self.upper = 200
#         self.lower = 200


if __name__ == '__main__':
    test_dataset = RibTest(set='val')
    # test_dataset = RibDataSet(tmp(), set='train')
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    for (img, img_id, img_shape) in tqdm(test_dataset):
        print(img)
        print(img_shape)
        print('------')
