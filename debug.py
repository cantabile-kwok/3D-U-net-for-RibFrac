from Unet_test import Unet
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.loss import DiceLoss
import os
import nibabel as nib
from tqdm import tqdm


# pred = nib.load(r'RibFrac447-img.nii.gz').get_data()


def normal_dice(output, target):
    inter = torch.sum(output[:, 0, :, :, :] * target[:, 0, :, :, :])
    union = torch.sum(output[:, 0, :, :, :]) + torch.sum(target[:, 0, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


device = torch.device('cuda')

model = Unet(in_channels=1, out_channels=1).to(device)
model = torch.nn.DataParallel(model, device_ids=[0,1])  # multi-GPU

# model = torch.load('best_model.pth', map_location=torch.device('cpu'))['net']
sd = torch.load('experiments/6-6-lr1e-2/latest_model_10.pth', map_location=torch.device('cuda'))['net']
model.load_state_dict(sd)

# sample_path = r'D:\学期文件\大二下\机器学习\3DUnet\3DUNet-Pytorch-master\dataset\fixed_data\train_data\RibFrac422-1-image.npy'
# x = np.load(sample_path)
# x[x >= 200] = 200
# x[x <= -200] = -200
# x = np.array(x, dtype=float)
# x /= 200
# target = np.load(sample_path.replace('image', 'label').replace('train_data', 'train_label'))
# target[target != 0] = 1
# target = torch.tensor(target).float()
# target = target[np.newaxis, :]
#
# x = x[np.newaxis, :]
# x = torch.tensor(x).float()
model.eval()
with torch.no_grad():
    dicelist = []
    for npy in tqdm(os.listdir(r'dataset/fixed_data/val_data')):
        x = np.load(os.path.join(r'dataset/fixed_data/val_data', npy))
        x[x >= 200] = 200
        x[x <= -200] = -200
        x = np.array(x, dtype=float)
        x /= 200
        x = x[np.newaxis, :]
        x = torch.tensor(x).float().to(device)
        target = np.load(os.path.join(r'dataset/fixed_data/val_data', npy).replace('image', 'label').replace('val_data',
                                                                                                             'val_label'))
        target[target != 0] = 1
        target = torch.tensor(target).float().to(device)
        target = target[np.newaxis, :]
        output = model(x)
        # output = x
        output = torch.sigmoid(output)
        # array = output.reshape(-1)
        # print(output.shape)
        # sample = random.sample(array.numpy().tolist(), 50000)
        # plt.figure()
        # plt.hist(sample, bins=100)
        # plt.show()
        # print(DiceLoss()(output, target))

        # output[output < 0.5] = 0
        # output[output >= 0.5] = 1
        dice = normal_dice(output, target)
        # print(dice)
        dicelist.append(dice.item())
    print(np.mean(dicelist))
