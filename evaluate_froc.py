# -*- coding: utf-8 -*-

# from mydataset import RibDataSet
import sys
from newdataset import RibDataSet, RibTest
import nibabel as nib
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as PPE
from functools import partial
from Unet_test import Unet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from evaluate.ribfrac.evaluation import evaluate
from FracNet.predict import _make_submission_files as make_submission
import pandas as pd
from newtrain import val_like_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-parameters management')

    parser.add_argument('--cpu', default=False)
    parser.add_argument('--block_size', type=int, default=64, help='size of sliding window')
    parser.add_argument('--save', default='Unet-6-5-new', help='save path of trained model')
    parser.add_argument('--dataset_path', default='dataset/fixed_data', help='fixed dataset root path')
    parser.add_argument('--gpu_id', type=list, default=[0, 1])
    # parser.add_argument('--exp_path', type=str, default='experiments/Unet-6-2-lr5e-3')

    args = parser.parse_args()
    save_path = os.path.join(os.getcwd(), 'experiments', args.save)
    res_path = os.path.join(save_path, 'val_results')
    device = torch.device('cpu' if args.cpu else 'cuda')

    model = Unet(in_channels=1, out_channels=1).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    val_like_test_dataset = RibTest(set='val', block_size=args.block_size)

    model_state = os.path.join(save_path, 'best_model_70.pth')
    model.load_state_dict(torch.load(model_state)['net'])
    val_test_log = val_like_test(model, val_like_test_dataset,
                                 os.path.join(os.getcwd(), 'dataset', 'raw_data', 'val_label'), res_path,
                                 device, args.block_size)
    key_recall = np.array(val_test_log['detection']['key_recall']).reshape(-1)
    # DEFAULT_KEY_FP = np.array([0.5, 1, 2, 4, 8])
    froc = key_recall.mean()
    print(f'=======Val FROC={froc}========')
    import json
    with open(os.path.join(res_path, 'val_log.json'), 'w') as f:
        f.write(json.dumps(val_test_log))
