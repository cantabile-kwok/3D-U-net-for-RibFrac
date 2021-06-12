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
from concurrent.futures import ThreadPoolExecutor as TPE
from functools import partial
from Unet_test import Unet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
# from evaluate.ribfrac.evaluation import evaluate
from FracNet.predict import _make_submission_files as make_submission
import pandas as pd


def execute_image(datalist_imgid_shape, res_img_path, model, device, block_size):
    # idx = idx_data_target[0]
    model.eval()
    with torch.no_grad():
        data = datalist_imgid_shape[0]
        img_id = datalist_imgid_shape[1]
        shape = datalist_imgid_shape[-1]
        prediction = np.zeros(shape, dtype=float)
        overlap_num = np.zeros(shape, dtype=float)
        for block_dict in tqdm(data):
            l_low = block_dict["bbox"][0]
            l_high = block_dict["bbox"][1]
            w_low = block_dict["bbox"][2]
            w_high = block_dict["bbox"][3]
            d_low = block_dict["bbox"][4]
            d_high = block_dict["bbox"][5]

            data_ = block_dict["data"].float()
            if data_.ndim == 4:
                data_ = data_[np.newaxis, :]
            data_ = data_.to(device)
            output = model(data_)
            output = torch.sigmoid(output).squeeze().cpu().detach().numpy()
            prediction[l_low:l_high, w_low:w_high, d_low:d_high] = prediction[l_low:l_high, w_low:w_high,
                                                                   d_low:d_high] + output
            overlap_num[l_low:l_high, w_low:w_high, d_low:d_high] = overlap_num[l_low:l_high, w_low:w_high,
                                                                   d_low:d_high] + 1
        prediction = prediction / overlap_num

    image = nib.Nifti1Image(prediction, np.eye(4))
    nib.save(image, os.path.join(res_img_path, img_id + '-img.nii.gz'))

    return img_id


def val_like_test(model, val_dataset, gt_label_path, data_path, device, block_size):
    res_pred_path = os.path.join(data_path, 'pred')
    res_lbl_path = os.path.join(data_path, 'lbl')
    # model.to(torch.device('cpu'))
    for p in [res_pred_path, res_lbl_path]:
        if not os.path.exists(p):
            os.makedirs(p)
    exe_img = partial(execute_image, res_img_path=res_pred_path, model=model, device=device,
                      block_size=block_size)
    model.eval()
    # -------------------------------------------
    # with PPE(max_workers=2) as executor:
    # for i in executor.map(exe_img, val_dataset):
    #  print(f'{i} is processed')
    # 上面这些是为了把一块一块的预测结果拼在一起
    # -------------------------------------------
    for item in tqdm(val_dataset):
        # break
        # if not "500" in item[1]:
            # continue
        exe_img(item)
    all_csv = pd.DataFrame(columns=['public_id', 'label_id', 'confidence', 'label_code'])
    for file in tqdm(os.listdir(res_pred_path)):
        if file.endswith('csv'):
            continue
        pred = nib.load(os.path.join(res_pred_path, file)).get_data()
        img_id = file.split('-')[0]
        pred_lbl_nii, pred_csv = make_submission(pred, img_id)  # 这时候的pred_nii就是label()过后的图了，每个区域的预测概率存储在pred_csv中
        nib.save(pred_lbl_nii, os.path.join(res_lbl_path, file.replace('img', 'label')))
        all_csv = pd.concat([all_csv, pred_csv], axis=0)
        print(all_csv)
    all_csv.to_csv(os.path.join(res_lbl_path, 'prediction.csv'), index=False)

    evaluate_results = evaluate(gt_label_path, res_lbl_path)

    return evaluate_results


def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # break
            data, target = data.float(), target.long()
            # target = common.to_one_hot_3d(target, n_labels)
            # target = target.unsqueeze(1)  # convert it to (B, 1, Depth, H, W)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            output[output>=0.5]=1.0
            output[output<0.5]=0.0
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice': val_dice.avg[1]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # break
        # print(idx)
        data, target = data.float(), target.long()  # both: (B, Channel, H, L, W);
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        output = torch.sigmoid(output)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        output[output>=0.5]=1.0
        output[output<0.5]=0.0
        train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice': train_dice.avg[1]})
    return val_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-parameters management')

    # Hardware options
    parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
    parser.add_argument('--cpu', default=False, action='store_true', help='use cpu only')
    parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--seed', type=int, default=2022, help='random seed')

    # Preprocess parameters
    parser.add_argument('--n_labels', type=int, default=2, help='number of classes')
    parser.add_argument('--upper', type=int, default=500, help='')
    parser.add_argument('--lower', type=int, default=-500, help='')

    # data in/out and dataset
    parser.add_argument('--dataset_path', default='dataset/fixed_data', help='fixed dataset root path')
    parser.add_argument('--test_data_path', default='dataset/LiTS/test', help='Testset path')
    parser.add_argument('--save', default='Unet-6-6-lr5e-2', help='save path of trained model')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size of trainset')

    # train
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
    parser.add_argument('--crop_size', type=int, default=48)
    parser.add_argument('--val_crop_max_size', type=int, default=96)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='which epoch to start,比如是20，就把19load进来，从20开始训.是0表示从头开始训')
    parser.add_argument('--val_interval', type=int, default=30)
    parser.add_argument('--losstype', default = 'hybrid', type = str)

    # test
    parser.add_argument('--block_size', type=int, default=64, help='size of sliding window')
    parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
    parser.add_argument('--postprocess', type=bool, default=False, help='post process')

    args = parser.parse_args()
    save_path = os.path.join(os.getcwd(), 'experiments', args.save)
    res_path = os.path.join(save_path, 'val_results')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=RibDataSet(args, set='train'), batch_size=args.batch_size,
                              # num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=RibDataSet(args, set='val'), batch_size=1,
                            # num_workers=args.n_threads,
                            shuffle=True)
    # val_like_test_loader = DataLoader(dataset=RibTest(set='val', block_size=args.block_size), batch_size=1,num_workers=args.n_threads, shuffle=True)
    val_like_test_dataset = RibTest(args, set = "val")
    # model info
    # model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model = Unet(in_channels=1, out_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    # loss = loss.TverskyLoss()
    # loss = loss.HybridLoss()
    if args.losstype == 'dice':
        loss = loss.DiceLoss()
    elif args.losstype == 'hybrid':
        loss = loss.HybridLoss()
    else:
        print('Unrecognized Loss Type, please check spelling.')
        exit(1)

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器

    if args.start_epoch != 0:
        model_path = os.path.join(save_path, f'latest_model_{args.start_epoch-1}.pth')
        model.load_state_dict(torch.load(model_path)['net'])
        optimizer.load_state_dict(torch.load(model_path)['optimizer'])

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch-args.start_epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, f'latest_model_{epoch}.pth'))
        trigger += 1

        if val_log['Val_dice'] > best[1]:
            print('Saving best model')
            for file in os.listdir(save_path):  # 删掉上一次的best model
                if file.startswith('best_model'):
                    os.remove(os.path.join(save_path, file))
                    break
            torch.save(state, os.path.join(save_path, f'best_model_{epoch}.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()

    print('DONE TRAINING')
    print('******************')
    print(f'save path:\t\t{args.save}')
    print('******************')
    exit(0)
