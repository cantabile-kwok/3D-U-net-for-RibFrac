import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
from functools import partial
import argparse
from Unet_test import Unet
from newdataset import RibTest


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
            # print(img_id)
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


def make_predictions(model, val_dataset, data_path, block_size, device):
    res_pred_path = os.path.join(data_path, 'pred')
    res_lbl_path = os.path.join(data_path, 'lbl')
    for p in [res_pred_path, res_lbl_path]:
        if not os.path.exists(p):
            os.makedirs(p)
    exe_img = partial(execute_image, res_img_path=res_pred_path, model=model, device=device,
                      block_size=block_size)
    model.eval()
    for item in tqdm(val_dataset):
        # break
        exe_img(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--exp_path', type=str, required=True, help='where to find best_model.pth')
    parser.add_argument('--raw_data_path', type=str, default="dataset/raw_data")
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--cpu_only', default=False)
    parser.add_argument('--upper', default = 200, type=int)
    parser.add_argument('--lower', default = -200, type=int)


    args = parser.parse_args()
    model = Unet(in_channels=1, out_channels=1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU

    flag = False
    for file in os.listdir(args.exp_path):
        if file.startswith("best_model"):
            model.load_state_dict(torch.load(os.path.join(args.exp_path, file))['net'])
            flag = True
            break
    assert flag, "Cannot find best model in exp_path"
    # val_like_test_dataset = RibTest(args, set='val')
    test_dataset=  RibTest(args, set='test')
    device = torch.device('cpu' if args.cpu_only else "cuda")
    model.to(device)
    # make_predictions(model, val_like_test_dataset, os.path.join(args.exp_path, 'val_results'), args.block_size, device)
    make_predictions(model, test_dataset, os.path.join(args.exp_path, 'test_results'), args.block_size, device)
    
    print('DONE')
    print('*************************')
    print(f"experiment: \t\t {args.exp_path}")
