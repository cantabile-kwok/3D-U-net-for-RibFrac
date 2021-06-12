import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm

from FracNet.dataset.fracnet_dataset import FracNetInferenceDataset
from FracNet.dataset import transforms as tsfm
from FracNet.model.unet import UNet


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
    max_region.bbox[0]:max_region.bbox[2],
    max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):
    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
                          bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)

            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                                 center_x - crop_size // 2:center_x + crop_size // 2,
                                 center_y - crop_size // 2:center_y + crop_size // 2,
                                 center_z - crop_size // 2:center_z + crop_size // 2
                                 ]
                pred[
                center_x - crop_size // 2:center_x + crop_size // 2,
                center_y - crop_size // 2:center_y + crop_size // 2,
                center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i],
                                                          cur_pred_patch), axis=0), output[i])

    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
                             bone_thresh, size_thresh)

    return pred


def _make_submission_files(pred, image_id, affine=np.eye(4)):
    pred_label = label(pred >= 0.5).astype(np.int16)  # 根据pred，做一个简单的聚类（不同簇有不同的标签）
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]  # 就是0加上各个label
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]  # 就是在每个label标出来的连通区域内，计算pred的平均值
    # placeholder for label class since classification isn't included
    pred_label_code = [0] + list(range(1, int(pred_label.max()+1)))  # 随便生成这些区域的label code，反正之后也不用
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),  # 表示有多行
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def predict(args):
    batch_size = 16
    num_workers = 4
    postprocess = True if args.postprocess == "True" else False

    model = UNet(1, 1, first_out_channels=16)
    model.eval()
    if args.model_path is not None:
        model_weights = torch.load(args.model_path)
        model.load_state_dict(model_weights)
    model = nn.DataParallel(model).cuda()

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]

    image_path_list = sorted([os.path.join(args.image_dir, file)
                              for file in os.listdir(args.image_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
                     for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = FracNetInferenceDataset(image_path, transforms=transforms)
        dataloader = FracNetInferenceDataset.get_dataloader(dataset,
                                                            batch_size, num_workers)
        pred_arr = _predict_single_image(model, dataloader, postprocess,
                                         args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_image, pred_info = _make_submission_files(pred_arr, image_id,
                                                       dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.pred_dir, f"{image_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(args.pred_dir, "pred_info.csv"),
                     index=False)


if __name__ == "__main__":
    import argparse

    prob_thresh = 0.1
    bone_thresh = 300
    size_thresh = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
                        help="The image nii directory.")
    parser.add_argument("--pred_dir", required=True,
                        help="The directory for saving predictions.")
    parser.add_argument("--model_path", default=None,
                        help="The PyTorch model weight path.")
    parser.add_argument("--prob_thresh", default=0.1,
                        help="Prediction probability threshold.")
    parser.add_argument("--bone_thresh", default=300,
                        help="Bone binarization threshold.")
    parser.add_argument("--size_thresh", default=100,
                        help="Prediction size threshold.")
    parser.add_argument("--postprocess", default="True",
                        help="Whether to execute post-processing.")
    args = parser.parse_args()
    predict(args)
