from skimage.measure import label, regionprops
import nibabel as nib
import numpy as np
from skimage.morphology import remove_small_objects
import argparse
from tqdm import tqdm
import pandas as pd
import os

def my_remove_small_objects(prediction, min_size=64, cnct=3):
    pred_lbl = label(prediction, connectivity=cnct)
    for region_idx in range(1, int(pred_lbl.max() + 1)):
        size = np.where(pred_lbl == region_idx)[0].shape[0]
        if size<=min_size:
            prediction = prediction - (pred_lbl == region_idx).astype(np.int8)
    pred_lbl = label(prediction, connectivity=cnct)
    return prediction, pred_lbl


def make_submission(pred, image_id, prob_thres, remove_area_thres, affine=np.eye(4)):
    pred_bin = pred >= prob_thres
    # pred_bin = remove_small_objects(pred_bin, min_size=remove_area_thres, connectivity=3)
    # pred_label = label(pred_bin).astype(np.int16)  # 根据pred，做一个简单的聚类（不同簇有不同的标签）
    pred_bin, pred_label = my_remove_small_objects(pred_bin, min_size=remove_area_thres, cnct=3)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]  # 就是0加上各个label
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]  # 就是在每个label标出来的连通区域内，计算pred的平均值
    # placeholder for label class since classification isn't included
    pred_label_code = [0] + [1 for i in range(int(pred_label.max()))]  # 随便生成这些区域的label code，反正之后也不用
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),  # 表示有多行
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--exp_path', type=str, required=True, help='where to store val_results/lbl')
    parser.add_argument('--prob_thres', type=float, default=0.5, help='where to store val_results/lbl')
    parser.add_argument('--remove_area_thres', type=int, default=512, help='where to store val_results/lbl')

    args = parser.parse_args()

    res_pred_path = os.path.join(args.exp_path, 'val_results', 'pred')
    res_lbl_path = os.path.join(args.exp_path, 'val_results', 'lbl')

    all_csv = pd.DataFrame(columns=['public_id', 'label_id', 'confidence', 'label_code'])
    for file in tqdm(os.listdir(res_pred_path)):
        if file.endswith('csv'):
            continue
        pred = nib.load(os.path.join(res_pred_path, file)).get_data()
        img_id = file.split('-')[0]
        pred_lbl_nii, pred_csv = make_submission(pred, img_id, prob_thres=args.prob_thres,
                                                 remove_area_thres=args.remove_area_thres)  # 这时候的pred_nii就是label()过后的图了，每个区域的预测概率存储在pred_csv中
        nib.save(pred_lbl_nii, os.path.join(res_lbl_path, file.replace('img', 'label')))
        all_csv = pd.concat([all_csv, pred_csv], axis=0)
        print("now number of labels is ", len(all_csv))
    all_csv.to_csv(os.path.join(res_lbl_path, 'prediction.csv'), index=False)
    print('DONE')
    print('*************************')
    print(f"experiment: \t\t {args.exp_path}")
    print(f"prob thres:\t\t{args.prob_thres}")
    print(f"area thres:\t\t{args.remove_area_thres}")
