import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm

from nii_dataset import NiiDataset

__all__ = ["froc", "plot_froc", "evaluate"]

# detection key FP values
DEFAULT_KEY_FP = (0.5, 1, 2, 4, 8)

# classification confusion matrix settings
label_code_dict = {
    0: "Background",
    1: "Displaced",
    2: "Nondisplaced",
    3: "Buckle",
    4: "Segmental",
    -1: "Ignore"
}
# for ground truth; FP: false positive detections;
# Ignore: undefined labels in annotations
clf_conf_mat_cols = ["Buckle", "Displaced", "Nondisplaced", "Segmental",
                     "FP", "Ignore"]
# for prediction; FN: false negative, hit no gt labels
clf_conf_mat_rows = ["Buckle", "Displaced", "Nondisplaced", "Segmental", "FN"]

pd.set_option("display.precision", 6)


def _get_gt_class(x):
    # if GT classification exists, use it
    if not pd.isna(x["gt_class"]):
        return x["gt_class"]
    # if the prediction doesn't hit anything, it's a false positive
    else:
        return "FP"


def _get_clf_confusion_matrix(gt_info, pred_metrics):
    """
    Calculate the classification confusion matrix.

    Parameters
    ----------
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.

    Returns
    -------
    conf_mat : pandas.DataFrame
        DataFrame of confusion matrix.
    """
    # set up the confusion matrix
    conf_mat = pd.DataFrame(np.zeros((len(clf_conf_mat_rows),
                                      len(clf_conf_mat_cols))), index=clf_conf_mat_rows,
                            columns=clf_conf_mat_cols)

    # iterate through all predictions and fill them in the confusion matrix
    for i in range(len(pred_metrics)):
        gt_class = pred_metrics.gt_class[i]
        pred_class = pred_metrics.pred_class[i]
        if pred_class not in ("Background", "Ignore"):
            if gt_class == "Background":
                gt_class = "FP"
            conf_mat.loc[pred_class, gt_class] += 1

    # iterate through all undetected GTs and fill them in the confusion matrix
    for i in range(len(gt_info)):
        if len(pred_metrics) == 0 \
                or gt_info.label_id[i] not in list(pred_metrics.hit_label):
            if gt_info.label_code[i] != "Background":
                conf_mat.loc["FN", gt_info["label_code"][i]] += 1

    return conf_mat


def _calculate_f1(conf_matrix):
    """
    Calculate classification macro F1 score.

    Parameters
    ----------
    conf_matrix : pandas.DataFrame
        DataFrame of confusion matrix.

    Returns
    -------
    f1_total : float
        Total classification macro F1 score including detection FP and FN.
    f1_target : float
        Classification macro F1 score excluding detection FP.
        This metric is target-aware, meaning that it only takes GT
        classification into account.
    f1_pred : float
        Classification macro F1 score exculding both detection FP and FN.
        This metric is prediction-aware, meaning that it calculates
        classification F1 solely on TP prediction thus completely decouples
        detection and classification.
    """
    conf_matrix_wo_ignore = conf_matrix.values[:, :-1]
    tp = np.diag(conf_matrix_wo_ignore)[:-1]
    # minus 1 since only four GT classes need to be calculated
    fp = np.array([conf_matrix_wo_ignore[i, :].sum() - tp[i]
                   for i in range(conf_matrix_wo_ignore.shape[0] - 1)])
    fn = np.array([conf_matrix_wo_ignore[:, i].sum() - tp[i]
                   for i in range(conf_matrix_wo_ignore.shape[0] - 1)])

    # calculate total F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_total = np.mean((2 * precision * recall) / (precision + recall + 1e-8))

    # calculate target-aware F1 which excludes detection FP
    precision = tp / (tp + (fp - conf_matrix_wo_ignore[:-1, -1]) + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_target = np.mean((2 * precision * recall) \
                        / (precision + recall + 1e-8))

    # calculate prediction-aware F1 which excludes both detection FP and FN
    precision = tp / (tp + (fp - conf_matrix_wo_ignore[:-1, -1]) + 1e-8)
    recall = tp / (tp + (fn - conf_matrix_wo_ignore[-1, :-1]) + 1e-8)
    f1_pred = np.mean((2 * precision * recall) / (precision + recall + 1e-8))

    return f1_total, f1_target, f1_pred


def _compile_pred_metrics(iou_matrix, gt_info, pred_info):
    """
    Compile prediction metrics into a Pandas DataFrame

    Parameters
    ----------
    iou_matrix : numpy.ndarray
        IoU array with shape of (n_pred, n_gt).
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    # meanings of each column:
    # pred_label --  The index of prediction
    # max_iou -- The highest IoU this prediction has with any certain GT
    # hit_label -- The GT label with which this prediction has the highest IoU
    # prob -- The confidence prediction of this prediction
    # pred_class -- The classification prediction
    # gt_class -- The ground-truth prediction
    # num_gt -- Total number of GT in this volume
    pred_metrics = pd.DataFrame(np.zeros((iou_matrix.shape[0], 3)),
                                columns=["pred_label", "max_iou", "hit_label"])
    pred_metrics["pred_label"] = np.arange(1, iou_matrix.shape[0] + 1)
    pred_metrics["max_iou"] = iou_matrix.max(axis=1)
    pred_metrics["hit_label"] = iou_matrix.argmax(axis=1) + 1

    # if max_iou == 0, this prediction doesn't hit any GT
    pred_metrics["hit_label"] = pred_metrics.apply(lambda x: x["hit_label"]
    if x["max_iou"] > 0 else 0, axis=1)

    # fill in the classification prediction and detection confidence
    pred_metrics = pred_metrics.merge(
        pred_info[["label_id", "label_code", "confidence"]],
        how="left", left_on="pred_label", right_on="label_id")
    pred_metrics.rename({"confidence": "prob", "label_code": "pred_class"},
                        axis=1, inplace=True)
    pred_metrics.drop("label_id", axis=1, inplace=True)

    # compare the classification prediction against GT
    pred_metrics = pred_metrics.merge(gt_info[["label_id", "label_code"]],
                                      how="left", left_on="hit_label", right_on="label_id")
    pred_metrics.rename({"label_code": "gt_class"}, axis=1, inplace=True)
    pred_metrics.drop("label_id", axis=1, inplace=True)
    pred_metrics["gt_class"] = pred_metrics.apply(_get_gt_class, axis=1)

    pred_metrics["num_gt"] = iou_matrix.shape[1]

    # compile the classification confusion matrix
    # clf_conf_mat = _get_clf_confusion_matrix(gt_info, pred_metrics)

    return pred_metrics, None


def evaluate_single_prediction(gt_label, pred_label, gt_info, pred_info):
    """
    Evaluate a single prediction.

    Parameters
    ----------
    gt_label : numpy.ndarray
        The numpy array of ground-truth labelled from 1 to n.
    pred_label : numpy.ndarray
        The numpy array of prediction labelled from 1 to n.
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    num_gt : int
        Number of GT in this case.
    clf_conf_mat : pandas.DataFrame
        Classification confusion matrix.
    inter_sum : int
        Number of gt-pred intersection voxels.
    union_sum : int
        Number of gt-pred union voxels.
    y_true_sum : int
        Number of GT voxels.
    y_pred_sum : int
        Number of prediction voxels.
    """
    gt_label = gt_label.astype(np.uint8)
    pred_label = pred_label.astype(np.uint8)

    # GT and prediction must have the same shape
    assert gt_label.shape == pred_label.shape, \
        "The prediction and ground-truth have different shapes. gt:" \
        f" {gt_label.shape} and pred: {pred_label.shape}."

    num_gt = gt_label.max()  # 这是GT的骨折区域的个数
    num_pred = pred_label.max()  # pred中的骨折区域的个数

    # if the prediction is empty, return empty pred_metrics
    # and confusion matrix
    if num_pred == 0:
        pred_metrics = pd.DataFrame()
        # clf_conf_mat = _get_clf_confusion_matrix(gt_info, pred_metrics)
        inter_sum = 0
        union_sum = (gt_label > 0).sum()
        y_true_sum = union_sum
        y_pred_sum = 0

        return pred_metrics, num_gt, None, inter_sum, union_sum, \
               y_true_sum, y_pred_sum

    # if GT is empty
    if num_gt == 0:
        pred_metrics = pd.DataFrame([
            {
                "pred_label": i,
                "max_iou": 0,
                "hit_label": 0,
                "gt_class": "FP",
                "num_gt": 0
            }
            for i in range(1, num_pred + 1)])
        pred_metrics = pred_metrics.merge(
            pred_info[["label_id", "label_code", "confidence"]],
            how="left", left_on="pred_label", right_on="label_id")
        pred_metrics.rename(
            {"label_code": "pred_class", "confidence": "prob"}, axis=1,
            inplace=True)
        pred_metrics.drop(["label_id"], axis=1, inplace=True)
        # clf_conf_mat = _get_clf_confusion_matrix(gt_info, pred_metrics)
        inter_sum = 0
        union_sum = (pred_label > 0).sum()
        y_true_sum = 0
        y_pred_sum = union_sum

        return pred_metrics, num_gt, None, inter_sum, union_sum, \
               y_true_sum, y_pred_sum

    # binarize the GT and prediction
    gt_bin = (gt_label > 0).astype(np.uint8)  # 就是把骨折区域都标成1
    pred_bin = (pred_label > 0).astype(np.uint8)

    num_pred = int(pred_label.max())  # tm之前不是写过这句啦？
    iou_matrix = np.zeros((num_gt, num_pred))

    intersection = np.logical_and(gt_bin, pred_bin)
    union = label(np.logical_or(gt_bin, pred_bin))  # 注意这里label操作过了

    # iterate through all intersection area and evaluate predictions
    for region in regionprops(label(intersection)):  # 就是交集中的每块
        # get an anchor point within the intersection to locate gt & pred
        anchor = tuple(region.coords[0].tolist())

        # get corresponding GT index, pred index and union index
        gt_idx = gt_label[anchor[0], anchor[1], anchor[2]]
        pred_idx = pred_label[anchor[0], anchor[1], anchor[2]]
        union_idx = union[anchor[0], anchor[1], anchor[2]]  # 这里的idx都是指预测的是第几个骨折区域

        if gt_idx == 0 or pred_idx == 0 or union_idx == 0:
            continue

        inter_area = region.area
        union_area = (union == union_idx).sum()
        iou = inter_area / (union_area + 1e-8)
        iou_matrix[gt_idx - 1, pred_idx - 1] = iou

    # 这个矩阵就是每个pred中的簇，与每个gt中的簇的iou
    iou_matrix = iou_matrix.T
    pred_metrics, clf_conf_mat = _compile_pred_metrics(iou_matrix,
                                                       gt_info, pred_info)
    # pred_metrics包含了这个CT照片含有的所有预测出来的簇的指标和信息
    inter_sum = intersection.sum()
    union_sum = (union > 0).sum()
    y_true_sum = gt_bin.sum()  # GT中有多少个voxel是骨折区域
    y_pred_sum = pred_bin.sum()  # 预测结果中多少个点是骨折区域

    # return pred_metrics, \
    #        num_gt, \
    #        clf_conf_mat, inter_sum, union_sum, \
    #        y_true_sum, y_pred_sum
    return pred_metrics, num_gt, None, inter_sum, union_sum, y_true_sum, y_pred_sum


def _froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh):
    """
    Calculate the FROC for a single confidence threshold.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of Pandas DataFrame of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    p_thresh : float
        The probability threshold of positive predictions.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".

    Returns
    -------
    fp : float
        False positives per scan for this threshold.
    recall : float
        Recall rate for this threshold.
    """
    EPS = 1e-8

    total_gt = sum(num_gts)  # 所有图片GT总共有多少骨折区域
    # collect all predictions above the probability threshold
    df_pos_pred = [df.loc[df["prob"] >= p_thresh] for df in df_list
                   if len(df) > 0]  # 所有图片的骨折区域中，那些置信度高于阈值的区域，被判定为真的骨折区域

    # calculate total true positives
    total_tp = sum([len(df.loc[df["max_iou"] > iou_thresh, "hit_label"] \
                        .unique()) for df in df_pos_pred])  # 在被判定为的骨折区域中，那些和GT最大IOU超过阈值（即判断对了的区域）个数

    # calculate total false positives
    total_fp = sum([len(df) - len(df.loc[df["max_iou"] > iou_thresh])
                    for df in df_pos_pred])

    fp = (total_fp + EPS) / (len(df_list) + EPS)
    recall = (total_tp + EPS) / (total_gt + EPS)

    return fp, recall


def _interpolate_recall_at_fp(fp_recall, key_fp):
    """
    Calculate recall at key_fp using interpolation.

    Parameters
    ----------
    fp_recall : pandas.DataFrame
        DataFrame of FP and recall.
    key_fp : float
        Key FP threshold at which the recall will be calculated.

    Returns
    -------
    recall_at_fp : float
        Recall at key_fp.
    """
    # get fp/recall interpolation points
    fp_recall_less_fp = fp_recall.loc[fp_recall.fp <= key_fp]
    fp_recall_more_fp = fp_recall.loc[fp_recall.fp >= key_fp]

    # if key_fp < min_fp, recall = 0
    if len(fp_recall_less_fp) == 0:
        return 0

    # if key_fp > max_fp, recall = max_recall
    if len(fp_recall_more_fp) == 0:
        return fp_recall.recall.max()

    fp_0 = fp_recall_less_fp["fp"].values[-1]
    fp_1 = fp_recall_more_fp["fp"].values[0]
    recall_0 = fp_recall_less_fp["recall"].values[-1]
    recall_1 = fp_recall_more_fp["recall"].values[0]
    recall_at_fp = recall_0 + (recall_1 - recall_0) \
                   * ((key_fp - fp_0) / (fp_1 - fp_0 + 1e-8))

    return recall_at_fp


def _get_key_recall(fp, recall, key_fp_list):
    """
    Calculate recall at a series of FP threshold.

    Parameters
    ----------
    fp : list of float
        List of FP at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_fp_list : list of float
        List of key FP values.

    Returns
    -------
    key_recall : list of float
        List of key recall at each key FP.
    """
    fp_recall = pd.DataFrame({"fp": fp, "recall": recall}).sort_values("fp")
    key_recall = [_interpolate_recall_at_fp(fp_recall, key_fp)
                  for key_fp in key_fp_list]

    return key_recall


def froc(df_list, num_gts, iou_thresh=0.2, key_fp=DEFAULT_KEY_FP):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    key_fp : tuple of float
        The key false positive per scan used in evaluating the sensitivity
        of the model.

    Returns
    -------
    fp : list of float
        List of false positives per scan at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_recall : list of float
        List of key recall corresponding to key FPs.
    avg_recall : float
        Average recall at key FPs. This is the evaluation metric we use
        in the detection track.
    """
    fp_recall = [_froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh)
                 for p_thresh in np.arange(0, 1, 0.01)]
    fp = [x[0] for x in fp_recall]
    recall = [x[1] for x in fp_recall]
    key_recall = _get_key_recall(fp, recall, key_fp)
    avg_recall = np.mean(key_recall)

    return fp, recall, key_recall, avg_recall


def plot_froc(fp, recall):
    """
    Plot the FROC curve.

    Parameters
    ----------
    fp : list of float
        List of false positive per scans at different confidence thresholds.
    recall : list of float
        List of recall at different confidence thresholds.
    """
    _, ax = plt.subplots()
    ax.plot(fp, recall)
    ax.set_title("FROC")
    plt.savefig("froc.jpg")


def _cal_seg_metrics(seg_results):
    """
    Calculate segmentation DICE score and IoU.

    Parameters
    ----------
    seg_results : list of tuple
        List of tuples containing intersection, gt sum and prediction sum
        of segmentation.

    Returns
    -------
    dice : float
        Segmentation DICE score.
    iou : float
        Segmentation IoU score.
    """
    inter, union, y_true_sum, y_pred_sum = np.array(seg_results).sum(axis=0) \
        .tolist()
    dice = (2 * inter) / (y_true_sum + y_pred_sum + 1e-8)
    iou = inter / (union + 1e-8)

    return dice, iou


def evaluate(gt_dir, pred_dir):
    """
    Evaluate predictions against the ground-truth.

    Parameters
    ----------
    gt_dir : str
        The ground-truth nii directory.
    pred_dir : str
        The prediction nii directory.

    Returns
    -------
    eval_results : dict
        Dictionary containing detection and classification results.
    """
    # GT and prediction iterator
    gt_iter = NiiDataset(gt_dir)
    pred_iter = NiiDataset(pred_dir)

    gt_csv_path = [os.path.join(gt_dir, x) for x in os.listdir(gt_dir)
                   if x.endswith(".csv")][0]
    pred_csv_path = [os.path.join(pred_dir, x) for x in os.listdir(pred_dir)
                     if x.endswith(".csv")][0]
    gt_info = pd.read_csv(gt_csv_path)
    pred_info = pd.read_csv(pred_csv_path)
    gt_info["label_code"] = gt_info["label_code"].map(label_code_dict)
    pred_info["label_code"] = pred_info["label_code"].map(label_code_dict)

    # GT and prediction directory sanity check
    assert len(pred_iter) == len(gt_iter), \
        "Unequal number of predictions and ground-truths."
    assert gt_iter.pid_list == pred_iter.pid_list, \
        "Unmatched file names in ground-truth and prediction directory."

    eval_results = []
    progress = tqdm(total=len(gt_iter))
    for i in range(len(gt_iter)):  # 取出GT的label，这玩意是一个0，1，2……组成的label volume
        # get GT array and information
        gt_arr, pid = gt_iter[i]
        cur_gt_info = gt_info.loc[gt_info.public_id == pid] \
            .reset_index(drop=True)  # 就是取出这张CT图片的info

        # get prediction array and information
        pred_arr, _ = pred_iter[i]
        cur_pred_info = pred_info.loc[pred_info.public_id == pid] \
            .reset_index(drop=True)

        # perform evaluation
        eval_results.append(evaluate_single_prediction(
            gt_arr, pred_arr, cur_gt_info, cur_pred_info))

        progress.update(1)

    progress.close()

    # detection results
    det_results = [x[0] for x in eval_results]  # 就是每张CT的pred_metric
    num_gts = [x[1] for x in eval_results]  # 每张图片的GT有多少骨折区域

    # classification results
    # clf_conf_mat = pd.DataFrame(np.sum([x[2].values for x in eval_results],
    #                                    axis=0), index=clf_conf_mat_rows, columns=clf_conf_mat_cols)

    # segmentation results
    seg_results = np.array([x[3:] for x in eval_results])
    dice, iou = _cal_seg_metrics(seg_results)

    # calculate the detection FROC
    fp, recall, key_recall, avg_recall = froc(det_results, num_gts)

    # calculate the classification macro F1
    # f1_total, f1_target, f1_pred = _calculate_f1(clf_conf_mat)

    eval_results = {
        "detection": {
            "fp": fp,
            "recall": recall,
            "key_recall": key_recall,
            "average_recall": avg_recall,
            "max_recall": max(recall),
            "average_fp_at_max_recall": max(fp),
        },
        # "classification": {
        #     "confusion_matrix": clf_conf_mat,
        #     # the overall classification performance
        #     "macro_average_F1": f1_total,
        #     # the target-aware classification performance excluding FPs
        #     "target_aware_F1": f1_target,
        #     # the predictiokn-aware classification performance
        #     # excluding both FPs and FNs
        #     "prediction_aware_F1": f1_pred
        # },
        "segmentation": {
            "dice": dice,
            "iou": iou
        }
    }

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--clf", default="False")
    args = parser.parse_args()
    eval_results = evaluate(args.gt_dir, args.pred_dir)

    # detection metrics
    print("\nDetection metrics")
    print("=" * 64)
    print("Recall at key FP")
    print(pd.DataFrame(np.array(eval_results["detection"]["key_recall"]) \
                       .reshape(1, -1), index=["Recall"],
                       columns=[f"FP={str(x)}" for x in DEFAULT_KEY_FP]))
    print("Average recall: {:.4f}".format(
        eval_results["detection"]["average_recall"]))
    print("Maximum recall: {:.4f}".format(
        eval_results["detection"]["max_recall"]
    ))
    print("Average FP per scan at maximum recall: {:.4f}".format(
        eval_results["detection"]["average_fp_at_max_recall"]
    ))
    # plot/print FROC curve
    print("FPR, Recall in FROC")
    for fp, recall in zip(reversed(eval_results["detection"]["fp"]),
                          reversed(eval_results["detection"]["recall"])):
        print(f"({fp:.8f}, {recall:.8f})")
    # plot_froc(eval_results["detection"]["fp"],
    #     eval_results["detection"]["recall"])

    if args.clf == "True":
        # classification metrics
        print("\nClassification metrics")
        print("=" * 64)
        print("Confusion matrix")
        print(eval_results["classification"]["confusion_matrix"])
        print("Macro-average F1: {:.4f}".format(
            eval_results["classification"]["macro_average_F1"]))
        print("Target-aware F1: {:.4f}".format(
            eval_results["classification"]["target_aware_F1"]))
        print("Prediction-aware F1: {:.4f}".format(
            eval_results["classification"]["prediction_aware_F1"]))

    # segmentation metrics
    print("\nSegmentation metrics")
    print("=" * 64)
    print(f"Dice score: {eval_results['segmentation']['dice']:.4f}")
    print(f"IoU: {eval_results['segmentation']['iou']:.4f}")
