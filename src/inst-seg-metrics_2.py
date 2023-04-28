from functools import partial
import pandas as pd
from pycocotools import mask, coco, cocoeval
import json
from copy import deepcopy
from typing import List
import sklearn.metrics
from itertools import chain

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import os


try:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
    from .compute_overlap import compute_overlap
except:
    print(
        "Couldn't import fast version of function compute_overlap, will use slow one. Check cython intallation"
    )
    from compute_overlap_slow import compute_overlap

import utils
from data_iterator import DataIteratorAPI


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = 20645
project_id_pred = 20644


# meta
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt))
categories = [{"id": i, "name": obj_cls.name} for i, obj_cls in enumerate(meta.obj_classes)]
category_name_to_id = {c["name"]: c["id"] for c in categories}
dataset_ids_gt = [ds.id for ds in api.dataset.get_list(project_id_gt)]
image_ids = []

# CM is of shape [GT x Pred]
NONE_CLS = "None"
cm_categories = list(map(lambda x: x["name"], categories)) + [NONE_CLS]

# stats
overall_stats = {}
per_image_stats = []
df = []


data_iterator = DataIteratorAPI(api)
iter_gt = data_iterator.iterate_project_images(project_id_gt)
iter_pred = data_iterator.iterate_project_images(project_id_pred)
for i, (image_item_gt, image_item_pred) in enumerate(zip(iter_gt, iter_pred)):
    image_id = image_item_gt.image_id
    dataset_id = image_item_gt.dataset_id
    image_ids.append(image_id)

    labels_gt, classes_gt, bboxes_gt, bitmaps_gt = utils.collect_labels(image_item_gt)
    labels_pred, classes_pred, bboxes_pred, bitmaps_pred = utils.collect_labels(image_item_pred)

    # [Pred, GT]
    pairwise_iou = compute_overlap(
        np.array(bboxes_pred, dtype=np.float64), np.array(bboxes_gt, dtype=np.float64)
    )

    # below this threshold we treat two bboxes don't match
    min_iou_threshold = 0.25
    matched_idxs, unmatched_idxs_gt, unmatched_idxs_pred, box_ious_matched = utils.match_bboxes(
        pairwise_iou, min_iou_threshold
    )
    matched_idxs_gt, matched_idxs_pred = list(zip(*matched_idxs))

    for i_gt, i_pred in matched_idxs:
        class_gt = classes_gt[i_gt]
        class_pred = classes_pred[i_pred]
        class_match = class_gt == class_pred
        mask1, mask2 = utils.join_bitmaps_tight(bitmaps_gt[i_gt], bitmaps_pred[i_pred])
        iou = utils.iou_numpy(mask1, mask2)
        gt_label_id = bitmaps_gt[i_gt].sly_id
        pred_label_id = bitmaps_pred[i_pred].sly_id
        row = [
            class_gt,
            class_pred,
            class_match,
            iou,
            image_id,
            dataset_id,
            gt_label_id,
            pred_label_id,
        ]
        df.append(row)

    for idx in unmatched_idxs_gt:
        cls = classes_gt[idx]
        gt_label_id = bitmaps_gt[idx].sly_id
        row = [cls, NONE_CLS, None, None, image_id, dataset_id, gt_label_id, -1]
        df.append(row)

    for idx in unmatched_idxs_pred:
        cls = classes_pred[idx]
        pred_label_id = bitmaps_pred[idx].sly_id
        row = [NONE_CLS, cls, None, None, image_id, dataset_id, -1, pred_label_id]
        df.append(row)


cols = [
    "gt_class",
    "pred_class",
    "class_match",
    "iou",
    "image_id",
    "dataset_id",
    "gt_label_id",
    "pred_label_id",
]
df = pd.DataFrame(df, columns=cols)

gt_classes = df["gt_class"].to_list()
pred_classes = df["pred_class"].to_list()
confusion_matrix = sklearn.metrics.confusion_matrix(gt_classes, pred_classes, labels=cm_categories)

# per-class + avg: P/R/F1
per_class_stats = sklearn.metrics.classification_report(
    gt_classes, pred_classes, labels=cm_categories, output_dict=True
)
per_class_stats.pop(NONE_CLS)
overall_keys = ["micro avg", "macro avg", "weighted avg"]
for key in overall_keys:
    overall_stats[key] = per_class_stats.pop(key)

class2image_ids = {}  # image_id in GT
for cls in cm_categories:
    if cls == NONE_CLS:
        continue
    cls_filtered = df[(df["gt_class"] == cls) | (df["pred_class"] == cls)]
    gt = cls_filtered["gt_class"].to_list()
    pred = cls_filtered["pred_class"].to_list()
    gt = [int(x == cls) for x in gt]
    pred = [int(x == cls) for x in pred]
    AP = sklearn.metrics.average_precision_score(gt, pred) if gt and pred else -1
    avg_iou = cls_filtered["iou"].mean()
    per_class_stats[cls]["IoU"] = avg_iou
    per_class_stats[cls]["AP"] = AP

    class2image_ids[cls] = list(set(cls_filtered["image_id"]))

overall_stats["mAP"] = np.mean([x["AP"] for x in per_class_stats.values() if x["AP"] != -1])
overall_stats["mIoU"] = np.nanmean([x["IoU"] for x in per_class_stats.values()])

per_dataset_stats = {}
for dataset_id in dataset_ids_gt:
    AP_per_class = []
    ious_per_class = []
    for cls in cm_categories:
        if cls == NONE_CLS:
            continue
        cls_filtered = df[
            (df["dataset_id"] == dataset_id) & ((df["gt_class"] == cls) | (df["pred_class"] == cls))
        ]
        gt = cls_filtered["gt_class"].to_list()
        pred = cls_filtered["pred_class"].to_list()
        gt = [int(x == cls) for x in gt]
        pred = [int(x == cls) for x in pred]
        if gt and pred:
            AP = sklearn.metrics.average_precision_score(gt, pred)
            AP_per_class.append(AP)
        avg_iou = cls_filtered["iou"].mean()
        ious_per_class.append(avg_iou)
    mAP = np.mean(AP_per_class)
    mIoU = np.nanmean(ious_per_class)
    per_dataset_stats[dataset_id] = {"mAP": mAP, "mIoU": mIoU}


# Per-image
def per_image_metrics(df, image_id):
    df_img = df[df["image_id"] == image_id]
    avg_iou = df_img["iou"].mean()
    TP = (df_img["gt_class"] == df_img["pred_class"]).sum()
    FP = (df_img["class_match"].isna() & (df_img["gt_class"] == NONE_CLS)).sum()
    FN = (df_img["class_match"].isna() & (df_img["pred_class"] == NONE_CLS)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return TP, FP, FN, precision, recall, avg_iou


print


# for cls, image_ids_gt in class2image_ids.items():
#     coco_api


# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
