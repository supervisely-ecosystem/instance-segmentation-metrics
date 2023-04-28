from functools import partial
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

    pyximport.install(
        setup_args={"include_dirs": np.get_include()}, reload_support=False
    )
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

# stats
overall_stats = []
per_class_stats = []
per_image_stats = []
confusion_matrix = []


data_iterator = DataIteratorAPI(api)
iter_gt = data_iterator.iterate_project_images(project_id_gt)
iter_pred = data_iterator.iterate_project_images(project_id_pred)
img_id2matches = {}
for i, (image_item_gt, image_item_pred) in enumerate(zip(iter_gt, iter_pred)):

    bboxes_gt, bboxes_pred = [], []
    labels_gt, labels_pred = [], []
    classes_gt, classes_pred = [], []
    bitmaps_gt, bitmaps_pred = [], []

    for item in image_item_gt.labels_iterator():
        item: DataIteratorAPI.LabelItem
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        label = item.label
        labels_gt.append(label)
        classes_gt.append(label.obj_class.name)
        rect = label.geometry.to_bbox()
        bbox = rect.left, rect.top, rect.right, rect.bottom
        bboxes_gt.append(bbox)
        bitmaps_gt.append(label.geometry)

    for item in image_item_pred.labels_iterator():
        item: DataIteratorAPI.LabelItem
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        label = item.label
        labels_pred.append(label)
        classes_pred.append(label.obj_class.name)
        rect = label.geometry.to_bbox()
        bbox = rect.left, rect.top, rect.right, rect.bottom
        bboxes_pred.append(bbox)
        bitmaps_pred.append(label.geometry)

    # 1. compute iou pairwise
    # 2. match by box_iou in sorted array
    # 3. PR-curve, AP

    pairwise_iou = compute_overlap(np.array(bboxes_pred, dtype=np.float64), np.array(bboxes_gt, dtype=np.float64))  # [Pred, GT]
    # matches_idxs_pred = matches.argmax(1)  # каждому pred ищем самый близкий GT
    # matches_idxs_gt = matches.argmax(0)
    
    min_threshold = 0.25
    
    n_pred, n_gt = pairwise_iou.shape
    m = pairwise_iou.flatten()
    match_2 = []
    ious_matched = []
    unmatched_idxs_pred, gt_rest = set(range(n_pred)), set(range(n_gt))
    for i in m.argsort()[::-1]:
        if m[i] < min_threshold or m[i] == 0:
            break
        pred_i = i//n_gt
        gt_i = i%n_gt
        if pred_i in unmatched_idxs_pred and gt_i in gt_rest:
            match_2.append([pred_i, gt_i])
            ious_matched.append(m[i])
            unmatched_idxs_pred.remove(pred_i)
            gt_rest.remove(gt_i)

    # class_matches = [[classes_pred[x[0]], classes_gt[x[1]]] for x in match_2]

    class_matches = [classes_pred[x[0]] == classes_gt[x[1]] for x in match_2]

    fp_idxs = list(unmatched_idxs_pred)
    fn_idxs = list(gt_rest)
    tp_idxs_2 = list(match_2)

    used_classes = list(set(classes_pred) & set(classes_gt))
    matched_pred, matched_gt = list(zip(*match_2))

    def filter_cls(target_cls, idxs, classes):
        return [idx for idx in idxs if classes[idx] == target_cls]

    for cls in used_classes:
        # pred, gt = [], []
        # pred += [idx for idx in chain(matched_pred, pred_rest) if classes_pred[idx] == cls]
        # gt += [idx for idx in matched_gt if classes_gt[idx] == cls]
        # gt += []

        matched_pred_cls = filter_cls(cls, matched_pred, classes_pred)
        pred_rest_cls = filter_cls(cls, unmatched_idxs_pred, classes_pred)
        ious_cls = filter_cls(cls, ious, classes_gt)
        matched_pred = filter_cls(cls, matched_pred, classes_pred)
        matched_pred = filter_cls(cls, matched_pred, classes_pred)
        matched_pred = filter_cls(cls, matched_pred, classes_pred)
        matched_pred = filter_cls(cls, matched_pred, classes_pred)

        y_true = [1]*len(match_2) + [1]*len(gt_rest) + [0]*len(unmatched_idxs_pred)
        ious = ious_matched + [0]*len(gt_rest) + [0]*len(unmatched_idxs_pred)
        p, r, th = sklearn.metrics.average_precision_score(y_true, ious)
        p, r, th = sklearn.metrics.recall_score(y_true, ious)


        
    
    # fp/fn потому что unmatched или потому что mis-class?

    # gt_cls, pred_cls, box_matched, iou, 

    threshold = 0.5



    sklearn.metrics.precision_recall_curve()
    sklearn.metrics.average_precision_score()

    break

    

    

# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
