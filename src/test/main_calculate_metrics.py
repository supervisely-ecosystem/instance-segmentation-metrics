from functools import partial
import pandas as pd
from typing import List

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import os

import utils
from data_iterator import DataIteratorAPI, ImageLabelsIterator
from src.globals import *


with open("ds_match.pkl", "rb") as f:
    ds_match = pickle.load(f)

# stats
df_rows = []

# COCO stats
images_coco = []
annotations_gt = []
annotations_pred = []

for i, (dataset_name, item) in enumerate(ds_match.items()):
    if item["dataset_matched"] != "both":
        continue
    dataset_id = dataset_name_to_id[dataset_name]
    for img_item in item["matched"]:
        img_gt, img_pred = img_item["left"], img_item["right"]
        img_gt: sly.ImageInfo
        image_id = img_gt.id
        image_ids.append(image_id)
        image_ids_gt2pred[img_gt.id] = img_pred.id

        labels_iter_gt = ImageLabelsIterator(api, img_gt.id, project_meta_gt, project_id_gt)
        labels_iter_pred = ImageLabelsIterator(api, img_pred.id, project_meta_pred, project_id_pred)
        image_item_gt = DataIteratorAPI.ImageItem(
            img_gt.id, labels_iter_gt.ann, project_id_gt, dataset_id, labels_iter_gt
        )
        image_item_pred = DataIteratorAPI.ImageItem(
            img_pred.id, labels_iter_pred.ann, project_id_pred, dataset_id, labels_iter_pred
        )

        # add DF rows
        new_rows = utils.collect_df_rows(image_item_gt, image_item_pred, NONE_CLS)
        df_rows += new_rows

        # add COCO annotations for pycocotools
        image_info, annotations = utils.collect_coco_annotations(
            image_item_gt, category_name_to_id, is_pred=False
        )
        annotations_gt += annotations
        images_coco.append(image_info)

        _, annotations = utils.collect_coco_annotations(
            image_item_pred,
            category_name_to_id,
            is_pred=True,
            image_id=image_id,
            dataset_id=dataset_id,
        )
        annotations_pred += annotations

coco_gt, coco_dt = utils.create_coco_apis(images_coco, annotations_gt, annotations_pred, categories)
df = utils.create_df(df_rows)

# Metrics
confusion_matrix = utils.calculate_confusion_matrix(df, cm_categories_selected)

overall_stats, per_dataset_stats, per_class_stats = utils.calculate_metrics(
    df, cm_categories_selected, dataset_ids, NONE_CLS
)

overall_coco = utils.overall_metrics_coco(coco_gt, coco_dt)

per_class_coco = {}
for cat_name in cm_categories_selected[:-1]:
    cat_id = category_name_to_id[cat_name]
    class_metrics = utils.per_class_metrics_coco(coco_gt, coco_dt, cat_id)
    per_class_coco[cat_id] = class_metrics

per_dataset_coco = {}
for dataset_id in dataset_ids:
    dataset_metrics = utils.per_dataset_metrics_coco(coco_gt, coco_dt, dataset_id, images_coco)
    per_dataset_coco[dataset_id] = dataset_metrics

import pickle

with open("metrics.pkl", "wb") as f:
    stats = [overall_stats, per_dataset_stats, per_class_stats]
    coco = [overall_coco, per_dataset_coco, per_class_coco]
    pickle.dump([confusion_matrix, stats, coco], f)

df.to_csv("df.csv")

with open("image_ids_gt2pred.pkl", "wb") as f:
    pickle.dump(image_ids_gt2pred, f)

# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
