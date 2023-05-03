from functools import partial
import pandas as pd
from typing import List

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
        "Couldn't import fast version of function compute_overlap, will use slow one. Check cython installation"
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
df_rows = []

# COCO stats
images_coco = []
annotations_gt = []
annotations_pred = []

data_iterator = DataIteratorAPI(api)
iter_gt = data_iterator.iterate_project_images(project_id_gt)
iter_pred = data_iterator.iterate_project_images(project_id_pred)
for i, (image_item_gt, image_item_pred) in enumerate(zip(iter_gt, iter_pred)):
    image_id = image_item_gt.image_id
    dataset_id = image_item_gt.dataset_id
    image_ids.append(image_id)

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
        image_item_pred, category_name_to_id, is_pred=True, image_id=image_id, dataset_id=dataset_id
    )
    annotations_pred += annotations

coco_gt, coco_dt = utils.create_coco_apis(images_coco, annotations_gt, annotations_pred, categories)
df = utils.create_df(df_rows)

confusion_matrix = utils.calculate_confusion_matrix(df, cm_categories)

overall_stats, per_dataset_stats, per_class_stats = utils.calculate_metrics(
    df, cm_categories, dataset_ids_gt, NONE_CLS
)

overall_coco = utils.overall_metrics_coco(coco_gt, coco_dt)

per_class_coco = {}
for category in categories:
    cat_id, cat_name = category["id"], category["name"]
    class_metrics = utils.per_class_metrics_coco(coco_gt, coco_dt, cat_id)
    per_class_coco[cat_id] = class_metrics

per_dataset_coco = {}
for dataset_id in dataset_ids_gt:
    dataset_metrics = utils.per_dataset_metrics_coco(coco_gt, coco_dt, dataset_id, images_coco)
    per_dataset_coco[dataset_id] = dataset_metrics

import pickle

with open("metrics.pkl", "wb") as f:
    stats = [overall_stats, per_dataset_stats, per_class_stats]
    coco = [overall_coco, per_dataset_coco, per_class_coco]
    pickle.dump([confusion_matrix, stats, coco], f)

with open("metrics.pkl", "rb") as f:
    confusion_matrix, stats, coco = pickle.load(f)

df.to_csv("df.csv")

# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
