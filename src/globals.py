import pandas as pd
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import os
import pickle


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = 20645
project_id_pred = 20644


# meta
project_meta_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt))
project_meta_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_id_pred))
# categories will contain all obj_classes in gt
categories = [
    {"id": i, "name": obj_cls.name} for i, obj_cls in enumerate(project_meta_gt.obj_classes)
]
category_name_to_id = {c["name"]: c["id"] for c in categories}
_datasets = api.dataset.get_list(project_id_gt)
dataset_ids_gt = [ds.id for ds in _datasets]
dataset_names_gt = pd.Series({ds.id: ds.name for ds in _datasets})
dataset_name_to_id = {d.name: d.id for d in _datasets}
image_ids = []
image_ids_gt2pred = {}
image_id_2_image_info = {img.id: img for ds in _datasets for img in api.image.get_list(ds.id)}

# CM is of shape [GT x Pred]
NONE_CLS = "None"
cm_categories = list(map(lambda x: x["name"], categories)) + [NONE_CLS]


df = pd.read_csv("df.csv", index_col=0)

with open("metrics.pkl", "rb") as f:
    confusion_matrix, stats, coco = pickle.load(f)
    overall_coco, per_dataset_coco, per_class_coco = coco
    overall_stats, per_dataset_stats, per_class_stats = stats

with open("image_ids_gt2pred.pkl", "rb") as f:
    image_ids_gt2pred = pickle.load(f)

# selected
selected_classes = ["car", "person"]
categories_selected = [c for c in categories if c["name"] in selected_classes]
cm_categories_selected = selected_classes + [NONE_CLS]
