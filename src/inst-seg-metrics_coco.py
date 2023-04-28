from functools import partial
from pycocotools import mask, coco, cocoeval
import json
from copy import deepcopy
from typing import List

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import os

import utils
from data_iterator import DataIteratorAPI


def create_coco_annotation(mask_np, id, image_id, category_id, score=None):
    segmentation = mask.encode(np.asfortranarray(mask_np))
    segmentation["counts"] = segmentation["counts"].decode()
    annotation = {
        "id": id,  # unique id for each annotation
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "iscrowd": 0,
    }
    if score is not None:
        annotation["score"] = score
    return annotation


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = 20645
project_id_pred = 20644


data_iterator = DataIteratorAPI(api)

# meta
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt))
categories = [{"id": i, "name": obj_cls.name} for i, obj_cls in enumerate(meta.obj_classes)]
category_name_to_id = {c["name"]: c["id"] for c in categories}
images = []

# Collect GT annotations
annotations_gt = []
for i, image_item in enumerate(data_iterator.iterate_project_images(project_id_gt)):
    image_id = image_item.image_id

    image = {
        "id": image_id,
        "height": image_item.image_height,
        "width": image_item.image_width,
    }
    images.append(image)

    for item in image_item.labels_iterator:
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        category_id = category_name_to_id[item.label.obj_class.name]
        mask_np = utils.uncrop_bitmap(item.label.geometry, item.image_width, item.image_height)
        label_id = item.label.geometry.sly_id
        annotation = create_coco_annotation(mask_np, label_id, image_id, category_id)
        annotations_gt.append(annotation)

# Create COCO format dictionary
coco_dict = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": images,
    "annotations": annotations_gt,
}

# Save the annotations to a JSON file
with open("ground_truth.json", "w") as f:
    json.dump(coco_dict, f)

coco_gt = coco.COCO("ground_truth.json")
ann_ids = coco_gt.getAnnIds()
coco_gt = coco_gt.loadRes(coco_gt.loadAnns(ann_ids))

# Collect Pred annotations
annotations_pred = []
for i, image_item in enumerate(data_iterator.iterate_project_images(project_id_pred)):
    image_id = images[i]["id"]

    for item in image_item.labels_iterator:
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        category_id = category_name_to_id[item.label.obj_class.name]
        mask_np = utils.uncrop_bitmap(item.label.geometry, item.image_width, item.image_height)
        label_id = item.label.geometry.sly_id
        score = item.label.tags.get("confidence").value
        id = None
        annotation = create_coco_annotation(mask_np, label_id, image_id, category_id, score)
        annotations_pred.append(annotation)

coco_dt = coco_gt.loadRes(annotations_pred)

e = cocoeval.COCOeval(coco_gt, coco_dt)
e.params.areaRng = [e.params.areaRng[0]]
e.params.areaRngLbl = [e.params.areaRngLbl[0]]
e.params.imgIds
e.evaluate()

t = [x for x in e.evalImgs if x]
t = sorted(t, key=lambda x: x["image_id"])
t2 = {k: x for k, x in e.ious.items() if len(x)}

e.accumulate()
e.summarize()


# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
