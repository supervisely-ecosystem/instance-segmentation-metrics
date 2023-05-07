import pandas as pd
import supervisely as sly
from dotenv import load_dotenv
import os


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = 20645
project_id_pred = 20644

# Interactive
is_classes_selected = False
clicked_image_id = None
clicked_confusion_pair = None
current_preview_table = None
current_info = None


# meta
def set_globals_meta():
    global project_meta_gt, project_meta_pred
    project_meta_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt))
    project_meta_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_id_pred))


if project_id_gt and project_id_pred:
    set_globals_meta()


image_ids = []
image_ids_gt2pred = {}
dataset_ids_matched = []
selected_classes = []  # ["car", "person"]
used_classes = []
cm_used_classes = []
ds_match = None

# calculated metrics
df = None
confusion_matrix = None
overall_stats, per_dataset_stats, per_class_stats = [None] * 3
overall_coco, per_dataset_coco, per_class_coco = [None] * 3


NONE_CLS = "None"


def init_globals_for_metrics():
    # has to be called before metrics calculation
    # needs: project_id_gt, project_meta_gt, selected_classes
    global categories, category_name_to_id, cm_categories, dataset_ids, dataset_names_gt, dataset_name_to_id, image_id_2_image_info, categories_selected, cm_categories_selected

    categories = [{"id": i, "name": cls_name} for i, cls_name in enumerate(selected_classes)]
    category_name_to_id = {c["name"]: c["id"] for c in categories}
    cm_categories = list(map(lambda x: x["name"], categories)) + [NONE_CLS]

    _datasets = api.dataset.get_list(project_id_gt)
    dataset_ids = [ds.id for ds in _datasets]
    dataset_names_gt = pd.Series({ds.id: ds.name for ds in _datasets})
    dataset_name_to_id = {d.name: d.id for d in _datasets}
    image_id_2_image_info = {img.id: img for ds in _datasets for img in api.image.get_list(ds.id)}

    # categories_selected = [c for c in categories if c["name"] in selected_classes]
    # cm_categories_selected = selected_classes + [NONE_CLS]
