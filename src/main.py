import pandas as pd
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import os
import pickle

from . import utils
from .data_iterator import DataIteratorAPI
from supervisely.app import DataJson
from supervisely.app.widgets import (
    ConfusionMatrix,
    Container,
    Card,
    SelectDataset,
    Button,
    MatchDatasets,
    Field,
    MatchTagMetas,
    Input,
    Table,
    Tabs,
    NotificationBox,
    GridGallery,
    Text,
    Checkbox,
    Switch,
)


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = 20645
project_id_pred = 20644


# meta
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt))
categories = [{"id": i, "name": obj_cls.name} for i, obj_cls in enumerate(meta.obj_classes)]
category_name_to_id = {c["name"]: c["id"] for c in categories}
_datasets = api.dataset.get_list(project_id_gt)
dataset_ids_gt = [ds.id for ds in _datasets]
dataset_names_gt = pd.Series({ds.id: ds.name for ds in _datasets})
image_ids = []
image_id_2_image_info = {img.id: img for ds in _datasets for img in api.image.get_list(ds.id)}

# CM is of shape [GT x Pred]
NONE_CLS = "None"
cm_categories = list(map(lambda x: x["name"], categories)) + [NONE_CLS]


df = pd.read_csv("df.csv", index_col=0)


with open("metrics.pkl", "rb") as f:
    confusion_matrix, stats, coco = pickle.load(f)
    overall_coco, per_dataset_coco, per_class_coco = coco
    overall_stats, per_dataset_stats, per_class_stats = stats


### -------------------- ###


### Confusion Matrix
confusion_matrix_widget = ConfusionMatrix()


### Overall
t = utils.collect_overall_metrics(overall_stats, overall_coco)
overall_table = Table(t)
t = utils.collect_per_dataset_metrics(
    per_dataset_stats, per_dataset_coco, dataset_ids_gt, dataset_names_gt
)
overall_per_dataset_table = Table(t)
overall_tab = Container([overall_table, overall_per_dataset_table])


### Per-class
t = utils.collect_per_class_metrics(per_class_stats, per_class_coco, category_name_to_id)
per_class_table = Table(t)
per_class_tab = Container([per_class_table])


metrics_tabs = Tabs(
    ["Confusion Matrix", "Overall", "Per-class"],
    [confusion_matrix_widget, overall_tab, per_class_tab],
)
tabs_card = Card(
    "Confusion Matrix & Metrics",
    collapsable=True,
    content=metrics_tabs,
)

inspect_table = Table()
inspect_card = Card("Per-image table", content=inspect_table, collapsable=True)

preview_gallery = GridGallery(2, enable_zoom=True, sync_views=True)
preview_table = Table()
card_gallery = Card("Preview image", content=preview_gallery, collapsable=True)
card_img_table = Card("Preview table", content=preview_table, collapsable=True)
preview_container = Container([card_gallery, card_img_table], "horizontal")


# Set up confusion_matrix_widget
confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=cm_categories, index=cm_categories)
confusion_matrix_widget._update_matrix_data(confusion_matrix_df)
confusion_matrix_widget.update_data()
DataJson().send_changes()


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    # show table with this confusion (per-object)
    gt_class = cell.row_name
    pred_class = cell.column_name
    df_match = df[(df["gt_class"] == gt_class) & (df["pred_class"] == pred_class)]
    t = utils.per_image_metrics_table(df_match, dataset_names_gt, NONE_CLS)
    inspect_table.read_pandas(t)


@per_class_table.click
def on_per_class_table_click(cell: Table.ClickedDataPoint):
    selected_class = cell.row["class"]
    df_match = df[(df["gt_class"] == selected_class) | (df["pred_class"] == selected_class)]
    t = utils.per_image_metrics_table(df_match, dataset_names_gt, NONE_CLS)
    inspect_table.read_pandas(t)


@overall_per_dataset_table.click
def on_overall_per_dataset_table_click(cell: Table.ClickedDataPoint):
    dataset_id = cell.row["dataset_id"]
    df_match = df[(df["dataset_id"] == dataset_id)]
    t = utils.per_image_metrics_table(df_match, dataset_names_gt, NONE_CLS)
    inspect_table.read_pandas(t)


@overall_table.click
def on_overall_table_click(cell: Table.ClickedDataPoint):
    t = utils.per_image_metrics_table(df, dataset_names_gt, NONE_CLS)
    inspect_table.read_pandas(t)


@inspect_table.click
def on_preview_table_click(cell: Table.ClickedDataPoint):
    image_id = cell.row["image_id"]
    show_preview(image_id)


def show_preview(image_id):
    preview_gallery.loading = True
    preview_gallery.clean_up()
    image_info = image_id_2_image_info[image_id]
    ann = sly.Annotation.from_json(api.annotation.download_json(image_id), meta)
    preview_gallery.append(image_info.preview_url, ann, title="Ground Truth")
    preview_gallery.append(image_info.preview_url, ann, title="Predicted")
    preview_gallery.loading = False

    t = utils.per_object_metrics(df, image_id)
    preview_table.read_pandas(t)


### FINAL APP
final_container = Container(
    [tabs_card, inspect_card, preview_container],
    gap=15,
)
app = sly.Application(final_container)


# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
