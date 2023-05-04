import pandas as pd
import numpy as np
import supervisely as sly

from src import globals as g

from . import utils
from .data_iterator import DataIteratorAPI
from supervisely.app import DataJson
from supervisely.app.widgets import (
    ConfusionMatrix,
    Container,
    Card,
    Table,
    Tabs,
    NotificationBox,
    GridGallery,
    Text,
    Checkbox,
    Switch,
)


### Confusion Matrix
confusion_matrix_widget = ConfusionMatrix()


### Overall
overall_table = Table()
overall_per_dataset_table = Table()
overall_tab = Container([overall_table, overall_per_dataset_table])


### Per-class
per_class_table = Table()
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


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    # show table with this confusion (per-object)
    gt_class = cell.row_name
    pred_class = cell.column_name
    df_match = g.df[(g.df["gt_class"] == gt_class) & (g.df["pred_class"] == pred_class)]
    t = utils.per_image_metrics_table(df_match, g.dataset_names_gt, g.NONE_CLS)
    inspect_table.read_pandas(t)
    inspect_card.uncollapse()


@per_class_table.click
def on_per_class_table_click(cell: Table.ClickedDataPoint):
    selected_class = cell.row["class"]
    df_match = g.df[(g.df["gt_class"] == selected_class) | (g.df["pred_class"] == selected_class)]
    t = utils.per_image_metrics_table(df_match, g.dataset_names_gt, g.NONE_CLS)
    inspect_table.read_pandas(t)
    inspect_card.uncollapse()


@overall_per_dataset_table.click
def on_overall_per_dataset_table_click(cell: Table.ClickedDataPoint):
    dataset_id = cell.row["dataset_id"]
    df_match = g.df[(g.df["dataset_id"] == dataset_id)]
    t = utils.per_image_metrics_table(df_match, g.dataset_names_gt, g.NONE_CLS)
    inspect_table.read_pandas(t)
    inspect_card.uncollapse()


@overall_table.click
def on_overall_table_click(cell: Table.ClickedDataPoint):
    t = utils.per_image_metrics_table(g.df, g.dataset_names_gt, g.NONE_CLS)
    inspect_table.read_pandas(t)
    inspect_card.uncollapse()


@inspect_table.click
def on_preview_table_click(cell: Table.ClickedDataPoint):
    image_id = cell.row["image_id"]
    show_preview(image_id)
    card_gallery.uncollapse()
    card_img_table.uncollapse()


def show_preview(image_id):
    preview_gallery.loading = True
    preview_gallery.clean_up()
    image_info = g.image_id_2_image_info[image_id]
    image_id_pred = g.image_ids_gt2pred[image_id]
    img_pred = g.api.image.get_info_by_id(image_id_pred)
    ann_gt = sly.Annotation.from_json(g.api.annotation.download_json(image_id), g.project_meta_gt)
    ann_pred = sly.Annotation.from_json(
        g.api.annotation.download_json(image_id_pred), g.project_meta_pred
    )
    preview_gallery.append(image_info.preview_url, ann_gt, title="Ground Truth")
    preview_gallery.append(img_pred.preview_url, ann_pred, title="Predicted")
    preview_gallery.loading = False

    t = utils.per_object_metrics(g.df, image_id)
    t.loc[t["gt_class"] == g.NONE_CLS, "gt_class"] = ""
    t.loc[t["pred_class"] == g.NONE_CLS, "pred_class"] = ""
    preview_table.read_pandas(t)


### Set up widgets
def setup_widgets():
    confusion_matrix_df = pd.DataFrame(
        g.confusion_matrix, columns=g.cm_used_classes, index=g.cm_used_classes
    )
    confusion_matrix_widget._update_matrix_data(confusion_matrix_df)
    confusion_matrix_widget.update_data()
    DataJson().send_changes()

    t = utils.collect_overall_metrics(g.overall_stats, g.overall_coco)
    overall_table.read_pandas(pd.DataFrame(t))

    t = utils.collect_per_dataset_metrics(
        g.per_dataset_stats, g.per_dataset_coco, g.dataset_ids_matched, g.dataset_names_gt
    )
    overall_per_dataset_table.read_pandas(pd.DataFrame(t))

    categories = [
        {"id": g.category_name_to_id[cls_name], "name": cls_name} for cls_name in g.used_classes
    ]
    t = utils.collect_per_class_metrics(g.per_class_stats, g.per_class_coco, categories)
    per_class_table.read_pandas(pd.DataFrame(t))

    tabs_card.uncollapse()


def reset_widgets():
    tabs_card.collapse()
    inspect_card.collapse()
    card_gallery.collapse()
    card_img_table.collapse()
