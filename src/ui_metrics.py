import pandas as pd
import supervisely as sly

from src import globals as g

from . import utils
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
    Switch,
    Field,
)


### Confusion Matrix
confusion_matrix_widget = ConfusionMatrix([[0]])


### Overall
overall_table = Table()
overall_table_f = Field(overall_table, "Overall metrics")
overall_per_dataset_table = Table()
overall_per_dataset_table_f = Field(overall_per_dataset_table, "Per-dataset metrics")
overall_tab = Container([overall_table_f, overall_per_dataset_table_f])


### Per-class
per_class_table = Table()
per_class_tab = Container([per_class_table])


### Tabs
metrics_tabs = Tabs(
    ["Confusion Matrix", "Overall", "Per-class"],
    [confusion_matrix_widget, overall_tab, per_class_tab],
)
tabs_card = Card(
    "Confusion Matrix & Metrics",
    collapsable=True,
    content=metrics_tabs,
)

### Per-image table
inspect_table = Table()
inspect_info = Text("", "info")
inspect_container = Container([inspect_info, inspect_table])
inspect_card = Card("Image list", content=inspect_container, collapsable=False)

### Preview image
preview_gallery = GridGallery(2, enable_zoom=True, sync_views=True, show_preview=True)
preview_image_name = Text("", "info")
preview_image_container = Container([preview_image_name, preview_gallery])
card_gallery = Card("Image preview", content=preview_image_container, collapsable=False)

### Preview table
switch_show_all_annotations = Switch()
switch_show_all_annotations_f = Field(
    switch_show_all_annotations,
    "Show all annotations",
    "By default the annotations are filtered by selected classes. Turn on to show all annotations for image instead.",
)
preview_table_info = NotificationBox(
    "", "Click on the table below to see only clicked annotations."
)
preview_table = Table()
preview_table_container = Container(
    [switch_show_all_annotations_f, preview_table_info, preview_table]
)
card_img_table = Card("Annotation list", content=preview_table_container, collapsable=False)

### preview container
preview_container = Container([card_gallery, card_img_table], "horizontal", fractions=[8, 5])


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    # show table with this confusion (per-object)
    gt_class = cell.row_name
    pred_class = cell.column_name
    df_match = g.df[(g.df["gt_class"] == gt_class) & (g.df["pred_class"] == pred_class)]
    t = utils.per_image_metrics_table(
        df_match, g.dataset_names_gt, g.image_id_2_image_info, g.NONE_CLS
    )
    inspect_table.read_pandas(t)
    inspect_info.text = (
        f'Showing images with ground truth class "{gt_class}" and predicted class "{pred_class}".'
    )
    g.clicked_confusion_pair = [gt_class, pred_class]


@per_class_table.click
def on_per_class_table_click(cell: Table.ClickedDataPoint):
    selected_class = cell.row["class"]
    df_match = g.df[(g.df["gt_class"] == selected_class) | (g.df["pred_class"] == selected_class)]
    t = utils.per_image_metrics_table(
        df_match, g.dataset_names_gt, g.image_id_2_image_info, g.NONE_CLS
    )
    inspect_table.read_pandas(t)
    inspect_info.text = (
        f'Showing images containing class "{selected_class}" either in ground truth or predictions.'
    )
    g.clicked_confusion_pair = selected_class


@overall_per_dataset_table.click
def on_overall_per_dataset_table_click(cell: Table.ClickedDataPoint):
    dataset_id = cell.row["dataset_id"]
    dataset_name = cell.row["dataset"]
    df_match = g.df[(g.df["dataset_id"] == dataset_id)]
    t = utils.per_image_metrics_table(
        df_match, g.dataset_names_gt, g.image_id_2_image_info, g.NONE_CLS
    )
    inspect_table.read_pandas(t)
    inspect_info.text = f'Showing images for dataset "{dataset_name}".'
    g.clicked_confusion_pair = None


@overall_table.click
def on_overall_table_click(cell: Table.ClickedDataPoint):
    t = utils.per_image_metrics_table(g.df, g.dataset_names_gt, g.image_id_2_image_info, g.NONE_CLS)
    inspect_table.read_pandas(t)
    inspect_info.text = f"Showing all images."
    g.clicked_confusion_pair = None


@inspect_table.click
def on_inspect_table_click(cell: Table.ClickedDataPoint):
    image_id = cell.row["image_id"]
    g.clicked_image_id = image_id
    g.current_info = get_image_infos(image_id)
    can_switch = g.clicked_confusion_pair is not None
    show_all_classes = can_switch & switch_show_all_annotations.is_switched()
    show_preview(image_id, g.clicked_confusion_pair, show_all_classes)

    if can_switch:
        switch_show_all_annotations_f.show()
    else:
        switch_show_all_annotations_f.hide()
    preview_image_name.text = f"Selected image: \"{cell.row['image_name']}\""
    preview_table_container.show()


@preview_table.click
def on_preview_table_click(cell: Table.ClickedDataPoint):
    index = cell.row["index"]
    row = g.current_preview_table.iloc[index]
    gt_label_id = row["gt_label_ids"]
    pred_label_id = row["pred_label_ids"]
    image_id_gt, image_id_pred, img_info_gt, img_info_pred, ann_gt, ann_pred = g.current_info
    update_image_preview(
        img_info_gt, img_info_pred, ann_gt, ann_pred, [gt_label_id], [pred_label_id]
    )


@switch_show_all_annotations.value_changed
def on_switch_all_annotations(is_switched):
    if g.clicked_image_id:
        show_preview(g.clicked_image_id, g.clicked_confusion_pair, is_switched)


def show_preview(image_id: int, confusion_pair, show_all_classes: bool):
    show_all_classes = show_all_classes or confusion_pair is None
    df = g.df
    if show_all_classes:
        df_match = df[df["image_id"] == image_id]
    else:
        if isinstance(confusion_pair, list):
            gt_class, pred_class = confusion_pair
            df_match = df[
                (df["image_id"] == image_id)
                & (df["gt_class"] == gt_class)
                & (df["pred_class"] == pred_class)
            ]
        elif isinstance(confusion_pair, str):
            selected_class = confusion_pair
            df_match = df[
                (df["image_id"] == image_id)
                & ((df["gt_class"] == selected_class) | (df["pred_class"] == selected_class))
            ]
    t, gt_label_ids, pred_label_ids = utils.per_object_metrics(df_match, g.NONE_CLS)
    t["index"] = list(range(len(t.index)))
    preview_table.read_pandas(t)
    g.current_preview_table = t
    g.current_preview_table["gt_label_ids"] = gt_label_ids
    g.current_preview_table["pred_label_ids"] = pred_label_ids

    image_id_gt, image_id_pred, img_info_gt, img_info_pred, ann_gt, ann_pred = g.current_info
    if show_all_classes:
        update_image_preview(img_info_gt, img_info_pred, ann_gt, ann_pred)
    else:
        update_image_preview(
            img_info_gt, img_info_pred, ann_gt, ann_pred, gt_label_ids, pred_label_ids
        )


def update_image_preview(
    img_info_gt: sly.ImageInfo,
    img_info_pred: sly.ImageInfo,
    ann_gt: sly.Annotation,
    ann_pred: sly.Annotation,
    gt_label_ids=None,
    pred_label_ids=None,
):
    preview_gallery.loading = True
    preview_gallery.clean_up()
    if gt_label_ids is not None:
        labels_gt = [
            ann_gt.get_label_by_id(label_id) for label_id in gt_label_ids if label_id != -1
        ]
        ann_gt = sly.Annotation(ann_gt.img_size, labels_gt)
    if pred_label_ids is not None:
        labels_pred = [
            ann_pred.get_label_by_id(label_id) for label_id in pred_label_ids if label_id != -1
        ]
        ann_pred = sly.Annotation(ann_pred.img_size, labels_pred)

    preview_gallery.append(img_info_gt.preview_url, ann_gt, title="Ground Truth")
    preview_gallery.append(img_info_pred.preview_url, ann_pred, title="Predicted")
    preview_gallery.loading = False


def get_image_infos(image_id_gt):
    img_info_gt = g.image_id_2_image_info[image_id_gt]
    image_id_pred = g.image_ids_gt2pred[image_id_gt]
    img_info_pred = g.api.image.get_info_by_id(image_id_pred)
    ann_gt = sly.Annotation.from_json(
        g.api.annotation.download_json(image_id_gt), g.project_meta_gt
    )
    ann_pred = sly.Annotation.from_json(
        g.api.annotation.download_json(image_id_pred), g.project_meta_pred
    )
    return image_id_gt, image_id_pred, img_info_gt, img_info_pred, ann_gt, ann_pred


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
    inspect_card.show()
    preview_container.show()


def reset_widgets():
    tabs_card.collapse()
    inspect_card.hide()
    preview_container.hide()
    preview_table_container.hide()
    inspect_info.text = "Click on the table above to show corresponding image list."
    preview_image_name.text = "Click on the table above to show the image."
    preview_gallery.clean_up()
    preview_table.read_pandas(pd.DataFrame())
    inspect_table.read_pandas(pd.DataFrame())
    g.clicked_image_id = None
    g.clicked_confusion_pair = None
