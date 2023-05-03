import os
import json
from collections import defaultdict
import numpy as np
from itertools import chain
import numpy as np

from dotenv import load_dotenv
import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.widgets import (
    ConfusionMatrix,
    Container,
    Card,
    SelectDataset,
    Button,
    MatchDatasets,
    Field,
    MatchObjClasses,
    Input,
    Table,
    Tabs,
    NotificationBox,
    GridGallery,
    Text,
    Checkbox,
    Switch,
)

from src import utils, dataset_matching
from src import globals as g
from src.globals import *


### 1. Select Datasets
select_dataset_gt = SelectDataset(project_id=project_id_gt, multiselect=True)
select_dataset_gt._all_datasets_checkbox.check()
select_dataset_pred = SelectDataset(project_id=project_id_pred, multiselect=True)
select_dataset_pred._all_datasets_checkbox.check()
card_gt = Card(
    "Ground Truth Project", "Select project with ground truth labels", content=select_dataset_gt
)
card_pred = Card(
    "Prediction Project", "Select project with predicted labels", content=select_dataset_pred
)
select_dataset_btn = Button("Select Datasets")
select_datasets_container = Container([card_gt, card_pred], "horizontal", gap=15)

change_datasets_btn = Button("Change Datasets", "info", "small", plain=True)

### 2. Match Datasets
match_datasets = MatchDatasets()
match_datasets_btn = Button("Match")
match_datasets_warn = NotificationBox(
    "Not matched.",
    "Datasets don't match. Please, check your dataset/image names. They must match",
    box_type="warning",
)
match_datasets_container = Container([match_datasets, match_datasets_btn, match_datasets_warn])
match_datasets_card = Card(
    "Match datasets",
    "Datasets and their images are compared by name. Only matched pairs of images are used in metrics.",
    True,
    match_datasets_container,
)


@match_datasets_btn.click
def on_match_datasets():
    match_datasets_warn.hide()
    project_id_gt = select_dataset_gt._project_selector.get_selected_id()
    project_id_pred = select_dataset_pred._project_selector.get_selected_id()
    if project_id_gt is None or project_id_pred is None:
        raise Exception("Please, select a project and datasets")
    ds_gt = api.dataset.get_list(project_id_gt)
    ds_pred = api.dataset.get_list(project_id_pred)

    match_datasets.set(ds_gt, ds_pred, "GT datasets", "Pred datasets")
    ds_matching = match_datasets.get_stat()
    if len(dataset_matching.validate_dataset_match(ds_matching)) == 0:
        match_datasets_warn.show()
        return

    rematch_tags()

    match_tags_card.uncollapse()
    change_datasets_btn.show()
    match_datasets_btn.disable()
    card_gt.lock()
    card_pred.lock()


@change_datasets_btn.click
def on_change_datasets():
    reset_widgets()


### 3. Match Tags
match_tags_input = Input("_nn")
match_tags_rematch_btn = Button("Rematch tags", button_size="small")
match_tags_rematch_c = Container([match_tags_input, match_tags_rematch_btn])
match_tags_input_f = Field(
    match_tags_rematch_c,
    "Input suffix for Pred tags",
    "If there is no matching you want due to suffix (like '_nn'), you can input it manually.",
)
match_tags = MatchObjClasses(selectable=True)
match_tags_btn = Button("Select")
match_tags_notif_note = NotificationBox("Note:", box_type="info")
match_tags_notif_warn = NotificationBox("Not selected.", box_type="warning")
match_tags_container = Container(
    [match_tags_input_f, match_tags, match_tags_btn, match_tags_notif_note, match_tags_notif_warn]
)
match_tags_card = Card(
    "Select tags",
    "Choose tags/classes that will be used for metrics.",
    True,
    match_tags_container,
)


@match_tags_rematch_btn.click
def rematch_tags():
    g.tags_gt = sly.ProjectMeta.from_json(api.project.get_meta(g.project_id_gt)).obj_classes
    g.tags_pred = sly.ProjectMeta.from_json(api.project.get_meta(g.project_id_pred)).obj_classes
    g.suffix = match_tags_input.get_value()
    g.tags_pred_filtered = dataset_matching.filter_tags_by_suffix(g.tags_pred, g.suffix)
    match_tags.set(g.tags_gt, g.tags_pred_filtered, "GT tags", "Pred tags", suffix=g.suffix)


@match_tags_btn.click
def on_select_tags():
    match_tags_notif_note.hide()
    match_tags_notif_warn.hide()
    selected_tags = match_tags.get_selected()
    selected_tags_matched = list(
        filter(lambda x: x[0] is not None and x[1] is not None, selected_tags)
    )
    if not g.is_tags_selected:
        if selected_tags_matched:
            match_tags_notif_note.description = (
                f"{len(selected_tags_matched)} matched tags will be used for metrics."
            )
            match_tags_notif_note.show()
            match_tags_btn.text = "Reselect tags"
            match_tags_btn._plain = True
            match_tags_btn._button_size = "small"
            match_tags_btn.update_data()
            DataJson().send_changes()
            metrics_card.uncollapse()
            metrics_btn.enable()
            match_tags_input.disable()
            match_tags_rematch_btn.disable()
            g.is_tags_selected = True
        else:
            match_tags_notif_warn.description = f"Please, select at least 1 matched tag."
            match_tags_notif_warn.show()
    else:
        metrics_card.collapse()
        metrics_btn.disable()
        match_tags_input.enable()
        match_tags_rematch_btn.enable()
        match_tags_btn.text = "Select"
        match_tags_btn._plain = False
        match_tags_btn._button_size = None
        match_tags_btn.update_data()
        DataJson().send_changes()
        g.is_tags_selected = False


### 4. Confusion Matrix & Metrics
task_notif_box = NotificationBox("Note:")
metrics_btn = Button("Calculate metrics")
confusion_matrix_widget = ConfusionMatrix()
metrics_overall_table = Table()
metrics_overall_table_f = Field(metrics_overall_table, "Overall project metrics")
metrics_per_class_table = Table()
metrics_per_class_table_f = Field(metrics_per_class_table, "Per-class metrics")
multilable_mode_switch = Switch(switched=False)
multilable_mode_text = Text(
    "<i style='color:gray;'>(more details in <a href='https://ecosystem.supervise.ly/apps/classification-metrics#Confusion-Matrix-implementation-details-for-multi-label-task' target='_blank'>Readme</a>)</i>"
)
multilable_mode_desc = Container([multilable_mode_text, multilable_mode_switch])
multilable_mode_switch_f = Field(
    multilable_mode_desc,
    "Count all combinations for misclassified tags",
    "Turning it on, you may get more insights about which classes the model most often confuses, "
    "but the values in the table will not actually mean the number of misclassified images, but rather misclassified tags.",
)
metrics_tab_confusion_matrix = Container(
    [confusion_matrix_widget, multilable_mode_switch_f], gap=20
)
metrics_tabs = Tabs(
    ["Confusion matrix", "Per class", "Overall"],
    [metrics_tab_confusion_matrix, metrics_per_class_table_f, metrics_overall_table_f],
)
metrics_card = Card(
    "Confusion Matrix & Metrics",
    "",
    collapsable=True,
    content=metrics_tabs,
)
metrics_per_image = Table()
per_image_notification_box = NotificationBox(
    title="Table for clicked datapoint from Confusion Matrix", description=""
)
card_per_image_table = Card(
    title="Per image metrics",
    description="Click on table row to preview image",
    collapsable=True,
    content=Container([per_image_notification_box, metrics_per_image], gap=5),
)
current_image_tag = Text()
images_gallery = GridGallery(
    2, show_opacity_slider=False, enable_zoom=True, resize_on_zoom=True, sync_views=True
)
card_img_preview = Card(
    title="Image preview",
    description="Ground Truth (left) and Prediction (right)",
    collapsable=True,
    content=Container([current_image_tag, images_gallery], gap=5),
)


@metrics_btn.click
def on_metrics_click():
    ds_matching = match_datasets.get_stat()
    selected_tags = match_tags.get_selected()


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    cls_gt = cell.row_name
    cls_pred = cell.column_name
    images_gallery.show()
    card_img_preview.uncollapse()


@multilable_mode_switch.value_changed
def on_mode_changed(is_checked):
    on_metrics_click()


@metrics_per_image.click
def select_image_row(cell: Table.ClickedDataPoint):
    image_name = cell.row["NAME"]
    set_img_to_gallery(image_name)


def get_sorted_image_tags(img_info, tag_meta_collection):
    img_tags = []
    for tag_json in img_info.tags:
        if "name" not in tag_json.keys():
            sly_id = tag_json["tagId"]
            for tag_meta in tag_meta_collection:
                tag_meta: sly.TagMeta
                if tag_meta.sly_id == sly_id:
                    tag_json["name"] = tag_meta.name
                    break
        tag = sly.Tag.from_json(tag_json, tag_meta_collection)
        img_tags.append(tag)

    # sort all tags with values
    img_tags_values = [tag for tag in img_tags if tag.meta.value_type == "any_number"]
    img_tags_values = sorted(img_tags_values, key=lambda item: item.value, reverse=True)
    return img_tags_values + [tag for tag in img_tags if tag.meta.value_type != "any_number"]


def set_img_to_gallery(image_name):
    images_gallery.loading = True
    current_image_tag.text = f"Image: {image_name}"

    img_info_gt = g.img_name_2_img_info_gt[image_name]
    img_info_pred = g.img_name_2_img_info_pred[image_name]

    img_tags_gt = get_sorted_image_tags(img_info_gt, g.tags_gt)
    img_tags_pred = get_sorted_image_tags(img_info_pred, g.tags_pred)

    images_for_preview = utils.get_preview_image_pair(
        img_info_gt,
        img_info_pred,
        img_tags_gt,
        img_tags_pred,
        g.is_multilabel,
    )

    images_gallery.clean_up()
    for current_image in images_for_preview:
        images_gallery.append(image_url=current_image["url"], title=current_image["title"])

    images_gallery.loading = False


def reset_widgets():
    change_datasets_btn.hide()
    card_gt.unlock()
    card_pred.unlock()
    match_datasets_btn.enable()
    match_tags_btn.enable()
    match_datasets.set()
    match_tags.set()
    match_tags_card.collapse()
    match_tags_notif_note.hide()
    match_tags_notif_warn.hide()
    metrics_btn.disable()
    task_notif_box.hide()
    confusion_matrix_widget.hide()
    metrics_overall_table.hide()
    metrics_per_class_table.hide()
    metrics_per_image.hide()
    images_gallery.hide()
    metrics_card.collapse()
    card_per_image_table.collapse()
    card_img_preview.collapse()
    match_tags_input.enable()
    g.is_tags_selected = False
    match_tags_btn.text = "Select"
    match_tags_btn._plain = False
    match_tags_btn._button_size = None
    match_tags_btn.update_data()
    DataJson().send_changes()
    match_tags_rematch_btn.enable()
    match_datasets_warn.hide()
    multilable_mode_switch.off()
    multilable_mode_switch_f.hide()


### FINAL APP
final_container = Container(
    [
        select_datasets_container,
        change_datasets_btn,
        match_datasets_card,
        match_tags_card,
        metrics_btn,
        task_notif_box,
        metrics_card,
        Container(
            [card_per_image_table, card_img_preview], direction="horizontal", fractions=[5, 5]
        ),
    ],
    gap=15,
)
app = sly.Application(final_container)
reset_widgets()
