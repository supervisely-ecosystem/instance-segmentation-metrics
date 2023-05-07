from supervisely.app import DataJson
from supervisely.app.widgets import (
    Container,
    Card,
    SelectDataset,
    Button,
    MatchDatasets,
    Field,
    MatchObjClasses,
    Input,
    NotificationBox,
)

from src import dataset_matching
from src import globals as g


### 1. Select Datasets
select_dataset_gt = SelectDataset(project_id=g.project_id_gt, multiselect=True)
select_dataset_gt._all_datasets_checkbox.check()
select_dataset_pred = SelectDataset(project_id=g.project_id_pred, multiselect=True)
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


### 3. Match Tags
match_tags_input = Input("_nn")
match_tags_rematch_btn = Button("Rematch classes", button_size="small")
match_tags_rematch_c = Container([match_tags_input, match_tags_rematch_btn])
match_tags_input_f = Field(
    match_tags_rematch_c,
    "Input suffix for predicted class names",
    "If there is no matching you want due to suffix (like '_nn'), you can input it manually (Usually you don't need to change this filed).",
)
match_tags = MatchObjClasses(selectable=True)
match_tags_btn = Button("Select")
match_tags_notif_note = NotificationBox("Note:", box_type="info")
match_tags_notif_warn = NotificationBox("Not selected.", box_type="warning")
match_tags_container = Container(
    [match_tags_input_f, match_tags, match_tags_btn, match_tags_notif_note, match_tags_notif_warn]
)
match_tags_card = Card(
    "Select classes",
    "Choose classes that will be used for metrics.",
    True,
    match_tags_container,
)


@match_datasets_btn.click
def on_match_datasets():
    match_datasets_warn.hide()
    project_id_gt = select_dataset_gt._project_selector.get_selected_id()
    project_id_pred = select_dataset_pred._project_selector.get_selected_id()
    if project_id_gt is None or project_id_pred is None:
        raise Exception("Please, select a project and datasets")
    ds_gt = g.api.dataset.get_list(project_id_gt)
    ds_pred = g.api.dataset.get_list(project_id_pred)

    match_datasets.set(ds_gt, ds_pred, "GT datasets", "Pred datasets")
    ds_matching = match_datasets.get_stat()
    if len(dataset_matching.validate_dataset_match(ds_matching)) == 0:
        match_datasets_warn.show()
        return

    g.project_id_gt = project_id_gt
    g.project_id_pred = project_id_pred
    g.ds_match = ds_matching
    g.set_globals_meta()

    rematch_tags()

    match_tags_card.uncollapse()
    change_datasets_btn.show()
    match_datasets_btn.disable()
    card_gt.lock()
    card_pred.lock()


@match_tags_rematch_btn.click
def rematch_tags():
    classes_gt = g.project_meta_gt.obj_classes
    classes_pred = g.project_meta_pred.obj_classes

    # match suffix
    suffix = match_tags_input.get_value()
    classes_pred_filtered = dataset_matching.filter_classes_by_suffix(classes_pred, suffix)

    # filter not acceptable geometry types
    classes_pred_filtered = dataset_matching.filter_classes_by_shape(classes_pred_filtered)
    classes_gt = dataset_matching.filter_classes_by_shape(classes_gt)

    match_tags.set(
        classes_gt,
        classes_pred_filtered,
        "Ground Truth classes",
        "Prediction classes",
        suffix=suffix,
    )


def on_select_tags():
    match_tags_notif_note.hide()
    match_tags_notif_warn.hide()
    selected_tags = match_tags.get_selected()
    # selected classes will be a union between selected GT and Pred
    selected_tags = list(set([cls for pair in selected_tags for cls in pair if cls is not None]))
    if not g.is_classes_selected:
        if selected_tags:
            match_tags_notif_note.description = (
                f"{len(selected_tags)} classes will be used for metrics."
            )
            match_tags_notif_note.show()
            match_tags_btn.text = "Reselect classes"
            match_tags_btn._plain = True
            match_tags_btn._button_size = "small"
            match_tags_btn.update_data()
            DataJson().send_changes()
            match_tags_input.disable()
            match_tags_rematch_btn.disable()
            g.is_classes_selected = True
            g.selected_classes = selected_tags
        else:
            match_tags_notif_warn.description = f"Please, select at least 1 matched tag."
            match_tags_notif_warn.show()
    else:
        match_tags_input.enable()
        match_tags_rematch_btn.enable()
        match_tags_btn.text = "Select"
        match_tags_btn._plain = False
        match_tags_btn._button_size = None
        match_tags_btn.update_data()
        DataJson().send_changes()
        g.is_classes_selected = False


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
    match_tags_input.enable()
    g.is_classes_selected = False
    match_tags_btn.text = "Select"
    match_tags_btn._plain = False
    match_tags_btn._button_size = None
    match_tags_btn.update_data()
    DataJson().send_changes()
    match_tags_rematch_btn.enable()
    match_datasets_warn.hide()
