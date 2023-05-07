import supervisely as sly
from supervisely.app.widgets import Container, Button, NotificationBox

from src import ui_match, ui_metrics, calculate_metrics_f
from src import globals as g

calculate_metrics_button = Button("Calculate metrics")
warn_ann_not_collected = NotificationBox(
    "Annotations are not collected.",
    "Can't collect at leaset one annotation for GT and Pred. Please select another classes.",
    "warning",
)


@calculate_metrics_button.click
def on_calculate_metrics():
    ui_metrics.reset_widgets()
    warn_ann_not_collected.hide()
    g.init_globals_for_metrics()
    r = calculate_metrics_f.calculate_metrics(g.ds_match)
    if r is None:
        sly.logger.warn("Annotations are not collected")
        warn_ann_not_collected.show()
        on_select_tags()
        return
    calculate_metrics_f.set_globals(*r)
    ui_metrics.setup_widgets()
    ui_metrics.tabs_card.uncollapse()


@ui_match.match_tags_btn.click
def on_select_tags():
    ui_match.on_select_tags()
    if g.is_classes_selected:
        calculate_metrics_button.enable()
    else:
        calculate_metrics_button.disable()


@ui_match.change_datasets_btn.click
def reset_widgets():
    ui_match.reset_widgets()
    ui_metrics.reset_widgets()
    calculate_metrics_button.disable()
    warn_ann_not_collected.hide()


reset_widgets()

### FINAL APP
final_container = Container(
    [
        ui_match.select_datasets_container,
        ui_match.change_datasets_btn,
        ui_match.match_datasets_card,
        ui_match.match_tags_card,
        calculate_metrics_button,
        warn_ann_not_collected,
        ui_metrics.tabs_card,
        ui_metrics.inspect_card,
        ui_metrics.preview_container,
    ],
    gap=15,
)
app = sly.Application(final_container)
