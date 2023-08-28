import supervisely as sly
from supervisely.app.widgets import Container

from src import utils, dataset_matching, ui_match
from src import globals as g
from src.globals import *


@ui_match.match_tags_btn.click
def on_select_tags():
    ui_match.on_select_tags()


### FINAL APP
final_container = Container(
    [
        ui_match.select_datasets_container,
        ui_match.change_datasets_btn,
        ui_match.match_datasets_card,
        ui_match.match_tags_card,
    ],
    gap=15,
)
app = sly.Application(final_container)
