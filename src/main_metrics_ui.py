import pandas as pd
import numpy as np
import supervisely as sly

from src import globals as g
from src import ui_metrics

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

### FINAL APP
final_container = Container(
    [ui_metrics.tabs_card, ui_metrics.inspect_card, ui_metrics.preview_container],
    gap=15,
)
app = sly.Application(final_container)
ui_metrics.reset_widgets()
ui_metrics.setup_widgets()


# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
