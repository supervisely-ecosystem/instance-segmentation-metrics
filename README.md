<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/115161827/236404883-d3b880bb-d2ac-4409-aad5-b8fd53285de2.jpg" />


# Instance Segmentation Metrics

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Use">How to use</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Screenshot">Screenshot</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/instance-segmentation-metrics)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/instance-segmentation-metrics)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/instance-segmentation-metrics.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/instance-segmentation-metrics.png)](https://supervise.ly)

</div>

## Overview
This app calculates the instance segmentation metrics and allows you to explore and understand the behavior of your model with interactive visualizations. It is achieved by comparing ground truth annotations with predictions.

## Key features:

- Calculate **mAP** (mean Average Precision) and **AR** (Average Recall) metrics the same way as it's calculated in COCO challenge.
- Calculate **IoU** to evaluate the mask accuracy.
- Calculate **Precision** and **Recall** to assess the classification correctness.
- Interact with the app, explore Confusion Matrix, per-class and per-image metrics getting more insights about your instance segmentation model.


## How to use

**Preparing the data:**
- You need 2 projects, one with ground truth (GT) annotations and another with predicted masks.
- The app supports only `bitmap` labels as shape.
- All predicted labels must have a `confidence` tag (the model confidence score).

**Step-by-step tutorial:**
1. Launch the app.
2. Select two projects: one with **ground truth** labels and another with the model **predictions**.
3. **Match projects**. The app expects the projects will contain datasets and images with the **same names**. If something doesn't match, you will be notified.
4. **Match classes**. This step determines the correspondence between ground truth classes and predicted classes. **Select** the classes you need to include in evaluation and click **Match**. The app will only evaluate the classes with a `bitmap` shape. Also, if your predicted class names have a suffix (like `_nn`), you can specify this suffix to match the classes despite the difference in name endings (example: `cat` and `cat_nn`).
5. Click **Calculate** button. This will run the calculations and can take a minute.
6. Explore the model predictions with a **Confusion Matrix**, **Overall**, **Per-dataset** and **Per-class** metrics. All the tables and Confusion Matrix are **clickable**, so you can explore it as you wish!
7. Explore in depth:
    - Click on cells in the **Confusion Matrix** to see the images corresponding your query. This will populate the **Per-image table**.
    - Click on a row in the **Per-image table** to load the image with annotations and predictions.
    - Check the **Overall** and **Per-class** tabs to see the metrics aggregated in various ways. All these tables are clickable too.


# Related apps

1. [Train MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/train) app to train classification model on your data 
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/train" src="https://i.imgur.com/mXG6njU.png" width="350px" style='padding-bottom: 10px'/>

2. [Serve MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/serve) app to load classification model to be applied to your project
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/serve" src="https://i.imgur.com/CU8XHdQ.png" width="350px" style='padding-bottom: 10px'/>

3. [Apply Classifier to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/apply-classification-model-to-project) app to apply classification model to your project
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-classification-model-to-project" src="https://github.com/supervisely-ecosystem/apply-classification-model-to-project/releases/download/v0.0.1/app-name-descrition.png" width="350px" style='padding-bottom: 10px'/>

# Screenshot

<img src="https://user-images.githubusercontent.com/97401023/216125292-2968dd8a-7e50-4c21-9f31-ebec4116b3f4.png" />
