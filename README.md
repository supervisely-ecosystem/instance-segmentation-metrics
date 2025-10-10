<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/115161827/236404883-d3b880bb-d2ac-4409-aad5-b8fd53285de2.jpg" />


# Instance Segmentation Metrics

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Use">How to Use</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Screenshot">Screenshot</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/instance-segmentation-metrics)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/instance-segmentation-metrics)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/instance-segmentation-metrics.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/instance-segmentation-metrics.png)](https://supervisely.com)

</div>

# Overview
This app calculates the instance segmentation metrics and allows you to explore and understand the behavior of your model with interactive visualizations. It is achieved by comparing ground truth annotations with predictions.

# Key features:

- Calculate **mAP** (mean Average Precision) and **AR** (Average Recall) metrics the same way as it's calculated in COCO challenge.
- Calculate **IoU** to evaluate the mask accuracy.
- Calculate **Precision** and **Recall** to assess classification correctness.
- Interact with the app, explore Confusion Matrix, per-class and per-image metrics to get more insights about your instance segmentation model.


# How to use

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


# Screenshot

<img src="https://user-images.githubusercontent.com/31512713/236821033-796e528e-a859-4394-9b2b-91919884d617.png" />


# Related apps

 - [Train MMDetection](../../../../../supervisely-ecosystem/mmdetection/train) - app to train an instance segmentation model on your data
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection/train" src="https://user-images.githubusercontent.com/115161827/236882770-782f2797-cd2e-46f4-849c-2c4f6a5d3e69.png" width="350px" style='padding-bottom: 10px'/>

 - [Train Detectron2](../../../../../../supervisely-ecosystem/detectron2/supervisely/train) - app to train an instance segmentation model on your data
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/train" src="https://user-images.githubusercontent.com/115161827/236882766-e7f4a518-44f0-42df-8f7a-e3d2eae7099f.png" width="350px" style='padding-bottom: 10px'/>

 - [Serve MMDetection](../../../../../supervisely-ecosystem/mmdetection/serve) - app to host a model that will be applied to your projects
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection/serve" src="https://user-images.githubusercontent.com/115161827/236882764-c5120744-50b2-4d82-be19-9a8825ff0d46.png" width="350px" style='padding-bottom: 10px'/>

 - [Serve Detectron2](../../../../../../../supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve) - app to host a model that will be applied to your projects
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve" src="https://user-images.githubusercontent.com/115161827/236882762-96d56c5f-33d4-4c28-8feb-21b245083989.png" width="350px" style='padding-bottom: 10px'/>

 - [Apply NN to Images Project](../../../../../supervisely-ecosystem/nn-image-labeling/project-dataset) app to apply the served model to your project with images
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://user-images.githubusercontent.com/115161827/236882758-7378d1e6-88b7-461a-bdc2-3a625f7fc21a.png" width="350px" style='padding-bottom: 10px'/>

 - [Apply NN to Videos Project](../../../../supervisely-ecosystem/apply-nn-to-videos-project) app to apply the served model to your project with videos
 <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://user-images.githubusercontent.com/115161827/236882750-98f53565-708c-4414-9554-98daf4b834eb.png" width="350px" style='padding-bottom: 10px'/>
