<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/97401023/216125503-54544e16-a36e-453f-9b23-ba7729adb3e8.png" />


# Classification metrics

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Use">How to use</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Screenshot">Screenshot</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/classification-metrics)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/classification-metrics)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/classification-metrics.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/classification-metrics.png)](https://supervise.ly)

</div>

# Overview
This app is an interactive visualization for classification metrics. It allows to estimate performance of a model by comparing target annotations with predictions. The app supports single-label and multi-label classification tasks and currently supports only image tags.

Application key points:

- Available single-label and multi-label classification tasks (that is auto-detected)
- Supports image tags
- Supports matching tags with suffixes
- Supports top-k predictions for single-label classification task
- You can explore multiclass confusion matrix, per class, per image and overall metrics.

# How to use
0. Be sure that you have two projects with the same dataset names and image names to compare. Unmatched datasets and image names will not be included to metrics calculation. Also be sure that images contain labels - image tags. One label should be on images from target project for single-label classification task and more than one for multi-label task. Images without tags in ground truth project will not be included to metrics calculation.
1. Launch the app. Select two projects labeled by image tags at first step: with ground truth labels and with model predictions.
2. The app matches images in both projects (step 2) and produces detailed report about matched pairs, allowing to validate the data.
3. Then you can match classes and select that will be used in metrics calculation at step 3. If your tags from prediction project have a suffix (e.g. "_nn") that was auto-generated after the [Apply Classifier App](https://ecosystem.supervise.ly/apps/apply-classification-model-to-project), you should make sure that they are matched too. If something went wrong and some tags weren't matched, adjust the suffix field and try again.
4. Click "Calculate" and explore the confusion matrix below (step 4). Click on the cells to see the per image stats for any interesting cases, where the model gets right or wrong. You can display images with target and predicted tags by click on rows of per images stats table.
5. Check the tabs "Overall" and "Per class" to see the metrics like precision, recall, f1-score and others.


## Confusion Matrix implementation details for multi-label task

Visualizing the Confusion Matrix for a multi-label task is ambiguous. For example in `scikit-learn` there is a [multilabel_confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html) method that returns a 3D `multi_confusion` matrix which can not be visualized in a table.

**What's the problem?**

Consider an example: we have an image with GT classes ["dog", "cat"] and predictions ["bird"]. There are two ways how to interpret this case and thus have two implementations in the app.

1. We might say the model haven't found the dog and the cat, and have wrongly detected a bird. This way the dog and cat are False Negatives and the bird is False Positive and we add it to "None" in confusion matrix:

    <img src="https://user-images.githubusercontent.com/31512713/219029394-ce851e44-6080-41f8-a58f-8ee48ba85a26.png" width=500/>

    This way only the diagonal (True Positives) values and None values will be non zero in the matrix. This may not be too useful for the model evaluation.


2. There is another interpretation. We might say the model have confused a bird with either the dog or the cat. But how to determine exactly which class was confused with which? There is no obvious way. So, we can only treat that the both dog and cat were confused with a bird. And the matrix will be as follows:

    <img src="https://user-images.githubusercontent.com/31512713/219029449-d732004c-5d7d-4c93-b839-51cda52859e1.png" width=500/>

    This way we will be adding excessively many misclassified values (especially in cases when you are working with a large number of classes. e.g: the model predicted 3 wrong classes and in GT were 3 another classes, there will be 3x3=9 misclassified values). So, use it carefully!


# Related apps

1. [Train MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/train) app to train classification model on your data 
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/train" src="https://i.imgur.com/mXG6njU.png" width="350px" style='padding-bottom: 10px'/>

2. [Serve MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/serve) app to load classification model to be applied to your project
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/serve" src="https://i.imgur.com/CU8XHdQ.png" width="350px" style='padding-bottom: 10px'/>

3. [Apply Classifier to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/apply-classification-model-to-project) app to apply classification model to your project
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-classification-model-to-project" src="https://github.com/supervisely-ecosystem/apply-classification-model-to-project/releases/download/v0.0.1/app-name-descrition.png" width="350px" style='padding-bottom: 10px'/>

# Screenshot

<img src="https://user-images.githubusercontent.com/97401023/216125292-2968dd8a-7e50-4c21-9f31-ebec4116b3f4.png" />
