import supervisely as sly
from src import utils
from src.data_iterator import DataIteratorAPI, ImageLabelsIterator

from src import globals as g


def calculate_metrics(ds_match):
    # populates globals: g.df, g.image_ids, g.image_ids_gt2pred, g.dataset_ids_matched

    df_rows = []
    images_coco = []
    annotations_gt = []
    annotations_pred = []

    g.dataset_ids_matched = []
    g.image_ids = []
    g.image_ids_gt2pred = {}

    for i, (dataset_name, item) in enumerate(ds_match.items()):
        if item["dataset_matched"] != "both":
            continue
        dataset_id = g.dataset_name_to_id[dataset_name]
        g.dataset_ids_matched.append(dataset_id)
        for img_item in item["matched"]:
            img_gt, img_pred = img_item["left"], img_item["right"]
            img_gt: sly.ImageInfo
            image_id = img_gt.id
            g.image_ids.append(image_id)
            g.image_ids_gt2pred[img_gt.id] = img_pred.id

            labels_iter_gt = ImageLabelsIterator(
                g.api, img_gt.id, g.project_meta_gt, g.project_id_gt
            )
            labels_iter_pred = ImageLabelsIterator(
                g.api, img_pred.id, g.project_meta_pred, g.project_id_pred
            )
            image_item_gt = DataIteratorAPI.ImageItem(
                img_gt.id, labels_iter_gt.ann, g.project_id_gt, dataset_id, labels_iter_gt
            )
            image_item_pred = DataIteratorAPI.ImageItem(
                img_pred.id, labels_iter_pred.ann, g.project_id_pred, dataset_id, labels_iter_pred
            )

            # add DF rows
            new_rows = utils.collect_df_rows(
                image_item_gt, image_item_pred, g.NONE_CLS, g.category_name_to_id
            )
            df_rows += new_rows

            # add COCO annotations for pycocotools
            image_info, annotations = utils.collect_coco_annotations(
                image_item_gt, g.category_name_to_id, is_pred=False
            )
            annotations_gt += annotations
            images_coco.append(image_info)

            _, annotations = utils.collect_coco_annotations(
                image_item_pred,
                g.category_name_to_id,
                is_pred=True,
                image_id=image_id,
                dataset_id=dataset_id,
            )
            annotations_pred += annotations

    # number of annotations must be > 0 to avoid a error from pycocotools
    if len(annotations_gt) == 0 or len(annotations_pred) == 0:
        return None

    coco_gt, coco_dt = utils.create_coco_apis(
        images_coco, annotations_gt, annotations_pred, g.categories
    )

    df = utils.create_df(df_rows)

    # Metrics
    confusion_matrix = utils.calculate_confusion_matrix(df, g.cm_categories_selected)

    overall_stats, per_dataset_stats, per_class_stats = utils.calculate_metrics(
        df, g.cm_categories_selected, g.dataset_ids_matched, g.NONE_CLS
    )

    overall_coco = utils.overall_metrics_coco(coco_gt, coco_dt)

    per_class_coco = {}
    for cat_name in g.cm_categories_selected[:-1]:
        cat_id = g.category_name_to_id[cat_name]
        class_metrics = utils.per_class_metrics_coco(coco_gt, coco_dt, cat_id)
        per_class_coco[cat_id] = class_metrics

    per_dataset_coco = {}
    for dataset_id in g.dataset_ids_matched:
        dataset_metrics = utils.per_dataset_metrics_coco(coco_gt, coco_dt, dataset_id, images_coco)
        per_dataset_coco[dataset_id] = dataset_metrics

    stats = [overall_stats, per_dataset_stats, per_class_stats]
    coco = [overall_coco, per_dataset_coco, per_class_coco]

    return df, confusion_matrix, stats, coco


def set_globals(df, confusion_matrix, stats, coco):
    g.df = df
    g.confusion_matrix = confusion_matrix
    g.overall_stats, g.per_dataset_stats, g.per_class_stats = stats
    g.overall_coco, g.per_dataset_coco, g.per_class_coco = coco


# import pickle

# with open("metrics.pkl", "wb") as f:
#     stats = [overall_stats, per_dataset_stats, per_class_stats]
#     coco = [overall_coco, per_dataset_coco, per_class_coco]
#     pickle.dump([confusion_matrix, stats, coco], f)

# df.to_csv("df.csv")

# with open("image_ids_gt2pred.pkl", "wb") as f:
#     pickle.dump(image_ids_gt2pred, f)

# 18: car
# 50: person
# 53: plant
# 9: bicycle
# 44: motorcycle
