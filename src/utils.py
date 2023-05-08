import pandas as pd
from src.data_iterator import DataIteratorAPI
import supervisely as sly
import numpy as np
import sklearn.metrics
from pycocotools import mask, cocoeval, coco


try:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
    from .compute_overlap import compute_overlap
except:
    print(
        "Couldn't import fast version of function compute_overlap, will use slow one. Check cython installation"
    )
    from compute_overlap_slow import compute_overlap


ROUND_N_DIGITS = 4


def uncrop_bitmap(bitmap: sly.Bitmap, image_width, image_height):
    data = bitmap.data
    h, w = data.shape
    o = bitmap.origin
    b = np.zeros((image_height, image_width), dtype=data.dtype)
    b[o.row : o.row + h, o.col : o.col + w] = data
    return b


def join_bitmaps_tight(bitmap1: sly.Bitmap, bitmap2: sly.Bitmap):
    r1 = bitmap1.to_bbox()
    bbox1 = r1.left, r1.top, r1.right, r1.bottom

    r2 = bitmap2.to_bbox()
    bbox2 = r2.left, r2.top, r2.right, r2.bottom

    fns = [min, min, max, max]
    x1, y1, x2, y2 = [f(v1, v2) for v1, v2, f in zip(bbox1, bbox2, fns)]
    w, h = x2 - x1 + 1, y2 - y1 + 1

    o1_x, o1_y = bbox1[0] - x1, bbox1[1] - y1
    o2_x, o2_y = bbox2[0] - x1, bbox2[1] - y1

    b1 = np.zeros((h, w), dtype=bitmap1.data.dtype)
    h1, w1 = bitmap1.data.shape
    b1[o1_y : o1_y + h1, o1_x : o1_x + w1] = bitmap1.data

    b2 = np.zeros((h, w), dtype=bitmap2.data.dtype)
    h2, w2 = bitmap2.data.shape
    b2[o2_y : o2_y + h2, o2_x : o2_x + w2] = bitmap2.data
    return b1, b2


def iou_numpy(mask_pred: np.ndarray, mask_gt: np.ndarray):
    # H x W
    assert len(mask_pred.shape) == 2 and len(mask_gt.shape) == 2
    SMOOTH = 1e-6

    intersection = (mask_pred & mask_gt).sum()
    union = (mask_pred | mask_gt).sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou


def collect_labels(image_item: DataIteratorAPI.ImageItem, category_name_to_id):
    labels, classes, bboxes, bitmaps = [], [], [], []

    for item in image_item.labels_iterator:
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        class_name = item.label.obj_class.name
        if class_name not in category_name_to_id:
            continue
        label = item.label
        rect = label.geometry.to_bbox()
        bbox = rect.left, rect.top, rect.right, rect.bottom
        labels.append(label)
        classes.append(label.obj_class.name)
        bboxes.append(bbox)
        bitmaps.append(label.geometry)

    return labels, classes, bboxes, bitmaps


def match_bboxes(pairwise_iou: np.ndarray, min_iou_threshold=0.25):
    # shape: [pred, gt]
    n_pred, n_gt = pairwise_iou.shape

    matched_idxs = []
    ious_matched = []
    unmatched_idxs_gt, unmatched_idxs_pred = set(range(n_gt)), set(range(n_pred))

    m = pairwise_iou.flatten()
    for i in m.argsort()[::-1]:
        if m[i] < min_iou_threshold or m[i] == 0:
            break
        pred_i = i // n_gt  # row
        gt_i = i % n_gt  # col
        if gt_i in unmatched_idxs_gt and pred_i in unmatched_idxs_pred:
            matched_idxs.append([gt_i, pred_i])
            ious_matched.append(m[i])
            unmatched_idxs_gt.remove(gt_i)
            unmatched_idxs_pred.remove(pred_i)

    return matched_idxs, list(unmatched_idxs_gt), list(unmatched_idxs_pred), ious_matched


def collect_df_rows(
    image_item_gt: DataIteratorAPI.ImageItem,
    image_item_pred: DataIteratorAPI.ImageItem,
    NONE_CLS: str,
    category_name_to_id: dict,
):
    # Remembering column names:
    # df_columns = [
    #     "gt_class",
    #     "pred_class",
    #     "class_match",
    #     "IoU",
    #     "image_id",
    #     "dataset_id",
    #     "gt_label_id",
    #     "pred_label_id",
    # ]

    rows = []
    image_id = image_item_gt.image_id
    dataset_id = image_item_gt.dataset_id

    labels_gt, classes_gt, bboxes_gt, bitmaps_gt = collect_labels(
        image_item_gt, category_name_to_id
    )
    labels_pred, classes_pred, bboxes_pred, bitmaps_pred = collect_labels(
        image_item_pred, category_name_to_id
    )

    if len(bboxes_gt) == 0 and len(bboxes_pred) == 0:
        return []

    if len(bboxes_gt) != 0 and len(bboxes_pred) != 0:
        # [Pred x GT]
        pairwise_iou = compute_overlap(
            np.array(bboxes_pred, dtype=np.float64), np.array(bboxes_gt, dtype=np.float64)
        )
        # below this threshold we treat two bboxes don't match
        min_iou_threshold = 0.25
        matched_idxs, unmatched_idxs_gt, unmatched_idxs_pred, box_ious_matched = match_bboxes(
            pairwise_iou, min_iou_threshold
        )
    else:
        matched_idxs = []
        unmatched_idxs_gt = list(range(len(bboxes_gt)))
        unmatched_idxs_pred = list(range(len(bboxes_pred)))

    for i_gt, i_pred in matched_idxs:
        class_gt = classes_gt[i_gt]
        class_pred = classes_pred[i_pred]
        class_match = class_gt == class_pred
        mask1, mask2 = join_bitmaps_tight(bitmaps_gt[i_gt], bitmaps_pred[i_pred])
        iou = iou_numpy(mask1, mask2)
        gt_label_id = bitmaps_gt[i_gt].sly_id
        pred_label_id = bitmaps_pred[i_pred].sly_id
        row = [
            class_gt,
            class_pred,
            class_match,
            iou,
            image_id,
            dataset_id,
            gt_label_id,
            pred_label_id,
        ]
        rows.append(row)

    for idx in unmatched_idxs_gt:
        cls = classes_gt[idx]
        gt_label_id = bitmaps_gt[idx].sly_id
        row = [cls, NONE_CLS, None, None, image_id, dataset_id, gt_label_id, -1]
        rows.append(row)

    for idx in unmatched_idxs_pred:
        cls = classes_pred[idx]
        pred_label_id = bitmaps_pred[idx].sly_id
        row = [NONE_CLS, cls, None, None, image_id, dataset_id, -1, pred_label_id]
        rows.append(row)

    return rows


def create_df(df_rows):
    df_columns = [
        "gt_class",
        "pred_class",
        "class_match",
        "IoU",
        "image_id",
        "dataset_id",
        "gt_label_id",
        "pred_label_id",
    ]
    df = pd.DataFrame(df_rows, columns=df_columns)
    return df


def calculate_confusion_matrix(df, cm_categories):
    gt_classes = df["gt_class"].to_list()
    pred_classes = df["pred_class"].to_list()
    confusion_matrix = sklearn.metrics.confusion_matrix(
        gt_classes, pred_classes, labels=cm_categories
    )
    return confusion_matrix


def get_unique_classes(gt_classes: list, pred_classes: list, NONE_CLS: str):
    used_classes = set(gt_classes) | set(pred_classes)
    if NONE_CLS in used_classes:
        used_classes.remove(NONE_CLS)
    used_classes = list(used_classes)
    return used_classes


# per-class + avg: P/R/F1
def calculate_metrics(df, used_classes: list, dataset_ids, NONE_CLS):
    gt_classes = df["gt_class"].to_list()
    pred_classes = df["pred_class"].to_list()

    # Get some per-class stats (classification_report)
    per_class_stats = sklearn.metrics.classification_report(
        gt_classes, pred_classes, labels=used_classes, output_dict=True
    )

    # Get some overall stats
    overall_stats = {}
    overall_keys = ["micro avg", "macro avg", "weighted avg", "accuracy"]
    for key in overall_keys:
        overall_stats[key] = per_class_stats.get(key, -1)
        if per_class_stats.get(key) is not None:
            per_class_stats.pop(key)

    # Per-class stats (IoU + AP)
    class2image_ids = {}  # image_id in GT
    for cls in used_classes:
        if cls == NONE_CLS:
            continue
        cls_filtered = df[(df["gt_class"] == cls) | (df["pred_class"] == cls)]
        # gt = cls_filtered["gt_class"].to_list()
        # pred = cls_filtered["pred_class"].to_list()
        # gt = [int(x == cls) for x in gt]
        # pred = [int(x == cls) for x in pred]
        # AP = sklearn.metrics.average_precision_score(gt, pred) if len(gt) and len(pred) else -1
        # per_class_stats[cls]["AP"] = AP
        avg_iou = cls_filtered["IoU"].mean()
        per_class_stats[cls]["Avg. IoU"] = avg_iou
        per_class_stats[cls]["N samples"] = per_class_stats[cls].pop("support")

        class2image_ids[cls] = list(set(cls_filtered["image_id"]))

    # Overall for per-class avg.
    # overall_stats["mAP"] = np.mean([x["AP"] for x in per_class_stats.values() if x["AP"] != -1])
    overall_stats["mIoU"] = np.nanmean([x["Avg. IoU"] for x in per_class_stats.values()])

    # Per-dataset stats (IoU + AP)
    per_dataset_stats = {}  # dataset_id to stats
    for dataset_id in dataset_ids:
        # AP_per_class = []
        ious_per_class = []
        dataset_mask = df["dataset_id"] == dataset_id
        for cls in used_classes:
            if cls == NONE_CLS:
                continue
            cls_filtered = df[dataset_mask & ((df["gt_class"] == cls) | (df["pred_class"] == cls))]
            # gt = cls_filtered["gt_class"].to_list()
            # pred = cls_filtered["pred_class"].to_list()
            # gt = [int(x == cls) for x in gt]
            # pred = [int(x == cls) for x in pred]
            # if len(gt) and len(pred):
            #     AP = sklearn.metrics.average_precision_score(gt, pred)
            #     AP_per_class.append(AP)
            avg_iou = cls_filtered["IoU"].mean()
            ious_per_class.append(avg_iou)
        # mAP = np.mean(AP_per_class)
        mIoU = np.nanmean(ious_per_class)

        ds_filtered = df[dataset_mask]
        ds_gt_classes = ds_filtered["gt_class"].to_list()
        ds_pred_classes = ds_filtered["pred_class"].to_list()
        ds_used_classes = get_unique_classes(ds_gt_classes, ds_pred_classes, NONE_CLS)
        ds_cls_report = sklearn.metrics.classification_report(
            ds_gt_classes, ds_pred_classes, labels=ds_used_classes, output_dict=True
        )

        # per_dataset_stats[dataset_id] = {"mAP": mAP, "mIoU": mIoU}
        per_dataset_stats[dataset_id] = {"mIoU": mIoU, "report": ds_cls_report["macro avg"]}

    return overall_stats, per_dataset_stats, per_class_stats


# Per-image
def per_image_metrics_table(df_match, dataset_names_gt, image_id_2_image_info, NONE_CLS):
    columns_metrics = ["TP", "FP", "FN", "precision", "recall", "Avg. IoU"]
    columns = ["image_id", "image_name", "dataset", *columns_metrics]
    rows = []
    for img_id, g in df_match.groupby("image_id"):
        row = per_image_metrics_for_group(g, NONE_CLS)
        row = [round(float(x), ROUND_N_DIGITS) if isinstance(x, float) else x for x in row]
        dataset_id = g["dataset_id"].values[0]
        dataset_name = dataset_names_gt[dataset_id]
        image_name = image_id_2_image_info[img_id].name
        row = [img_id, image_name, dataset_name, *row]
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    return table


def per_image_metrics_for_group(df_group, NONE_CLS):
    TP = (df_group["gt_class"] == df_group["pred_class"]).sum()
    FP = (df_group["class_match"].isna() & (df_group["gt_class"] == NONE_CLS)).sum()
    FN = (df_group["class_match"].isna() & (df_group["pred_class"] == NONE_CLS)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    avg_iou = df_group["IoU"].mean()
    return TP, FP, FN, precision, recall, avg_iou


# Per-object
def per_object_metrics(df_match, NONE_CLS):
    gt_label_ids = df_match["gt_label_id"].to_list()
    pred_label_ids = df_match["pred_label_id"].to_list()
    df_match = df_match.drop(columns=["gt_label_id", "pred_label_id", "dataset_id", "image_id"])
    df_match.loc[df_match["gt_class"] == NONE_CLS, "gt_class"] = ""
    df_match.loc[df_match["pred_class"] == NONE_CLS, "pred_class"] = ""
    df_match["IoU"] = df_match["IoU"].values.astype(float)
    df_match["IoU"] = np.round(df_match["IoU"], ROUND_N_DIGITS)
    return df_match, gt_label_ids, pred_label_ids


def create_coco_annotation(mask_np, id, image_id, category_id, confidence=None):
    segmentation = mask.encode(np.asfortranarray(mask_np))
    segmentation["counts"] = segmentation["counts"].decode()
    annotation = {
        "id": id,  # unique id for each annotation
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "iscrowd": 0,
    }
    if confidence is not None:
        annotation["score"] = confidence
    return annotation


def collect_coco_annotations(
    image_item: DataIteratorAPI.ImageItem,
    category_name_to_id: dict,
    is_pred,
    image_id=None,
    dataset_id=None,
):
    if image_id is None:
        image_id = image_item.image_id

    if dataset_id is None:
        dataset_id = image_item.dataset_id

    image_info = {
        "dataset_id": dataset_id,
        "id": image_id,
        "height": image_item.image_height,
        "width": image_item.image_width,
    }

    image_annotations = []

    for item in image_item.labels_iterator:
        if not isinstance(item.label.geometry, sly.Bitmap):
            continue
        class_name = item.label.obj_class.name
        if class_name not in category_name_to_id:
            continue
        category_id = category_name_to_id[class_name]
        mask_np = uncrop_bitmap(item.label.geometry, item.image_width, item.image_height)
        label_id = item.label.geometry.sly_id
        confidence = None
        if is_pred:
            conf_tag = item.label.tags.get("confidence")
            if conf_tag is None:
                raise Exception('Predicted labels must have tag "confidence".')
            confidence = conf_tag.value
        annotation = create_coco_annotation(
            mask_np, label_id, image_id, category_id, confidence=confidence
        )
        image_annotations.append(annotation)

    return image_info, image_annotations


def create_coco_apis(images_coco, annotations_gt, annotations_pred, categories):
    # Create COCO format dictionary
    coco_json = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": images_coco,
        "annotations": annotations_gt,
    }

    # Save annotations to json
    sly.json.dump_json_file(coco_json, "ground_truth.json", indent=None)

    # GT
    coco_gt = coco.COCO("ground_truth.json")
    ann_ids = coco_gt.getAnnIds()
    coco_gt = coco_gt.loadRes(coco_gt.loadAnns(ann_ids))

    # Pred
    coco_dt = coco_gt.loadRes(annotations_pred)

    return coco_gt, coco_dt


# COCO Per-dataset
def per_dataset_metrics_coco(coco_gt, coco_dt, dataset_id, images_coco):
    img_ids_for_dataset = [img["id"] for img in images_coco if img["dataset_id"] == dataset_id]
    e = cocoeval.COCOeval(coco_gt, coco_dt)
    e.params.areaRng = [e.params.areaRng[0]]
    e.params.areaRngLbl = [e.params.areaRngLbl[0]]
    e.params.imgIds = img_ids_for_dataset
    e.evaluate()
    e.accumulate()
    e.summarize()
    return e.stats


# COCO Per-class
def per_class_metrics_coco(coco_gt, coco_dt, category_id):
    e = cocoeval.COCOeval(coco_gt, coco_dt)
    e.params.areaRng = [e.params.areaRng[0]]
    e.params.areaRngLbl = [e.params.areaRngLbl[0]]
    e.params.catIds = [category_id]
    e.evaluate()
    e.accumulate()
    e.summarize()
    return e.stats


# COCO overall
def overall_metrics_coco(coco_gt, coco_dt):
    e = cocoeval.COCOeval(coco_gt, coco_dt)
    e.params.areaRng = [e.params.areaRng[0]]
    e.params.areaRngLbl = [e.params.areaRngLbl[0]]
    e.evaluate()
    e.accumulate()
    e.summarize()
    return e.stats


def coco_stats_to_dict(coco_stats: np.ndarray):
    return {
        "mAP@0.50:0.95 (COCO)": coco_stats[0],
        "mAP@0.50 (COCO)": coco_stats[1],
        "mAP@0.75 (COCO)": coco_stats[2],
        "Average Recall (COCO)": coco_stats[8],
    }


def format_table(t: dict, n_digits=4):
    # col : [rows]
    return {
        k: [round(float(x), n_digits) if isinstance(x, float) else x for x in rows]
        for k, rows in t.items()
    }


def format_table2(t: dict, n_digits=4):
    # [rows]
    return [
        {k: round(float(x), n_digits) if isinstance(x, float) else x for k, x in row.items()}
        for row in t
    ]


def collect_overall_metrics(overall_stats: dict, overall_coco: np.ndarray):
    macro = overall_stats["macro avg"]
    res = {
        **coco_stats_to_dict(overall_coco),
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1-score": macro["f1-score"],
        # "mAP": overall_stats["mAP"],
        "mIoU (mask)": overall_stats["mIoU"],
        "N samples": macro["support"],
    }
    res = {k: [v] for k, v in res.items()}
    res = format_table(res, ROUND_N_DIGITS)
    return res


def collect_per_dataset_metrics(
    per_dataset_stats: dict,
    per_dataset_coco: np.ndarray,
    dataset_ids: list,
    dataset_names_gt: dict,
):
    res = [
        {
            "dataset_id": dataset_id,
            "dataset": dataset_names_gt[dataset_id],
            **coco_stats_to_dict(per_dataset_coco[dataset_id]),
            "precision": metrics["report"]["precision"],
            "recall": metrics["report"]["recall"],
            "f1-score": metrics["report"]["f1-score"],
            "mIoU (mask)": metrics["mIoU"],
            "N samples": metrics["report"]["support"],
        }
        for dataset_id, metrics in per_dataset_stats.items()
    ]
    res = format_table2(res, ROUND_N_DIGITS)
    return res


def collect_per_class_metrics(per_class_stats: dict, per_class_coco: dict, categories: list):
    # per_class_stats - class_name : dict
    # per_class_coco - class_id : np.array
    res = [
        {
            "class": cat["name"],
            **coco_stats_to_dict(per_class_coco[cat["id"]]),
            **per_class_stats[cat["name"]],
        }
        for cat in categories
    ]
    res = format_table2(res, ROUND_N_DIGITS)
    return res
