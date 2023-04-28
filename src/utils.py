from src.data_iterator import DataIteratorAPI
import supervisely as sly
import numpy as np


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


def collect_labels(image_item: DataIteratorAPI.ImageItem):
    labels, classes, bboxes, bitmaps = [], [], [], []

    for item in image_item.labels_iterator:
        if not isinstance(item.label.geometry, sly.Bitmap):
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
