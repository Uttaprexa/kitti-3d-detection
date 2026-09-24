"""
Evaluation metrics: 3D IoU-based matching, precision-recall, and mAP broken
down by distance-from-sensor threshold (near / mid / far) -- mirroring how
KITTI's own eval separates "easy/moderate/hard" partly by distance/occlusion,
and how the resume bullet ("mAP by distance threshold") is meant to be read.
"""

import numpy as np
from sklearn.metrics import auc

from utils.boxes import boxes_3d_iou


DISTANCE_BUCKETS = {
    "near_0_30m": (0.0, 30.0),
    "mid_30_50m": (30.0, 50.0),
    "far_50m_plus": (50.0, np.inf),
}


def _box_distance(boxes):
    return np.hypot(boxes[:, 0], boxes[:, 1])


def average_precision(matched_flags, scores, num_gt):
    """
    Standard AP via precision-recall curve area (not the 11-point KITTI
    interpolation, for simplicity/transparency -- documented as such).

    Args:
        matched_flags: (N,) bool, whether each (confidence-sorted) prediction
                        was a true positive
        scores: (N,) confidence scores, same order as matched_flags
        num_gt: int, total number of ground-truth boxes for this class/bucket
    Returns:
        ap: float
        precision, recall: arrays for plotting
    """
    if num_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if len(scores) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    order = np.argsort(-scores)
    matched_flags = np.asarray(matched_flags)[order]

    tp_cumsum = np.cumsum(matched_flags)
    fp_cumsum = np.cumsum(~matched_flags)

    recall = tp_cumsum / num_gt
    precision = tp_cumsum / np.clip(tp_cumsum + fp_cumsum, 1, None)

    # prepend (recall=0, precision=1) so the AUC integrates from the origin
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    ap = auc(recall, precision)
    return ap, precision, recall


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Greedy matching: sort predictions by score descending, match each to the
    highest-IoU unmatched GT box above threshold.

    Returns:
        matched_flags: (N,) bool aligned with pred_boxes/pred_scores input order
    """
    n_pred = len(pred_boxes)
    matched_flags = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or len(gt_boxes) == 0:
        return matched_flags

    iou = boxes_3d_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
    order = np.argsort(-pred_scores)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)

    for idx in order:
        row = iou[idx].copy()
        row[gt_used] = -1  # can't reuse a GT box
        best_gt = np.argmax(row)
        if row[best_gt] >= iou_threshold:
            matched_flags[idx] = True
            gt_used[best_gt] = True

    return matched_flags


def evaluate_mAP_by_distance(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5):
    """
    Compute AP separately within each distance bucket, across a dataset of
    scenes.

    Args:
        all_pred_boxes:  list of (Ni, 7) arrays, one per scene
        all_pred_scores: list of (Ni,) arrays, one per scene
        all_gt_boxes:    list of (Mi, 7) arrays, one per scene
        iou_threshold: float, 3D IoU required for a true positive (0.5/0.7 typical for Car)
    Returns:
        dict: {bucket_name: {"AP": float, "num_gt": int, "num_pred": int}}
    """
    results = {}
    for bucket_name, (d_min, d_max) in DISTANCE_BUCKETS.items():
        bucket_pred_boxes, bucket_pred_scores, bucket_matched = [], [], []
        num_gt = 0

        for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
            pred_boxes = np.asarray(pred_boxes).reshape(-1, 7)
            gt_boxes = np.asarray(gt_boxes).reshape(-1, 7)

            if len(gt_boxes) > 0:
                gt_dist = _box_distance(gt_boxes)
                gt_mask = (gt_dist >= d_min) & (gt_dist < d_max)
                gt_in_bucket = gt_boxes[gt_mask]
            else:
                gt_in_bucket = gt_boxes

            num_gt += len(gt_in_bucket)

            if len(pred_boxes) == 0:
                continue
            pred_dist = _box_distance(pred_boxes)
            pred_mask = (pred_dist >= d_min) & (pred_dist < d_max)
            preds_in_bucket = pred_boxes[pred_mask]
            scores_in_bucket = np.asarray(pred_scores)[pred_mask]

            if len(preds_in_bucket) == 0:
                continue

            matched = match_predictions_to_gt(preds_in_bucket, scores_in_bucket, gt_in_bucket, iou_threshold)
            bucket_pred_boxes.append(preds_in_bucket)
            bucket_pred_scores.append(scores_in_bucket)
            bucket_matched.append(matched)

        if bucket_matched:
            all_matched = np.concatenate(bucket_matched)
            all_scores = np.concatenate(bucket_pred_scores)
        else:
            all_matched = np.array([], dtype=bool)
            all_scores = np.array([])

        ap, _, _ = average_precision(all_matched, all_scores, num_gt)
        results[bucket_name] = {
            "AP": ap,
            "num_gt": num_gt,
            "num_pred": len(all_scores),
        }
    return results
