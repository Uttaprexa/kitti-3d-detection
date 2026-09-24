"""
Loss functions for anchor-based 3D detection.
  - Focal loss: handles the massive foreground/background anchor imbalance
    (most anchors in a scene are background).
  - Smooth L1: standard box regression loss on the encoded 7-DoF deltas.

Requires PyTorch (see model/pointpillars.py docstring for sandbox note).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017), reduces easy-negative dominance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, valid_mask=None):
        """
        logits: (...,) raw scores (pre-sigmoid)
        targets: (...,) same shape, in {0, 1}
        valid_mask: (...,) bool, positions to include in the loss
                    (e.g. exclude "don't care" anchors near GT boxes)
        """
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if valid_mask is not None:
            loss = loss[valid_mask]
        return loss.mean() if loss.numel() > 0 else loss.sum()


class BoxRegressionLoss(nn.Module):
    """Smooth L1 over the 7 encoded box deltas, averaged over positive anchors."""

    def __init__(self, beta=1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_deltas, target_deltas, positive_mask):
        """
        pred_deltas, target_deltas: (N, 7)
        positive_mask: (N,) bool -- only anchors matched to a GT box contribute
        """
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_deltas.device, requires_grad=True)
        diff = pred_deltas[positive_mask] - target_deltas[positive_mask]
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=self.beta, reduction="none")
        return loss.sum(dim=-1).mean()


def assign_anchor_targets(anchors_flat, gt_boxes, gt_ious_fn, pos_iou_thresh=0.6, neg_iou_thresh=0.45,
                           search_margin=2.0):
    """
    Label each anchor as positive (matched to a GT box), negative (background),
    or ignored (in between the two thresholds -- standard anchor-based
    detection practice to avoid penalizing ambiguous anchors).

    Performance note: with a dense anchor grid (e.g. 500x440x2 = 440,000
    anchors), computing full (A, M) 3D IoU against every anchor is
    prohibitively slow -- the polygon-clipping IoU in boxes.py is pure
    Python and the vast majority of anchors are nowhere near any GT box.
    So we first prune to anchors within a simple circular radius of each
    GT box's center (radius = half the sum of anchor+GT diagonals, plus a
    margin) using cheap vectorized distance checks, and only run the
    expensive rotated-IoU computation on that much smaller candidate set.

    Args:
        anchors_flat: (A, 7) numpy array
        gt_boxes: (M, 7) numpy array
        gt_ious_fn: callable(anchors, gt_boxes) -> (A, M) IoU matrix
                    (pass utils.boxes.boxes_3d_iou)
    Returns:
        labels: (A,) numpy int array in {0 (neg), 1 (pos), -1 (ignore)}
        matched_gt_idx: (A,) numpy int array, index into gt_boxes for
                        positive anchors (else -1)
    """
    A = anchors_flat.shape[0]
    labels = np.zeros(A, dtype=np.int64)          # default: background
    matched_gt_idx = -np.ones(A, dtype=np.int64)  # default: unmatched

    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    if len(gt_boxes) == 0:
        return labels, matched_gt_idx

    anchor_centers = anchors_flat[:, :2]                              # (A, 2)
    anchor_diag = np.sqrt(anchors_flat[:, 3] ** 2 + anchors_flat[:, 4] ** 2)  # (A,)
    gt_diag = np.sqrt(gt_boxes[:, 3] ** 2 + gt_boxes[:, 4] ** 2)              # (M,)

    candidate_mask = np.zeros(A, dtype=bool)
    per_gt_candidates = []
    for gi in range(len(gt_boxes)):
        max_dist = (anchor_diag + gt_diag[gi]) / 2.0 + search_margin
        dist = np.hypot(anchor_centers[:, 0] - gt_boxes[gi, 0],
                         anchor_centers[:, 1] - gt_boxes[gi, 1])
        mask = dist <= max_dist
        per_gt_candidates.append(np.where(mask)[0])
        candidate_mask |= mask

    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) > 0:
        iou = gt_ious_fn(anchors_flat[candidate_idx], gt_boxes)  # (C, M), C << A
        max_iou = iou.max(axis=1)
        local_matched_gt = iou.argmax(axis=1)

        local_labels = np.zeros(len(candidate_idx), dtype=np.int64)
        local_labels[(max_iou > neg_iou_thresh) & (max_iou <= pos_iou_thresh)] = -1
        local_labels[max_iou > pos_iou_thresh] = 1

        labels[candidate_idx] = local_labels
        pos_local = local_labels == 1
        matched_gt_idx[candidate_idx[pos_local]] = local_matched_gt[pos_local]

    # Force-assign the single best nearby anchor per GT box as positive, so
    # small/rare objects always get at least one positive anchor even if no
    # anchor cleared pos_iou_thresh. Only searches within that GT's already-
    # pruned candidate set (cheap), with a same-size-anchor fallback if a GT
    # box ended up with zero candidates (e.g. an unusually large box).
    for gi, cand in enumerate(per_gt_candidates):
        if len(cand) == 0:
            continue
        local_iou = gt_ious_fn(anchors_flat[cand], gt_boxes[gi:gi + 1])[:, 0]
        best_idx = cand[np.argmax(local_iou)]
        labels[best_idx] = 1
        matched_gt_idx[best_idx] = gi

    return labels, matched_gt_idx
