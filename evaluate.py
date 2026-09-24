"""
Evaluation script: runs a trained PointPillars checkpoint over a dataset
and reports mAP broken down by distance bucket (utils/eval/metrics.py).

Usage:
    python evaluate.py --checkpoint checkpoint_epoch20.pt --synthetic
    python evaluate.py --checkpoint checkpoint_epoch40.pt --root_dir /path/to/kitti --split_file val.txt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.kitti_dataset import KittiDetectionDataset, collate_single_scene
from data.synthetic_generator import POINT_CLOUD_RANGE
from model.pointpillars import PointPillars
from utils.anchors import generate_anchors, decode_boxes
from eval.metrics import evaluate_mAP_by_distance


def nms_bev(boxes, scores, iou_threshold=0.1):
    from utils.boxes import boxes_3d_iou
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = boxes_3d_iou(boxes[i:i+1], boxes[rest])[0]
        order = rest[ious < iou_threshold]
    return keep


@torch.no_grad()
def run_inference(model, points, anchors_flat, feat_h, feat_w, score_threshold=0.3, device="cpu",
                   max_pre_nms=300):
    model.eval()
    cls_preds, reg_preds, feat_hw = model(points.to(device))
    assert feat_hw == (feat_h, feat_w)

    cls_scores = torch.sigmoid(cls_preds.reshape(-1)).cpu().numpy()
    reg_deltas = reg_preds.reshape(-1, 7).cpu().numpy()

    keep_mask = cls_scores > score_threshold
    if keep_mask.sum() == 0:
        return np.zeros((0, 7)), np.zeros((0,))

    idx = np.where(keep_mask)[0]
    scores_all = cls_scores[idx]
    if len(idx) > max_pre_nms:
        top_order = np.argsort(-scores_all)[:max_pre_nms]
        idx = idx[top_order]
        scores_all = scores_all[top_order]

    decoded = decode_boxes(reg_deltas[idx], anchors_flat[idx])
    scores = scores_all

    keep_idx = nms_bev(decoded, scores)
    return decoded[keep_idx], scores[keep_idx]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        dataset = KittiDetectionDataset(synthetic=True, num_synthetic_scenes=args.num_eval_scenes,
                                         synthetic_seed=999)
    else:
        split_ids = [line.strip() for line in open(args.split_file)]
        dataset = KittiDetectionDataset(root_dir=args.root_dir, split_ids=split_ids)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single_scene)

    pc_range = POINT_CLOUD_RANGE
    model = PointPillars(pc_range).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    feat_h, feat_w = None, None
    dummy_points = torch.zeros(10, 4, device=device)
    dummy_points[:, 0] = torch.linspace(1.0, 60.0, 10)
    with torch.no_grad():
        _, _, (feat_h, feat_w) = model(dummy_points)
    anchors = generate_anchors(pc_range, feature_map_size=(feat_h, feat_w))
    anchors_flat = anchors.reshape(-1, 7)

    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    for sample in loader:
        pred_boxes, pred_scores = run_inference(
            model, sample["points"], anchors_flat, feat_h, feat_w,
            score_threshold=args.score_threshold, device=device,
            max_pre_nms=args.max_pre_nms,
        )
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_gt_boxes.append(sample["gt_boxes"].numpy())

    results = evaluate_mAP_by_distance(all_pred_boxes, all_pred_scores, all_gt_boxes,
                                        iou_threshold=args.iou_threshold)
    print(f"\nmAP by distance (3D IoU >= {args.iou_threshold}):")
    for bucket, r in results.items():
        print(f"  {bucket:16s} AP={r['AP']:.3f}  (gt={r['num_gt']}, pred={r['num_pred']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--num_eval_scenes", type=int, default=100)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_pre_nms", type=int, default=300,
                         help="Cap on candidate boxes entering NMS, kept by score. Prevents "
                              "O(N^2) pure-Python NMS from blowing up at low score thresholds.")
    args = parser.parse_args()
    evaluate(args)
