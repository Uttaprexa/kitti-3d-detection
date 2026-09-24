"""
End-to-end demo of every part of this pipeline that does NOT require PyTorch:
  1. Generate a batch of synthetic LiDAR scenes with GT boxes
  2. Build the anchor grid used by the detection head
  3. Simulate a detector's output (anchors nudged toward nearby GT + noise +
     confidence scores + some false positives), standing in for a trained
     model's predictions
  4. Run NMS, then IoU-based matching against GT
  5. Compute mAP broken down by distance bucket
  6. Visualize one scene: LiDAR points, GT boxes, and predicted boxes in BEV

This script has zero dependence on PyTorch/CUDA -- it's the part of the
project that's fully verified and runnable in this environment right now.
Swap step 3 for real model inference once train.py has been run with torch.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.synthetic_generator import generate_synthetic_dataset, POINT_CLOUD_RANGE
from utils.anchors import generate_anchors
from utils.boxes import boxes_to_bev_corners, boxes_3d_iou
from eval.metrics import evaluate_mAP_by_distance


def simulate_detector_output(gt_boxes, rng, miss_rate=0.1, false_positive_rate=1,
                              pos_noise_std=0.15, size_noise_std=0.1, heading_noise_std=0.05):
    """
    Stand-in for a trained model's output: perturbs GT boxes (simulating
    localization error), randomly drops some (simulating missed detections),
    and adds random false-positive boxes -- enough realism to make the mAP
    numbers meaningful for testing the eval code without needing torch.
    """
    pred_boxes, pred_scores = [], []

    for box in gt_boxes:
        if rng.random() < miss_rate:
            continue
        noisy = box.copy()
        noisy[0:2] += rng.normal(0, pos_noise_std, 2)
        noisy[2] += rng.normal(0, pos_noise_std * 0.5)
        noisy[3:6] *= 1 + rng.normal(0, size_noise_std, 3)
        noisy[6] += rng.normal(0, heading_noise_std)
        pred_boxes.append(noisy)
        # confidence correlated with how small the perturbation was
        err = np.linalg.norm(noisy[:3] - box[:3])
        pred_scores.append(np.clip(0.95 - err * 0.3, 0.05, 0.98))

    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    for _ in range(rng.poisson(false_positive_rate)):
        fp = np.array([
            rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), -1.0,
            rng.uniform(3.0, 4.5), rng.uniform(1.4, 1.8), rng.uniform(1.4, 1.6),
            rng.uniform(-np.pi, np.pi),
        ])
        pred_boxes.append(fp)
        pred_scores.append(rng.uniform(0.1, 0.5))  # FPs tend to have lower confidence

    if not pred_boxes:
        return np.zeros((0, 7)), np.zeros((0,))
    return np.array(pred_boxes), np.array(pred_scores)


def main():
    rng = np.random.default_rng(123)

    print("1. Generating synthetic dataset...")
    scenes = generate_synthetic_dataset(num_scenes=50, seed=123)
    total_objects = sum(len(gt) for _, gt, _ in scenes)
    print(f"   {len(scenes)} scenes, {total_objects} total objects, "
          f"{total_objects/len(scenes):.1f} objects/scene avg")

    print("\n2. Building anchor grid...")
    feature_map_size = (200, 176)  # e.g. 500x440 pillar grid downsampled 2.5x by the backbone
    anchors = generate_anchors(POINT_CLOUD_RANGE, feature_map_size)
    print(f"   anchor grid shape: {anchors.shape} -> {np.prod(anchors.shape[:3])} total anchors")

    print("\n3. Simulating detector output on all scenes...")
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    for points, gt_boxes, gt_labels in scenes:
        pred_boxes, pred_scores = simulate_detector_output(gt_boxes, rng)
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_gt_boxes.append(gt_boxes)

    print("\n4. Computing mAP by distance bucket (3D IoU >= 0.5)...")
    results = evaluate_mAP_by_distance(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5)
    for bucket, r in results.items():
        print(f"   {bucket:16s} AP={r['AP']:.3f}  (gt={r['num_gt']:3d}, pred={r['num_pred']:3d})")

    print("\n5. Rendering one example scene to demo_scene_result.png...")
    points, gt_boxes, gt_labels = scenes[0]
    pred_boxes, pred_scores = all_pred_boxes[0], all_pred_scores[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(points[:, 0], points[:, 1], s=1, c="lightgray", label="LiDAR points")

    if len(gt_boxes) > 0:
        for c in boxes_to_bev_corners(gt_boxes):
            poly = np.concatenate([c, c[:1]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], "g-", linewidth=2.5, label="_nolegend_")
    if len(pred_boxes) > 0:
        for c, s in zip(boxes_to_bev_corners(pred_boxes), pred_scores):
            poly = np.concatenate([c, c[:1]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], "r--", linewidth=1.5, alpha=max(0.3, s), label="_nolegend_")

    ax.plot([], [], "g-", linewidth=2.5, label="Ground truth")
    ax.plot([], [], "r--", linewidth=1.5, label="Predicted (simulated)")
    ax.scatter([0], [0], c="blue", marker="*", s=200, label="Sensor origin")
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.set_title("KITTI-style 3D detection demo (BEV) -- synthetic scene")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("demo_scene_result.png", dpi=110)
    print("   saved demo_scene_result.png")


if __name__ == "__main__":
    main()
