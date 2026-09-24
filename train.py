"""
Training loop for PointPillars.

Usage:
    python train.py --synthetic --epochs 20         # develop/verify on synthetic data
    python train.py --root_dir /path/to/kitti --epochs 40   # real KITTI

Requires PyTorch + a GPU for reasonable speed. Not runnable in the sandbox
this repo was developed in (no torch/network access there) -- see
demo_geometry.py for the parts already verified without torch.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.kitti_dataset import KittiDetectionDataset, collate_single_scene
from data.synthetic_generator import POINT_CLOUD_RANGE
from model.pointpillars import PointPillars
from utils.anchors import generate_anchors, encode_boxes
from utils.losses import FocalLoss, BoxRegressionLoss, assign_anchor_targets
from utils.boxes import boxes_3d_iou


def build_dataset(args):
    if args.synthetic:
        return KittiDetectionDataset(synthetic=True, num_synthetic_scenes=args.num_synthetic_scenes)
    split_ids = [line.strip() for line in open(args.split_file)]
    return KittiDetectionDataset(root_dir=args.root_dir, split_ids=split_ids)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single_scene)

    pc_range = POINT_CLOUD_RANGE
    model = PointPillars(pc_range).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", args.start_epoch)
            print(f"resumed from {args.resume_from} at epoch {start_epoch} (optimizer state restored)")
        else:
            model.load_state_dict(ckpt)
            start_epoch = args.start_epoch
            print(f"resumed model weights only (old checkpoint format, no optimizer state) "
                  f"from {args.resume_from}; starting epoch counter at {start_epoch} "
                  f"(pass --start_epoch to override if this run's checkpoint came from a "
                  f"different epoch count than {start_epoch})")

    focal_loss = FocalLoss()
    reg_loss_fn = BoxRegressionLoss()

    dummy_points = torch.zeros(10, 4, device=device)
    dummy_points[:, 0] = torch.linspace(1.0, 60.0, 10)
    dummy_points[:, 1] = 0.0
    with torch.no_grad():
        _, _, (feat_h, feat_w) = model(dummy_points)
    anchors = generate_anchors(pc_range, feature_map_size=(feat_h, feat_w))
    anchors_flat = anchors.reshape(-1, 7)
    print(f"anchor grid: {feat_h}x{feat_w}x2 = {feat_h*feat_w*2} anchors "
          f"(pillar grid is {model.pillar_net.grid_h}x{model.pillar_net.grid_w})")

    model.train()
    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        epoch_cls_loss, epoch_reg_loss, n_batches = 0.0, 0.0, 0

        for sample in loader:
            points = sample["points"].to(device)
            gt_boxes = sample["gt_boxes"].numpy()

            cls_preds, reg_preds, feat_hw = model(points)
            assert feat_hw == (feat_h, feat_w), (
                f"Anchor grid {(feat_h, feat_w)} doesn't match model feature map {feat_hw}; "
                "recompute anchors from the model's actual grid size before training."
            )

            labels, matched_gt_idx = assign_anchor_targets(anchors_flat, gt_boxes, boxes_3d_iou)
            labels_t = torch.from_numpy(labels).to(device)
            valid_mask = labels_t != -1
            cls_targets = (labels_t == 1).float()

            cls_preds_flat = cls_preds.reshape(-1)
            c_loss = focal_loss(cls_preds_flat, cls_targets, valid_mask=valid_mask)

            pos_mask = labels_t == 1
            if pos_mask.sum() > 0 and len(gt_boxes) > 0:
                matched_gt_boxes = gt_boxes[matched_gt_idx[pos_mask.cpu().numpy()]]
                target_deltas = encode_boxes(matched_gt_boxes, anchors_flat[pos_mask.cpu().numpy()])
                target_deltas_t = torch.from_numpy(target_deltas).float().to(device)
                reg_preds_flat = reg_preds.reshape(-1, 7)
                r_loss = torch.nn.functional.smooth_l1_loss(
                    reg_preds_flat[pos_mask], target_deltas_t, beta=1 / 9, reduction="mean"
                )
            else:
                r_loss = torch.tensor(0.0, device=device)

            loss = c_loss + 2.0 * r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls_loss += c_loss.item()
            epoch_reg_loss += r_loss.item() if torch.is_tensor(r_loss) else float(r_loss)
            n_batches += 1

        print(f"epoch {epoch+1}/{total_epochs} | cls_loss {epoch_cls_loss/n_batches:.4f} "
              f"| reg_loss {epoch_reg_loss/n_batches:.4f}")

        is_last = (epoch + 1) == total_epochs
        if (epoch + 1) % args.save_every == 0 or is_last:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }, f"checkpoint_epoch{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--num_synthetic_scenes", type=int, default=500)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a checkpoint to resume from. --epochs then means "
                              "'how many additional epochs to run', not a new total.")
    parser.add_argument("--start_epoch", type=int, default=0,
                         help="Only used when resuming from an old-format checkpoint with no "
                              "embedded epoch count (new checkpoints auto-detect this).")
    args = parser.parse_args()
    train(args)
