# 3D Point Cloud Object Detection on KITTI

A PointPillars-based 3D object detection pipeline for LiDAR point clouds:
data ingestion (LiDAR + camera calibration), coordinate transforms between
sensor frames, model, training, and evaluation with standard 3D detection
metrics (3D IoU, mAP by distance threshold).

## Results (trained on real KITTI data)

Trained on a 1000-scene subset of the real KITTI training set (850 train /
150 held-out val), on an AWS EC2 `g4dn.xlarge` (Tesla T4 GPU), resuming
training across sessions via a checkpoint format that saves optimizer state
alongside model weights.

| Metric | Epoch 20 | Epoch 80 | Epoch 140 (final) |
|---|---|---|---|
| Regression loss (box-fitting error) | 0.242 | 0.061 | **0.034** |
| Near-field AP, 0-30m (loose IoU>=0.1) | 0.023 | 0.266 | **0.359** |
| Mid-range AP, 30-50m (loose IoU>=0.1) | 0.015 | 0.053 | **0.094** |
| Far-field AP, 50m+ (loose IoU>=0.1) | 0.003 | 0.009 | **0.009** |

Near-field detection improved ~15.6x and mid-range ~6.3x from epoch 20 to
140, with box-regression loss dropping 86% overall. Far-field AP plateaued
between epoch 80 and 140 (0.009 -> 0.009) -- consistent with LiDAR point
density falling off sharply with distance (fewer laser returns hit distant
objects), so far-field objects have fundamentally less signal to learn
from at this scene count, regardless of additional epochs on the same data.

At the standard strict benchmark bar (score>=0.3 confidence, 3D IoU>=0.5),
3 predictions cleared the bar at epoch 140 (up from 0 at epoch 80) --
still far from reliable detection by that standard, but the first sign of
any prediction clearing it at all. Training was stopped at epoch 140 given
visibly diminishing returns (near-field growth slowed from 11x in the
first 60 additional epochs to 1.4x in the next 60) and a plateaued
far-field metric -- further improvement would more likely need more
training scenes than more epochs on the same 850.

Measured training throughput on this setup: **~7.5 minutes/epoch** on 850
scenes. That rate is dominated by anchor-to-groundtruth matching, which is
pure numpy/CPU regardless of GPU (see the bug-fix log below) -- the actual
GPU forward/backward pass is not the bottleneck here.

## What's here

```
data/
  synthetic_generator.py   Synthetic LiDAR scenes for dev/testing (no KITTI download needed)
  transforms.py             Camera <-> LiDAR frame transforms, KITTI calib parsing
  kitti_dataset.py          PyTorch Dataset (real KITTI files, or synthetic mode)
utils/
  boxes.py                  3D box corners, rotated BEV IoU, full 3D IoU
  anchors.py                 Anchor grid generation, box encode/decode
  losses.py                  Focal loss (classification), smooth L1 (box regression)
model/
  pointpillars.py            PillarFeatureNet + 2D CNN backbone + detection head
eval/
  metrics.py                 3D-IoU matching, AP, mAP broken down by distance
train.py                     Training loop (real KITTI or --synthetic; supports --resume_from)
evaluate.py                  Inference + NMS + mAP evaluation
demo_geometry.py             Runnable without torch: full non-torch pipeline demo
```

## Status: what's verified vs. what needs torch

This was built in a sandbox with **no internet access and no PyTorch
installed**, so I split the work into two tiers:

**Verified and tested right now** (pure numpy/scipy, no torch needed):
- `utils/boxes.py` -- rotated 3D IoU, checked against 5 hand-calculated
  cases (identical boxes, rotated boxes, disjoint boxes, height-disjoint
  boxes, a half-overlap cube) -- all exact.
- `data/transforms.py` -- camera/LiDAR round-trip tested against a real
  KITTI calibration matrix (velo->cam->velo error: 1.8e-15, i.e. exact).
  The full label round-trip (camera-frame box -> LiDAR box -> back) has a
  ~8mm residual, which comes from the real calibration's small sensor tilt
  interacting with the standard KITTI devkit approximation (shift bottom-
  center by h/2 along the camera's local "up") -- this is the same
  approximation the official devkit uses, not a bug in this code.
- `utils/anchors.py` -- anchor grid shape check, and an encode/decode
  round-trip on 20 random boxes (max error 1.5e-5, floating-point noise).
- `eval/metrics.py` -- a near-perfect simulated detector scores AP=1.0 in
  every distance bucket; random noise predictions score AP=0.0. Both as
  expected.
- `demo_geometry.py` -- runs the entire non-torch pipeline end to end:
  generates 50 synthetic scenes, builds the anchor grid, simulates
  realistic detector output (localization noise + misses + false
  positives), computes mAP by distance bucket, and renders a BEV
  visualization (`demo_scene_result.png`).

**Written to spec but NOT executed here** (needs `pip install torch`):
- `model/pointpillars.py` -- the actual learned model.
- `utils/losses.py` -- torch-based loss wiring.
- `train.py`, `evaluate.py` -- the training/inference loops.

These were written carefully (shapes traced through by hand, channel
counts double-checked -- see comments), and `pointpillars.py` has a
`__main__` self-test block for a quick shape check once torch is
available. But treat them as a strong first draft to run and debug, not
as already-proven code the way the numpy pieces above are.

## Fixed after real-world testing (2026-09-22)

Two bugs surfaced once this was actually run on a real machine with torch
installed, both now fixed and the fix verified in this sandbox (the
`assign_anchor_targets` fix is pure numpy, so it was re-tested directly
here; the backbone/model fix could only be traced through by hand, not
executed, since this sandbox still has no torch):

1. **`assign_anchor_targets` was computing rotated 3D IoU between every
   anchor and every GT box** -- with the original backbone (see #2 below)
   that meant 440,000 anchors x ~5 GT boxes = ~2.2M pure-Python
   polygon-clip calls per scene, which never finished. Fixed by spatially
   pruning to anchors within a radius of each GT box first (vectorized
   numpy distance check), then only running the expensive IoU on that
   much smaller candidate set. Verified: with the anchor count from fix
   #2 below, this now takes ~0.9s/scene instead of hanging indefinitely.
   Also fixed a latent type bug where the empty-GT-boxes branch returned
   torch tensors while the normal path returned numpy arrays -- would
   have crashed `train.py`'s `torch.from_numpy(labels)` call on any scene
   with zero GT boxes.

2. **The backbone ran the detection head at full pillar resolution**
   (500x440x2 = 440,000 anchors for this repo's default point cloud
   range) because it upsampled every multi-scale feature back up to full
   resolution before the head. Real PointPillars runs its head on a
   downsampled feature map specifically to avoid this. Fixed by having
   `Backbone2D` pool the shallower features DOWN to the coarsest scale
   (stride 4, ~125x110 for the default range -> 27,500 anchors, a ~16x
   reduction) instead of upsampling everything up. Also fixed a related
   bug this uncovered: `PointPillars.forward` was returning the pillar
   pseudo-image's resolution as the "feature map size" instead of the
   backbone's actual output resolution -- harmless before (they happened
   to match), but would have silently misaligned anchors after this fix
   if left in place. `train.py`/`evaluate.py` now read the true feature
   map size off a dummy forward pass rather than assuming a downsample
   factor, so this can't drift out of sync again if the architecture
   changes further.

**Practical note for training on a CPU (e.g. a MacBook, no CUDA):** even
after these fixes, anchor assignment runs at roughly ~0.9s/scene in pure
Python/numpy. A full epoch over hundreds of scenes will take minutes, and
the CNN forward/backward pass adds more on top. For a first sanity check,
use a small scene count and few epochs:
```bash
python train.py --synthetic --num_synthetic_scenes 50 --epochs 5
```
before scaling up. Real training on the full KITTI training set is
realistically a GPU job (local CUDA GPU or Colab), not something to run
end-to-end on a laptop CPU.

## Two more bugs found once real GPU training actually ran (2026-09-23)

1. **`PillarFeatureNet.pillarize` used a Python for-loop with `.item()`
   calls per point.** Harmless on a few thousand synthetic points, but
   real KITTI scans have ~100,000-120,000 points per scene, and each
   `.item()` call forces a GPU<->CPU sync -- with ~240,000 such calls per
   scene, this would have made real-data training impractically slow
   regardless of GPU compute. Rewrote fully vectorized (stable-sort +
   cumulative-group-offset trick to assign each point a slot within its
   pillar, then vectorized scatter/reduction for the centroid and
   pillar-center offset features). Verified byte-for-byte identical
   output against the original loop logic on random test data before
   this was ever run on real hardware.

2. **NMS blew up at low confidence thresholds.** `evaluate.py`'s
   non-max-suppression compares every surviving candidate box against
   every other pairwise, using the same pure-Python polygon-clip IoU as
   elsewhere in this repo. At a normal confidence threshold this stayed
   small, but a diagnostic run at a much lower threshold (`--score_threshold
   0.02`, used to check whether the model had "quiet" signal below the
   normal bar) let thousands of candidates through per scene, turning a
   should-be-quick eval into a multi-hour hang. Fixed with a standard
   `max_pre_nms` cap -- keep only the top-scoring N candidates (default
   300) before running the expensive pairwise step, exactly like real
   detectors do.

Also added checkpoint resume support (`--resume_from`, `--start_epoch`) to
`train.py` once real training moved to a paid cloud GPU, so a multi-hour
run could be extended across sessions rather than restarted from scratch
every time. Checkpoints now save optimizer state alongside model weights;
old-format checkpoints (raw state_dict, no optimizer state) still load
fine, just restart the optimizer's momentum/adaptive-LR history.

## The infrastructure path, briefly

CPU (Mac) for pipeline development and the synthetic-data sanity checks
-> Google Colab (free T4 GPU) for a first real-data attempt, which is
where the pillar-resolution/anchor-count bug above became a hard blocker
-> AWS EC2 `g4dn.xlarge` (paid T4 GPU, ~$0.53/hr on-demand) for the actual
multi-hour training runs, using `tmux` so training survives dropped SSH
connections. Total real-data training: 140 epochs on 850 scenes, in three
sessions (20 -> 80 -> 140) resumed via checkpoint, ~7.5 min/epoch,
~$9-10 of total compute.

## Running it

```bash
# Right now, no setup needed:
pip install numpy scipy matplotlib scikit-learn
python demo_geometry.py

# Sanity-check training on synthetic data (any machine with torch):
pip install torch
python train.py --synthetic --epochs 20
python evaluate.py --checkpoint checkpoint_epoch20.pt --synthetic

# On real KITTI, after downloading velodyne/, label_2/, calib/ from
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# (realistically a GPU job -- see throughput note above):
python train.py --root_dir /path/to/kitti --split_file train.txt --epochs 140
python evaluate.py --checkpoint checkpoint_epoch140.pt --root_dir /path/to/kitti --split_file val.txt \
    --score_threshold 0.02 --iou_threshold 0.1   # loose thresholds show early-training signal

# Resume a long run across sessions instead of restarting from scratch:
python train.py --root_dir /path/to/kitti --split_file train.txt \
    --resume_from checkpoint_epoch80.pt --start_epoch 80 --epochs 60
```

## Design notes worth knowing for an interview

- **Box convention**: `(x, y, z, dx, dy, dz, heading)` in the LiDAR frame,
  z = box centroid (not bottom, unlike raw KITTI labels), heading about
  the LiDAR z-axis. `kitti_label_to_velo_box` handles the conversion from
  KITTI's camera-frame/bottom-center convention.
- **3D IoU**: computed as BEV polygon intersection (via Sutherland-Hodgman
  clipping of the two rotated rectangles) times height overlap, divided by
  total volume. This is exact for the KITTI setting since real objects
  don't pitch/roll -- it's the same simplification used in PointPillars/
  SECOND/OpenPCDet-style eval code, not a shortcut unique to this repo.
- **Anchors**: one size per class (car defaults: l=3.9, w=1.6, h=1.56m,
  z=-1.0m) at two headings (0, 90 deg) per grid location -- the original
  PointPillars anchor design, before rotation-based data augmentation
  tricks used in later work.
- **mAP by distance**: buckets ground truth/predictions by range from the
  sensor (0-30m / 30-50m / 50m+) before computing AP per bucket -- this is
  the natural way to show a model's accuracy degrades with range (fewer
  LiDAR returns hit distant objects), which is a more informative story
  than a single dataset-wide mAP number.
