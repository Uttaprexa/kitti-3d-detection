# 3D Point Cloud Object Detection on KITTI

A PointPillars-based 3D object detection pipeline for LiDAR point clouds:
data ingestion (LiDAR + camera calibration), coordinate transforms between
sensor frames, model, training, and evaluation with standard 3D detection
metrics (3D IoU, mAP by distance threshold).

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
train.py                     Training loop (real KITTI or --synthetic)
evaluate.py                  Inference + NMS + mAP evaluation
demo_geometry.py             Runnable NOW: full non-torch pipeline demo
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

## Running it

```bash
# Right now, no setup needed:
pip install numpy scipy matplotlib scikit-learn
python demo_geometry.py

# Once you have torch + a GPU (local or Colab):
pip install torch
python train.py --synthetic --epochs 20        # sanity-check training on synthetic data
python evaluate.py --checkpoint checkpoint_epoch20.pt --synthetic

# On real KITTI, after downloading velodyne/, label_2/, calib/ from
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d :
python train.py --root_dir /path/to/kitti --split_file train.txt --epochs 40
python evaluate.py --checkpoint checkpoint_epoch40.pt --root_dir /path/to/kitti --split_file val.txt
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
