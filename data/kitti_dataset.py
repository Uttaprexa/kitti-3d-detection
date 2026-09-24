"""
Dataset class for KITTI 3D object detection, with a `synthetic=True` mode
that generates scenes on the fly instead of reading real KITTI files --
useful for developing/testing the pipeline before the ~12GB+ real dataset
is downloaded (see README for the real-KITTI directory layout expected).

Requires PyTorch. See model/pointpillars.py docstring for sandbox note.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import load_kitti_calib, kitti_label_to_velo_box
from data.synthetic_generator import generate_synthetic_scene


KITTI_CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def load_kitti_bin_pointcloud(bin_path):
    """KITTI velodyne .bin files: flat float32 array of [x,y,z,intensity]."""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def load_kitti_label_file(label_path, calib, classes=("Car",)):
    """
    Parse a KITTI label .txt file into LiDAR-frame boxes.
    Each line: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
    """
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return np.zeros((0, 7)), np.zeros((0,), dtype=np.int64)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] not in classes or parts[0] == "DontCare":
                continue
            obj_type = parts[0]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            rotation_y = float(parts[14])

            box = kitti_label_to_velo_box(
                location_cam=[x, y, z], dims_hwl=[h, w, l],
                rotation_y=rotation_y, calib=calib,
            )
            boxes.append(box)
            labels.append(KITTI_CLASSES[obj_type])

    if not boxes:
        return np.zeros((0, 7)), np.zeros((0,), dtype=np.int64)
    return np.array(boxes), np.array(labels, dtype=np.int64)


class KittiDetectionDataset(Dataset):
    """
    Args:
        root_dir: expected layout for real KITTI mode:
            root_dir/velodyne/{id}.bin
            root_dir/label_2/{id}.txt
            root_dir/calib/{id}.txt
        split_ids: list of sample id strings (e.g. ["000000", "000001", ...])
        synthetic: if True, ignores root_dir/split_ids and generates scenes
                   on the fly (for pipeline development without real data)
        num_synthetic_scenes: dataset length when synthetic=True
        synthetic_seed: RNG seed for reproducible synthetic scenes
    """

    def __init__(self, root_dir=None, split_ids=None, synthetic=False,
                 num_synthetic_scenes=200, synthetic_seed=0):
        self.synthetic = synthetic
        if synthetic:
            self.num_synthetic_scenes = num_synthetic_scenes
            self._rng = np.random.default_rng(synthetic_seed)
            # pre-generate for a stable, indexable dataset
            self._scenes = [
                generate_synthetic_scene(rng=self._rng)
                for _ in range(num_synthetic_scenes)
            ]
        else:
            assert root_dir is not None and split_ids is not None, \
                "root_dir and split_ids are required when synthetic=False"
            self.root_dir = root_dir
            self.split_ids = split_ids

    def __len__(self):
        return self.num_synthetic_scenes if self.synthetic else len(self.split_ids)

    def __getitem__(self, idx):
        if self.synthetic:
            points, gt_boxes, gt_labels = self._scenes[idx]
        else:
            sample_id = self.split_ids[idx]
            bin_path = os.path.join(self.root_dir, "velodyne", f"{sample_id}.bin")
            label_path = os.path.join(self.root_dir, "label_2", f"{sample_id}.txt")
            calib_path = os.path.join(self.root_dir, "calib", f"{sample_id}.txt")

            points = load_kitti_bin_pointcloud(bin_path)
            calib = load_kitti_calib(calib_path)
            gt_boxes, gt_labels = load_kitti_label_file(label_path, calib)

        return {
            "points": torch.from_numpy(points).float(),
            "gt_boxes": torch.from_numpy(gt_boxes).float(),
            "gt_labels": torch.from_numpy(gt_labels).long(),
        }


def collate_single_scene(batch):
    """
    PointPillars in this repo processes one scene at a time (batch_size=1
    through the pillar scatter step -- see PillarFeatureNet.forward note),
    so this collate function just unwraps a batch of 1 rather than padding
    variable-length point clouds/boxes into a dense tensor.
    """
    assert len(batch) == 1, "This repo's pillar scatter step assumes batch_size=1; wrap training in a loop over batch=1 samples, or extend PillarFeatureNet to carry a batch index per pillar for true batching."
    return batch[0]
