"""
Dense anchor grid generation, PointPillars-style: one anchor size per class,
placed at every location of the (downsampled) BEV feature map, at a small
set of fixed rotations.
"""

import numpy as np


def generate_anchors(
    point_cloud_range,
    feature_map_size,   # (H, W) of the BEV feature map the anchors align to
    anchor_size=(3.9, 1.6, 1.56),   # (l, w, h) - "Car" class default
    anchor_z=-1.0,                  # anchor center height (KITTI ground ~ -1.0m to -1.7m)
    rotations=(0.0, np.pi / 2),
):
    """
    Returns:
        anchors: (H, W, R, 7) array of [x, y, z, dx, dy, dz, heading]
    """
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    H, W = feature_map_size

    x_centers = np.linspace(x_min, x_max, W, endpoint=False) + (x_max - x_min) / (2 * W)
    y_centers = np.linspace(y_min, y_max, H, endpoint=False) + (y_max - y_min) / (2 * H)

    xx, yy = np.meshgrid(x_centers, y_centers)  # each (H, W)
    R = len(rotations)

    anchors = np.zeros((H, W, R, 7), dtype=np.float32)
    for r_idx, rot in enumerate(rotations):
        anchors[:, :, r_idx, 0] = xx
        anchors[:, :, r_idx, 1] = yy
        anchors[:, :, r_idx, 2] = anchor_z
        anchors[:, :, r_idx, 3] = anchor_size[0]
        anchors[:, :, r_idx, 4] = anchor_size[1]
        anchors[:, :, r_idx, 5] = anchor_size[2]
        anchors[:, :, r_idx, 6] = rot
    return anchors


def encode_boxes(gt_boxes, anchors):
    """
    Encode GT boxes relative to matched anchors (PointPillars/SECOND-style
    residual encoding), for use as regression targets.

    Args:
        gt_boxes: (N, 7)
        anchors:  (N, 7)  -- already matched 1:1 with gt_boxes
    Returns:
        targets: (N, 7) [dx, dy, dz, dl, dw, dh, dheading]
    """
    xa, ya, za, la, wa, ha, ra = [anchors[:, i] for i in range(7)]
    xg, yg, zg, lg, wg, hg, rg = [gt_boxes[:, i] for i in range(7)]

    diag = np.sqrt(la ** 2 + wa ** 2)  # diagonal normalizer (SECOND paper)
    dx = (xg - xa) / (diag + 1e-6)
    dy = (yg - ya) / (diag + 1e-6)
    dz = (zg - za) / (ha + 1e-6)
    dl = np.log(lg / (la + 1e-6) + 1e-6)
    dw = np.log(wg / (wa + 1e-6) + 1e-6)
    dh = np.log(hg / (ha + 1e-6) + 1e-6)
    dr = rg - ra

    return np.stack([dx, dy, dz, dl, dw, dh, dr], axis=1)


def decode_boxes(deltas, anchors):
    """Inverse of encode_boxes: turn network output deltas back into boxes."""
    xa, ya, za, la, wa, ha, ra = [anchors[:, i] for i in range(7)]
    dx, dy, dz, dl, dw, dh, dr = [deltas[:, i] for i in range(7)]

    diag = np.sqrt(la ** 2 + wa ** 2)
    xg = dx * diag + xa
    yg = dy * diag + ya
    zg = dz * ha + za
    lg = np.exp(dl) * la
    wg = np.exp(dw) * wa
    hg = np.exp(dh) * ha
    rg = dr + ra

    return np.stack([xg, yg, zg, lg, wg, hg, rg], axis=1)
