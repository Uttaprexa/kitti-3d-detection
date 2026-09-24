"""
Coordinate transforms between KITTI's camera and LiDAR (velodyne) frames.

KITTI frames:
  - LiDAR (velo):  x=forward, y=left,  z=up
  - Camera (cam0-rect): x=right,  y=down,  z=forward

Calibration files provide:
  - Tr_velo_to_cam: (3,4) rigid transform, velo -> unrectified cam
  - R0_rect:        (3,3) rectification rotation, unrectified cam -> rectified cam
  - P2:              (3,4) projection matrix, rectified cam -> image pixels (left color cam)

KITTI 3D labels give box location (x,y,z) and rotation_y in the RECTIFIED
camera frame, with location = bottom-center of the box (not the centroid!)
and rotation_y measured around the camera's y-axis (down).
"""

import numpy as np


def load_kitti_calib(calib_path):
    """
    Parse a KITTI calib .txt file into a dict of numpy matrices.
    Expected keys in file: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
    """
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            nums = np.array([float(v) for v in value.strip().split()])
            calib[key.strip()] = nums

    def reshape(key, shape):
        if key in calib:
            calib[key] = calib[key].reshape(shape)

    reshape("P0", (3, 4))
    reshape("P1", (3, 4))
    reshape("P2", (3, 4))
    reshape("P3", (3, 4))
    reshape("R0_rect", (3, 3))
    reshape("Tr_velo_to_cam", (3, 4))
    reshape("Tr_imu_to_velo", (3, 4))
    return calib


def _to_homogeneous(points):
    """(N,3) -> (N,4) by appending a column of ones."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


def velo_to_cam_rect(points_velo, calib):
    """
    Transform points from LiDAR frame to rectified camera frame.
    points_velo: (N,3)
    """
    Tr = calib["Tr_velo_to_cam"]          # (3,4): velo -> unrect cam
    R0 = calib["R0_rect"]                 # (3,3): unrect cam -> rect cam
    pts_h = _to_homogeneous(points_velo)  # (N,4)
    cam_unrect = pts_h @ Tr.T              # (N,3)
    cam_rect = cam_unrect @ R0.T           # (N,3)
    return cam_rect


def cam_rect_to_velo(points_cam_rect, calib):
    """Inverse of velo_to_cam_rect: rectified camera frame -> LiDAR frame."""
    Tr = calib["Tr_velo_to_cam"]  # (3,4)
    R0 = calib["R0_rect"]         # (3,3)

    R0_inv = np.linalg.inv(R0)
    cam_unrect = points_cam_rect @ R0_inv.T  # (N,3)

    # Tr: velo->cam is [R|t] (3,4). Invert the rigid transform:
    R = Tr[:, :3]
    t = Tr[:, 3]
    R_inv = np.linalg.inv(R)
    points_velo = (cam_unrect - t) @ R_inv.T
    return points_velo


def project_to_image(points_cam_rect, P):
    """
    Project rectified-camera-frame 3D points into image pixel coordinates.
    points_cam_rect: (N,3)
    P: (3,4) projection matrix (e.g. calib['P2'])
    Returns: (N,2) pixel coords, (N,) depth (z in camera frame, for masking behind-camera pts)
    """
    pts_h = _to_homogeneous(points_cam_rect)  # (N,4)
    proj = pts_h @ P.T                         # (N,3)
    depth = proj[:, 2]
    pixels = proj[:, :2] / np.clip(depth[:, None], 1e-6, None)
    return pixels, depth


def kitti_label_to_velo_box(location_cam, dims_hwl, rotation_y, calib):
    """
    Convert a single KITTI label (camera-frame bottom-center location,
    dims as (h, w, l), rotation_y about camera y-axis) into the LiDAR-frame
    box convention used by boxes.py: (x, y, z, dx, dy, dz, heading), where
    z is the box CENTROID and heading is rotation about the LiDAR z-axis.

    Args:
        location_cam: (3,) [x, y, z] bottom-center, rectified camera frame
        dims_hwl: (3,) [h, w, l] KITTI order
        rotation_y: float, radians
        calib: dict from load_kitti_calib
    Returns:
        (7,) array [x, y, z, dx, dy, dz, heading] in LiDAR frame
    """
    h, w, l = dims_hwl
    loc_cam = np.asarray(location_cam, dtype=np.float64).reshape(1, 3)

    # KITTI location is bottom-center; shift up by h/2 to get the centroid
    # before transforming, since the transform is rigid (rotation+translation)
    # and centroid-in-cam maps directly to centroid-in-velo.
    centroid_cam = loc_cam.copy()
    centroid_cam[0, 1] -= h / 2.0  # camera y points DOWN, so "up" is -y

    centroid_velo = cam_rect_to_velo(centroid_cam, calib)[0]  # (3,)

    # rotation_y is measured around the camera's y-axis (pointing down).
    # The LiDAR z-axis points up, i.e. roughly opposite. For the common case
    # of a Tr_velo_to_cam with no roll/pitch (only a yaw + axis permutation),
    # heading_velo = -rotation_y - pi/2 recovers the standard PointPillars
    # convention where heading=0 means the box's length (dx) points along
    # the LiDAR +x (forward) axis. This is the standard KITTI devkit relation.
    heading_velo = -rotation_y - np.pi / 2.0

    box = np.array([
        centroid_velo[0], centroid_velo[1], centroid_velo[2],
        l, w, h,
        heading_velo,
    ])
    return box
