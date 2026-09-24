"""
Generates synthetic LiDAR scenes that mimic KITTI's structure closely enough
to exercise the whole pipeline (data loading -> voxelization -> model ->
loss -> eval) without needing to download the real (dozens-of-GB) dataset.

Each scene: a ground plane of sparse points, plus a handful of car-shaped
point clusters at random positions/headings, with matching ground-truth
boxes in the LiDAR-frame 7-DoF convention used throughout this repo.
"""

import numpy as np


CAR_DIMS_MEAN = np.array([3.9, 1.6, 1.56])  # l, w, h (typical KITTI car)
CAR_DIMS_STD = np.array([0.4, 0.15, 0.1])

POINT_CLOUD_RANGE = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 1.0])  # x_min,y_min,z_min,x_max,y_max,z_max


def _sample_box_surface_points(box, n_points, rng, noise_std=0.02):
    """
    Sample points on the surface of a rotated box, roughly mimicking how a
    LiDAR only sees the near-facing faces of an object (not the far side).
    box: (x,y,z,dx,dy,dz,heading)
    """
    x, y, z, dx, dy, dz, heading = box
    # Sample points on 3 visible faces (front + two sides), biased by a
    # simple visibility heuristic: faces whose outward normal has negative
    # dot product with the vector from sensor (origin) to box are "near".
    faces = []
    half = np.array([dx, dy, dz]) / 2.0

    candidate_faces = [
        (np.array([1, 0, 0]), np.array([0, half[1], half[2]])),   # +x face
        (np.array([-1, 0, 0]), np.array([0, half[1], half[2]])),  # -x face
        (np.array([0, 1, 0]), np.array([half[0], 0, half[2]])),   # +y face
        (np.array([0, -1, 0]), np.array([half[0], 0, half[2]])),  # -y face
    ]

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])

    center = np.array([x, y, z])
    to_center = center  # vector from sensor origin to box center
    to_center_dir = to_center / (np.linalg.norm(to_center) + 1e-9)

    for normal_local, extent_local in candidate_faces:
        normal_world = R @ normal_local
        # keep faces roughly facing the sensor
        if np.dot(normal_world, -to_center_dir) > -0.2:
            faces.append((normal_local, extent_local, half - np.abs(normal_local) * half))

    if not faces:
        faces = [candidate_faces[0] + (half,)]

    pts = []
    per_face = max(1, n_points // max(1, len(faces)))
    for normal_local, extent_local, face_half in faces:
        u = rng.uniform(-1, 1, size=per_face)
        v = rng.uniform(-1, 1, size=per_face)
        local = np.zeros((per_face, 3))
        offset_axes = [i for i in range(3) if normal_local[i] == 0]
        local[:, offset_axes[0]] = u * half[offset_axes[0]]
        local[:, offset_axes[1]] = v * half[offset_axes[1]]
        fixed_axis = [i for i in range(3) if normal_local[i] != 0][0]
        local[:, fixed_axis] = normal_local[fixed_axis] * half[fixed_axis]
        world = local @ R.T + center
        world += rng.normal(0, noise_std, size=world.shape)
        pts.append(world)
    return np.concatenate(pts, axis=0)


def generate_synthetic_scene(num_objects=None, rng=None, ground_points=3000, max_objects=8):
    """
    Generate one synthetic LiDAR scene.

    Returns:
        points: (N, 4) array [x, y, z, intensity]
        gt_boxes: (M, 7) array [x, y, z, dx, dy, dz, heading]
        gt_labels: (M,) array of ints (all 0 = "Car" for this simplified demo)
    """
    rng = rng or np.random.default_rng()
    if num_objects is None:
        num_objects = rng.integers(2, max_objects + 1)

    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    ground_z = -1.6  # typical KITTI velo-frame ground height

    # --- ground plane, sparse, with mild noise ---
    gx = rng.uniform(x_min, x_max, ground_points)
    gy = rng.uniform(y_min, y_max, ground_points)
    gz = ground_z + rng.normal(0, 0.03, ground_points)
    ground_pts = np.stack([gx, gy, gz], axis=1)

    # --- object boxes: reject overlapping placements ---
    boxes = []
    attempts = 0
    while len(boxes) < num_objects and attempts < num_objects * 20:
        attempts += 1
        bx = rng.uniform(x_min + 5, x_max - 5)
        by = rng.uniform(y_min + 5, y_max - 5)
        bz = ground_z + CAR_DIMS_MEAN[2] / 2.0  # sits on the ground
        dims = np.clip(rng.normal(CAR_DIMS_MEAN, CAR_DIMS_STD), CAR_DIMS_MEAN * 0.5, None)
        heading = rng.uniform(-np.pi, np.pi)
        candidate = np.array([bx, by, bz, dims[0], dims[1], dims[2], heading])

        ok = True
        for existing in boxes:
            dist = np.hypot(candidate[0] - existing[0], candidate[1] - existing[1])
            if dist < (candidate[3] + existing[3]) / 2.0 + 1.0:
                ok = False
                break
        if ok:
            boxes.append(candidate)

    gt_boxes = np.array(boxes) if boxes else np.zeros((0, 7))

    # --- points on each object's visible surface ---
    obj_point_lists = []
    for box in gt_boxes:
        dist = np.hypot(box[0], box[1])
        n_pts = int(np.clip(400 - dist * 4, 20, 400))  # farther objects, fewer returns
        pts = _sample_box_surface_points(box, n_pts, rng)
        obj_point_lists.append(pts)

    all_xyz = [ground_pts] + obj_point_lists if obj_point_lists else [ground_pts]
    xyz = np.concatenate(all_xyz, axis=0)
    intensity = rng.uniform(0.0, 1.0, size=(xyz.shape[0], 1))
    points = np.concatenate([xyz, intensity], axis=1)

    # keep points within the defined point cloud range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    points = points[mask]

    gt_labels = np.zeros(len(gt_boxes), dtype=np.int64)  # single class: Car
    return points, gt_boxes, gt_labels


def generate_synthetic_dataset(num_scenes, seed=0):
    """Generate a list of (points, gt_boxes, gt_labels) scenes."""
    rng = np.random.default_rng(seed)
    return [generate_synthetic_scene(rng=rng) for _ in range(num_scenes)]
