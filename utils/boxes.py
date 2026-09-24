"""
3D box math for LiDAR-frame bounding boxes.

Box parameterization: (x, y, z, dx, dy, dz, heading)
  - x, y, z   : center of box in LiDAR frame (x=forward, y=left, z=up)
  - dx, dy, dz: box length (along x before rotation), width (along y), height (z)
  - heading   : rotation about the z-axis, radians, right-hand rule

This is the standard PointPillars / OpenPCDet box convention, and it's the
frame KITTI labels get converted into after camera->LiDAR transform.
"""

import numpy as np


def boxes_to_bev_corners(boxes):
    """
    Compute the 4 bird's-eye-view (BEV) corners of each box (ignoring z/height).

    Args:
        boxes: (N, 7) array [x, y, z, dx, dy, dz, heading]
    Returns:
        corners: (N, 4, 2) array of (x, y) corners, ordered counter-clockwise
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    N = boxes.shape[0]
    x, y = boxes[:, 0], boxes[:, 1]
    dx, dy = boxes[:, 3], boxes[:, 4]
    heading = boxes[:, 6]

    # Corners in the box's own frame (before rotation), CCW order
    half_l = dx[:, None] / 2.0
    half_w = dy[:, None] / 2.0
    local_x = np.concatenate([half_l, half_l, -half_l, -half_l], axis=1)   # (N,4)
    local_y = np.concatenate([half_w, -half_w, -half_w, half_w], axis=1)   # (N,4)

    cos_h = np.cos(heading)[:, None]
    sin_h = np.sin(heading)[:, None]

    rot_x = local_x * cos_h - local_y * sin_h + x[:, None]
    rot_y = local_x * sin_h + local_y * cos_h + y[:, None]

    corners = np.stack([rot_x, rot_y], axis=-1)  # (N, 4, 2)
    return corners


def _polygon_area(poly):
    """Shoelace formula. poly: (M, 2) ordered vertices (either winding)."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon(subject_poly, clip_poly):
    """
    Sutherland-Hodgman polygon clipping. Clips subject_poly against the
    convex polygon clip_poly. Both must be given as CCW vertex lists.
    Returns the clipped polygon (possibly empty).
    """
    def inside(p, a, b):
        # True if p is on the left side of directed edge a->b (CCW convex clip)
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    output = list(subject_poly)
    for i in range(len(clip_poly)):
        if not output:
            break
        a, b = clip_poly[i], clip_poly[(i + 1) % len(clip_poly)]
        input_list = output
        output = []
        for j in range(len(input_list)):
            cur = input_list[j]
            prev = input_list[j - 1]
            cur_in = inside(cur, a, b)
            prev_in = inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, a, b))
    return np.array(output) if output else np.empty((0, 2))


def _ensure_ccw(poly):
    """Sutherland-Hodgman needs both polygons wound the same way (CCW here)."""
    area_signed = np.sum(
        (poly[:, 0] * np.roll(poly[:, 1], -1)) - (np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    return poly if area_signed >= 0 else poly[::-1]


def bev_iou(boxes_a, boxes_b):
    """
    Pairwise BEV (bird's-eye-view) IoU between two sets of rotated boxes.

    Args:
        boxes_a: (N, 7)
        boxes_b: (M, 7)
    Returns:
        iou: (N, M)
    """
    corners_a = boxes_to_bev_corners(boxes_a)
    corners_b = boxes_to_bev_corners(boxes_b)
    N, M = len(boxes_a), len(boxes_b)
    iou = np.zeros((N, M))

    for i in range(N):
        poly_a = _ensure_ccw(corners_a[i])
        area_a = _polygon_area(poly_a)
        for j in range(M):
            poly_b = _ensure_ccw(corners_b[j])
            area_b = _polygon_area(poly_b)
            inter_poly = _clip_polygon(poly_a, poly_b)
            inter_area = _polygon_area(inter_poly)
            union = area_a + area_b - inter_area
            iou[i, j] = inter_area / union if union > 1e-9 else 0.0
    return iou


def boxes_3d_iou(boxes_a, boxes_b):
    """
    Pairwise full 3D IoU: BEV polygon intersection * height overlap,
    divided by total volume. Standard KITTI/PointPillars approximation
    (exact for axis-aligned height, which matches the KITTI box convention
    since real-world objects don't pitch/roll).

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading] -- z is box CENTER
        boxes_b: (M, 7)
    Returns:
        iou_3d: (N, M)
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    N, M = len(boxes_a), len(boxes_b)

    corners_a = boxes_to_bev_corners(boxes_a)
    corners_b = boxes_to_bev_corners(boxes_b)

    z_a_min = boxes_a[:, 2] - boxes_a[:, 5] / 2.0
    z_a_max = boxes_a[:, 2] + boxes_a[:, 5] / 2.0
    z_b_min = boxes_b[:, 2] - boxes_b[:, 5] / 2.0
    z_b_max = boxes_b[:, 2] + boxes_b[:, 5] / 2.0

    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    iou = np.zeros((N, M))
    for i in range(N):
        poly_a = _ensure_ccw(corners_a[i])
        area_a = _polygon_area(poly_a)
        for j in range(M):
            poly_b = _ensure_ccw(corners_b[j])
            area_b = _polygon_area(poly_b)

            inter_poly = _clip_polygon(poly_a, poly_b)
            inter_area = _polygon_area(inter_poly)
            if inter_area <= 1e-9:
                continue

            h_overlap = min(z_a_max[i], z_b_max[j]) - max(z_a_min[i], z_b_min[j])
            if h_overlap <= 0:
                continue

            inter_vol = inter_area * h_overlap
            union_vol = vol_a[i] + vol_b[j] - inter_vol
            iou[i, j] = inter_vol / union_vol if union_vol > 1e-9 else 0.0
    return iou


def box3d_to_corners_3d(boxes):
    """
    Full 8-corner 3D box representation (for visualization / export).

    Args:
        boxes: (N, 7)
    Returns:
        corners: (N, 8, 3), order: 4 bottom corners CCW, then 4 top corners CCW
    """
    bev = boxes_to_bev_corners(boxes)  # (N,4,2)
    z = boxes[:, 2]
    dz = boxes[:, 5]
    z_bottom = (z - dz / 2.0)[:, None]
    z_top = (z + dz / 2.0)[:, None]

    bottom = np.concatenate([bev, np.repeat(z_bottom, 4, axis=1)[:, :, None]], axis=-1)
    top = np.concatenate([bev, np.repeat(z_top, 4, axis=1)[:, :, None]], axis=-1)
    return np.concatenate([bottom, top], axis=1)  # (N, 8, 3)
