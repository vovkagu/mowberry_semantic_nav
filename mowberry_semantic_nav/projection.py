"""Pure geometry helpers for associating camera detections with LiDAR ranges.

This module has no ROS runtime dependency (only numpy and duck-typed
geometry_msgs objects), so it can be unit tested without a running graph.

Coordinate frame conventions used throughout
--------------------------------------------
* Laser frame (e.g. ``laser``): REP-103 body frame. +X forward, +Y left,
  +Z up. A 2D scan lies in the Z = 0 plane of this frame.
* Camera optical frame (e.g. ``usb_camera_optical_link``): REP-103 optical
  convention. +Z forward (along the lens axis), +X right, +Y down.
* Image pixels: column ``u`` grows to the right, row ``v`` grows downwards.

Pinhole model (horizontal axis only)
------------------------------------
A 3D point ``(x, y, z)`` expressed in the optical frame lands on image column::

    u = cx + fx * x / z          (only meaningful for z > 0)

``fx`` is the focal length in pixels and ``cx`` the principal point column.
Without a calibration, both can be estimated from the horizontal FOV::

    fx = (width / 2) / tan(hfov / 2),    cx = width / 2

Why we project laser points into the image (and not the other way round)
------------------------------------------------------------------------
The camera and the LiDAR are mounted at different positions on the robot.
Turning a pixel into a bearing angle and reading the laser beam at that same
angle ignores this offset (parallax) and picks the wrong beam for nearby
objects. Instead every laser hit is transformed into the camera optical
frame with the full TF (translation + rotation, so camera tilt is handled
too) and projected onto the image. Hits whose column falls inside a bounding
box are, by construction, the parts of the scan plane the camera sees inside
that box.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Horizontal pinhole parameters for a specific image resolution."""

    fx: float
    cx: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width, height, hfov_rad):
        """Estimate intrinsics from the horizontal field of view."""
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        return cls(fx=fx, cx=width / 2.0, width=width, height=height)

    @classmethod
    def from_camera_info(cls, k, info_width, info_height, width, height):
        """Build intrinsics from a CameraInfo K matrix.

        If the CameraInfo resolution differs from the image the VLM actually
        processed, ``fx`` and ``cx`` are scaled linearly with the width.
        """
        scale = width / float(info_width)
        return cls(fx=k[0] * scale, cx=k[2] * scale,
                   width=width, height=height)


def transform_to_matrix(transform):
    """Convert a ``geometry_msgs/Transform`` into a 4x4 homogeneous matrix.

    The matrix maps points expressed in the transform's *child* (source)
    frame into its *parent* (target) frame: ``p_target = M @ p_source``.
    """
    t = transform.translation
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # Standard unit-quaternion to rotation-matrix conversion.
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (t.x, t.y, t.z)
    return matrix


def apply_transform(matrix, points):
    """Apply a 4x4 homogeneous transform to an ``(N, 3)`` array of points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """Convert LaserScan polar ranges to ``(N, 3)`` Cartesian laser points.

    Invalid returns (NaN, inf, or outside the sensor's valid interval) are
    dropped, so the result may hold fewer points than there are beams.
    """
    r = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(r.size) * angle_increment
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    r, angles = r[valid], angles[valid]
    # Polar -> Cartesian in the scan plane (Z = 0 of the laser frame).
    return np.column_stack(
        (r * np.cos(angles), r * np.sin(angles), np.zeros_like(r)))


def locate_object_in_scan(points_laser, laser_to_optical, intrinsics,
                          u_min, u_max, range_percentile=20.0,
                          cluster_tolerance=0.3, min_depth=0.1):
    """Find the laser hits belonging to an object seen between two columns.

    Args:
        points_laser: ``(N, 3)`` scan points in the laser frame.
        laser_to_optical: 4x4 matrix mapping laser points into the camera
            optical frame.
        intrinsics: :class:`PinholeIntrinsics` of the analysed image.
        u_min, u_max: horizontal pixel span of the bounding box.
        range_percentile: the object is assumed to be the *near* part of
            what is visible inside the box (background shows through on
            the sides of a trunk or door frame), so the reference distance
            is this low percentile of the in-box distances.
        cluster_tolerance: hits within this many metres of the reference
            distance are considered part of the object.
        min_depth: ignore points closer than this along the lens axis (also
            excludes everything behind the camera).

    Returns:
        ``(centroid_laser, distance_m)``, where ``centroid_laser`` is the
        object position in the laser frame and ``distance_m`` its horizontal
        distance from the camera, or ``None`` if no laser hit falls inside
        the box (object above/below the scan plane, or out of range).
    """
    if points_laser.size == 0:
        return None

    points_cam = apply_transform(laser_to_optical, points_laser)
    x_cam, z_cam = points_cam[:, 0], points_cam[:, 2]

    # Project onto the image column. Points behind / too close to the lens
    # get a dummy divisor so we never divide by zero; they are masked out.
    in_front = z_cam > min_depth
    u = intrinsics.cx + intrinsics.fx * x_cam / np.where(in_front, z_cam, 1.0)
    in_box = in_front & (u >= u_min) & (u <= u_max)
    if not in_box.any():
        return None

    # Horizontal (ground-plane) distance from the camera. The optical Y axis
    # points down, so it is left out of the distance.
    distances = np.hypot(x_cam[in_box], z_cam[in_box])
    # method='lower' makes the reference an actual sample, so the cluster
    # always contains at least one hit. Interpolating could land in the gap
    # between the object and the background and leave the cluster empty.
    reference = np.percentile(distances, range_percentile, method='lower')
    cluster = np.abs(distances - reference) <= cluster_tolerance

    centroid = points_laser[in_box][cluster].mean(axis=0)
    return centroid, float(distances[cluster].mean())


def standoff_goal(robot_xy, object_xy, standoff):
    """Compute a goal that stops ``standoff`` metres short of the object.

    Nav2 cannot plan *into* an obstacle (a tree trunk is lethal in the
    costmap), so the goal is pulled back along the straight line from the
    robot to the object and oriented to face the object.

    Returns:
        ``(x, y, yaw)`` in the same frame as the inputs (``map``).
    """
    dx = object_xy[0] - robot_xy[0]
    dy = object_xy[1] - robot_xy[1]
    yaw = math.atan2(dy, dx)
    travel = max(0.0, math.hypot(dx, dy) - standoff)
    return (robot_xy[0] + travel * math.cos(yaw),
            robot_xy[1] + travel * math.sin(yaw),
            yaw)


def yaw_to_quaternion(yaw):
    """Return the ``(x, y, z, w)`` quaternion for a rotation about +Z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def describe_relative_position(u_center, image_width, distance,
                               far_threshold):
    """Coarse text position for Jev, e.g. ``front_left`` or ``far_center``.

    The image is split into equal thirds horizontally; objects beyond
    ``far_threshold`` metres are prefixed ``far`` instead of ``front``.
    """
    if u_center < image_width / 3.0:
        side = 'left'
    elif u_center > 2.0 * image_width / 3.0:
        side = 'right'
    else:
        side = 'center'
    depth = 'far' if distance > far_threshold else 'front'
    return f'{depth}_{side}'
