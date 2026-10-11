import math
from types import SimpleNamespace

from mowberry_semantic_nav import projection
import numpy as np
import pytest

# Laser frame (x fwd, y left, z up) -> optical frame (x right, y down,
# z fwd), co-located. Optical axes expressed in laser coordinates:
# x_o = -y_l, y_o = -z_l, z_o = x_l.
LASER_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
INTRINSICS = projection.PinholeIntrinsics.from_hfov(320, 240, math.radians(90))


def _tf(x=0.0, y=0.0, z=0.0, yaw=0.0):
    qx, qy, qz, qw = projection.yaw_to_quaternion(yaw)
    return SimpleNamespace(
        translation=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw))


def test_transform_to_matrix_yaw_and_translation():
    matrix = projection.transform_to_matrix(_tf(1.0, 2.0, 0.0, math.pi / 2))
    point = projection.apply_transform(matrix, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(point[0], [1.0, 3.0, 0.0], atol=1e-9)


def test_scan_to_points_drops_invalid():
    points = projection.scan_to_points(
        [1.0, float('inf'), float('nan'), 0.01, 2.0],
        angle_min=0.0, angle_increment=math.pi / 2,
        range_min=0.1, range_max=10.0)
    # Only index 0 (1 m at 0 rad) and index 4 (2 m at 2*pi rad) survive.
    np.testing.assert_allclose(points, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                               atol=1e-9)


def test_hfov_focal_length():
    # 90 deg HFOV: half-width equals focal length.
    assert INTRINSICS.fx == pytest.approx(160.0)
    assert INTRINSICS.cx == pytest.approx(160.0)


def test_locate_picks_near_object_not_background():
    # Object 2 m dead ahead, wall 6 m behind it showing at the box edges.
    angles = np.radians(np.linspace(-5, 5, 21))
    ranges = np.where(np.abs(angles) < np.radians(3), 2.0, 6.0)
    points = np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles),
         np.zeros_like(ranges)))
    hit = projection.locate_object_in_scan(
        points, LASER_TO_OPTICAL, INTRINSICS, u_min=140, u_max=180)
    assert hit is not None
    centroid, distance = hit
    assert distance == pytest.approx(2.0, abs=0.01)
    assert centroid[0] == pytest.approx(2.0, abs=0.01)
    assert centroid[1] == pytest.approx(0.0, abs=0.01)


def test_locate_reference_never_falls_between_clusters():
    # 3 object hits at 3 m and 9 background hits at 10 m. An interpolated
    # 20th percentile (~4.4 m) would match neither group.
    angles = np.radians(np.linspace(-6, 6, 12))
    ranges = np.where(np.arange(12) < 3, 3.0, 10.0)
    points = np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles),
         np.zeros_like(ranges)))
    hit = projection.locate_object_in_scan(
        points, LASER_TO_OPTICAL, INTRINSICS, u_min=0, u_max=320)
    assert hit is not None
    assert hit[1] == pytest.approx(3.0, abs=0.05)


def test_locate_left_box_sees_left_points():
    # Point at +45 deg (left) projects to the left image edge (u ~ 0).
    points = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    hit = projection.locate_object_in_scan(
        points, LASER_TO_OPTICAL, INTRINSICS, u_min=-5, u_max=10)
    assert hit is not None
    np.testing.assert_allclose(hit[0], [1.0, 1.0, 0.0])


def test_locate_ignores_points_behind_camera():
    points = np.array([[-2.0, 0.0, 0.0]])
    assert projection.locate_object_in_scan(
        points, LASER_TO_OPTICAL, INTRINSICS, 0, 320) is None


def test_standoff_goal_stops_short_and_faces_object():
    x, y, yaw = projection.standoff_goal((0.0, 0.0), (3.0, 4.0), 1.0)
    assert math.hypot(x, y) == pytest.approx(4.0)
    assert yaw == pytest.approx(math.atan2(4.0, 3.0))


def test_standoff_goal_when_already_close():
    x, y, _ = projection.standoff_goal((1.0, 1.0), (1.5, 1.0), 1.0)
    assert (x, y) == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize('u, distance, expected', [
    (10, 1.0, 'front_left'),
    (160, 1.0, 'front_center'),
    (300, 1.0, 'front_right'),
    (160, 9.0, 'far_center'),
])
def test_describe_relative_position(u, distance, expected):
    assert projection.describe_relative_position(
        u, 320, distance, far_threshold=4.0) == expected
