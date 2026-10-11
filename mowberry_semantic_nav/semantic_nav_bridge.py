#!/usr/bin/env python3
r"""Semantic navigation bridge: natural-language command -> Nav2 goal.

Runs on the remote PC, next to the robot on the same ROS 2 domain.

Pipeline (triggered by every message on ``command_topic``)
---------------------------------------------------------
1. Snapshot the latest compressed camera frame and LaserScan, rejecting
   stale data.
2. VLM: detect labelled bounding boxes in the frame (stub for now).
3. Localise every detection: project the laser hits into the image to find
   which ones fall inside each box (see ``projection.py``), then transform
   the matching hits into the ``map`` frame with tf2.
4. Jev: describe the localised objects as text and ask which one matches the
   user's intent (or ``unknown_abort``).
5. Publish a ``PoseStamped`` goal on ``/goal_pose``, stopping short of the
   object and facing it.

Fail-safe
---------
If any stage fails, or Jev answers ``unknown_abort``, or its confidence is
below ``min_confidence``, the node cancels all Nav2 NavigateToPose goals and
publishes a burst of zero ``Twist`` commands so the robot stops.

Interfaces
----------
Subscribes:
    ``image_topic``        sensor_msgs/CompressedImage
    ``camera_info_topic``  sensor_msgs/CameraInfo (optional, for calibration)
    ``scan_topic``         sensor_msgs/LaserScan
    ``command_topic``      std_msgs/String (user text command)
Publishes:
    ``goal_topic``         geometry_msgs/PoseStamped (Nav2 bt_navigator)
    ``cmd_vel_topic``      geometry_msgs/Twist (zero velocity on abort only)
    ``status_topic``       std_msgs/String (JSON pipeline status)
Service client:
    ``nav_cancel_service`` action_msgs/CancelGoal (cancel all Nav2 goals)

Example::

    ros2 run mowberry_semantic_nav semantic_nav_bridge
    ros2 topic pub --once /semantic_nav/command std_msgs/String \
        "{data: 'go straight to the shed door'}"
"""

import json
import math
import threading
import time
import types

from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, Twist
from mowberry_semantic_nav import projection
from mowberry_semantic_nav.jev_client import ABORT_CHOICE, JevClient
from mowberry_semantic_nav.vlm_stub import VlmStub
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
from std_msgs.msg import String
import tf2_ros

# Parameter name -> default value. Topic and frame defaults match the
# linorobot2 Mowberry setup (USB camera downscaled network stream).
DEFAULT_PARAMETERS = {
    # Topics / services
    'image_topic': '/usb_camera/downscaled/image_raw/compressed',
    'camera_info_topic': '/usb_camera/downscaled/camera_info',
    'scan_topic': '/scan',
    'command_topic': '/semantic_nav/command',
    'status_topic': '/semantic_nav/status',
    'goal_topic': '/goal_pose',
    'cmd_vel_topic': '/cmd_vel',
    'nav_cancel_service': '/navigate_to_pose/_action/cancel_goal',
    # Frames
    'map_frame': 'map',
    'base_frame': 'base_link',
    'camera_frame': 'usb_camera_optical_link',
    # Camera model fallback when CameraInfo is missing or uncalibrated
    'camera_hfov_deg': 60.0,
    # Laser / image association
    'bbox_margin_ratio': 0.1,
    'range_percentile': 20.0,
    'cluster_tolerance_m': 0.3,
    'far_threshold_m': 4.0,
    # Goal shaping
    'standoff_distance_m': 0.8,
    # Safety / timing
    'max_sensor_age_s': 1.0,
    'tf_timeout_s': 0.5,
    'min_confidence': 0.75,
    'stop_burst_count': 5,
    # Jev
    'jev_endpoint': 'http://localhost:8000/v1/decide',
    'jev_timeout_s': 5.0,
}

JEV_QUESTION = 'Which object ID matches the intent of the user command?'


class PipelineError(Exception):
    """A pipeline stage could not produce a safe result; triggers a stop."""


class SemanticNavBridge(Node):
    """ROS 2 node turning text commands into Nav2 goals (see module doc)."""

    def __init__(self):
        super().__init__('semantic_nav_bridge')
        for name, default in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default)
        self._cfg = types.SimpleNamespace(**{
            name: self.get_parameter(name).value
            for name in DEFAULT_PARAMETERS})
        cfg = self._cfg

        # Latest sensor messages, stored with the *local* monotonic receive
        # time. Staleness is judged on receive time rather than header
        # stamps, so clock skew between the Pi and this PC cannot make fresh
        # data look stale (or stale data look fresh).
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_scan = None
        self._camera_info = None

        # AI layers. Swap VlmStub for the real VLM client once available.
        self._vlm = VlmStub()
        self._jev = JevClient(cfg.jev_endpoint, timeout_s=cfg.jev_timeout_s)

        # TF: the listener spins its own thread, so blocking lookups with a
        # timeout from the pipeline callback are safe.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Sensors and commands use separate callback groups. Under the
        # MultiThreadedExecutor, frames and scans keep arriving while a
        # (slow, network-bound) VLM/Jev pipeline runs. Commands are
        # mutually exclusive, so they are processed one at a time in order.
        sensor_group = MutuallyExclusiveCallbackGroup()
        command_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            CompressedImage, cfg.image_topic, self._on_image,
            qos_profile_sensor_data, callback_group=sensor_group)
        self.create_subscription(
            CameraInfo, cfg.camera_info_topic, self._on_camera_info,
            qos_profile_sensor_data, callback_group=sensor_group)
        self.create_subscription(
            LaserScan, cfg.scan_topic, self._on_scan,
            qos_profile_sensor_data, callback_group=sensor_group)
        self.create_subscription(
            String, cfg.command_topic, self._on_command, 10,
            callback_group=command_group)

        self._goal_pub = self.create_publisher(PoseStamped, cfg.goal_topic, 10)
        self._cmd_vel_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self._status_pub = self.create_publisher(String, cfg.status_topic, 10)
        self._cancel_client = self.create_client(
            CancelGoal, cfg.nav_cancel_service)

        self.get_logger().info(
            f'Ready. Listening for commands on {cfg.command_topic} '
            f'(image: {cfg.image_topic}, scan: {cfg.scan_topic})')

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def _on_image(self, msg):
        with self._lock:
            self._latest_image = (msg, time.monotonic())

    def _on_scan(self, msg):
        with self._lock:
            self._latest_scan = (msg, time.monotonic())

    def _on_camera_info(self, msg):
        with self._lock:
            self._camera_info = msg

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------
    def _on_command(self, msg):
        command = msg.data.strip()
        if not command:
            return
        self.get_logger().info(f'Command received: "{command}"')
        self._publish_status('processing', command=command)

        try:
            goal, target, decision = self._run_pipeline(command)
        except PipelineError as exc:
            self._trigger_failsafe(str(exc), command)
            return
        except Exception as exc:  # noqa: BLE001 - never leave robot moving
            self.get_logger().error(f'Unexpected pipeline error: {exc!r}')
            self._trigger_failsafe(f'internal error: {exc!r}', command)
            return

        self._goal_pub.publish(goal)
        self.get_logger().info(
            f'Goal sent for object {target["id"]} ({target["label"]}) at '
            f'map ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}), '
            f'Jev confidence {decision.confidence:.2f}')
        self._publish_status(
            'goal_sent', command=command, target=target,
            confidence=decision.confidence,
            goal={'x': goal.pose.position.x, 'y': goal.pose.position.y})

    def _run_pipeline(self, command):
        """Run perception -> localisation -> decision -> goal.

        Returns:
            ``(PoseStamped goal, chosen object dict, JevDecision)``.

        Raises:
            PipelineError: on any condition where moving would be unsafe.
        """
        cfg = self._cfg
        image, scan = self._snapshot_sensors()

        # Stage 2: VLM detection on the captured frame.
        try:
            vlm_result = self._vlm.detect(bytes(image.data), command)
        except ValueError as exc:
            raise PipelineError(f'VLM failed: {exc}') from exc
        if not vlm_result.detections:
            raise PipelineError('VLM returned no detections')

        # Stage 3: give every detection a range and a map position.
        intrinsics = self._intrinsics_for(
            vlm_result.image_width, vlm_result.image_height)
        objects = self._localize_detections(vlm_result, scan, intrinsics)
        if not objects:
            raise PipelineError(
                'no detection could be matched to a laser range')

        # Stage 4: ask Jev which object the user means. Only text goes out;
        # map coordinates stay local.
        state = {
            'user_command': command,
            'detected_objects': [
                {k: o[k] for k in
                 ('id', 'label', 'relative_position', 'distance_m')}
                for o in objects],
        }
        options = [str(o['id']) for o in objects] + [ABORT_CHOICE]
        decision = self._jev.decide(state, JEV_QUESTION, options)
        self.get_logger().info(
            f'Jev decision: {decision.choice} '
            f'(confidence {decision.confidence:.2f}) {decision.reason}')

        if decision.choice == ABORT_CHOICE:
            raise PipelineError(f'Jev aborted: {decision.reason}')
        if decision.confidence < cfg.min_confidence:
            raise PipelineError(
                f'Jev confidence {decision.confidence:.2f} below '
                f'{cfg.min_confidence:.2f}')
        target = next(
            (o for o in objects if str(o['id']) == decision.choice), None)
        if target is None:
            raise PipelineError(
                f'Jev chose "{decision.choice}", not one of {options}')

        # Stage 5: goal pose short of the object, facing it.
        robot_xy = self._robot_xy_in_map()
        x, y, yaw = projection.standoff_goal(
            robot_xy, target['map_xy'], cfg.standoff_distance_m)
        return self._make_goal(x, y, yaw), target, decision

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------
    def _snapshot_sensors(self):
        """Return the latest (image, scan), raising if missing or stale."""
        with self._lock:
            image_entry, scan_entry = self._latest_image, self._latest_scan
        now = time.monotonic()
        max_age = self._cfg.max_sensor_age_s
        for name, entry in (('camera frame', image_entry),
                            ('laser scan', scan_entry)):
            if entry is None:
                raise PipelineError(f'no {name} received yet')
            age = now - entry[1]
            if age > max_age:
                raise PipelineError(
                    f'{name} is stale ({age:.1f}s > {max_age:.1f}s)')
        return image_entry[0], scan_entry[0]

    def _intrinsics_for(self, width, height):
        """Prefer calibrated CameraInfo; fall back to the configured HFOV."""
        with self._lock:
            info = self._camera_info
        # Uncalibrated drivers publish an all-zero K matrix.
        if info is not None and info.k[0] > 0.0 and info.width > 0:
            return projection.PinholeIntrinsics.from_camera_info(
                info.k, info.width, info.height, width, height)
        return projection.PinholeIntrinsics.from_hfov(
            width, height, math.radians(self._cfg.camera_hfov_deg))

    def _localize_detections(self, vlm_result, scan, intrinsics):
        """Attach laser distance and map coordinates to each detection.

        Two transforms are needed:

        * laser -> camera optical frame: projects laser hits into the image.
          This is a static mounting transform, so the latest one is used.
        * laser -> map at the scan's timestamp: places the matched hits in
          the world. Using the scan time (not "now") keeps the result
          consistent even if the robot moved since the scan was taken.

        Detections without laser support are skipped (logged) because they
        cannot be navigated to safely.
        """
        cfg = self._cfg
        laser_frame = scan.header.frame_id
        laser_to_optical = self._lookup_matrix(
            cfg.camera_frame, laser_frame, Time())
        laser_to_map = self._lookup_matrix(
            cfg.map_frame, laser_frame, Time.from_msg(scan.header.stamp))

        points = projection.scan_to_points(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max)

        objects = []
        for index, det in enumerate(vlm_result.detections, start=1):
            # Shrink the box horizontally so hits that only graze its edges
            # (usually background) are ignored.
            margin = (det.x_max - det.x_min) * cfg.bbox_margin_ratio
            hit = projection.locate_object_in_scan(
                points, laser_to_optical, intrinsics,
                det.x_min + margin, det.x_max - margin,
                range_percentile=cfg.range_percentile,
                cluster_tolerance=cfg.cluster_tolerance_m)
            if hit is None:
                self.get_logger().warn(
                    f'Detection {index} ({det.label}) has no laser hits '
                    'inside its box; skipping')
                continue

            point_laser, distance = hit
            map_point = projection.apply_transform(
                laser_to_map, point_laser.reshape(1, 3))[0]
            objects.append({
                'id': index,
                'label': det.label,
                'relative_position': projection.describe_relative_position(
                    det.u_center, intrinsics.width, distance,
                    cfg.far_threshold_m),
                'distance_m': round(distance, 2),
                'map_xy': (float(map_point[0]), float(map_point[1])),
            })
        return objects

    def _lookup_matrix(self, target_frame, source_frame, stamp):
        """Look up ``source -> target`` as a 4x4 matrix.

        If the transform is not available at ``stamp`` (typical when the
        Pi's and this PC's clocks drift, or AMCL lags), fall back to the
        latest available transform with a warning.
        """
        timeout = Duration(seconds=self._cfg.tf_timeout_s)
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, timeout)
        except tf2_ros.ExtrapolationException:
            self.get_logger().warn(
                f'TF {source_frame}->{target_frame} unavailable at scan '
                'time; using latest')
            try:
                tf = self._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time(), timeout)
            except tf2_ros.TransformException as exc:
                raise PipelineError(f'TF lookup failed: {exc}') from exc
        except tf2_ros.TransformException as exc:
            raise PipelineError(f'TF lookup failed: {exc}') from exc
        return projection.transform_to_matrix(tf.transform)

    def _robot_xy_in_map(self):
        """Return the robot position (base frame origin) in the map frame."""
        matrix = self._lookup_matrix(
            self._cfg.map_frame, self._cfg.base_frame, Time())
        return float(matrix[0, 3]), float(matrix[1, 3])

    def _make_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = self._cfg.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        (goal.pose.orientation.x, goal.pose.orientation.y,
         goal.pose.orientation.z, goal.pose.orientation.w) = \
            projection.yaw_to_quaternion(yaw)
        return goal

    # ------------------------------------------------------------------
    # Fail-safe and status
    # ------------------------------------------------------------------
    def _trigger_failsafe(self, reason, command=''):
        """Stop the robot: cancel all Nav2 goals and command zero velocity.

        Cancelling is the real stop. While a goal is active, Nav2's
        controller publishes /cmd_vel continuously and would override a
        lone zero Twist. The zero burst covers the time until the cancel
        lands, and also the case where Nav2 is not running at all.
        """
        self.get_logger().error(f'FAIL-SAFE STOP: {reason}')

        if self._cancel_client.service_is_ready():
            # An empty request (zero goal UUID, zero stamp) cancels ALL
            # goals on the NavigateToPose action server.
            self._cancel_client.call_async(CancelGoal.Request())
        else:
            self.get_logger().warn(
                f'{self._cfg.nav_cancel_service} not available; '
                'sending zero velocity only')

        stop = Twist()
        for _ in range(self._cfg.stop_burst_count):
            self._cmd_vel_pub.publish(stop)

        self._publish_status('aborted', command=command, reason=reason)

    def _publish_status(self, state, **fields):
        self._status_pub.publish(
            String(data=json.dumps({'state': state, **fields})))


def main(args=None):
    rclpy.init(args=args)
    node = SemanticNavBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
