#!/usr/bin/env python3
"""
column_detector.py  --  ERC 2026 Library Assistant Robot, stage 2 of 4
"""

import base64
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

HANDOFF_FILE = '/tmp/erc_target_column.json'

DIGIT_TEMPLATES_B64 = {
    1: "iVBORw0KGgoAAAANSUhEUgAAACUAAAAmCAAAAABmYQ1xAAAAvElEQVQ4EY3BAVLDMBDAQOn/jxaBHrE9hUx3DeQ/8SLP4pvJowzkXVzkJQk5xC+5hVzindxCIP4gt5BLvJNbCMQf5BZyiTeyhBxiyBJyihdZQg4xZAk5xJAl5BBDlpBDDFlCDjFkCTnEkCVMNjFkCZNNDFnCZBNDljDZxJAl5BBDNskhhmySQwzZJIcYskkOMWSTHGLITg4xZCeH+CUbuQQIxEYWgXgkmMQjQSAeCXKJH7KJISCfkE/IJ74AbIdGDB6jZUAAAAAASUVORK5CYII=",
    2: "iVBORw0KGgoAAAANSUhEUgAAACUAAAAmCAAAAABmYQ1xAAAAw0lEQVQ4EY3BgWGDMAADMPn/o70EQgsbZZVSQ3xWpOIflcZ/Kk7qJN7iUL/FIXZ1J3axqXuxiak+iSmG2sVSLzHEUFOc1BJDUFNc1BIENcVFLUFQU1zUEgQ1xUUtQVBTXNQuhrhXSwxxr3YxxZ1aYhN3aolN/FVLLGlc1SEOaVzUEm9pnNUuzuKsdvHSIN5qF4caQrzUJt6KIJbaxVsNQWzqoyCGehAE9SCGoB7EENSDGIJ6EEN8VkMM8agxxTfiG/GNH5xHNCRBTtK8AAAAAElFTkSuQmCC",
    3: "iVBORw0KGgoAAAANSUhEUgAAACMAAAAlCAAAAADt6w+YAAAAw0lEQVQ4EYXBAWKcMAADMPn/j/YSEjhoWU9KU/FfRSr+VKn4W1NxqktcGqf6IbbY6pdYYqk3McWh3sUQUy2x1BZDTDXFRy1BDDXFTS1BDDXFTS1BDDXFTR1iiHd1iCFe1SGmeFGHWOKhLnGKh7rEKR7qJpZ4qKcY4lVtQbyrLcRUQ9zVEmKoKe5qCUEtcVNLCGqJm1pCUFt81BJiqCUutYUYaouttiCm+kh9xBCHehVTbPVLLPFRd3GKH4p4iO/iu/juHy8INSK/Q5tTAAAAAElFTkSuQmCC",
    4: "iVBORw0KGgoAAAANSUhEUgAAACMAAAAlCAAAAADt6w+YAAAAx0lEQVQ4EYXBgXHCQBDAQKn/ohX+8QHGGbNrHORfoTHkKh4MkxthyJcEkqcw5CQOsoXJSQw5mJzEkIPJp3iRLU3e4oNsafIWbzJMXuKDDHmJRYhFhrzEg0AsMuQQm0AsMuQpNnmIRYZssckSiwxZYpMtFhnyEIscYpEhEIuMWGQIxCIjFhkCcUMQiBuCQNwQBOKGIBA3BIHkJBaBQJCrWGTIVSwy5CoWGXIViwzD5CQ2OZh8iSc5mHyJJzmYfItNDia/yG/y2x+Ok0Qit9iWhwAAAABJRU5ErkJggg==",
    5: "iVBORw0KGgoAAAANSUhEUgAAACQAAAAlCAAAAAAPNxThAAAAuElEQVQ4EY3BgWECMQwEwd3+i77IsiFPeBFmDMhHwYB8FIz8y8gkgECMlDCQYqSEgRRpYSBFWhhIkRYGUqSFgRRpochfkUVaKDKQFooMpIUiA2lhkXvSwht5khbuyCYt3JNFljCRIksAeQgXAnIvPAgyCIcgk7AJMgmbICUs8io0QUpo8iI0QSAcchWaIBAOuQibICUc8hQ2ASnhQY6wSZElDGSRLdyRJkd4I4f8ClfyJF+QL8gXfgAVWjAj9uFI0gAAAABJRU5ErkJggg==",
}
TEMPLATE_CANONICAL_SIZE = (40, 40)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class ColumnDetector(Node):

    # --- Template matching ---------------------------------------------
    MIN_TEMPLATE_SCORE = 0.37
    MIN_TEMPLATE_MARGIN = 0.06
    MIN_BLOB_AREA = 400
    MAX_BLOB_AREA = 2500
    MIN_BLOB_ASPECT = 0.5
    MAX_BLOB_ASPECT = 2.0
    CROP_FRACTION = 0.4

    # --- Search rotation -----------------------------------------------
    ROTATE_SPEED = 0.35
    ROTATE_MAX_RADIANS = 2 * math.pi * 1.15
    SEARCH_CONFIRM_FRAMES = 3

    # --- Centring --------------------------------------------------------
    CENTRING_SPEED = 0.15
    MIN_MARKERS_TO_STOP = 4
    CENTRE_TOLERANCE_PX = 80
    CENTRING_TIMEOUT_SEC = 20.0
    MAX_LINE_FIT_RETRIES = 2

    HOLD_KP = 1.8
    HOLD_KI = 0.6
    HOLD_DEADBAND = 0.02
    HOLD_INTEGRAL_CLAMP = 0.3
    HOLD_MAX_SPEED = 0.20
    ABORT_DRIFT = 0.14
    TICK = 0.1

    # --- Confirmation ---------------------------------------------------
    CONFIRM_FRAMES = 4
    PIXEL_TOLERANCE = 25
    CONFIRM_TIMEOUT_SEC = 5.0
    PAN_FALLBACK = [0.0, -0.5, 0.5, -1.0, 1.0]
    MAX_TF_RETRIES = 8

    # --- Geometry sanity ------------------------------------------------
    EXPECTED_MARKER_PITCH = 1.0
    PITCH_TOLERANCE = 0.35

    def __init__(self):
        super().__init__('column_detector')

        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('erc_images_dir', '/erc_images')
        # Set to true only when you want a clean slate.
        self.declare_parameter('purge_images_dir', False)
        self.declare_parameter('marker_tilt', 0.10)
        self.declare_parameter('rotate_clockwise', True)
        self.declare_parameter('standoff_from_marker', 0.60)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('save_debug_frames', False)
        self.declare_parameter('annotate_other_markers', False)

        self.target = int(self.get_parameter('shelf_column_number').value)
        self.images_dir = str(self.get_parameter('erc_images_dir').value)
        self.marker_tilt = float(self.get_parameter('marker_tilt').value)
        self.rotate_sign = -1.0 if self.get_parameter('rotate_clockwise').value else 1.0
        self.standoff = float(self.get_parameter('standoff_from_marker').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.debug = bool(self.get_parameter('save_debug_frames').value)
        self.annotate_others = bool(self.get_parameter('annotate_other_markers').value)

        if not 1 <= self.target <= 5:
            raise ValueError(f'shelf_column_number must be 1-5, got {self.target}')
        self.get_logger().info(f'Target shelf column: {self.target}')


        if os.path.exists(HANDOFF_FILE):
            try:
                os.remove(HANDOFF_FILE)
                self.get_logger().info(f'Cleared stale {HANDOFF_FILE}.')
            except OSError as e:
                self.get_logger().warn(f'Could not clear {HANDOFF_FILE}: {e}')

        self._prepare_images_dir()

        self.bridge = CvBridge()
        self.templates = self._load_templates()
        self.camera_matrix = None
        self.should_exit = False
        self.exit_code = 0


        self.phase = 'search'
        self.search_streak = 0
        self.centring_started = 0.0
        self.line_fit_retries = 0
        self.image_width = 640
        self.history = []
        self.pan_index = 0
        self.phase_started = time.time()
        self.tf_retries = 0
        self.last_candidates = []
        self._debug_counter = 0


        self.cumulative_rotation = 0.0
        self.last_imu_stamp = None
        self.yaw = None
        self.start_xy = None
        self.current_xy = None
        self.integral = [0.0, 0.0]
        self._log_tick = 0


        self._latest_color = None
        self._latest_depth = None
        self._color_stamp = 0.0
        self._depth_stamp = 0.0

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.column_pub = self.create_publisher(
            Int32, '/erc/shelf_column_identification', latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, '/head_front_camera/head_front_camera/color/camera_info',
            self.camera_info_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Imu, '/base_imu', self.imu_cb, 10)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self._color_cb, 5)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self._depth_cb, 5)
        # Wait for the map frame before starting
        wait_start = time.time()
        while time.time() - wait_start < 30.0:
            try:
                self.tf_buffer.lookup_transform(
                    self.map_frame, 'base_link', RclTime(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                self.get_logger().info(
                    'TF map -> base_link available. Starting detection.')
                break
            except Exception:
                time.sleep(0.5)
        else:
            self.get_logger().warn(
                'TF map -> base_link not available after 30 s. Continuing anyway - '
                'localisation will fail if AMCL is still starting up.')

        self.send_head(0.0, self.marker_tilt)

        self.send_head(0.0, self.marker_tilt)
        self.timer = self.create_timer(self.TICK, self.tick)

        self.get_logger().info(
            f'Rotating {"clockwise" if self.rotate_sign < 0 else "counter-clockwise"} '
            f'to bring the shelf markers into view (head tilted to '
            f'{self.marker_tilt:.2f} rad so the 2.26 m marker plates are framed).')

    # ------------------------------------------------------------------
    # Frame callbacks (latest-frame cache)
    # ------------------------------------------------------------------
    def _color_cb(self, msg):
        self._latest_color = msg
        self._color_stamp = time.time()
        self._try_process_frame()

    def _depth_cb(self, msg):
        self._latest_depth = msg
        self._depth_stamp = time.time()

    def _try_process_frame(self):
        if self._latest_color is None or self._latest_depth is None:
            return
        now = time.time()
        if now - self._color_stamp > 0.5 or now - self._depth_stamp > 0.5:
            return
        self.image_cb(self._latest_color, self._latest_depth)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _prepare_images_dir(self):
        """Ensure the images directory exists.  Only purges if the caller
        explicitly opted in with purge_images_dir:=true - by default the
        folder is left alone so past runs' annotated images are preserved
        for the organising committee to inspect."""
        os.makedirs(self.images_dir, exist_ok=True)
        if not bool(self.get_parameter('purge_images_dir').value):
            return
        removed = 0
        for name in os.listdir(self.images_dir):
            path = os.path.join(self.images_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError as e:
                    self.get_logger().warn(f'Could not remove {path}: {e}')
        self.get_logger().info(
            f'Competition image folder {self.images_dir} cleared on request '
            f'({removed} old file(s) removed).')

    @staticmethod
    def _load_templates():
        out = {}
        for digit, b64 in DIGIT_TEMPLATES_B64.items():
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            out[digit] = cv2.resize(img, TEMPLATE_CANONICAL_SIZE,
                                    interpolation=cv2.INTER_LINEAR)
        return out

    def send_head(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(pan), float(tilt)]
        pt.time_from_start.sec = 1
        traj.points.append(pt)
        self.head_pub.publish(traj)

    def stop_base(self):
        stop = Twist()
        for _ in range(3):
            self.cmd_pub.publish(stop)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info('Camera intrinsics captured.')

    def odom_cb(self, msg):
        self.current_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.yaw is None:
            self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.phase == 'search' and self.start_xy is None:
            self.start_xy = self.current_xy

    def imu_cb(self, msg):
        stamp = RclTime.from_msg(msg.header.stamp)
        if self.last_imu_stamp is not None and self.yaw is not None:
            dt = (stamp - self.last_imu_stamp).nanoseconds / 1e9
            if dt > 0:
                delta = msg.angular_velocity.z * dt
                self.yaw = normalize_angle(self.yaw + delta)
                if self.phase == 'search':
                    self.cumulative_rotation += abs(delta)
        self.last_imu_stamp = stamp

    # ------------------------------------------------------------------
    # Motion tick
    # ------------------------------------------------------------------
    def tick(self):
        if self.phase == 'confirm':
            self.confirm_tick()
            return
        if self.phase == 'centring':
            if time.time() - self.centring_started > self.CENTRING_TIMEOUT_SEC:
                self.get_logger().warn(
                    'Centring timed out - settling for whatever markers are in frame. '
                    'The shelf-line fit may be based on too few points.')
                self.stop_base()
                self.phase = 'confirm'
                self.phase_started = time.time()
                self.history.clear()
                return
        elif self.phase != 'search':
            return

        drift = 0.0
        if self.start_xy and self.current_xy:
            drift = math.hypot(self.start_xy[0] - self.current_xy[0],
                               self.start_xy[1] - self.current_xy[1])

        self._log_tick += 1
        if self._log_tick % 20 == 0:
            self.get_logger().info(
                f'Searching... {math.degrees(self.cumulative_rotation):.0f} deg rotated, '
                f'{drift:.3f} m drift.')

        if drift > self.ABORT_DRIFT:
            self.get_logger().error(
                f'Drifted {drift:.2f} m while rotating - stopping before the base reaches '
                'the table. Check that the arms are tucked and that the drift hold is '
                'actually being applied.')
            self.stop_base()
            self.should_exit = True
            self.exit_code = 1
            return

        if self.cumulative_rotation >= self.ROTATE_MAX_RADIANS:
            self.get_logger().error(
                f'Rotated {math.degrees(self.cumulative_rotation):.0f} deg without seeing a '
                'marker. Re-run with save_debug_frames:=true and inspect what the camera '
                'is actually framing.')
            self.stop_base()
            self.should_exit = True
            self.exit_code = 1
            return

        twist = Twist()
        speed = self.CENTRING_SPEED if self.phase == 'centring' else self.ROTATE_SPEED
        twist.angular.z = self.rotate_sign * speed

        if self.start_xy and self.current_xy and self.yaw is not None and drift > self.HOLD_DEADBAND:
            ex = self.start_xy[0] - self.current_xy[0]
            ey = self.start_xy[1] - self.current_xy[1]
            self.integral[0] = max(-self.HOLD_INTEGRAL_CLAMP,
                                   min(self.HOLD_INTEGRAL_CLAMP,
                                       self.integral[0] + ex * self.TICK))
            self.integral[1] = max(-self.HOLD_INTEGRAL_CLAMP,
                                   min(self.HOLD_INTEGRAL_CLAMP,
                                       self.integral[1] + ey * self.TICK))
            wx = self.HOLD_KP * ex + self.HOLD_KI * self.integral[0]
            wy = self.HOLD_KP * ey + self.HOLD_KI * self.integral[1]
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            bx, by = wx * c + wy * s, -wx * s + wy * c
            twist.linear.x = max(-self.HOLD_MAX_SPEED, min(self.HOLD_MAX_SPEED, bx))
            twist.linear.y = max(-self.HOLD_MAX_SPEED, min(self.HOLD_MAX_SPEED, by))

        self.cmd_pub.publish(twist)

    def confirm_tick(self):
        if time.time() - self.phase_started < self.CONFIRM_TIMEOUT_SEC:
            return
        self.pan_index += 1
        if self.pan_index < len(self.PAN_FALLBACK):
            pan = self.PAN_FALLBACK[self.pan_index]
            self.get_logger().warn(
                f'Column {self.target} not confirmed here - panning head to {pan:+.2f} rad '
                f'({self.pan_index}/{len(self.PAN_FALLBACK) - 1}).')
            self.send_head(pan, self.marker_tilt)
            self.history.clear()
            self.phase_started = time.time()
            return
        self.get_logger().warn(
            'Head pan exhausted without confirming the target column - resuming base rotation.')
        self.pan_index = 0
        self.send_head(0.0, self.marker_tilt)
        self.history.clear()
        self.search_streak = 0
        self.start_xy = self.current_xy
        self.integral = [0.0, 0.0]
        self.phase = 'search'

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def image_cb(self, color_msg, depth_msg):
        if self.should_exit or self.camera_matrix is None:
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        markers = self.detect_markers(color)
        self.save_debug_frame(color, markers)

        self.image_width = color.shape[1]

        if self.phase == 'search':
            self.search_streak = self.search_streak + 1 if markers else 0
            if self.search_streak >= self.SEARCH_CONFIRM_FRAMES:
                self.get_logger().info(
                    f'Shelf edge in view ({len(markers)} marker(s): '
                    f'{sorted(m["digit"] for m in markers)}) - slowing down and turning '
                    'until the whole shelf is framed.')
                self.phase = 'centring'
                self.centring_started = time.time()
            return

        if self.phase == 'centring':
            if not markers:
                return
            centroid = sum(m['cx'] for m in markers) / float(len(markers))
            offset = centroid - self.image_width / 2.0
            if (len(markers) >= self.MIN_MARKERS_TO_STOP
                    and abs(offset) < self.CENTRE_TOLERANCE_PX):
                self.get_logger().info(
                    f'Shelf centred: {len(markers)} markers '
                    f'{sorted(m["digit"] for m in markers)}, centroid {offset:+.0f} px from '
                    'image centre. Stopping rotation.')
                self.stop_base()
                self.phase = 'confirm'
                self.phase_started = time.time()
                self.history.clear()
            return

        if self.phase != 'confirm':
            return

        hit = next((m for m in markers if m['digit'] == self.target), None)
        if hit is None:
            self.history.clear()
            return

        self.history.append((hit['cx'], hit['cy']))
        if len(self.history) < self.CONFIRM_FRAMES:
            return
        recent = self.history[-self.CONFIRM_FRAMES:]
        if (max(x for x, _ in recent) - min(x for x, _ in recent) >= self.PIXEL_TOLERANCE or
                max(y for _, y in recent) - min(y for _, y in recent) >= self.PIXEL_TOLERANCE):
            return

        self.get_logger().info(
            f'CONFIRMED column {self.target} at pixel ({hit["cx"]},{hit["cy"]}) '
            f'across {self.CONFIRM_FRAMES} frames. Localising the shelf...')
        self.finalise(color, depth, color_msg.header, markers, hit)

    def detect_markers(self, color_img):
        h, w = color_img.shape[:2]
        crop = color_img[0:int(h * self.CROP_FRACTION), 0:w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 10)
        self.last_candidates = []
        if cv2.countNonZero(thresh) < 150:
            return []

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_per_digit = {}
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self.MIN_BLOB_AREA or area > self.MAX_BLOB_AREA:
                continue
            aspect = bw / float(bh) if bh > 0 else 0.0
            if aspect < self.MIN_BLOB_ASPECT or aspect > self.MAX_BLOB_ASPECT:
                continue

            blob = cv2.resize(thresh[y:y + bh, x:x + bw], TEMPLATE_CANONICAL_SIZE,
                              interpolation=cv2.INTER_LINEAR)
            scores = {d: float(cv2.matchTemplate(blob, t, cv2.TM_CCOEFF_NORMED)[0, 0])
                      for d, t in self.templates.items()}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            digit, score = ranked[0]
            margin = score - ranked[1][1]
            self.last_candidates.append((digit, score, margin, x, y, bw, bh))

            if score < self.MIN_TEMPLATE_SCORE or margin < self.MIN_TEMPLATE_MARGIN:
                continue
            cand = {'digit': digit, 'score': score,
                    'cx': x + bw // 2, 'cy': y + bh // 2,
                    'box': (x, y, bw, bh)}
            if digit not in best_per_digit or score > best_per_digit[digit]['score']:
                best_per_digit[digit] = cand

        return sorted(best_per_digit.values(), key=lambda m: m['cx'])

    # ------------------------------------------------------------------
    # Localisation + output
    # ------------------------------------------------------------------
    def finalise(self, color, depth, header, markers, hit):
        located = []
        for m in markers:
            p = self.marker_to_map(depth, header, m['cx'], m['cy'])
            if p is not None:
                located.append((m, p))

        if not any(m['digit'] == self.target for m, _ in located):
            self.tf_retries += 1
            if self.tf_retries > self.MAX_TF_RETRIES:
                self.get_logger().error(
                    'Could not place the target marker in the map frame after '
                    f'{self.MAX_TF_RETRIES} attempts - is AMCL publishing map->odom?')
                self.stop_base()
                self.should_exit = True
                self.exit_code = 1
                return
            self.get_logger().warn('Target marker not localisable this frame - retrying.')
            self.history.clear()
            return

        pts = np.array([[p[0], p[1]] for _, p in located])
        target_pt = next(p for m, p in located if m['digit'] == self.target)

        direction, normal = self.fit_shelf_line(pts)
        target_xy = np.array(target_pt[:2])

        if direction is None and self.line_fit_retries < self.MAX_LINE_FIT_RETRIES:
            self.line_fit_retries += 1
            self.get_logger().warn(
                f'Only {len(located)} marker(s) localised - not enough for a shelf-line fit. '
                f'Turning further to reframe (attempt {self.line_fit_retries}/'
                f'{self.MAX_LINE_FIT_RETRIES}).')
            self.history.clear()
            self.phase = 'centring'
            self.centring_started = time.time()
            return

        if direction is not None:
            centroid = pts.mean(axis=0)
            target_xy = centroid + direction * float(np.dot(target_xy - centroid, direction))
            self.check_marker_pitch(pts, direction)
        else:
            self.get_logger().warn(
                f'Only {len(located)} marker(s) localised - cannot fit the shelf line. '
                'Falling back to "face the marker from wherever the robot is", which is '
                'less precisely perpendicular.')
            robot = self.robot_xy()
            if robot is None:
                self.get_logger().error('No robot pose available for the fallback - aborting.')
                self.should_exit = True
                self.exit_code = 1
                return
            v = target_xy - robot
            n = v / (np.linalg.norm(v) + 1e-9)
            normal = -n

        robot = self.robot_xy()
        if robot is not None and float(np.dot(normal, robot - target_xy)) < 0:
            normal = -normal

        self.stop_base()

        msg = Int32()
        msg.data = self.target
        self.column_pub.publish(msg)
        self.get_logger().info(
            f'Published column {self.target} on /erc/shelf_column_identification '
            '(latched, so a late subscriber still receives it).')

        self.save_competition_image(color, markers, hit)

        payload = {
            'column': self.target,
            'frame': self.map_frame,
            'marker_xy': [float(target_xy[0]), float(target_xy[1])],
            'marker_z': float(target_pt[2]),
            'shelf_normal': [float(normal[0]), float(normal[1])],
            'standoff': self.standoff,
            'markers_seen': sorted(m['digit'] for m, _ in located),
            'stamp': time.time(),
        }
        try:
            with open(HANDOFF_FILE, 'w') as f:
                json.dump(payload, f, indent=2)
            self.get_logger().info(f'Wrote {HANDOFF_FILE} for approach_column.py.')
        except OSError as e:
            self.get_logger().error(f'Could not write {HANDOFF_FILE}: {e}')
            self.exit_code = 1

        heading = math.degrees(math.atan2(-normal[1], -normal[0]))
        self.get_logger().info(
            f'Shelf face normal {normal[0]:+.3f},{normal[1]:+.3f} -> the robot must end up '
            f'facing {heading:.1f} deg in {self.map_frame} to be perpendicular to column '
            f'{self.target}, standing {self.standoff:.2f} m out from the marker plane.')
        self.should_exit = True

    def marker_to_map(self, depth, header, cx, cy):
        z = self.sample_depth(depth, cx, cy)
        if z is None:
            return None
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        px, py = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        pt = PointStamped()
        pt.header.frame_id = header.frame_id
        pt.header.stamp = RclTime().to_msg()
        pt.point.x = (cx - px) * z / fx
        pt.point.y = (cy - py) * z / fy
        pt.point.z = float(z)
        try:
            out = self.tf_buffer.transform(
                pt, self.map_frame, timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().warn(f'TF {header.frame_id} -> {self.map_frame} failed: {e}')
            return None
        return (out.point.x, out.point.y, out.point.z)

    @staticmethod
    def fit_shelf_line(pts):
        if len(pts) < 3:
            return None, None
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        direction = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
        normal = np.array([-direction[1], direction[0]])
        return direction, normal

    def check_marker_pitch(self, pts, direction):
        s = sorted(float(np.dot(p - pts.mean(axis=0), direction)) for p in pts)
        gaps = [b - a for a, b in zip(s, s[1:])]
        if not gaps:
            return
        worst = max(abs(g - self.EXPECTED_MARKER_PITCH) for g in gaps)
        desc = ', '.join(f'{g:.2f}' for g in gaps)
        if worst > self.PITCH_TOLERANCE:
            self.get_logger().warn(
                f'Marker spacing came out as [{desc}] m but should be '
                f'{self.EXPECTED_MARKER_PITCH:.1f} m. Something in the depth/intrinsics/TF '
                'chain is off - treat the goal position below with suspicion.')
        else:
            self.get_logger().info(
                f'Marker spacing sanity check passed: [{desc}] m (expected '
                f'{self.EXPECTED_MARKER_PITCH:.1f} m apart).')

    def robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, 'base_link', RclTime(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception:
            return None
        return np.array([tf.transform.translation.x, tf.transform.translation.y])

    @staticmethod
    def sample_depth(depth, cx, cy, window=3):
        h, w = depth.shape[:2]
        patch = depth[max(0, cy - window):min(h, cy + window),
                      max(0, cx - window):min(w, cx + window)].astype(np.float32)
        patch = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(patch)) if patch.size else None

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def save_competition_image(self, color, markers, hit):
        img = color.copy()
        h, w = img.shape[:2]

        xs = sorted(m['cx'] for m in markers)
        pitch = int(np.median(np.diff(xs))) if len(xs) >= 2 else int(w * 0.18)
        half = max(20, pitch // 2)

        x, y, bw, bh = hit['box']
        left, right = max(0, hit['cx'] - half), min(w - 1, hit['cx'] + half)
        top = max(0, y - 10)

        if self.annotate_others:
            for m in markers:
                if m['digit'] == self.target:
                    continue
                mx, my, mw, mh = m['box']
                cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (200, 200, 200), 1)

        cv2.rectangle(img, (left, top), (right, h - 1), (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f'Target column: {self.target}', (left, max(18, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, stamp, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4)
        cv2.putText(img, stamp, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        path = os.path.join(self.images_dir,
                            f'column_{self.target}_{time.strftime("%Y%m%d_%H%M%S")}.png')
        if cv2.imwrite(path, img):
            self.get_logger().info(f'Saved competition image: {path}')
        else:
            self.get_logger().error(f'Failed to write {path} - check permissions.')
            self.exit_code = 1

    def save_debug_frame(self, color, markers):
        if not self.debug:
            return
        self._debug_counter += 1
        if self._debug_counter % 5:
            return
        img = color.copy()
        if self.last_candidates:
            for digit, score, margin, x, y, w, h in self.last_candidates:
                ok = score >= self.MIN_TEMPLATE_SCORE and margin >= self.MIN_TEMPLATE_MARGIN
                colour = (0, 255, 0) if ok else (0, 140, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(img, f'{digit}:{score:.2f}/m{margin:.2f}', (x, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        else:
            cv2.putText(img, 'NO CANDIDATES', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f'phase={self.phase} seen={[m["digit"] for m in markers]}',
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        out_dir = '/tmp/erc_debug_frames'
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f'col_{time.strftime("%H%M%S")}_{self._debug_counter}.jpg'),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 70])


def main(args=None):
    rclpy.init(args=args)
    node = ColumnDetector()
    code = 0
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.stop_base()
        time.sleep(0.3)
        code = node.exit_code
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
