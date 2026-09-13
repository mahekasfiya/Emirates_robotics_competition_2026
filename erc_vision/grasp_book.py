#!/usr/bin/env python3
"""
grasp_book.py  --  ERC 2026 Library Assistant Robot, stage 5 of 6.

Picks the target book off the shelf and folds it into a carry pose.

Frame handling:
  Latest-frame cache. The robot is stationary during the re-detect step,
  so the most recent frame from each camera is what we want.

Grasp verification:
  Position + contact only. Effort readings from this position-controlled
  gripper are unreliable (typically 0.00-0.05 N even on a firm grasp,
  because the position controller has no motion to oppose). The honest
  signals are:
    finger joint close to 0  - pads fully shut
    contact=True             - pads touching a book's mesh
"""

import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_vision.erc_arm_kinematics import ArmKinematics, rotation_from_columns

COLUMN_FILE = '/tmp/erc_target_column.json'
BOOK_FILE = '/tmp/erc_target_book.json'
CARRY_FILE = '/tmp/erc_carry_state.json'

COLOUR_RANGES = {
    'red':    [((0, 110, 50), (9, 255, 255)), ((170, 110, 50), (179, 255, 255))],
    'green':  [((45, 90, 40), (88, 255, 255))],
    'blue':   [((100, 110, 40), (132, 255, 255))],
    'yellow': [((22, 110, 70), (36, 255, 255))],
}

ARM_JOINTS = [f'arm_right_{i}_joint' for i in range(1, 8)]


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GraspBook(Node):

    # --- Base placement ---------------------------------------------------
    GRASP_STANDOFF = 0.54
    RETREAT_STANDOFF = 0.85
    SHOULDER_OFFSET = 0.159
    PAD_CENTRE_BEHIND_FRAME = 0.056
    GRASP_DEPTH_INTO_BOOK = 0.13
    PREGRASP_OFFSET = 0.10
    STAGING_EXTRA = 0.15
    LIFT_HEIGHT = 0.03
    WITHDRAW = 0.22
    WITHDRAW_SEC = 2.5
    IK_RESTARTS = 30

    # --- Torso ------------------------------------------------------------
    SURVEY_TORSO = 0.25
    TORSO_SETTLE_SEC = 14.0
    TORSO_TOLERANCE = 0.006
    TORSO_MAX_SAFE = 0.28
    ROW_HEIGHTS = {1: 1.595, 2: 1.265, 3: 0.935, 4: 0.605}
    MAX_ROW_Z_DEVIATION = 0.08
    STANDOFF_SHRINK = 0.05
    MIN_STANDOFF = 0.46

    # --- Gripper ----------------------------------------------------------
    GRIP_OPEN = 0.068
    GRIP_CLOSED = 0.0
    GRIP_MOVE_SEC = 1.5
    GRIP_OPEN_CONFIRM = 0.055
    BOOK_SPINE = 0.02
    GRIP_STALL_MAX = 0.015
    GRIP_HOLD_PERIOD = 0.5
    GRIP_CLOSE_DURATION = 0.35
    GRIP_CLOSE_TIMEOUT = 6.0
    GRIP_SETTLE_TOL = 0.0004
    GRIP_HOLD_DURATION = 0.08
    BUMP_GUARD_ENABLED = False

    # --- Alignment --------------------------------------------------------
    XY_TOLERANCE = 0.020
    YAW_TOLERANCE = 0.010
    XY_ACCEPT = 0.050
    FORWARD_ACCEPT = 0.045
    LATERAL_ACCEPT = 0.060
    YAW_ACCEPT = 0.070
    ALIGN_PATIENCE = 10.0
    KP_LINEAR = 1.2
    KP_ANGULAR = 1.5
    MAX_LINEAR = 0.12
    MAX_ANGULAR = 0.45
    MIN_LINEAR = 0.035
    MIN_ANGULAR = 0.09
    ALIGN_TIMEOUT = 30.0
    MAX_CREEP = 0.75

    # --- Detection --------------------------------------------------------
    MIN_CONTOUR_AREA = 500
    MAX_CONTOUR_AREA = 120000
    DETECT_SETTLE_SEC = 1.2
    DETECT_TIMEOUT_SEC = 6.0
    SEARCH_RADIUS = 0.25

    MAX_ATTEMPTS = 3

    def __init__(self):
        super().__init__('grasp_book')
        self.declare_parameter('grasp_standoff', self.GRASP_STANDOFF)
        self.declare_parameter('retreat_standoff', self.RETREAT_STANDOFF)
        self.declare_parameter('max_attempts', self.MAX_ATTEMPTS)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('save_debug_frames', False)

        self.grasp_standoff = float(self.get_parameter('grasp_standoff').value)
        self.retreat_standoff = float(self.get_parameter('retreat_standoff').value)
        self.max_attempts = int(self.get_parameter('max_attempts').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.debug = bool(self.get_parameter('save_debug_frames').value)

        self.bridge = CvBridge()
        self.kin = None
        self.camera_matrix = None
        self.joint_state = {}
        self.book_contact = False
        self.joint_effort = {}
        self.grip_hold_value = None
        self.last_failure = None
        self.exit_code = 0

        # Latest-frame cache
        self._latest_colour = None
        self._latest_depth = None
        self._colour_stamp = 0.0
        self._depth_stamp = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            JointTrajectory, '/gripper_right_controller/joint_trajectory', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, '/robot_description',
                                 self.urdf_cb, latched)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(CameraInfo,
                                 '/head_front_camera/head_front_camera/color/camera_info',
                                 self.info_cb, 10)
        self.create_subscription(Contacts, '/contacts', self.contacts_cb, 10)
        self.create_subscription(
            Image, '/head_front_camera/head_front_camera/color/image_raw',
            self._colour_cb, 5)
        self.create_subscription(
            Image, '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self._depth_cb, 5)

    def _colour_cb(self, msg):
        self._latest_colour = msg
        self._colour_stamp = time.time()

    def _depth_cb(self, msg):
        self._latest_depth = msg
        self._depth_stamp = time.time()

    def urdf_cb(self, msg):
        if self.kin is None:
            try:
                self.kin = ArmKinematics(msg.data)
                self.get_logger().info(
                    f'Kinematic chain loaded from /robot_description: '
                    f'{len(self.kin.joint_names)} joints '
                    f'({", ".join(self.kin.joint_names)}).')
            except Exception as e:
                self.get_logger().error(f'Could not parse the URDF: {e}')

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_state[name] = msg.position[i]
            if msg.effort and i < len(msg.effort):
                self.joint_effort[name] = msg.effort[i]

    def info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def contacts_cb(self, msg):
        for c in msg.contacts:
            names = f'{c.collision1.name} {c.collision2.name}'.lower()
            if 'gripper_right' in names and 'book' in names:
                self.book_contact = True
                return

    def wait_ready(self, timeout=25.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.kin is not None and self.camera_matrix is not None and self.joint_state:
                break
            time.sleep(0.1)
        else:
            self.get_logger().error('Timed out waiting for inputs.')
            return False

        tf_deadline = time.time() + 20.0
        while time.time() < tf_deadline:
            if self.robot_pose(0.5) is not None:
                self.get_logger().info('map -> base_link available; TF buffer primed.')
                return True
            time.sleep(0.2)
        self.get_logger().error('No map -> base_link transform after 20 s.')
        return False

    def current_q(self):
        names = self.kin.joint_names
        return np.array([self.joint_state.get(n, 0.0) for n in names])

    def robot_pose(self, timeout=1.0):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, RclTime(),
                timeout=rclpy.duration.Duration(seconds=timeout))
        except Exception:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y,
                yaw_from_quat(tf.transform.rotation))

    @staticmethod
    def _cmd(raw, error, tolerance, floor, ceiling):
        out = float(np.clip(raw, -ceiling, ceiling))
        if error > tolerance * 1.5 and abs(out) < floor:
            out = math.copysign(min(floor, ceiling), raw if raw != 0 else 1.0)
        return out

    def sim_now(self):
        t = self.get_clock().now().nanoseconds / 1e9
        return t if t > 0 else time.time()

    def stop_base(self):
        stop = Twist()
        for _ in range(3):
            self.cmd_pub.publish(stop)
            time.sleep(0.02)

    def send_traj(self, pub, names, positions, seconds, extra_points=None):
        traj = JointTrajectory()
        traj.joint_names = list(names)
        points = extra_points if extra_points else [(positions, seconds)]
        for pos, t in points:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in pos]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)
        pub.publish(traj)

    def move_arm(self, q_list, seconds_each=2.0, wait=True):
        pts = []
        t = 0.0
        for q in q_list:
            t += seconds_each
            pts.append((q[1:], t))
        self.send_traj(self.arm_pub, ARM_JOINTS, None, 0, extra_points=pts)
        if wait:
            self.sleep_sim(t + 0.6)

    def move_torso(self, value, wait=True):
        value = float(np.clip(value, -0.001, 0.35))
        self.send_traj(self.torso_pub, ['torso_lift_joint'], [value],
                       max(2, int(self.TORSO_SETTLE_SEC)))
        if not wait:
            return True
        start = self.get_clock().now().nanoseconds / 1e9
        use_sim = start > 0
        wall_deadline = time.time() + (self.TORSO_SETTLE_SEC + 6.0) * 25
        last = None
        while time.time() < wall_deadline:
            actual = self.joint_state.get('torso_lift_joint')
            if actual is not None and abs(actual - value) < self.TORSO_TOLERANCE:
                self.get_logger().info(f'Torso at {actual:.3f} m.')
                return True
            if use_sim:
                elapsed = self.get_clock().now().nanoseconds / 1e9 - start
                if elapsed > self.TORSO_SETTLE_SEC + 6.0:
                    if last is not None and actual is not None and abs(actual - last) < 0.001:
                        break
                    last = actual
                    start = self.get_clock().now().nanoseconds / 1e9
            time.sleep(0.1)
        actual = self.joint_state.get('torso_lift_joint', float('nan'))
        self.get_logger().warn(
            f'Torso stopped at {actual:.3f} m, short of {value:.3f} m.')
        return False

    def move_head(self, pan, tilt, wait=True):
        self.send_traj(self.head_pub, ['head_1_joint', 'head_2_joint'],
                       [pan, tilt], 1)
        if wait:
            self.sleep_sim(1.6)

    def set_gripper(self, value, wait=True, duration=None):
        value = float(np.clip(value, 0.0, 0.069))
        self.send_traj(self.grip_pub, ['gripper_right_finger_joint'],
                       [value], duration if duration is not None
                       else max(1, int(self.GRIP_MOVE_SEC)))
        if wait:
            self.sleep_sim(self.GRIP_MOVE_SEC + 0.4)

    def close_gripper_and_wait(self):
        self.set_gripper(self.GRIP_CLOSED, wait=False,
                         duration=self.GRIP_CLOSE_DURATION)
        start = self.sim_now()
        last = None
        still = 0
        while self.sim_now() - start < self.GRIP_CLOSE_TIMEOUT:
            pos = self.joint_state.get('gripper_right_finger_joint')
            if pos is not None:
                if last is not None and abs(pos - last) < self.GRIP_SETTLE_TOL:
                    still += 1
                    if still >= 6:
                        break
                else:
                    still = 0
                last = pos
            time.sleep(0.05)
        pos = self.joint_state.get('gripper_right_finger_joint', 0.0)
        return float(pos), self.grip_effort()

    def start_grip_hold(self, value):
        self.grip_hold_value = float(np.clip(value, 0.0, 0.069))
        if getattr(self, '_grip_timer', None) is None:
            self._grip_timer = self.create_timer(
                self.GRIP_HOLD_PERIOD, self._grip_hold_tick)

    def _grip_hold_tick(self):
        if self.grip_hold_value is None:
            return
        self.send_traj(self.grip_pub, ['gripper_right_finger_joint'],
                       [self.grip_hold_value], self.GRIP_HOLD_DURATION)

    def stop_grip_hold(self):
        self.grip_hold_value = None

    def grip_effort(self):
        e = self.joint_effort.get('gripper_right_finger_joint')
        return abs(e) if e is not None and np.isfinite(e) else None

    def confirm_open(self):
        deadline = time.time() + 8.0
        while time.time() < deadline:
            pos = self.joint_state.get('gripper_right_finger_joint')
            if pos is not None and pos >= self.GRIP_OPEN_CONFIRM:
                return True
            time.sleep(0.1)
        pos = self.joint_state.get('gripper_right_finger_joint', float('nan'))
        self.get_logger().error(f'Gripper did not open: finger joint at {pos:.4f}.')
        return False

    def sleep_sim(self, seconds):
        start = self.get_clock().now().nanoseconds / 1e9
        if start == 0:
            time.sleep(seconds)
            return
        while rclpy.ok():
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start >= seconds:
                return
            time.sleep(0.05)

    def drive_to(self, goal_xy, goal_yaw, label=''):
        pose = self.robot_pose(3.0)
        if pose is None:
            self.get_logger().error('No map->base_link transform - cannot move.')
            return False
        dist = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
        if dist > self.MAX_CREEP:
            self.get_logger().error(f'{label} target past the creep limit.')
            return False

        started = self.sim_now()
        wall_cap = time.time() + self.ALIGN_TIMEOUT * 40
        settled = None
        best = (1e9, 1e9)
        while self.sim_now() - started < self.ALIGN_TIMEOUT and time.time() < wall_cap:
            pose = self.robot_pose(0.2)
            if pose is None:
                time.sleep(0.05)
                continue
            ex, ey = goal_xy[0] - pose[0], goal_xy[1] - pose[1]
            eyaw = normalize_angle(goal_yaw - pose[2])
            exy = math.hypot(ex, ey)
            best = (min(best[0], exy), min(best[1], abs(eyaw)))
            waited = self.sim_now() - started
            c0, s0 = math.cos(goal_yaw), math.sin(goal_yaw)
            e_fwd = abs(ex * c0 + ey * s0)
            e_lat = abs(-ex * s0 + ey * c0)
            if waited < self.ALIGN_PATIENCE:
                fwd_ok = lat_ok = self.XY_TOLERANCE
                yaw_ok = self.YAW_TOLERANCE
            else:
                fwd_ok, lat_ok = self.FORWARD_ACCEPT, self.LATERAL_ACCEPT
                yaw_ok = self.YAW_ACCEPT
            xy_ok = max(fwd_ok, lat_ok)
            if e_fwd < fwd_ok and e_lat < lat_ok and abs(eyaw) < yaw_ok:
                self.stop_base()
                if settled is None:
                    settled = time.time()
                elif time.time() - settled > 0.4:
                    self.get_logger().info(
                        f'{label} reached: {exy * 100:.1f} cm, '
                        f'{math.degrees(abs(eyaw)):.2f} deg residual.')
                    return True
                time.sleep(0.05)
                continue
            settled = None
            c, s = math.cos(pose[2]), math.sin(pose[2])
            bx, by = ex * c + ey * s, -ex * s + ey * c
            tw = Twist()
            tw.linear.x = self._cmd(self.KP_LINEAR * bx, exy, xy_ok,
                                    self.MIN_LINEAR * abs(bx) / (exy + 1e-9),
                                    self.MAX_LINEAR)
            tw.linear.y = self._cmd(self.KP_LINEAR * by, exy, xy_ok,
                                    self.MIN_LINEAR * abs(by) / (exy + 1e-9),
                                    self.MAX_LINEAR)
            tw.angular.z = self._cmd(self.KP_ANGULAR * eyaw, abs(eyaw), yaw_ok,
                                     self.MIN_ANGULAR, self.MAX_ANGULAR)
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.stop_base()
        pose = self.robot_pose(1.0)
        if pose is not None:
            ex, ey = goal_xy[0] - pose[0], goal_xy[1] - pose[1]
            c0, s0 = math.cos(goal_yaw), math.sin(goal_yaw)
            e_fwd = abs(ex * c0 + ey * s0)
            e_lat = abs(-ex * s0 + ey * c0)
            self.get_logger().warn(
                f'{label} alignment timed out. '
                f'Final split: fwd {e_fwd * 100:.1f} cm (limit '
                f'{self.FORWARD_ACCEPT * 100:.1f}), lat {e_lat * 100:.1f} cm '
                f'(limit {self.LATERAL_ACCEPT * 100:.1f}), '
                f'yaw {math.degrees(abs(normalize_angle(goal_yaw - pose[2]))):.2f} deg '
                f'(limit {math.degrees(self.YAW_ACCEPT):.2f}).')
        return False

    def redetect_book(self, colour, expected_map_xyz):
        started = self.sim_now()
        wall_cap = time.time() + self.DETECT_TIMEOUT_SEC * 40
        best = None
        last_stamp = None
        frames = 0
        while self.sim_now() - started < self.DETECT_TIMEOUT_SEC and time.time() < wall_cap:
            if self._latest_colour is None or self._latest_depth is None:
                time.sleep(0.05)
                continue
            cmsg = self._latest_colour
            dmsg = self._latest_depth
            stamp = self._colour_stamp
            if time.time() - stamp > 0.5 or stamp == last_stamp:
                time.sleep(0.05)
                continue
            last_stamp = stamp
            frames += 1
            try:
                color = self.bridge.imgmsg_to_cv2(cmsg, 'bgr8')
                depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding='passthrough')
            except Exception:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = None
            for lo, hi in COLOUR_RANGES[colour]:
                part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                m = np.zeros(depth.shape[:2], np.uint8)
                cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
                vals = depth[m > 0].astype(np.float32)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size < 20:
                    continue
                z = float(np.median(vals))
                pos = self.pixel_to_map(cmsg.header, x + w // 2, y + h // 2, z)
                if pos is None:
                    continue
                err = float(np.linalg.norm(pos - np.asarray(expected_map_xyz)))
                if err > self.SEARCH_RADIUS:
                    continue
                if best is None or err < best[1]:
                    best = (pos, err)
            if best is not None:
                self.get_logger().info(
                    f'Re-detected the {colour} book: '
                    f'({best[0][0]:.3f}, {best[0][1]:.3f}, {best[0][2]:.3f}), '
                    f'{best[1] * 100:.1f} cm from the estimate.')
                return best[0]
            time.sleep(0.05)

        self.get_logger().warn(
            f'Could not re-detect the book in {frames} frames - using the detection-pose '
            'estimate.')
        return None

    def pixel_to_map(self, header, cx, cy, z):
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
        except Exception:
            return None
        return np.array([out.point.x, out.point.y, out.point.z])

    def map_to_base(self, xyz):
        pt = PointStamped()
        pt.header.frame_id = self.map_frame
        pt.header.stamp = RclTime().to_msg()
        pt.point.x, pt.point.y, pt.point.z = [float(v) for v in xyz]
        try:
            out = self.tf_buffer.transform(
                pt, 'base_footprint', timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'map -> base_footprint failed: {e}')
            return None
        return np.array([out.point.x, out.point.y, out.point.z])

    def run(self):
        if not self.wait_ready():
            self.exit_code = 1
            return
        try:
            with open(BOOK_FILE) as f:
                book = json.load(f)
            with open(COLUMN_FILE) as f:
                col = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Could not read the handoff files: {e}')
            self.exit_code = 1
            return

        colour = book['colour']
        row = int(book['row'])
        book_xyz = np.array(book['book_xyz'], float)
        normal = np.array(col['shelf_normal'], float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        shelf_dir = np.array([-normal[1], normal[0]])

        self.get_logger().info(
            f'Grasping the {colour} book, column {book["column"]}, row {row}.')

        standoff = self.grasp_standoff
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                self.get_logger().warn(
                    f'--- Grasp attempt {attempt}/{self.max_attempts}, '
                    f'standoff {standoff:.2f} m ---')
            if self.attempt_grasp(colour, row, book_xyz, normal, shelf_dir, standoff):
                self.retreat(book_xyz, normal, shelf_dir)
                self.record_carry(book, True)
                return
            if self.last_failure == 'gripper':
                self.get_logger().error('Gripper never opened - stopping.')
                break
            self.recover_after_failure()
            if self.last_failure == 'ik':
                standoff = max(self.MIN_STANDOFF, standoff - self.STANDOFF_SHRINK)
                self.get_logger().info(f'Next attempt will park at {standoff:.2f} m.')

        self.get_logger().error(
            f'Failed to grasp the {colour} book after {self.max_attempts} attempts.')
        self.record_carry(book, False)
        self.exit_code = 1

    def attempt_grasp(self, colour, row, book_xyz, normal, shelf_dir, standoff):
        goal_xy = (book_xyz[:2] + normal * standoff
                   - shelf_dir * self.SHOULDER_OFFSET)
        goal_yaw = math.atan2(-normal[1], -normal[0])
        if not self.drive_to(goal_xy, goal_yaw, label='Grasp standoff'):
            self.last_failure = 'align'
            return False

        self.move_torso(self.SURVEY_TORSO)
        base_pt = self.map_to_base(book_xyz)
        if base_pt is None:
            return False
        cam_h = 1.147 + self.SURVEY_TORSO
        tilt = float(np.clip(math.atan2(base_pt[2] - cam_h,
                                        max(0.2, base_pt[0])), -1.0, 0.34))
        self.move_head(0.0, tilt)
        self.sleep_sim(self.DETECT_SETTLE_SEC)

        refined = self.redetect_book(colour, book_xyz)
        target_map = refined if refined is not None else book_xyz
        base_pt = self.map_to_base(target_map)
        if base_pt is None:
            return False

        row_z = self.ROW_HEIGHTS.get(row)
        grasp_z = base_pt[2]
        if row_z is not None:
            delta = grasp_z - row_z
            if abs(delta) > self.MAX_ROW_Z_DEVIATION:
                self.get_logger().warn(
                    f'Measured book height {grasp_z:.3f} m is {delta * 100:+.1f} cm from '
                    f'row {row}\'s shelf height - beyond the trust band. Using the row '
                    'height instead.')
                grasp_z = row_z
            else:
                self.get_logger().info(
                    f'Book height {grasp_z:.3f} m ({delta * 100:+.1f} cm from row nominal).')

        grasp_p = np.array([base_pt[0] + self.GRASP_DEPTH_INTO_BOOK,
                            base_pt[1], grasp_z])
        pregrasp_p = grasp_p - np.array([self.PREGRASP_OFFSET, 0.0, 0.0])
        staging_p = pregrasp_p - np.array([self.STAGING_EXTRA, 0.0, 0.0])
        self.get_logger().info(
            f'Book face at x={base_pt[0]:.3f}; grasping frame target x={grasp_p[0]:.3f}.')

        seed = self.current_q()
        safe_torso = (np.array([-0.001] + [-9.0] * 7),
                      np.array([self.TORSO_MAX_SAFE] + [9.0] * 7))
        shelf_plane = np.array([base_pt[0], 0.0, 0.0])

        best = None
        for label, R in (('upright', rotation_from_columns((1, 0, 0), (0, 1, 0), (0, 0, 1))),
                         ('rolled', rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1)))):
            q_st, pe_st, re_st = self.kin.solve(staging_p, R, seed, bounds=safe_torso,
                                                restarts=self.IK_RESTARTS)
            if q_st is None or not self.kin.reached(pe_st, re_st):
                continue
            q, pe, re = self.kin.solve(pregrasp_p, R, q_st, bounds=safe_torso,
                                       restarts=self.IK_RESTARTS)
            if q is None or not self.kin.reached(pe, re):
                continue
            travel = float(np.linalg.norm(q_st - seed) + np.linalg.norm(q - q_st))
            self.get_logger().info(
                f'  {label} wrist: staging {pe_st * 1000:.2f} mm, pre-grasp '
                f'{pe * 1000:.2f} mm, joint travel {travel:.2f} rad.')
            if best is None or travel < best[4]:
                best = (q, R, label, q_st, travel)
        if best is None:
            self.get_logger().error('No IK solution for the pre-grasp.')
            self.last_failure = 'ik'
            return False
        q_pre, R_grasp, roll_label, q_stage, _ = best
        torso_target = float(q_pre[0])
        self.get_logger().info(
            f'Pre-grasp chosen: {roll_label} wrist, torso {torso_target:.3f} m.')

        self.move_torso(torso_target)
        torso_now = float(self.joint_state.get('torso_lift_joint', torso_target))
        torso_bounds = (np.array([torso_now - 1e-4] + [-9.0] * 7),
                        np.array([torso_now + 1e-4] + [9.0] * 7))

        if abs(torso_now - torso_target) > self.TORSO_TOLERANCE:
            q_fix, pe, re = self.kin.solve(pregrasp_p, R_grasp, q_pre,
                                           bounds=torso_bounds, lock=[0])
            if q_fix is not None and self.kin.reached(pe, re):
                q_pre = q_fix
                self.get_logger().info(
                    f'Re-solved pre-grasp at torso {torso_now:.3f} m.')
            else:
                self.get_logger().error(
                    f'Cannot reach the book with the torso at {torso_now:.3f} m.')
                self.last_failure = 'ik'
                return False

        self.book_contact = False
        self.set_gripper(self.GRIP_OPEN)
        if not self.confirm_open():
            self.last_failure = 'gripper'
            return False

        self.move_arm([q_stage], seconds_each=3.0)
        approach, ok = self.kin.cartesian_path(q_stage, [pregrasp_p], R_grasp,
                                               lock=[0], bounds=torso_bounds,
                                               restarts=self.IK_RESTARTS)
        if not ok:
            self.get_logger().error('Could not solve staging -> pre-grasp.')
            return False
        self.move_arm(approach, seconds_each=2.0)

        steps = 5
        waypoints = [pregrasp_p + (grasp_p - pregrasp_p) * (i + 1) / steps
                     for i in range(steps)]
        qs, ok = self.kin.cartesian_path(approach[-1], waypoints, R_grasp,
                                         lock=[0], bounds=torso_bounds,
                                         restarts=self.IK_RESTARTS)
        if not ok:
            self.get_logger().error('Could not solve straight-line entry.')
            return False
        for q in qs:
            clear, link = self.kin.clearance_ok(
                q, plane_point=np.array([base_pt[0], 0.0, 0.0]),
                plane_normal=np.array([1.0, 0.0, 0.0]))
            if not clear:
                self.get_logger().error(
                    f'Entry path would put {link} past the shelf face.')
                return False
        self.move_arm(qs, seconds_each=1.0)

        if self.book_contact and self.BUMP_GUARD_ENABLED:
            self.get_logger().warn('Bump guard: backing out.')
            self.last_failure = 'align'
            return False

        self.book_contact = False
        finger, effort = self.close_gripper_and_wait()
        self.start_grip_hold(self.GRIP_CLOSED)
        self.get_logger().info(
            f'Closed on the spine: finger joint {finger:.4f}, effort '
            f'{("%.2f N" % effort) if effort is not None else "n/a"}, '
            f'contact={self.book_contact}.')

        # Contacts fire briefly during the close. Give the sensor a moment to
        # latch before reading.
        self.sleep_sim(0.5)

        # Reject if the jaws stayed too wide - the pads met something much
        # thicker than the spine or never finished closing.
        if finger > self.GRIP_STALL_MAX:
            self.get_logger().warn(
                f'Finger joint at {finger:.4f} - wider than a 2 cm spine. Miss.')
            self.stop_grip_hold()
            return False

        # Position + contact only. Effort readings from this position-controlled
        # gripper are unreliable (0.00-0.05 N even on a firm grasp, because the
        # position controller has no motion to oppose). The honest signals are
        # finger < GRIP_STALL_MAX (pads shut) AND /contacts fired (pads touching
        # a book mesh).
        if not self.book_contact:
            self.get_logger().warn(
                f'No gripper-to-book contact (finger joint at {finger:.4f}, '
                f'effort {("%.2f N" % effort) if effort is not None else "n/a"}). Miss.')
            self.stop_grip_hold()
            return False

        self.get_logger().info(
            f'Book in hand: finger {finger:.4f}, contact=True '
            f'(effort {("%.2f N" % effort) if effort is not None else "n/a"} - '
            'not trusted on this gripper).')

        lift_p = grasp_p + np.array([0.0, 0.0, self.LIFT_HEIGHT])
        out_p = lift_p - np.array([self.WITHDRAW, 0.0, 0.0])
        qs_out, ok = self.kin.cartesian_path(qs[-1], [lift_p, out_p], R_grasp,
                                             lock=[0], bounds=torso_bounds,
                                             restarts=self.IK_RESTARTS)
        if not ok:
            qs_out = list(reversed(qs[:-1])) + [q_pre]
        self.book_contact = False
        self.move_arm(qs_out, seconds_each=self.WITHDRAW_SEC)

        self.sleep_sim(1.0)
        finger = self.joint_state.get('gripper_right_finger_joint', 0.0)
        # Position + contact only, same as the post-close check.
        lost = finger > self.GRIP_STALL_MAX or not self.book_contact
        if lost:
            self.get_logger().warn(
                f'Book lost on the way out (finger {finger:.4f}, contact '
                f'{self.book_contact}). Retrying.')
            self.stop_grip_hold()
            return False
        self.get_logger().info('Book still held after withdrawal.')
        return True

    def recover_after_failure(self):
        self.stop_grip_hold()
        self.set_gripper(self.GRIP_OPEN, wait=False)
        seed = self.current_q()
        target = np.array([0.30, -0.20, 1.05])
        R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
        q, pe, re = self.kin.solve(target, R, seed)
        if q is not None and self.kin.reached(pe, re):
            self.move_arm([q], seconds_each=3.0)
        else:
            self.get_logger().warn('No IK for the recovery pose.')

    def retreat(self, book_xyz, normal, shelf_dir):
        seed = self.current_q()
        R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
        for target in (np.array([0.34, -0.19, 1.05]),
                       np.array([0.30, -0.22, 1.00]),
                       np.array([0.38, -0.16, 1.10])):
            q, pe, re = self.kin.solve(target, R, seed)
            if q is not None and self.kin.reached(pe, re):
                self.move_arm([q], seconds_each=3.0)
                self.get_logger().info(
                    f'Carry pose set, gripper at ({target[0]:.2f}, {target[1]:.2f}, '
                    f'{target[2]:.2f}) in base_footprint.')
                break
        else:
            self.get_logger().warn('No IK for any carry pose.')

        goal_xy = (book_xyz[:2] + normal * self.retreat_standoff
                   - shelf_dir * self.SHOULDER_OFFSET)
        goal_yaw = math.atan2(-normal[1], -normal[0])
        self.drive_to(goal_xy, goal_yaw, label='Retreat standoff')

    def record_carry(self, book, holding):
        payload = dict(book)
        payload['holding'] = bool(holding)
        payload['stamp'] = time.time()
        try:
            with open(CARRY_FILE, 'w') as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            self.get_logger().error(f'Could not write {CARRY_FILE}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GraspBook()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    code = 0
    try:
        node.run()
        code = node.exit_code
    finally:
        node.stop_base()
        time.sleep(0.3)
        if rclpy.ok():
            rclpy.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
    sys.exit(code)


if __name__ == '__main__':
    main()
