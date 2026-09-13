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

Swing routing:
  The tuck-to-staging move is a joint-space interpolation across ~4.3 rad
  of arm travel. Interpolating that in one shot bulges the gripper forward
  past the book face mid-move. The swing is split: tuck -> RETRACT (gripper
  well behind the book face) -> staging.

Entry step distribution:
  Fine steps on the COMMIT, coarse on the creep to the checkpoint.

Recovery:
  Pull the arm straight back along the approach axis BEFORE folding to
  tuck, so the fold happens in open air rather than sweeping through the
  region where the book sits.

Lateral aim:
  The re-detect's Y is cross-checked against the far detection. When they
  disagree by more than LATERAL_CROSSCHECK_M, the far detection's Y wins.
  The disturb guard compares consecutive re-detects; threshold raised to
  0.080 so AMCL drift (3-4 cm between attempts) doesn't false-trigger.
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
RIGHT_TUCK = [-0.3, -1.9, 0.3, -2.35, 0.0, 0.0, 0.0]


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
    GRASP_DEPTH_INTO_BOOK = 0.07
    PREGRASP_OFFSET = 0.15
    STAGING_EXTRA = 0.15
    TIP_BEHIND_FRAME = 0.027
    # Empirical lateral trim, metres in base_footprint y (+ is the robot's
    # LEFT). Use this only if a residual bias survives the depth-frame fix:
    # the 'lateral residual' line logged at the hover checkpoint gives both
    # the sign and the size. Positive shifts the gripper left, which is what
    # you want if the RIGHT finger is the one catching the spine.
    LATERAL_AIM_BIAS = 0.0
    TIP_CLEARANCE = 0.020
    POSTURE_LEASH = 2.6
    LIFT_HEIGHT = 0.03
    WITHDRAW = 0.22
    WITHDRAW_SEC = 2.5
    IK_RESTARTS = 30
    RETRACT_BEHIND_BOOK = 0.35
    RETRACT_BELOW_BOOK = 0.15
    RECOVER_RETRACT_DISTANCE = 0.45
    RECOVER_RETRACT_MIN_X = 0.15
    ENTRY_STEPS_APPROACH = 2
    ENTRY_STEPS_COMMIT = 6

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
    GRIP_CLOSE_DURATION = 0.9
    GRIP_CLOSE_TIMEOUT = 6.0
    GRIP_SETTLE_TOL = 0.0004
    GRIP_HOLD_DURATION = 0.08
    BUMP_GUARD_ENABLED = True

    # --- Alignment --------------------------------------------------------
    XY_TOLERANCE = 0.020
    YAW_TOLERANCE = 0.010
    XY_ACCEPT = 0.050
    FORWARD_ACCEPT = 0.050
    LATERAL_ACCEPT = 0.040
    YAW_ACCEPT = 0.070
    PROCEED_FWD = 0.070
    PROCEED_LAT = 0.045
    PROCEED_YAW = 0.090
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
    MIN_CONTOUR_AREA = 250
    MAX_CONTOUR_AREA = 120000
    DETECT_SETTLE_SEC = 1.2
    DETECT_TIMEOUT_SEC = 6.0
    SEARCH_RADIUS = 0.25
    REDETECT_FRAMES = 6
    REDETECT_MIN_FRAMES = 3
    REDETECT_MAX_SPREAD = 0.030
    OUTLIER_REJECT = 0.030
    DEPTH_ERODE = 2
    BORDER_MARGIN = 3
    MIN_BLOB_WIDTH_M = 0.008
    MAX_BLOB_WIDTH_M = 0.110
    # A book that has been genuinely knocked over shows up as a 10+ cm jump
    # in the map-frame measurement. AMCL drift between attempts is 3-4 cm on
    # this setup, which the previous 3 cm threshold was misreading as book
    # motion. 8 cm is high enough to ignore AMCL drift, low enough to catch
    # a real disturbance.
    DISTURBED_THRESHOLD = 0.080
    # Drift in the 4-8 cm range gets logged but does not abort.
    DISTURBED_LOG = 0.040
    LATERAL_CROSSCHECK_M = 0.025

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
        self.depth_matrix = None
        self.joint_state = {}
        self.book_contact = False
        self.joint_effort = {}
        self.grip_hold_value = None
        self.last_failure = None
        self.last_measured = None
        self.exit_code = 0

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
        self.create_subscription(CameraInfo,
                                 '/head_front_camera/head_front_camera/depth/camera_info',
                                 self.depth_info_cb, 10)
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

    def depth_info_cb(self, msg):
        if self.depth_matrix is None:
            self.depth_matrix = np.array(msg.k).reshape(3, 3)

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

    ARM_VEL_LIMIT = 1.95
    ARM_VEL_MARGIN = 0.30

    def leashed(self, bounds):
        lo, hi = bounds
        lo = np.array(lo, float).copy()
        hi = np.array(hi, float).copy()
        tuck = np.array(RIGHT_TUCK, float)
        lo[1:] = np.maximum(lo[1:], tuck - self.POSTURE_LEASH)
        hi[1:] = np.minimum(hi[1:], tuck + self.POSTURE_LEASH)
        return lo, hi

    def arm_move_time(self, q_target, minimum):
        now = self.current_q()
        travel = float(np.max(np.abs(np.asarray(q_target)[1:] - now[1:])))
        needed = travel / (self.ARM_VEL_LIMIT * self.ARM_VEL_MARGIN)
        return max(minimum, needed)

    def arm_error(self, q_target):
        now = self.current_q()
        return float(np.max(np.abs(np.asarray(q_target)[1:] - now[1:])))

    ARM_SETTLE_TOL = 0.020
    ARM_SETTLE_STILL = 6
    DROOP_CORRECTIONS = 8
    DROOP_GAIN = 1.8
    DROOP_FINAL_TOL = 0.012

    def move_arm(self, q_list, seconds_each=2.0, wait=True, precise=False):
        pts = []
        t = 0.0
        for q in q_list:
            t += seconds_each
            pts.append((q[1:], t))
        self.send_traj(self.arm_pub, ARM_JOINTS, None, 0, extra_points=pts)
        if not wait:
            return True

        target = np.asarray(q_list[-1])
        if self._settle(target, t) or not q_list:
            return True
        if not precise and self.arm_error(target) < 0.08:
            return True

        if not precise:
            return True
        tol = self.DROOP_FINAL_TOL
        rounds = self.DROOP_CORRECTIONS
        bias = np.zeros(7)
        best = self.arm_error(target)
        for attempt in range(rounds):
            actual = self.current_q()
            err = target[1:] - actual[1:]
            worst = float(np.max(np.abs(err)))
            best = min(best, worst)
            if worst < tol:
                if attempt:
                    self.get_logger().info(
                        f'Arm converged to {worst:.3f} rad after {attempt} correction(s).')
                return True
            bias = np.clip(bias + self.DROOP_GAIN * err, -0.6, 0.6)
            corrected = target.copy()
            corrected[1:] = np.clip(target[1:] + bias,
                                    self.kin.lower[1:], self.kin.upper[1:])
            self.send_traj(self.arm_pub, ARM_JOINTS, None, 0,
                           extra_points=[(corrected[1:], max(1.2, t * 0.35))])
            self._settle(target, max(1.2, t * 0.35))
        final = self.arm_error(target)
        self.get_logger().warn(
            f'Arm still {final:.3f} rad short after {rounds} droop corrections '
            f'(best {best:.3f}).')
        return False

    def _settle(self, target, t):
        start = self.sim_now()
        budget = t * 1.5 + 5.0
        cap = time.time() + budget * 40
        last, still = None, 0
        while time.time() < cap:
            err = self.arm_error(target)
            if err < self.ARM_SETTLE_TOL:
                return True
            now = self.current_q()[1:]
            if last is not None and float(np.max(np.abs(now - last))) < 0.002:
                still += 1
                if still >= self.ARM_SETTLE_STILL and self.sim_now() - start > t * 0.4:
                    return False
            else:
                still = 0
            last = now
            if self.sim_now() - start > budget:
                return False
            time.sleep(0.05)
        return False

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
            eyaw_abs = abs(normalize_angle(goal_yaw - pose[2]))
            if (e_fwd <= self.PROCEED_FWD and e_lat <= self.PROCEED_LAT
                    and eyaw_abs <= self.PROCEED_YAW):
                self.get_logger().warn(
                    f'Still within the workable band (fwd <= {self.PROCEED_FWD * 100:.0f} cm, '
                    f'lat <= {self.PROCEED_LAT * 100:.0f} cm) - going ahead. The entry '
                    'reaches further to compensate.')
                return True
        return False

    def dump_view(self, colour):
        try:
            if self._latest_colour is None:
                return
            img = self.bridge.imgmsg_to_cv2(self._latest_colour, 'bgr8')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = None
            for lo, hi in COLOUR_RANGES[colour]:
                part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)
            cv2.imwrite('/tmp/erc_grasp_view.png',
                        np.hstack([img, cv2.addWeighted(img, 0.5, overlay, 0.5, 0)]))
        except Exception as e:
            self.get_logger().warn(f'Could not write the diagnostic view: {e}')

    def redetect_book(self, colour, expected_map_xyz):
        started = self.sim_now()
        wall_cap = time.time() + self.DETECT_TIMEOUT_SEC * 40
        best = None
        samples = []
        last_stamp = None
        frames = 0
        stats = {'contours': 0, 'biggest': 0.0, 'area': 0, 'border': 0, 'depth': 0,
                 'width': 0, 'widths': [], 'tf': 0, 'radius': 0, 'nearest': 9.9}
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
            stats['contours'] += len(contours)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                stats['biggest'] = max(stats['biggest'], area)
                if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                    stats['area'] += 1
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                ih, iw = depth.shape[:2]
                if (x <= self.BORDER_MARGIN or y <= self.BORDER_MARGIN
                        or x + w >= iw - self.BORDER_MARGIN
                        or y + h >= ih - self.BORDER_MARGIN):
                    stats['border'] += 1
                    continue
                m = np.zeros(depth.shape[:2], np.uint8)
                cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
                inner = cv2.erode(m, np.ones((self.DEPTH_ERODE * 2 + 1,) * 2, np.uint8))
                src = inner if cv2.countNonZero(inner) >= 20 else m
                vals = depth[src > 0].astype(np.float32)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size < 20:
                    stats['depth'] += 1
                    continue
                z = float(np.percentile(vals, 30))
                width_m = w * z / (self.depth_matrix if self.depth_matrix is not None
                                   else self.camera_matrix)[0, 0]
                if not self.MIN_BLOB_WIDTH_M <= width_m <= self.MAX_BLOB_WIDTH_M:
                    stats['width'] += 1
                    stats['widths'].append(round(width_m, 4))
                    continue
                mom = cv2.moments(cnt)
                if mom['m00'] <= 0:
                    continue
                pos = self.pixel_to_map(dmsg.header, int(mom['m10'] / mom['m00']),
                                        int(mom['m01'] / mom['m00']), z,
                                        matrix=self.depth_matrix)
                if pos is None:
                    stats['tf'] += 1
                    continue
                err = float(np.linalg.norm(pos - np.asarray(expected_map_xyz)))
                stats['nearest'] = min(stats['nearest'], err)
                if err > self.SEARCH_RADIUS:
                    stats['radius'] += 1
                    continue
                if best is None or err < best[1]:
                    best = (pos, err)
            if best is not None:
                samples.append(best[0])
                best = None
                if len(samples) >= self.REDETECT_FRAMES:
                    break
            time.sleep(0.05)

        if len(samples) >= self.REDETECT_MIN_FRAMES:
            arr = np.array(samples)
            med = np.median(arr, axis=0)
            keep = arr[np.linalg.norm(arr - med, axis=1) <= self.OUTLIER_REJECT]
            if len(keep) < self.REDETECT_MIN_FRAMES:
                keep = arr
            pos = np.median(keep, axis=0)
            spread = float(np.max(np.linalg.norm(keep - pos, axis=1)))
            err = float(np.linalg.norm(pos - np.asarray(expected_map_xyz)))
            self.get_logger().info(
                f'Re-detected the {colour} book over {len(keep)}/{len(samples)} frames: '
                f'({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), {err * 100:.1f} cm from the '
                f'estimate, spread {spread * 100:.1f} cm.')
            if spread > self.REDETECT_MAX_SPREAD:
                self.get_logger().warn(
                    f'Frames disagree by {spread * 100:.1f} cm, over the '
                    f'{self.REDETECT_MAX_SPREAD * 100:.0f} cm limit. The jaws only clear '
                    'the spine by about 3 cm a side, so entering on this would probably '
                    'knock the book. Not entering.')
                return None
            return pos

        near = ('n/a' if stats['nearest'] > 9 else f"{stats['nearest'] * 100:.0f} cm")
        self.get_logger().warn(
            f'Could not re-detect the {colour} book over {frames} frames. '
            f'{stats["contours"]} contour(s) found, biggest {stats["biggest"]:.0f} px2. '
            f'Rejected: area {stats["area"]}, border-clipped {stats["border"]}, '
            f'depth {stats["depth"]}, width {stats["width"]} '
            f'{stats["widths"][:4]}, TF {stats["tf"]}, too-far {stats["radius"]} '
            f'(nearest {near}). Using the detection-pose estimate.')
        if stats['contours'] == 0:
            self.get_logger().warn(
                f'Zero {colour} contours at all - the HSV window is not matching, or the '
                'head is not pointing at the book. Check /tmp/erc_grasp_view.png.')
        self.dump_view(colour)
        return None

    def pixel_to_map(self, header, cx, cy, z, matrix=None):
        """Unproject a pixel to the map frame.

        The range comes from the DEPTH image, so the ray must be the DEPTH
        camera's - head_front_camera_color_frame sits 0.015 m to the side of
        head_front_camera_depth_frame in the URDF. Using the colour ray with a
        depth range mixes two cameras separated by that baseline and produces
        a systematic ~1.5 cm lateral error, always the same direction. The
        fingers only clear a 2 cm spine by about 3 cm a side, so a constant
        1.5 cm bias is most of the margin.
        """
        K = self.camera_matrix if matrix is None else matrix
        fx, fy = K[0, 0], K[1, 1]
        px, py = K[0, 2], K[1, 2]
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
            if self.last_failure == 'disturbed':
                self.get_logger().error(
                    'Book already disturbed - not attempting again.')
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
        pan = float(np.clip(math.atan2(base_pt[1], max(0.2, base_pt[0])),
                            -1.25, 1.25))
        horiz = max(0.2, math.hypot(base_pt[0], base_pt[1]))
        tilt = float(np.clip(math.atan2(base_pt[2] - cam_h, horiz), -1.0, 0.34))
        self.get_logger().info(
            f'Aiming the head at the book: pan {math.degrees(pan):+.1f} deg, '
            f'tilt {math.degrees(tilt):+.1f} deg.')
        self.move_head(pan, tilt)
        self.sleep_sim(self.DETECT_SETTLE_SEC)

        refined = self.redetect_book(colour, book_xyz)
        if refined is None:
            self.get_logger().warn(
                'No measurement good enough to aim the entry. Skipping this attempt '
                'rather than entering blind.')
            self.last_failure = 'measure'
            return False

        if self.last_measured is not None:
            moved = float(np.linalg.norm(refined - self.last_measured))
            if moved > self.DISTURBED_THRESHOLD:
                self.get_logger().error(
                    f'The book has shifted {moved * 100:.1f} cm since the last attempt '
                    f'(threshold {self.DISTURBED_THRESHOLD * 100:.0f} cm) - the previous '
                    'approach probably knocked it over. Stopping.')
                self.last_failure = 'disturbed'
                return False
            if moved > self.DISTURBED_LOG:
                self.get_logger().info(
                    f'Re-detect drifted {moved * 100:.1f} cm since the last attempt - '
                    'likely AMCL drift, not book motion. Continuing.')
        self.last_measured = np.array(refined, float)

        base_pt_far = self.map_to_base(book_xyz)
        base_pt_re = self.map_to_base(refined)
        if base_pt_far is not None and base_pt_re is not None:
            y_error = abs(base_pt_far[1] - base_pt_re[1])
            if y_error > self.LATERAL_CROSSCHECK_M:
                self.get_logger().warn(
                    f'Re-detect lateral Y disagrees with far detection by '
                    f'{y_error * 100:.1f} cm (limit '
                    f'{self.LATERAL_CROSSCHECK_M * 100:.1f}) - using the far detection for '
                    'lateral aim.')
                base_pt = np.array([base_pt_re[0], base_pt_far[1], base_pt_re[2]])
            else:
                base_pt = base_pt_re
        elif base_pt_re is not None:
            base_pt = base_pt_re
        elif base_pt_far is not None:
            base_pt = base_pt_far
        else:
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
                            base_pt[1] + self.LATERAL_AIM_BIAS, grasp_z])
        if abs(self.LATERAL_AIM_BIAS) > 1e-6:
            self.get_logger().info(
                f'Lateral aim trimmed by {self.LATERAL_AIM_BIAS * 100:+.1f} cm.')
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
            q_st, pe_st, re_st = self.kin.solve(staging_p, R, seed,
                                                bounds=safe_torso,
                                                restarts=self.IK_RESTARTS)
            if q_st is None or not self.kin.reached(pe_st, re_st):
                continue
            q, pe, re = self.kin.solve(pregrasp_p, R, q_st,
                                       bounds=safe_torso,
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
                                           bounds=torso_bounds,
                                           lock=[0], restarts=self.IK_RESTARTS)
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

        # --- SWING: tuck -> retract -> staging ---
        retract_x = float(base_pt[0]) - self.RETRACT_BEHIND_BOOK
        retract_z = float(base_pt[2]) - self.RETRACT_BELOW_BOOK
        retract_p = np.array([retract_x, -self.SHOULDER_OFFSET, retract_z])
        retract_bounds = (np.array([torso_now - 1e-4] + [-9.0] * 7),
                          np.array([torso_now + 1e-4] + [9.0] * 7))
        q_retract, pe_r, re_r = self.kin.solve(
            retract_p, R_grasp, self.current_q(),
            bounds=retract_bounds, lock=[0], restarts=self.IK_RESTARTS)
        if q_retract is not None and self.kin.reached(pe_r, re_r):
            self.get_logger().info(
                f'Swing split: tuck -> retract (x={retract_x:.2f}, z={retract_z:.2f}) -> '
                'staging.')
            self.book_contact = False
            self.move_arm([q_retract], seconds_each=self.arm_move_time(q_retract, 2.5))
            if self.book_contact:
                self.get_logger().warn(
                    'Contact during the TUCK -> RETRACT phase of the swing. Both '
                    'endpoints are behind the book face, so this should not be possible '
                    '- check whether the book was already leaning.')
                self.last_failure = 'bump'
                return False
            self.book_contact = False
            self.move_arm([q_stage], seconds_each=self.arm_move_time(q_stage, 3.0))
        else:
            self.get_logger().warn(
                f'Could not solve the retract waypoint (pe {pe_r * 1000:.0f} mm) - '
                'swinging directly to staging. The arc may bulge forward; if it does the '
                'book may be knocked.')
            self.book_contact = False
            self.move_arm([q_stage], seconds_each=self.arm_move_time(q_stage, 3.0))
        if self.book_contact:
            self.get_logger().warn(
                'Contact during the RETRACT -> STAGING swing, before the entry even '
                'began. The retract endpoint is behind the book face, so this means the '
                'arc bulges past it; the entry is not the problem.')
            self.last_failure = 'bump'
            return False

        approach, ok = self.kin.cartesian_path(q_stage, [pregrasp_p], R_grasp,
                                               lock=[0], bounds=torso_bounds,
                                               restarts=self.IK_RESTARTS)
        if not ok:
            self.get_logger().error('Could not solve staging -> pre-grasp.')
            return False
        self.book_contact = False
        self.move_arm(approach, seconds_each=max(
            2.0, self.arm_move_time(approach[-1], 2.0) / max(1, len(approach))))
        if self.book_contact:
            self.get_logger().warn(
                'Contact during the STAGING -> PRE-GRASP lead-in. Backing out.')
            self.last_failure = 'bump'
            return False

        hover_x = float(base_pt[0]) + self.TIP_BEHIND_FRAME - self.TIP_CLEARANCE
        hover_p = np.array([hover_x, grasp_p[1], grasp_p[2]])
        na = self.ENTRY_STEPS_APPROACH
        nc = self.ENTRY_STEPS_COMMIT
        waypoints = [pregrasp_p + (hover_p - pregrasp_p) * (i + 1) / na
                     for i in range(na)]
        waypoints += [hover_p + (grasp_p - hover_p) * (i + 1) / nc
                      for i in range(nc)]
        hover_idx = na - 1
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
        self.book_contact = False
        for i, q in enumerate(qs):
            speed = 2.5 if i >= len(qs) - 3 else 1.2
            last = (i == len(qs) - 1)
            self.move_arm([q], seconds_each=self.arm_move_time(q, speed),
                          precise=last)
            if i == hover_idx:
                self.sleep_sim(0.6)
                if self.book_contact and self.BUMP_GUARD_ENABLED:
                    self.get_logger().warn(
                        'Contact at the hover checkpoint, with the tips still clear of '
                        'the spine - so the aim is off laterally, not too deep. '
                        'Backing out before committing.')
                    self.last_failure = 'bump'
                    return False
                here = self.kin.fk(self.current_q())[:3, 3]
                lat = float(here[1] - hover_p[1])
                self.get_logger().info(
                    f'Hover checkpoint clear - tips 1 cm short of the spine. '
                    f'Lateral residual {lat * 100:+.1f} cm '
                    f'({"gripper left of target" if lat > 0 else "gripper right of target"}), '
                    f'depth {here[0] - hover_p[0]:+.3f} m. Committing.')
                if abs(lat) > 0.015:
                    self.get_logger().warn(
                        f'{abs(lat) * 100:.1f} cm of lateral residual against ~3 cm of '
                        'finger clearance - if a finger catches the spine, set '
                        f'LATERAL_AIM_BIAS to {-lat:+.3f}.')
                continue
            if i > hover_idx and i < len(qs) - 1 and self.BUMP_GUARD_ENABLED:
                if self.book_contact:
                    self.get_logger().warn(
                        f'Contact during the commit at waypoint '
                        f'{i - hover_idx}/{len(qs) - 1 - hover_idx} - aborting before '
                        'the gripper base reaches the spine.')
                    self.last_failure = 'bump'
                    return False

        joint_err = self.arm_error(qs[-1])
        actual = self.kin.fk(self.current_q())[:3, 3]
        offset = float(np.linalg.norm(actual - grasp_p))
        self.get_logger().info(
            f'Arm settled: gripper at ({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) '
            f'vs target ({grasp_p[0]:.3f}, {grasp_p[1]:.3f}, {grasp_p[2]:.3f}) - '
            f'{offset * 100:.1f} cm off, worst joint {joint_err:.3f} rad.')
        if offset > 0.020:
            self.get_logger().warn(
                f'The arm is {offset * 100:.1f} cm from where it was sent. The fingertips '
                'lead the pads by several centimetres, so they meet the book before the '
                'pads do - this is what knocks it over. Not closing.')
            self.last_failure = 'tracking'
            return False

        if self.book_contact and self.BUMP_GUARD_ENABLED:
            self.get_logger().warn(
                'Bump guard: contact fired during the final commit waypoint. The base of '
                'the hand may have met the spine. Backing out before closing.')
            self.last_failure = 'bump'
            return False

        self.book_contact = False
        finger, effort = self.close_gripper_and_wait()
        self.start_grip_hold(self.GRIP_CLOSED)
        self.get_logger().info(
            f'Closed on the spine: finger joint {finger:.4f}, effort '
            f'{("%.2f N" % effort) if effort is not None else "n/a"}, '
            f'contact={self.book_contact}.')

        self.sleep_sim(0.5)

        if finger > self.GRIP_STALL_MAX:
            self.get_logger().warn(
                f'Finger joint at {finger:.4f} - wider than a 2 cm spine. Miss.')
            self.stop_grip_hold()
            return False

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
        try:
            cur_q = self.current_q()
            cur_xyz = self.kin.fk(cur_q)[:3, 3]
            back_x = max(self.RECOVER_RETRACT_MIN_X,
                         float(cur_xyz[0]) - self.RECOVER_RETRACT_DISTANCE)
            back_p = np.array([back_x, float(cur_xyz[1]), float(cur_xyz[2])])
            R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
            safe_torso = (np.array([-0.001] + [-9.0] * 7),
                          np.array([self.TORSO_MAX_SAFE] + [9.0] * 7))
            q_back, pe, re = self.kin.solve(
                back_p, R, cur_q, bounds=safe_torso,
                lock=[0], restarts=self.IK_RESTARTS)
            if q_back is not None and self.kin.reached(pe, re):
                self.move_arm([q_back],
                              seconds_each=self.arm_move_time(q_back, 2.5))
                self.get_logger().info(
                    f'Arm retracted to x={back_x:.2f} before tucking.')
            else:
                self.get_logger().warn(
                    f'Could not solve the retract waypoint (pe {pe * 1000:.0f} mm) - '
                    'folding directly to tuck. This may drag the book.')
        except Exception as e:
            self.get_logger().warn(f'Retract before tuck failed: {e}')
        self.send_traj(self.arm_pub, ARM_JOINTS, RIGHT_TUCK, 4.0)
        self.sleep_sim(4.6)
        self.get_logger().info(
            'Arm returned to the tuck pose - clear of the head camera.')

    def retreat(self, book_xyz, normal, shelf_dir):
        seed = self.current_q()
        R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
        torso_now = float(self.joint_state.get('torso_lift_joint', seed[0]))
        carry_band = (np.array([torso_now - 1e-4] + [-9.0] * 7),
                      np.array([torso_now + 1e-4] + [9.0] * 7))
        for target in (np.array([0.34, -0.19, 1.05]),
                       np.array([0.30, -0.22, 1.00]),
                       np.array([0.38, -0.16, 1.10])):
            q, pe, re = self.kin.solve(target, R, seed, bounds=carry_band,
                                       lock=[0], restarts=self.IK_RESTARTS)
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
