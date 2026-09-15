#!/usr/bin/env python3
"""
return_and_place.py  --  ERC 2026 Library Assistant Robot, stage 6 of 6.

Carries the book back to the Start/End Zone, finds the collection bin with
the camera, and lowers the book into it.

"""

import json
import math
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_vision.erc_arm_kinematics import ArmKinematics, rotation_from_columns

CARRY_FILE = '/tmp/erc_carry_state.json'
ARM_JOINTS = [f'arm_right_{i}_joint' for i in range(1, 8)]

BIN_RED = [((0, 110, 40), (10, 255, 255)), ((168, 110, 40), (179, 255, 255))]


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ReturnAndPlace(Node):

    SHOULDER_OFFSET = 0.159
    BOOK_HALF_HEIGHT = 0.125
    CARRY_YAWS_DEG = (30, -30, -60, -90, 60, 90, 0)
    CARRY_TARGETS = ((0.25, -0.25, 1.05), (0.35, -0.16, 0.95),
                     (0.30, -0.20, 1.00), (0.40, -0.12, 0.95))
    TABLE_TOP_Z = 0.73
    CARRY_TABLE_MARGIN = 0.08
    CARRY_LIFT_EXTRA = 0.02
    PLACE_CLEARANCE = 0.04
    BIN_DEPTH = 0.21
    APPROACH_ABOVE_RIM = 0.12
    MAX_BASE_Y = 0.15
    PLACE_REACH_LIMIT = 0.88

    XY_TOLERANCE = 0.025
    YAW_TOLERANCE = 0.012
    KP_LINEAR = 0.8
    KP_ANGULAR = 1.2
    MAX_LINEAR = 0.10
    MAX_ANGULAR = 0.40
    ALIGN_TIMEOUT = 30.0
    MAX_CREEP = 0.60

    GRIP_OPEN = 0.068
    MIN_BIN_AREA = 4000
    MIN_BIN_DEPTH = 0.55
    MAX_BIN_DEPTH = 1.60
    DETECT_TIMEOUT_SEC = 5.0
    MAX_ATTEMPTS = 3
    TORSO_MAX_SAFE = 0.28
    TORSO_SETTLE_SEC = 10.0
    TORSO_TOLERANCE = 0.006
    BIN_TRUST_RADIUS = 0.40
    BIN_RIM_MIN = 0.80
    BIN_RIM_MAX = 1.10
    PLACE_STANDOFF = 0.35
    AIM_INSETS = [0.0, 0.05, 0.10]

    CONTACT_FRESHNESS_SEC = 1.5
    FINGER_EMPTY_THRESHOLD = 0.00015

    def __init__(self):
        super().__init__('return_and_place')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 1.5708)
        self.declare_parameter('bin_fallback_rim_z', 0.95)

        self.declare_parameter('nominal_bin_range', 1.00)
        self.declare_parameter('max_attempts', self.MAX_ATTEMPTS)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.start_x = float(self.get_parameter('start_x').value)
        self.start_y = float(self.get_parameter('start_y').value)
        self.start_yaw = float(self.get_parameter('start_yaw').value)
        self.fallback_rim = float(self.get_parameter('bin_fallback_rim_z').value)
        self.nominal_bin_range = float(self.get_parameter('nominal_bin_range').value)
        self.max_attempts = int(self.get_parameter('max_attempts').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.bridge = CvBridge()
        self.kin = None
        self.camera_matrix = None
        self.joint_state = {}
        self.latest_frames = None
        self.bin_contact = False

        self.book_contact = False
        self.last_book_contact_time = 0.0
        self.exit_code = 0


        self._latest_color = None
        self._latest_depth = None
        self._color_stamp = 0.0
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
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, '/robot_description', self.urdf_cb, latched)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(CameraInfo,
                                 '/head_front_camera/head_front_camera/color/camera_info',
                                 self.info_cb, 10)
        self.create_subscription(Contacts, '/bin_contacts', self.bin_cb, 10)
        # Same topic the grasp stage uses to confirm a successful pick.
        self.create_subscription(Contacts, '/contacts', self.contacts_cb, 10)

        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self._color_cb, 5)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self._depth_cb, 5)

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

    def urdf_cb(self, msg):
        if self.kin is None:
            try:
                self.kin = ArmKinematics(msg.data)
            except Exception as e:
                self.get_logger().error(f'Could not parse the URDF: {e}')

    def joint_cb(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joint_state[n] = p

    def info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def bin_cb(self, msg):
        if msg.contacts:
            self.bin_contact = True

    def contacts_cb(self, msg):
        #Track gripper-to-book contact

        for c in msg.contacts:
            names = f'{c.collision1.name} {c.collision2.name}'.lower()
            if 'gripper_right' in names and 'book' in names:
                self.book_contact = True
                self.last_book_contact_time = time.time()
                return

    def image_cb(self, c, d):
        self.latest_frames = (c, d, time.time())

    # ------------------------------------------------------------------
    def sleep_sim(self, seconds):
        start = self.get_clock().now().nanoseconds / 1e9
        if start == 0:
            time.sleep(seconds)
            return
        while rclpy.ok():
            if self.get_clock().now().nanoseconds / 1e9 - start >= seconds:
                return
            time.sleep(0.05)

    def current_q(self):
        return np.array([self.joint_state.get(n, 0.0) for n in self.kin.joint_names])

    def robot_pose(self, timeout=1.0):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, RclTime(),
                timeout=rclpy.duration.Duration(seconds=timeout))
        except Exception:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y,
                yaw_from_quat(tf.transform.rotation))

    def stop_base(self):
        s = Twist()
        for _ in range(3):
            self.cmd_pub.publish(s)
            time.sleep(0.02)

    def send_traj(self, pub, names, points):
        traj = JointTrajectory()
        traj.joint_names = list(names)
        for pos, t in points:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in pos]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)
        pub.publish(traj)

    def move_arm(self, q_list, seconds_each=2.0, wait=True):
        pts, t = [], 0.0
        for q in q_list:
            t += seconds_each
            pts.append((q[1:], t))
        self.send_traj(self.arm_pub, ARM_JOINTS, pts)
        if wait:
            self.sleep_sim(t + 0.5)

    def sim_now(self):

        t = self.get_clock().now().nanoseconds / 1e9
        return t if t > 0 else time.time()

    def move_torso(self, value, wait=True):

        value = float(np.clip(value, -0.001, self.TORSO_MAX_SAFE))
        # Skip if already there.
        current = self.joint_state.get('torso_lift_joint')
        if current is not None and abs(current - value) < self.TORSO_TOLERANCE:
            self.get_logger().info(
                f'Torso already at {current:.3f} m (target {value:.3f}) - skipping.')
            return True
        self.send_traj(self.torso_pub, ['torso_lift_joint'],
                       [([value], max(2, int(self.TORSO_SETTLE_SEC)))])
        if not wait:
            return True
        start = self.sim_now()
        wall_cap = time.time() + (self.TORSO_SETTLE_SEC + 4.0) * 5
        last = None
        while time.time() < wall_cap:
            actual = self.joint_state.get('torso_lift_joint')
            if actual is not None and abs(actual - value) < self.TORSO_TOLERANCE:
                self.get_logger().info(f'Torso at {actual:.3f} m.')
                return True
            if self.sim_now() - start > self.TORSO_SETTLE_SEC + 2.0:
                if last is not None and actual is not None and abs(actual - last) < 0.001:
                    break
                last = actual
                start = self.sim_now()
            time.sleep(0.1)
        actual = self.joint_state.get('torso_lift_joint', float('nan'))
        self.get_logger().warn(
            f'Torso stopped at {actual:.3f} m, short of {value:.3f} m.')
        return False

    def move_head(self, pan, tilt, wait=True):
        self.send_traj(self.head_pub, ['head_1_joint', 'head_2_joint'],
                       [([pan, tilt], 1)])
        if wait:
            self.sleep_sim(1.0)

    def set_gripper(self, value, wait=True):
        value = float(np.clip(value, 0.0, 0.069))
        self.send_traj(self.grip_pub, ['gripper_right_finger_joint'], [([value], 2)])
        if wait:
            self.sleep_sim(1.6)

    # ------------------------------------------------------------------
    def navigate_home(self):
        if not self.nav.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Nav2 action server unavailable.')
            return False
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.start_x
        pose.pose.position.y = self.start_y
        pose.pose.orientation.z = math.sin(self.start_yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.start_yaw / 2.0)
        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(
            f'Returning to the Start/End Zone at ({self.start_x:.2f}, {self.start_y:.2f}) '
            f'facing {math.degrees(self.start_yaw):.0f} deg, book in hand.')

        fut = self.nav.send_goal_async(goal)
        handle = self._await(fut, 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().error('Nav2 rejected the return goal.')
            return False
        result = self._await(handle.get_result_async(), 180.0)
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Back in the start zone.')
            return True
        self.get_logger().warn(
            'Nav2 did not report success on the return leg; continuing so the place '
            'attempt can still run if we are close enough.')
        return False

    def _await(self, future, timeout):
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def drive_to(self, goal_xy, goal_yaw, label=''):
        pose = self.robot_pose(3.0)
        if pose is None:
            self.get_logger().error('No map->base_link transform.')
            return False
        if math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1]) > self.MAX_CREEP:
            self.get_logger().error(f'{label} target beyond the creep limit.')
            return False
        deadline = time.time() + self.ALIGN_TIMEOUT
        settled = None
        while time.time() < deadline:
            pose = self.robot_pose(0.2)
            if pose is None:
                time.sleep(0.05)
                continue
            ex, ey = goal_xy[0] - pose[0], goal_xy[1] - pose[1]
            eyaw = normalize_angle(goal_yaw - pose[2])
            if math.hypot(ex, ey) < self.XY_TOLERANCE and abs(eyaw) < self.YAW_TOLERANCE:
                self.stop_base()
                if settled is None:
                    settled = time.time()
                elif time.time() - settled > 0.4:
                    self.get_logger().info(f'{label} reached.')
                    return True
                time.sleep(0.05)
                continue
            settled = None
            c, s = math.cos(pose[2]), math.sin(pose[2])
            tw = Twist()
            tw.linear.x = float(np.clip(self.KP_LINEAR * (ex * c + ey * s),
                                        -self.MAX_LINEAR, self.MAX_LINEAR))
            tw.linear.y = float(np.clip(self.KP_LINEAR * (-ex * s + ey * c),
                                        -self.MAX_LINEAR, self.MAX_LINEAR))
            tw.angular.z = float(np.clip(self.KP_ANGULAR * eyaw,
                                         -self.MAX_ANGULAR, self.MAX_ANGULAR))
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.stop_base()
        self.get_logger().warn(f'{label} alignment timed out.')
        return False

    # ------------------------------------------------------------------
    def find_bin(self):
        deadline = time.time() + self.DETECT_TIMEOUT_SEC
        best = None
        while time.time() < deadline:
            if self.latest_frames is None:
                time.sleep(0.05)
                continue
            cmsg, dmsg, stamp = self.latest_frames
            if time.time() - stamp > 0.5:
                time.sleep(0.05)
                continue
            try:
                color = self.bridge.imgmsg_to_cv2(cmsg, 'bgr8')
                depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding='passthrough')
            except Exception:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = None
            for lo, hi in BIN_RED:
                part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            k = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                if cv2.contourArea(cnt) < self.MIN_BIN_AREA:
                    break
                m = np.zeros(depth.shape[:2], np.uint8)
                cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
                ys, xs = np.nonzero(m)
                vals = depth[m > 0].astype(np.float32)
                good = np.isfinite(vals) & (vals > 0)
                if good.sum() < 200:
                    continue
                med = float(np.median(vals[good]))
                if not self.MIN_BIN_DEPTH <= med <= self.MAX_BIN_DEPTH:
                    continue

                idx = np.nonzero(good)[0]
                if idx.size > 400:
                    idx = idx[np.linspace(0, idx.size - 1, 400).astype(int)]
                pts = []
                for i in idx:
                    p = self.pixel_to_map(cmsg.header, int(xs[i]), int(ys[i]),
                                          float(vals[i]))
                    if p is not None:
                        pts.append(p)
                if len(pts) < 50:
                    continue
                arr = np.array(pts)

                centre = np.median(arr, axis=0)
                rim = float(np.percentile(arr[:, 2], 95))
                if not self.BIN_RIM_MIN <= rim <= self.BIN_RIM_MAX:
                    self.get_logger().warn(
                        f'Discarding a red blob whose rim came out at {rim:.3f} m - '
                        f'outside the plausible {self.BIN_RIM_MIN:.2f}-{self.BIN_RIM_MAX:.2f} m '
                        'band for the collection bin.')
                    continue
                best = (centre, rim, med)
                break
            if best is not None:
                self.get_logger().info(
                    f'Collection bin located by vision at ({best[0][0]:.2f}, '
                    f'{best[0][1]:.2f}), rim height {best[1]:.3f} m, '
                    f'{best[2]:.2f} m from the camera.')
                return best[0], best[1]
            time.sleep(0.05)
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
                pt, self.map_frame, timeout=rclpy.duration.Duration(seconds=0.3))
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
        except Exception:
            return None
        return np.array([out.point.x, out.point.y, out.point.z])

    # ------------------------------------------------------------------
    def run(self):
        deadline = time.time() + 25.0
        while time.time() < deadline and (self.kin is None or self.camera_matrix is None
                                          or not self.joint_state):
            time.sleep(0.1)
        if self.kin is None or self.camera_matrix is None:
            self.get_logger().error('Kinematics or camera never became available.')
            self.exit_code = 1
            return

        holding = True
        try:
            with open(CARRY_FILE) as f:
                holding = bool(json.load(f).get('holding', True))
        except Exception:
            self.get_logger().warn(
                f'No {CARRY_FILE}; assuming a book is in hand and continuing.')
        if not holding:
            self.get_logger().error(
                'The grasp stage reported no book in hand. Returning to the start zone '
                'anyway, but skipping the place.')

        if holding:
            self.ensure_carry_height()

        self.navigate_home()
        self.drive_to((self.start_x, self.start_y), self.start_yaw, label='Start zone')
        if not holding:
            return

        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                self.get_logger().warn(f'--- Place attempt {attempt}/{self.max_attempts} ---')
            if not self.still_holding():
                finger = self.joint_state.get('gripper_right_finger_joint', 0.0)
                self.get_logger().error(
                    f'The gripper is empty (finger joint {finger:.5f}, last book contact '
                    f'{time.time() - self.last_book_contact_time:.1f} s ago) - the book '
                    'was lost somewhere on the return leg. Not attempting a place: '
                    '/bin_contacts fires on any contact with the bin, so going through '
                    'the motions would report a delivery that did not happen.')
                self.exit_code = 1
                return
            if self.attempt_place():
                self.get_logger().info('Book delivered to the collection bin.')
                self.park_arm()
                return
        self.get_logger().error(
            f'Could not place the book after {self.max_attempts} attempts.')
        self.exit_code = 1

    def ensure_carry_height(self):

        try:
            gripper_z = float(self.kin.fk(self.current_q())[2, 3])
        except Exception as e:
            self.get_logger().warn(f'Could not read the gripper height: {e}')
            return
        book_bottom = gripper_z - self.BOOK_HALF_HEIGHT
        needed_bottom = self.TABLE_TOP_Z + self.CARRY_TABLE_MARGIN

        if book_bottom >= needed_bottom:
            self.get_logger().info(
                f'Carry height fine: gripper at {gripper_z:.3f} m puts the book\'s lower '
                f'edge at {book_bottom:.3f} m, {(book_bottom - self.TABLE_TOP_Z) * 100:.0f} '
                'cm above the table. No lift needed.')
            return

        shortfall = needed_bottom - book_bottom + self.CARRY_LIFT_EXTRA
        torso_now = float(self.joint_state.get('torso_lift_joint', 0.0))
        target = float(np.clip(torso_now + shortfall, -0.001, 0.28))
        gained = target - torso_now
        self.get_logger().warn(
            f'Book\'s lower edge is at {book_bottom:.3f} m, below the '
            f'{needed_bottom:.3f} m needed to clear the table. Raising the torso '
            f'{shortfall * 100:.1f} cm (from {torso_now:.3f} to {target:.3f} m).')
        actual = self.move_torso(target)
        actual = float(self.joint_state.get('torso_lift_joint', target))
        gained = actual - torso_now
        new_bottom = book_bottom + gained

        if new_bottom < needed_bottom:
            self.get_logger().warn(
                f'Torso alone only reached a book bottom of {new_bottom:.3f} m - it had '
                f'{(actual - torso_now) * 100:.1f} cm of travel left. Folding the arm '
                'higher instead.')
            if self.fold_arm_high(actual):
                return
            self.get_logger().error(
                'Could not fold the arm higher either. The book is below the table top '
                'and the table edge will sweep it out of the gripper on the way in.')
        else:
            self.get_logger().info(
                f'Torso now {actual:.3f} m - book\'s lower edge at {new_bottom:.3f} m, '
                f'{(new_bottom - self.TABLE_TOP_Z) * 100:.0f} cm above the table.')

    @staticmethod
    def _carry_rotation(theta):

        return rotation_from_columns((math.cos(theta), math.sin(theta), 0.0),
                                     (math.sin(theta), -math.cos(theta), 0.0),
                                     (0.0, 0.0, -1.0))

    def fold_arm_high(self, torso):

        seed = self.current_q()
        band = (np.array([torso - 1e-4] + [-9.0] * 7),
                np.array([torso + 1e-4] + [9.0] * 7))
        best = None
        for yaw_deg in self.CARRY_YAWS_DEG:
            R = self._carry_rotation(math.radians(yaw_deg))
            for (x, y, z) in self.CARRY_TARGETS:
                if z - self.BOOK_HALF_HEIGHT < self.TABLE_TOP_Z + self.CARRY_TABLE_MARGIN:
                    continue
                q, pe, re = self.kin.solve(np.array([x, y, z]), R, seed,
                                           bounds=band, lock=[0], restarts=12)
                if q is None or not self.kin.reached(pe, re):
                    continue
                travel = float(np.linalg.norm(q[1:] - seed[1:]))
                if best is None or travel < best[0]:
                    best = (travel, q, yaw_deg, x, y, z)
                break
            if best is not None and best[0] < 4.0:
                break
        if best is None:
            return False
        travel, q, yaw_deg, x, y, z = best
        self.get_logger().info(
            f'Folding the arm to a carry pose: yaw {yaw_deg:+d} deg, gripper '
            f'({x:.2f}, {y:+.2f}, {z:.2f}), {travel:.2f} rad of travel. Book\'s lower '
            f'edge will sit at {z - self.BOOK_HALF_HEIGHT:.3f} m, '
            f'{(z - self.BOOK_HALF_HEIGHT - self.TABLE_TOP_Z) * 100:.0f} cm above the '
            'table.')
        self.move_arm([q], seconds_each=4.0)
        try:
            reached_z = float(self.kin.fk(self.current_q())[2, 3])
            self.get_logger().info(
                f'Carry pose reached: gripper at {reached_z:.3f} m, book bottom '
                f'{reached_z - self.BOOK_HALF_HEIGHT:.3f} m.')
        except Exception:
            pass
        return True

    def still_holding(self):

        now = time.time()
        if (self.book_contact
                and (now - self.last_book_contact_time) < self.CONTACT_FRESHNESS_SEC):
            return True

        finger = self.joint_state.get('gripper_right_finger_joint')
        if finger is None:
            return True
        if finger > self.FINGER_EMPTY_THRESHOLD:
            return True

        self.get_logger().warn(
            f'No gripper-to-book contact in the last {self.CONTACT_FRESHNESS_SEC:.1f} s '
            f'and finger joint at {finger:.5f} - treating as empty.')
        return False

    def attempt_place(self):
        self.move_head(0.0, -0.45)
        self.sleep_sim(0.5)
        found = self.find_bin()
        if found is None:
            self.get_logger().warn(
                'Bin not seen. Falling back to the nominal rim height and the bin pose '
                'implied by the start zone.')
            pose = self.robot_pose(2.0)
            if pose is None:
                return False
            bin_centre = np.array([pose[0] + math.cos(pose[2]) * self.nominal_bin_range,
                                   pose[1] + math.sin(pose[2]) * self.nominal_bin_range,
                                   self.fallback_rim])
            rim_z = self.fallback_rim
        else:
            bin_centre, rim_z = found

        pose = self.robot_pose(2.0)
        if pose is None:
            return False
        heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
        left = np.array([-heading[1], heading[0]])


        nominal = np.array([pose[0] + heading[0] * self.nominal_bin_range,
                            pose[1] + heading[1] * self.nominal_bin_range])
        drift = float(np.linalg.norm(bin_centre[:2] - nominal))
        if drift > self.BIN_TRUST_RADIUS:
            self.get_logger().warn(
                f'Vision puts the bin {drift:.2f} m from where it should be '
                f'({bin_centre[0]:.2f}, {bin_centre[1]:.2f}) vs '
                f'({nominal[0]:.2f}, {nominal[1]:.2f}). Beyond the '
                f'{self.BIN_TRUST_RADIUS:.2f} m trust radius, so using the nominal pose.')
            bin_centre = np.array([nominal[0], nominal[1], bin_centre[2]])


        target_xy = (bin_centre[:2] - heading * self.PLACE_STANDOFF
                     + left * self.SHOULDER_OFFSET)
        if target_xy[1] > self.MAX_BASE_Y:
            target_xy[1] = self.MAX_BASE_Y
        reach = float(np.dot(bin_centre[:2] - target_xy, heading))
        self.get_logger().info(
            f'Place pose ({target_xy[0]:.2f}, {target_xy[1]:.2f}) leaves the bin centre '
            f'{reach:.2f} m ahead of base_link'
            + (' - at the table-clearance limit.' if abs(target_xy[1] - self.MAX_BASE_Y) < 1e-6
               else '.'))
        self.drive_to(target_xy, pose[2], label='Place pose')

        base_bin = self.map_to_base(np.array([bin_centre[0], bin_centre[1], rim_z]))
        if base_bin is None:
            return False

        floor_z = rim_z - self.BIN_DEPTH
        hover_z = rim_z + self.APPROACH_ABOVE_RIM + self.BOOK_HALF_HEIGHT
        release_z = floor_z + self.PLACE_CLEARANCE + self.BOOK_HALF_HEIGHT
        base_floor = base_bin[2] - (rim_z - floor_z)

        hover_p = np.array([base_bin[0], base_bin[1],
                            base_bin[2] + self.APPROACH_ABOVE_RIM + self.BOOK_HALF_HEIGHT])
        release_p = np.array([base_bin[0], base_bin[1],
                              base_floor + self.PLACE_CLEARANCE + self.BOOK_HALF_HEIGHT])
        self.get_logger().info(
            f'Bin rim {rim_z:.3f} m, floor {floor_z:.3f} m. Hovering at gripper height '
            f'{hover_z:.3f} m, releasing at {release_z:.3f} m so the book\'s lower edge '
            f'sits {self.PLACE_CLEARANCE * 100:.0f} cm above the floor.')

        R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
        seed = self.current_q()
        safe_torso = (np.array([-0.001] + [-9.0] * 7),
                      np.array([self.TORSO_MAX_SAFE] + [9.0] * 7))

        q_hover = None
        for inset in self.AIM_INSETS:
            h = hover_p - np.array([inset, 0.0, 0.0])
            r = release_p - np.array([inset, 0.0, 0.0])
            q, pe, re = self.kin.solve(h, R, seed, bounds=safe_torso, restarts=30)
            if q is not None and self.kin.reached(pe, re):
                q_hover, hover_p, release_p = q, h, r
                if inset:
                    self.get_logger().info(
                        f'Hover solved with the aim point pulled {inset * 100:.0f} cm '
                        'toward the bin\'s near rim.')
                break
        if q_hover is None:
            self.get_logger().error(
                f'No IK for the hover pose above the bin at ({hover_p[0]:.2f}, '
                f'{hover_p[1]:.2f}, {hover_p[2]:.2f}) at any aim inset, with the torso '
                f'capped at {self.TORSO_MAX_SAFE:.2f} m. The base is too far back - check '
                'the bin position reported above.')
            return False

        torso_target = float(q_hover[0])
        self.move_torso(torso_target)
        torso_now = float(self.joint_state.get('torso_lift_joint', torso_target))
        if abs(torso_now - torso_target) > self.TORSO_TOLERANCE:
            locked = (np.array([torso_now - 1e-4] + [-9.0] * 7),
                      np.array([torso_now + 1e-4] + [9.0] * 7))
            q_fix, pe, re = self.kin.solve(hover_p, R, q_hover, bounds=locked,
                                           lock=[0], restarts=30)
            if q_fix is None or not self.kin.reached(pe, re):
                self.get_logger().error(
                    f'Torso stopped at {torso_now:.3f} m and the hover pose is no longer '
                    'reachable from there.')
                return False
            q_hover = q_fix
            self.get_logger().info(
                f'Re-solved the hover at torso {torso_now:.3f} m ({pe * 1000:.2f} mm).')
        self.move_arm([q_hover], seconds_each=3.0)

        steps = 4
        waypoints = [hover_p + (release_p - hover_p) * (i + 1) / steps
                     for i in range(steps)]
        qs, ok = self.kin.cartesian_path(q_hover, waypoints, R)
        if not ok:
            self.get_logger().warn(
                'Could not solve the full descent into the bin. Releasing from the hover '
                'height instead - this scores as a drop rather than a gentle place.')
            self.set_gripper(self.GRIP_OPEN)
            self.sleep_sim(1.0)
            return self.confirm_placed()
        self.move_arm(qs, seconds_each=1.2)

        self.bin_contact = False
        self.set_gripper(self.GRIP_OPEN)
        self.sleep_sim(1.0)

        placed = self.confirm_placed()

        qs_up, ok = self.kin.cartesian_path(qs[-1] if ok else q_hover,
                                            [hover_p], R)
        if ok:
            self.move_arm(qs_up, seconds_each=1.6)
        return placed

    def confirm_placed(self):
        if self.bin_contact:
            self.get_logger().info('/bin_contacts fired - the book is in the bin.')
            return True
        self.get_logger().warn(
            'No /bin_contacts event after releasing. The book may have missed, or the '
            'contact simply was not reported.')
        return False

    def park_arm(self):
        seed = self.current_q()
        R = rotation_from_columns((1, 0, 0), (0, -1, 0), (0, 0, -1))
        q, pe, re = self.kin.solve(np.array([0.30, -0.20, 1.00]), R, seed)
        if q is not None and self.kin.reached(pe, re):
            self.move_arm([q], seconds_each=3.0)
        self.send_traj(self.torso_pub, ['torso_lift_joint'], [([0.0], 10)])


def main(args=None):
    rclpy.init(args=args)
    node = ReturnAndPlace()
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
