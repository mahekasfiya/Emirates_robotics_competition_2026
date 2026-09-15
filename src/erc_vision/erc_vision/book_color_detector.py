#!/usr/bin/env python3
"""
book_color_detector.py  --  ERC 2026 Library Assistant Robot, stage 4 of 5.

Runs once the robot is parked perpendicular to the target shelf column.
"""

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

COLUMN_FILE = '/tmp/erc_target_column.json'
BOOK_FILE = '/tmp/erc_target_book.json'

COLOUR_RANGES = {
    'red':    [((0, 110, 50), (9, 255, 255)), ((170, 110, 50), (179, 255, 255))],
    'green':  [((45, 90, 40), (88, 255, 255))],
    'blue':   [((100, 110, 40), (132, 255, 255))],
    'yellow': [((22, 110, 70), (36, 255, 255))],
}

DEFAULT_ROW_HEIGHTS = [1.595, 1.265, 0.935, 0.605]


class BookColourDetector(Node):

    # --- Blob filtering --------------------------------------------------
    MIN_CONTOUR_AREA = 300
    MAX_CONTOUR_AREA = 60000
    MIN_ASPECT = 0.05
    MAX_ASPECT = 1.40
    MIN_SOLIDITY = 0.65
    MORPH_KERNEL = 3

    # --- Sweep -----------------------------------------------------------
    DEFAULT_TILTS = [0.349, 0.10, -0.25, -0.55, -0.85]
    DWELL_SEC = 2.2
    HEAD_SETTLE_SEC = 0.9
    MAX_PASSES = 2

    # --- Confirmation ----------------------------------------------------
    CONFIRM_FRAMES = 3
    PIXEL_TOLERANCE = 30

    # --- 3D gating -------------------------------------------------------
    MIN_BOOK_DEPTH = 0.25
    MAX_BOOK_DEPTH = 1.60
    MAX_ROW_RESIDUAL = 0.13

    def __init__(self):
        super().__init__('book_color_detector')

        self.declare_parameter('book_colour', 'red')
        self.declare_parameter('erc_images_dir', '/erc_images')
        self.declare_parameter('purge_images_dir', False)
        self.declare_parameter('column_tolerance', 0.35)
        self.declare_parameter('row_heights', DEFAULT_ROW_HEIGHTS)
        self.declare_parameter('rows_numbered_top_down', True)
        self.declare_parameter('tilt_angles', self.DEFAULT_TILTS)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('save_debug_frames', False)

        self.colour = str(self.get_parameter('book_colour').value).strip().lower()
        if self.colour not in COLOUR_RANGES:
            raise ValueError(
                f'book_colour must be one of {sorted(COLOUR_RANGES)}, got "{self.colour}"')
        self.images_dir = str(self.get_parameter('erc_images_dir').value)
        self.column_tolerance = float(self.get_parameter('column_tolerance').value)
        self.row_heights = [float(z) for z in self.get_parameter('row_heights').value]
        self.top_down = bool(self.get_parameter('rows_numbered_top_down').value)
        self.tilts = [float(t) for t in self.get_parameter('tilt_angles').value]
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.debug = bool(self.get_parameter('save_debug_frames').value)

        os.makedirs(self.images_dir, exist_ok=True)
        if bool(self.get_parameter('purge_images_dir').value):
            for name in os.listdir(self.images_dir):
                path = os.path.join(self.images_dir, name)
                if os.path.isfile(path):
                    os.remove(path)

        self.target_s = None
        self.shelf_dir = None
        self.shelf_origin = None
        self.column = None
        self._load_column_handoff()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.should_exit = False
        self.exit_code = 0

        self.tilt_index = 0
        self.sweep_pass = 0
        self.tilt_started = 0.0
        self.history = []
        self.image_width = 640
        self._debug_counter = 0
        self._last_reject_reason = ''

        # Latest-frame cache
        self._latest_color = None
        self._latest_depth = None
        self._color_stamp = 0.0
        self._depth_stamp = 0.0

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.row_pub = self.create_publisher(
            Int32, '/erc/shelf_row_identification', latched)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, '/head_front_camera/head_front_camera/color/camera_info',
            self.camera_info_cb, 10)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self._color_cb, 5)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self._depth_cb, 5)

        self.goto_tilt(0)
        self.create_timer(0.2, self.tick)
        self.get_logger().info(
            f'Looking for the {self.colour} book in column {self.column}. Sweeping head tilt '
            f'{self.tilts} - the four rows span 0.99 m but the camera only frames ~0.68 m '
            'at this range.')

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
    def _load_column_handoff(self):
        try:
            with open(COLUMN_FILE) as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f'Could not read {COLUMN_FILE}: {e} - column_detector.py must run first.')

        self.column = data['column']
        marker = np.array(data['marker_xy'], dtype=float)
        normal = np.array(data['shelf_normal'], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        self.shelf_dir = np.array([-normal[1], normal[0]])
        self.shelf_origin = marker
        self.target_s = 0.0

    def along_shelf(self, xy):
        return float(np.dot(np.asarray(xy) - self.shelf_origin, self.shelf_dir))

    def camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def goto_tilt(self, index):
        self.tilt_index = index
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [0.0, float(self.tilts[index])]
        pt.time_from_start.sec = 1
        traj.points.append(pt)
        self.head_pub.publish(traj)
        self.tilt_started = time.time()
        self.history.clear()
        self.get_logger().info(
            f'Head tilt -> {self.tilts[index]:+.2f} rad '
            f'({index + 1}/{len(self.tilts)}, pass {self.sweep_pass + 1}/{self.MAX_PASSES}).')

    def tick(self):
        if self.should_exit:
            return
        if time.time() - self.tilt_started < self.DWELL_SEC:
            return
        nxt = self.tilt_index + 1
        if nxt < len(self.tilts):
            self.goto_tilt(nxt)
            return
        self.sweep_pass += 1
        if self.sweep_pass < self.MAX_PASSES:
            self.get_logger().warn(
                f'No {self.colour} book confirmed in this column on pass {self.sweep_pass} '
                '- sweeping again.')
            self.goto_tilt(0)
            return
        self.get_logger().error(
            f'Could not find a {self.colour} book in column {self.column} after '
            f'{self.MAX_PASSES} sweeps. Last rejection: {self._last_reject_reason or "none"}. '
            'Re-run with save_debug_frames:=true and check the HSV windows against a real '
            'frame before assuming the book is not there.')
        self.exit_code = 1
        self.should_exit = True

    # ------------------------------------------------------------------
    def image_cb(self, color_msg, depth_msg):
        if self.should_exit or self.camera_matrix is None:
            return
        if time.time() - self.tilt_started < self.HEAD_SETTLE_SEC:
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        self.image_width = color.shape[1]
        blobs = self.find_colour_blobs(color)
        self.save_debug_frame(color, blobs)
        if not blobs:
            self.history.clear()
            return

        best = self.pick_target_blob(depth, color_msg.header, blobs)
        if best is None:
            self.history.clear()
            return

        blob, position, row, residual = best
        self.history.append((blob['cx'], blob['cy']))
        if len(self.history) < self.CONFIRM_FRAMES:
            return
        recent = self.history[-self.CONFIRM_FRAMES:]
        if (max(x for x, _ in recent) - min(x for x, _ in recent) >= self.PIXEL_TOLERANCE or
                max(y for _, y in recent) - min(y for _, y in recent) >= self.PIXEL_TOLERANCE):
            return

        self.finalise(color, blob, position, row, residual)

    def find_colour_blobs(self, color):
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in COLOUR_RANGES[self.colour]:
            part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        kernel = np.ones((self.MORPH_KERNEL, self.MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0:
                continue
            aspect = w / float(h)
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue
            if area / float(w * h) < self.MIN_SOLIDITY:
                continue
            out.append({'contour': cnt, 'box': (x, y, w, h), 'area': area,
                        'cx': x + w // 2, 'cy': y + h // 2})
        return out

    def pick_target_blob(self, depth, header, blobs):
        scored = []
        rejects = []
        for blob in blobs:
            z = self.blob_depth(depth, blob)
            if z is None:
                rejects.append('no valid depth')
                continue
            if not self.MIN_BOOK_DEPTH <= z <= self.MAX_BOOK_DEPTH:
                rejects.append(f'depth {z:.2f} m out of range')
                continue
            pos = self.to_map(header, blob['cx'], blob['cy'], z)
            if pos is None:
                rejects.append('TF failed')
                continue
            offset = abs(self.along_shelf(pos[:2]) - self.target_s)
            if offset > self.column_tolerance:
                rejects.append(f'{offset:.2f} m off the column centre')
                continue
            row, residual = self.snap_row(pos[2])
            if residual > self.MAX_ROW_RESIDUAL:
                rejects.append(
                    f'height {pos[2]:.2f} m is {residual:.2f} m from any shelf row')
                continue
            scored.append((offset, blob, pos, row, residual))

        if not scored:
            if rejects:
                self._last_reject_reason = '; '.join(sorted(set(rejects)))
            return None
        scored.sort(key=lambda s: s[0])
        _, blob, pos, row, residual = scored[0]
        return blob, pos, row, residual

    def blob_depth(self, depth, blob):
        mask = np.zeros(depth.shape[:2], np.uint8)
        cv2.drawContours(mask, [blob['contour']], -1, 255, thickness=cv2.FILLED)
        vals = depth[mask > 0].astype(np.float32)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.median(vals)) if vals.size >= 10 else None

    def to_map(self, header, cx, cy, z):
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

    def snap_row(self, height):
        residuals = [abs(height - z) for z in self.row_heights]
        idx = int(np.argmin(residuals))
        row = idx + 1 if self.top_down else len(self.row_heights) - idx
        return row, residuals[idx]

    # ------------------------------------------------------------------
    def finalise(self, color, blob, position, row, residual):
        msg = Int32()
        msg.data = int(row)
        self.row_pub.publish(msg)
        self.get_logger().info(
            f'Found the {self.colour} book in column {self.column}, row {row}. '
            f'Measured height {position[2]:.3f} m vs shelf row height '
            f'{self.row_heights[row - 1 if self.top_down else len(self.row_heights) - row]:.3f} m '
            f'(residual {residual * 100:.1f} cm). Published on '
            '/erc/shelf_row_identification (latched).')

        self.save_competition_image(color, blob, row)

        payload = {
            'colour': self.colour,
            'column': self.column,
            'row': int(row),
            'frame': self.map_frame,
            'book_xyz': [float(v) for v in position],
            'row_residual': float(residual),
            'stamp': time.time(),
        }
        try:
            with open(BOOK_FILE, 'w') as f:
                json.dump(payload, f, indent=2)
            self.get_logger().info(f'Wrote {BOOK_FILE} for the grasp stage.')
        except OSError as e:
            self.get_logger().error(f'Could not write {BOOK_FILE}: {e}')
            self.exit_code = 1

        self.should_exit = True

    def save_competition_image(self, color, blob, row):
        img = color.copy()
        h, w = img.shape[:2]
        x, y, bw, bh = blob['box']
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        label = f'{self.colour} book - column {self.column}, row {row}'
        cv2.putText(img, label, (max(4, x), max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (max(4, x), max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, stamp, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, stamp, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        path = os.path.join(
            self.images_dir,
            f'book_{self.colour}_col{self.column}_row{row}_'
            f'{time.strftime("%Y%m%d_%H%M%S")}.png')
        if cv2.imwrite(path, img):
            self.get_logger().info(f'Saved competition image: {path}')
        else:
            self.get_logger().error(f'Failed to write {path} - check permissions.')
            self.exit_code = 1

    def save_debug_frame(self, color, blobs):
        if not self.debug:
            return
        self._debug_counter += 1
        if self._debug_counter % 4:
            return
        img = color.copy()
        for b in blobs:
            x, y, w, h = b['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, f'a={int(b["area"])}', (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img, f'{self.colour} tilt={self.tilts[self.tilt_index]:+.2f} '
                         f'blobs={len(blobs)}',
                    (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        out_dir = '/tmp/erc_debug_frames'
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f'book_{self._debug_counter}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])


def main(args=None):
    rclpy.init(args=args)
    node = BookColourDetector()
    code = 0
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.3)
        code = node.exit_code
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
