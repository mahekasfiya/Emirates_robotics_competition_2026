#!/usr/bin/env python3
"""
approach_column.py  --  ERC 2026 Library Assistant Robot, stage 3 of 4.

Drives the robot to a pose squarely in front of the target shelf column and
finishes facing it dead-on.
"""

import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time as RclTime
from tf2_ros import Buffer, TransformListener

HANDOFF_FILE = '/tmp/erc_target_column.json'


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ApproachColumn(Node):

    # --- Stage B: precision alignment -----------------------------------
    XY_TOLERANCE = 0.025          # m
    YAW_TOLERANCE = 0.010         # rad (~0.6 deg)
    KP_LINEAR = 0.8
    KP_ANGULAR = 1.2
    MAX_LINEAR = 0.12             # m/s - deliberately gentle this close in
    MAX_ANGULAR = 0.45            # rad/s
    ALIGN_TIMEOUT_SEC = 25.0
    ALIGN_RATE = 20.0             # Hz

    MAX_ALIGN_DISTANCE = 0.60     # m
    SETTLE_SEC = 0.5

    NAV_SERVER_TIMEOUT = 15.0
    NAV_ACCEPT_TIMEOUT = 15.0
    NAV_RESULT_TIMEOUT = 180.0

    def __init__(self):
        super().__init__('approach_column')
        self.declare_parameter('standoff_override', 0.0)   # 0
        self.declare_parameter('skip_precision_align', False)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.skip_align = bool(self.get_parameter('skip_precision_align').value)
        self.standoff_override = float(self.get_parameter('standoff_override').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.exit_code = 0

    # ------------------------------------------------------------------
    def _await(self, future, timeout_sec):

        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def robot_pose(self, timeout_sec=1.0):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, RclTime(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
        except Exception:
            return None
        return (tf.transform.translation.x,
                tf.transform.translation.y,
                yaw_from_quat(tf.transform.rotation))

    def stop_base(self):
        stop = Twist()
        for _ in range(3):
            self.cmd_pub.publish(stop)
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def run(self):
        try:
            with open(HANDOFF_FILE) as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().error(
                f'Could not read {HANDOFF_FILE}: {e} - column_detector.py must succeed '
                'before this node runs. Not navigating.')
            self.exit_code = 1
            return

        column = data['column']
        marker = np.array(data['marker_xy'], dtype=float)
        normal = np.array(data['shelf_normal'], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        standoff = self.standoff_override if self.standoff_override > 0 else float(data['standoff'])
        frame = data.get('frame', self.map_frame)
        if frame != self.map_frame:
            self.get_logger().warn(
                f'Handoff is in frame "{frame}" but this node is configured for '
                f'"{self.map_frame}". Using "{frame}".')
            self.map_frame = frame

        goal_xy = marker + normal * standoff
        goal_yaw = math.atan2(-normal[1], -normal[0])

        self.get_logger().info(
            f'Column {column} marker at ({marker[0]:.2f}, {marker[1]:.2f}); standing off '
            f'{standoff:.2f} m along the shelf normal gives goal ({goal_xy[0]:.2f}, '
            f'{goal_xy[1]:.2f}) facing {math.degrees(goal_yaw):.1f} deg - perpendicular to '
            'the shelf face by construction.')

        if not self.nav.wait_for_server(timeout_sec=self.NAV_SERVER_TIMEOUT):
            self.get_logger().error(
                'Nav2 navigate_to_pose action server never appeared - is the full navigation '
                'stack up, not just localisation?')
            self.exit_code = 1
            return

        if not self.send_nav_goal(goal_xy, goal_yaw):

            self.exit_code = 1

        if self.skip_align:
            self.get_logger().info('Precision alignment disabled by parameter.')
        else:
            if self.precision_align(goal_xy, goal_yaw):
                self.exit_code = 0

        self.report(goal_xy, goal_yaw, column)

    # ------------------------------------------------------------------
    def send_nav_goal(self, goal_xy, goal_yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(goal_xy[0])
        pose.pose.position.y = float(goal_xy[1])
        pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        handle = self._await(self.nav.send_goal_async(goal), self.NAV_ACCEPT_TIMEOUT)
        if handle is None or not handle.accepted:
            self.get_logger().error('Nav2 rejected the goal (or timed out accepting it).')
            return False

        self.get_logger().info('Nav2 accepted the goal - driving to the column...')
        result = self._await(handle.get_result_async(), self.NAV_RESULT_TIMEOUT)
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 reports arrival. Refining the final pose...')
            return True

        status = result.status if result is not None else 'timed out'
        self.get_logger().error(
            f'Nav2 did not succeed (status={status}). Attempting precision alignment anyway '
            'in case the robot is already close.')
        return False

    def precision_align(self, goal_xy, goal_yaw):

        pose = self.robot_pose(timeout_sec=3.0)
        if pose is None:
            self.get_logger().error(
                'No map->base_link transform - cannot align. Is AMCL running and converged?')
            return False

        dist = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
        if dist > self.MAX_ALIGN_DISTANCE:
            self.get_logger().error(
                f'Still {dist:.2f} m from the goal, beyond the {self.MAX_ALIGN_DISTANCE:.2f} m '
                'alignment limit. Refusing to creep in open-loop without the costmap in the '
                'loop - fix the navigation failure instead.')
            return False

        period = 1.0 / self.ALIGN_RATE
        deadline = time.time() + self.ALIGN_TIMEOUT_SEC
        settled_since = None

        while time.time() < deadline:
            pose = self.robot_pose(timeout_sec=0.2)
            if pose is None:
                time.sleep(period)
                continue

            ex = goal_xy[0] - pose[0]
            ey = goal_xy[1] - pose[1]
            eyaw = normalize_angle(goal_yaw - pose[2])
            exy = math.hypot(ex, ey)

            if exy < self.XY_TOLERANCE and abs(eyaw) < self.YAW_TOLERANCE:
                self.stop_base()
                if settled_since is None:
                    settled_since = time.time()
                elif time.time() - settled_since >= self.SETTLE_SEC:
                    self.get_logger().info(
                        f'Aligned: {exy * 100:.1f} cm from the target point, '
                        f'{math.degrees(abs(eyaw)):.2f} deg off perpendicular.')
                    return True
                time.sleep(period)
                continue
            settled_since = None

            c, s = math.cos(pose[2]), math.sin(pose[2])
            bx = ex * c + ey * s           # forward, in the robot's own frame
            by = -ex * s + ey * c          # left
            twist = Twist()
            twist.linear.x = float(np.clip(self.KP_LINEAR * bx, -self.MAX_LINEAR, self.MAX_LINEAR))
            twist.linear.y = float(np.clip(self.KP_LINEAR * by, -self.MAX_LINEAR, self.MAX_LINEAR))
            twist.angular.z = float(np.clip(self.KP_ANGULAR * eyaw,
                                            -self.MAX_ANGULAR, self.MAX_ANGULAR))
            self.cmd_pub.publish(twist)
            time.sleep(period)

        self.stop_base()
        self.get_logger().warn(
            f'Alignment timed out after {self.ALIGN_TIMEOUT_SEC:.0f} s without reaching '
            'tolerance. If this recurs, AMCL is probably jittering - check /amcl_pose '
            'covariance while stationary.')
        return False

    def report(self, goal_xy, goal_yaw, column):
        self.stop_base()
        time.sleep(0.2)
        pose = self.robot_pose(timeout_sec=2.0)
        if pose is None:
            self.get_logger().warn('Final pose unavailable.')
            return
        exy = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
        eyaw = math.degrees(abs(normalize_angle(goal_yaw - pose[2])))
        self.get_logger().info(
            f'Final pose for column {column}: ({pose[0]:.3f}, {pose[1]:.3f}) at '
            f'{math.degrees(pose[2]):.2f} deg. Residual: {exy * 100:.1f} cm, {eyaw:.2f} deg. '
            'Robot is parked facing the column.')


def main(args=None):
    rclpy.init(args=args)
    node = ApproachColumn()

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
