#!/usr/bin/env python3
"""
set_initial_pose.py  --  ERC 2026 Library Assistant Robot, stage 1b of 4.

Replaces the manual "2D Pose Estimate"
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class SetInitialPose(Node):

    REPUBLISH_COUNT = 5
    REPUBLISH_PERIOD = 1.0
    CONVERGE_TIMEOUT_SEC = 20.0

    CONVERGED_POS_VAR = 0.5

    STAMP_BACKDATE_SEC = 0.3

    def __init__(self):
        super().__init__('set_initial_pose')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('xy_variance', 0.10)
        self.declare_parameter('yaw_variance', 0.05)
        self.declare_parameter('map_frame', 'map')

        self.x = float(self.get_parameter('initial_x').value)
        self.y = float(self.get_parameter('initial_y').value)
        self.yaw = float(self.get_parameter('initial_yaw').value)
        self.xy_var = float(self.get_parameter('xy_variance').value)
        self.yaw_var = float(self.get_parameter('yaw_variance').value)
        self.map_frame = str(self.get_parameter('map_frame').value)

        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        amcl_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.latest_pose = None
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self.amcl_cb, amcl_qos)

    def amcl_cb(self, msg):
        self.latest_pose = msg

    def build_msg(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        now = self.get_clock().now()
        backdate = rclpy.duration.Duration(seconds=self.STAMP_BACKDATE_SEC)
        msg.header.stamp = (now - backdate).to_msg() if now.nanoseconds > 0 else now.to_msg()
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        cov = [0.0] * 36
        cov[0] = self.xy_var        # x
        cov[7] = self.xy_var        # y
        cov[35] = self.yaw_var      # yaw
        msg.pose.covariance = cov
        return msg

    def run(self):
        self.get_logger().info(
            f'Seeding AMCL at ({self.x:.2f}, {self.y:.2f}, '
            f'{math.degrees(self.yaw):.1f} deg) in "{self.map_frame}".')


        clock_wait = 0.0
        while self.get_clock().now().nanoseconds == 0 and clock_wait < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            clock_wait += 0.1

        waited = 0.0
        while self.pub.get_subscription_count() == 0 and waited < 20.0:
            rclpy.spin_once(self, timeout_sec=0.2)
            waited += 0.2
        if self.pub.get_subscription_count() == 0:
            self.get_logger().error(
                'Nothing is subscribed to /initialpose after 20 s - AMCL is not running. '
                'Localisation will be wrong for the entire run.')
            return 1

        for i in range(self.REPUBLISH_COUNT):
            self.pub.publish(self.build_msg())
            deadline = time.time() + self.REPUBLISH_PERIOD
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.converged():
                break
            self.get_logger().info(
                f'Waiting for AMCL to settle ({i + 1}/{self.REPUBLISH_COUNT})...')

        if self.converged():
            p = self.latest_pose.pose.pose.position
            self.get_logger().info(
                f'AMCL converged at ({p.x:.2f}, {p.y:.2f}). Localisation ready.')
            return 0

        self.get_logger().warn(
            'AMCL has not reported a tight pose estimate. Continuing anyway - the rotation '
            'sweep in the next stage gives it more scan diversity to settle with - but if '
            'navigation then goes to the wrong place, this is the first thing to check '
            '(ros2 topic echo /amcl_pose).')
        return 0

    def converged(self):
        if self.latest_pose is None:
            return False
        cov = self.latest_pose.pose.covariance
        return (cov[0] + cov[7]) < self.CONVERGED_POS_VAR


def main(args=None):
    rclpy.init(args=args)
    node = SetInitialPose()
    code = 0
    try:
        code = node.run()
    finally:
        time.sleep(0.3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
