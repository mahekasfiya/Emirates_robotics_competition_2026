#!/usr/bin/env python3
"""
set_initial_pose.py  --  ERC 2026 Library Assistant Robot, stage 1b of 4.

Replaces the manual "2D Pose Estimate" click in RViz, and replaces the
earlier global-localization approach.

WHY NOT GLOBAL LOCALIZATION
  /reinitialize_global_localization scatters particles over the whole map
  and relies on motion to disambiguate.  Two things make that a bad bet
  here: the arena is a near-symmetric 10 x 10 m box, and the robot's very
  next action is a rotation IN PLACE - which gives AMCL plenty of angular
  information but almost no translation, the thing it actually needs to
  resolve position.  Meanwhile the spawn pose is not unknown at all:
  simulation.launch.py spawns TIAGo at a hardcoded pose.  Seeding AMCL
  with a pose we already know turns a gamble into a certainty.

FRAME NOTE - READ THIS BEFORE CHANGING THE DEFAULTS
  The defaults below are (0, 0, 0), which is correct for a map built by
  running SLAM from the robot's spawn pose: that map's origin IS the spawn
  pose, so the robot starts at the map origin regardless of where it sits
  in Gazebo's world frame.  If you later switch to a map generated in the
  Gazebo world frame, change these to the world spawn pose instead
  (x=0, y=0, yaw=1.5708 for the stock simulation.launch.py).

The covariance is deliberately non-zero: we know the spawn pose exactly,
but the map itself has some distortion, so leaving AMCL a little room to
settle onto the real scan match converges better than pinning it hard.
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
    # Trace of the position covariance below which we call AMCL converged.
    # This must sit ABOVE the seed covariance we publish (2 * xy_variance),
    # or the check can never pass: AMCL only shrinks covariance once the
    # robot has moved past update_min_d / update_min_a, which has not
    # happened yet when this node runs.  What we are really checking here
    # is that AMCL accepted the pose and is publishing at all.
    CONVERGED_POS_VAR = 0.5
    # AMCL looks up base_footprint->odom AT the message stamp.  Stamping
    # with now() races the transform broadcaster and loses - every seed
    # attempt logged "Failed to transform initial pose in time". Back-dating
    # slightly asks for a transform that has already been published.
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

        # AMCL subscribes to /initialpose with default (volatile) QoS, so a
        # latched publisher would not help a subscriber that appears later -
        # we republish a few times instead, below.
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

        # Wait for AMCL to be listening, otherwise the first message lands
        # nowhere and AMCL sits on its default pose for the whole run.
        # Wait for a valid sim clock first, so the back-dated stamp above is
        # computed against sim time rather than against zero.
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
