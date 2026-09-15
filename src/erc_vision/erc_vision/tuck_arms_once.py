#!/usr/bin/env python3
"""
tuck_arms_once.py  --  ERC 2026 Library Assistant Robot, stage 1a of 4.

Folds both arms in against the torso, lowers the torso, waits for the
motion to finish.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# arm_*_1..7.  Mirrored on joints 1 and 3 between left and right.
LEFT_TUCK = [0.3, -1.9, -0.3, -2.35, 0.0, 0.0, 0.0]
RIGHT_TUCK = [-0.3, -1.9, 0.3, -2.35, 0.0, 0.0, 0.0]

MOTION_SEC = 3
SETTLE_SEC = 4.0
TORSO_DOWN = 0.0
PUBLISHER_MATCH_TIMEOUT = 10.0


class TuckArmsOnce(Node):

    def __init__(self):
        super().__init__('tuck_arms_once')
        self.declare_parameter('lower_torso', True)
        self.lower_torso = bool(self.get_parameter('lower_torso').value)

        self.arm_left = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.torso = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)

    @staticmethod
    def _traj(names, positions, seconds):
        traj = JointTrajectory()
        traj.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start.sec = int(seconds)
        traj.points.append(pt)
        return traj

    def wait_for_controllers(self):

        needed = [('arm_left', self.arm_left), ('arm_right', self.arm_right)]
        if self.lower_torso:
            needed.append(('torso', self.torso))
        deadline = time.time() + PUBLISHER_MATCH_TIMEOUT
        while time.time() < deadline:
            missing = [n for n, p in needed if p.get_subscription_count() == 0]
            if not missing:
                self.get_logger().info('Controllers connected.')
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        missing = [n for n, p in needed if p.get_subscription_count() == 0]
        self.get_logger().error(
            f'Controller(s) {missing} never subscribed after {PUBLISHER_MATCH_TIMEOUT:.0f} s. '
            'Sending the command anyway, but the arms may not move - check that the '
            'controller spawners in simulation.launch.py completed.')
        return False

    def tuck(self):
        self.arm_left.publish(self._traj(
            [f'arm_left_{i}_joint' for i in range(1, 8)], LEFT_TUCK, MOTION_SEC))
        self.arm_right.publish(self._traj(
            [f'arm_right_{i}_joint' for i in range(1, 8)], RIGHT_TUCK, MOTION_SEC))
        if self.lower_torso:

            self.torso.publish(self._traj(['torso_lift_joint'], [TORSO_DOWN], MOTION_SEC))
        self.get_logger().info(
            'Tucking both arms'
            + (' and lowering the torso' if self.lower_torso else '')
            + ' - this pose holds for the rest of the run, until the grasp stage.')


def main(args=None):
    rclpy.init(args=args)
    node = TuckArmsOnce()
    try:
        node.wait_for_controllers()
        node.tuck()


        clock_deadline = time.time() + 10.0
        while rclpy.ok() and node.get_clock().now().nanoseconds == 0:
            if time.time() > clock_deadline:
                node.get_logger().warn(
                    'No /clock after 10 s - is use_sim_time set and the sim running? '
                    'Falling back to the wall clock for the settle wait.')
                break
            rclpy.spin_once(node, timeout_sec=0.1)


        deadline = node.get_clock().now().nanoseconds / 1e9 + SETTLE_SEC
        while rclpy.ok() and node.get_clock().now().nanoseconds / 1e9 < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info('Arm tuck complete - the launch sequence can continue.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
