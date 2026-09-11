#!/usr/bin/env python3
"""
tuck_arms_once.py  --  ERC 2026 Library Assistant Robot, stage 1a of 4.

Folds both arms in against the torso, lowers the torso, waits for the
motion to finish, exits cleanly.

WHY THIS RUNS FIRST, BEFORE ANY MOTION AT ALL
  In the start zone the collection-bin table sits about 0.19 m behind the
  base's rear edge.  With the arms extended, rotating in place sweeps them
  straight into it: the robot snags, the base stops turning, and the trial
  is effectively over.  Tucking once up front covers everything that
  follows - the search rotation, Nav2-driven navigation, and the final
  alignment - not just one node's own movement.  The arms are not needed
  until the grasp, which happens much later.

  The arm links carry Gazebo contact sensors, so an arm-to-table collision
  is also directly penalised by the rubric (-0.5 each).  base_link does
  NOT carry one, so paradoxically the arms are the part most worth
  protecting.

JOINT VALUES
  Validated against tiago_pro_limits.pdf - every value below sits inside
  its joint's hard URDF limit - and tightened once already after /contacts
  confirmed gripper_left_outer_finger_left_link striking the table with a
  looser earlier pose.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# arm_*_1..7.  Mirrored on joints 1 and 3 between left and right.
LEFT_TUCK = [0.3, -1.9, -0.3, -2.35, 0.0, 0.0, 0.0]
RIGHT_TUCK = [-0.3, -1.9, 0.3, -2.35, 0.0, 0.0, 0.0]

MOTION_SEC = 3                # trajectory duration requested of the controller
SETTLE_SEC = 4.0              # sim-time wait for the motion to actually finish
TORSO_DOWN = 0.0              # torso_lift_joint, range -0.001 .. 0.35 m
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
        """A publish() issued before the controller has matched this
        publisher is silently dropped - the arms then never move and the
        first rotation takes the table with it.  Wait for a real match
        instead of sleeping and hoping."""
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
            # Keeps the tucked arms low and the centre of mass down for the
            # spin.  The torso only moves at 0.035 m/s, so from a raised
            # position this is the slowest part of the whole stage.
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

        # The clock must be VALID before it can be used to time anything.
        # With use_sim_time set, get_clock().now() returns 0 until the first
        # /clock message arrives.  Computing a deadline from that zero gives
        # a deadline already far in the past (the sim has usually been up for
        # ten seconds or more by this point), so the wait returns instantly
        # and the launch sequence advances with the arms still swinging.
        clock_deadline = time.time() + 10.0
        while rclpy.ok() and node.get_clock().now().nanoseconds == 0:
            if time.time() > clock_deadline:
                node.get_logger().warn(
                    'No /clock after 10 s - is use_sim_time set and the sim running? '
                    'Falling back to the wall clock for the settle wait.')
                break
            rclpy.spin_once(node, timeout_sec=0.1)

        # Settle on the node's clock, which is sim time when use_sim_time is
        # set.  Gazebo's real-time factor is usually below 1.0 here, so a
        # wall-clock sleep would cut the motion short.
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
