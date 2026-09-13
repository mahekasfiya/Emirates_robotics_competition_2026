#!/usr/bin/env python3
"""
erc_arm_kinematics.py  --  forward and inverse kinematics for TIAGo Pro's
right arm, with no MoveIt dependency.

WHY NOT MOVEIT
  There is no move_group in the competition image's running graph, and
  standing up a MoveIt config is a multi-day job.  What the grasp and place
  stages actually need is narrower than motion planning: given a target
  pose for the gripper, find joint angles that reach it.  That is a
  well-conditioned 8-DoF problem (torso + 7 arm joints) and damped least
  squares solves it in milliseconds.

THE CHAIN IS READ FROM /robot_description AT RUNTIME
  Nothing here is hardcoded.  Joint origins, axes and limits all come from
  the URDF the simulation is actually running, so the module cannot drift
  out of sync with the robot.

MULTI-RESTART IS NOT OPTIONAL
  Single-seed damped least squares gets stuck in local minima on this arm,
  and the failure is silent and misleading: targets that are comfortably
  reachable report 20-60 cm of residual error and look like they are out of
  range.  Several of the "unreachable" conclusions during development were
  this, not geometry.  solve() therefore tries the seed first and then
  random restarts, and keeps the best result.

SOLUTIONS ARE SCORED ON JOINT TRAVEL, NOT JUST ERROR
  Reaching the same Cartesian pose is possible in very different arm
  postures - arm_right_1 ranges from -3.18 to +0.42 rad across the four
  shelf rows.  Jumping between two such solutions would sweep the elbow
  through whatever is in between, which next to a shelf means through the
  shelf.  So among solutions that meet tolerance, solve() prefers the one
  closest to the seed, and callers chain their seeds along a path.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def _tighter(limit, safety, key, soft_key, pick):
    """The stricter of a joint's URDF limit and its safety_controller soft
    limit, since the soft limit is the one ros2_control enforces."""
    hard = None
    if limit is not None and limit.get(key) is not None:
        hard = float(limit.get(key))
    soft = None
    if safety is not None and safety.get(soft_key) is not None:
        soft = float(safety.get(soft_key))
    if hard is None and soft is None:
        return 0.0
    if hard is None:
        return soft
    if soft is None:
        return hard
    return pick(hard, soft)


def rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def axis_angle_matrix(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def transform(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def rotation_from_columns(x_axis, y_axis, z_axis):
    """Build a rotation matrix from the three axis directions it should have.

    For the gripper's grasping frame the convention that matters is:
      x = approach direction (the way the fingers point)
      y = finger closing axis
      z = x cross y
    """
    return np.column_stack([np.asarray(x_axis, float),
                            np.asarray(y_axis, float),
                            np.asarray(z_axis, float)])


def rotation_error(R_target, R_current):
    dR = R_target @ R_current.T
    return np.array([dR[2, 1] - dR[1, 2],
                     dR[0, 2] - dR[2, 0],
                     dR[1, 0] - dR[0, 1]]) / 2.0


class ArmKinematics:

    def __init__(self, urdf_string,
                 tip='gripper_right_grasping_link',
                 root='base_footprint'):
        root_xml = ET.fromstring(urdf_string)
        self.joints = {}
        for j in root_xml.findall('joint'):
            child = j.find('parent'), j.find('child')
            if child[0] is None or child[1] is None:
                continue
            origin = j.find('origin')
            axis = j.find('axis')
            limit = j.find('limit')
            safety = j.find('safety_controller')
            self.joints[j.find('child').get('link')] = {
                'name': j.get('name'),
                'type': j.get('type'),
                'parent': j.find('parent').get('link'),
                'xyz': np.array([float(v) for v in (
                    origin.get('xyz', '0 0 0').split() if origin is not None
                    else ['0', '0', '0'])]),
                'rpy': [float(v) for v in (
                    origin.get('rpy', '0 0 0').split() if origin is not None
                    else ['0', '0', '0'])],
                'axis': [float(v) for v in axis.get('xyz').split()]
                        if axis is not None else [0.0, 0.0, 1.0],
                # Use the TIGHTER of the URDF limit and the safety_controller
                # soft limit. The soft limits are what ros2_control actually
                # enforces - roughly 0.07 rad inside the URDF values on this
                # arm - so a solution sitting exactly on a URDF limit is a
                # trajectory point the controller can refuse.
                'lower': _tighter(limit, safety, 'lower', 'soft_lower_limit', max),
                'upper': _tighter(limit, safety, 'upper', 'soft_upper_limit', min),
            }

        chain = []
        link = tip
        while link != root:
            if link not in self.joints:
                raise ValueError(f'Could not walk from {tip} back to {root}; '
                                 f'stuck at "{link}".')
            chain.append(link)
            link = self.joints[link]['parent']
        chain.reverse()
        self.chain = chain

        self.movable = [l for l in chain
                        if self.joints[l]['type'] in ('revolute', 'prismatic')]
        self.joint_names = [self.joints[l]['name'] for l in self.movable]
        self.lower = np.array([self.joints[l]['lower'] for l in self.movable])
        self.upper = np.array([self.joints[l]['upper'] for l in self.movable])
        self.n = len(self.movable)
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def fk(self, q, upto=None):
        M = np.eye(4)
        k = 0
        for link in self.chain:
            d = self.joints[link]
            M = M @ transform(rpy_to_matrix(*d['rpy']), d['xyz'])
            if d['type'] == 'revolute':
                M = M @ transform(axis_angle_matrix(d['axis'], q[k]), np.zeros(3))
                k += 1
            elif d['type'] == 'prismatic':
                M = M @ transform(np.eye(3), np.asarray(d['axis'], float) * q[k])
                k += 1
            if upto is not None and link == upto:
                return M
        return M

    def link_origins(self, q):
        """Origin of every link along the chain, for coarse clearance checks."""
        out = {}
        M = np.eye(4)
        k = 0
        for link in self.chain:
            d = self.joints[link]
            M = M @ transform(rpy_to_matrix(*d['rpy']), d['xyz'])
            if d['type'] == 'revolute':
                M = M @ transform(axis_angle_matrix(d['axis'], q[k]), np.zeros(3))
                k += 1
            elif d['type'] == 'prismatic':
                M = M @ transform(np.eye(3), np.asarray(d['axis'], float) * q[k])
                k += 1
            out[link] = M[:3, 3].copy()
        return out

    def jacobian(self, q, eps=1e-6):
        base = self.fk(q)
        J = np.zeros((6, self.n))
        for i in range(self.n):
            dq = np.array(q, dtype=float)
            dq[i] += eps
            pert = self.fk(dq)
            J[:3, i] = (pert[:3, 3] - base[:3, 3]) / eps
            J[3:, i] = rotation_error(pert[:3, :3], base[:3, :3]) / eps
        return J

    # ------------------------------------------------------------------
    def solve(self, target_p, target_R, seed,
              restarts=12, iters=180, rot_weight=0.35, damping=0.06,
              pos_tol=0.006, rot_tol=0.10,
              lock=None, bounds=None):
        """Damped least squares IK.

        lock    - indices held fixed at their seed value (e.g. the torso,
                  once it has been commanded, so a slow 0.035 m/s joint is
                  not asked to move again mid-trajectory).
        bounds  - optional (lo, hi) arrays narrowing the search, used to
                  keep the torso inside a small band around its nominal.
        Returns (q, position_error, rotation_error) - always a best effort,
        so callers must check the errors rather than assume success.
        """
        target_p = np.asarray(target_p, float)
        lo = self.lower.copy() if bounds is None else np.maximum(self.lower, bounds[0])
        hi = self.upper.copy() if bounds is None else np.minimum(self.upper, bounds[1])
        seed = np.clip(np.asarray(seed, float), lo, hi)
        lock = set(lock or [])

        best = (None, 1e9, 1e9, 1e9)
        for attempt in range(restarts):
            if attempt == 0:
                q = seed.copy()
            else:
                q = self._rng.uniform(lo, hi)
                for i in lock:
                    q[i] = seed[i]

            for _ in range(iters):
                M = self.fk(q)
                pos_err = target_p - M[:3, 3]
                rot_err = rotation_error(target_R, M[:3, :3])
                if np.linalg.norm(pos_err) < pos_tol * 0.3 and \
                        np.linalg.norm(rot_err) < rot_tol * 0.3:
                    break
                e = np.concatenate([pos_err, rot_weight * rot_err])
                J = self.jacobian(q)
                J[3:, :] *= rot_weight
                for i in lock:
                    J[:, i] = 0.0
                dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
                q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
                for i in lock:
                    q[i] = seed[i]

            M = self.fk(q)
            pe = float(np.linalg.norm(target_p - M[:3, 3]))
            re = float(np.linalg.norm(rotation_error(target_R, M[:3, :3])))
            travel = float(np.linalg.norm(q - seed))

            if pe < pos_tol and re < rot_tol:
                # Among solutions that reach, prefer the one that moves least:
                # a big posture change next to a shelf is a swept collision.
                if best[1] >= pos_tol or best[2] >= rot_tol or travel < best[3]:
                    best = (q.copy(), pe, re, travel)
                if travel < 1.0:
                    break
            elif best[0] is None or (pe + 0.05 * re) < (best[1] + 0.05 * best[2]):
                if best[1] >= pos_tol or best[2] >= rot_tol:
                    best = (q.copy(), pe, re, travel)

        return best[0], best[1], best[2]

    def reached(self, pos_err, rot_err, pos_tol=0.006, rot_tol=0.10):
        return pos_err < pos_tol and rot_err < rot_tol

    # ------------------------------------------------------------------
    def cartesian_path(self, start_q, waypoints, target_R, lock=None, bounds=None,
                       restarts=12):
        """IK a sequence of Cartesian points, seeding each solve from the
        previous solution.  This is what keeps a straight-line entry into
        the shelf actually straight in joint space instead of letting the
        arm take an arbitrary route between two valid endpoints.

        Returns (list_of_q, ok).  ok is False as soon as any waypoint fails,
        with the successful prefix still returned so the caller can decide.

        restarts matters more than it looks.  The default of 12 is not enough
        for this arm: a pre-grasp pose that solves to 1.7 mm with 30 restarts
        can report failure with 12, because damped least squares settles into
        a local minimum and the caller cannot tell that from a genuinely
        unreachable target.  Callers working near the shelf should pass their
        own restart count.
        """
        qs = []
        q = np.asarray(start_q, float).copy()
        for p in waypoints:
            q_new, pe, re = self.solve(p, target_R, q, lock=lock, bounds=bounds,
                                       restarts=restarts)
            if q_new is None or not self.reached(pe, re):
                return qs, False
            qs.append(q_new)
            q = q_new
        return qs, True

    def clearance_ok(self, q, plane_point, plane_normal, margin=0.02,
                     exempt_prefixes=('gripper_', 'arm_right_tool')):
        """Coarse guard: no arm link origin may cross the shelf face plane.

        Only the gripper and the wrist are allowed past it, since those are
        the parts that legitimately reach in to the book. This does not
        replace real collision checking - it catches the specific failure
        that matters here, an elbow swinging into the shelf.
        """
        origins = self.link_origins(q)
        n = np.asarray(plane_normal, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        for link, xyz in origins.items():
            if any(link.startswith(p) for p in exempt_prefixes):
                continue
            if float(np.dot(xyz - np.asarray(plane_point, float), n)) > -margin:
                return False, link
        return True, None
