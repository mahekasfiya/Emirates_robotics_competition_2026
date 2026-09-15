#!/usr/bin/env python3
"""
erc_arm_kinematics.py  --  forward and inverse kinematics for TIAGo Pro's
right arm
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

        origins = self.link_origins(q)
        n = np.asarray(plane_normal, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        for link, xyz in origins.items():
            if any(link.startswith(p) for p in exempt_prefixes):
                continue
            if float(np.dot(xyz - np.asarray(plane_point, float), n)) > -margin:
                return False, link
        return True, None
