"""Observation assembly. SHARED between training and deployment.

This module is **pure**: numpy in, numpy out, no ROS import, no state, no
threads. The ROS subscriber that feeds it lives in ``observation_node.py``.

The split is deliberate and is a departure from the original scaffold, which
put both in this file. Purity here is what makes the contract enforceable: the
nan-handling, the min-pooling, and the bearing wrap are all testable without a
simulator, in milliseconds, and ``policy_node.py`` (Phase 4) can import this
module on a robot that has no Gazebo at all. Importing ``observation`` must
never transitively import ``rclpy``.

``assemble_observation`` is the single implementation required by
``CONTRACTS.md``. Training and inference both call it. Neither reimplements any
part of it.
"""

from __future__ import annotations

import math

import numpy as np

from robot_rl_env import contract


def wrap_angle(a: float) -> float:
    """Wrap to ``[-pi, pi]``.

    Exactly-antipodal inputs may come back as either ``+pi`` or ``-pi``
    depending on float rounding. That is harmless here because the only
    consumers are ``sin`` and ``cos``, which agree on both.
    """
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw (Z rotation) of a quaternion.

    The robot is planar, so roll and pitch are noise and are discarded rather
    than fed through a full Euler conversion whose gimbal behaviour would only
    ever be a liability here.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def to_body_frame(
    point: tuple[float, float], origin_xy: tuple[float, float], origin_yaw: float
) -> tuple[float, float]:
    """Express a point given in some frame F in a child frame at ``origin``."""
    dx = point[0] - origin_xy[0]
    dy = point[1] - origin_xy[1]
    c, s = math.cos(-origin_yaw), math.sin(-origin_yaw)
    return c * dx - s * dy, s * dx + c * dy


def from_body_frame(
    point: tuple[float, float], origin_xy: tuple[float, float], origin_yaw: float
) -> tuple[float, float]:
    """Inverse of :func:`to_body_frame`."""
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return origin_xy[0] + c * point[0] - s * point[1], origin_xy[1] + s * point[0] + c * point[1]


def pool_ranges(ranges) -> np.ndarray:
    """Reduce a raw 360-beam scan to ``contract.N_BEAMS`` by **min**-pooling.

    Mean-pooling averages a chair leg between two open beams into free space and
    the policy drives into it. Minimum is the only correct reduction for a
    collision-avoidance observation. See CONTRACTS.md.

    Returns metres, clipped to ``[0, LIDAR_MAX]`` -- not normalized. The env
    needs the metric value for the collision test and the info dict, and it must
    be the *same* array the observation was built from rather than a second,
    subtly different reduction.
    """
    r = np.asarray(ranges, dtype=np.float32)
    if r.shape != (contract.N_RAW_BEAMS,):
        raise ValueError(
            f"expected {contract.N_RAW_BEAMS} raw beams, got {r.shape}. The "
            f"LiDAR <samples> in models/diffbot disagrees with contract."
        )
    # Order matters: nan/inf must die before the clip, or they propagate into
    # the network and produce nan actions with no error message anywhere.
    r = np.nan_to_num(r, nan=contract.LIDAR_MAX, posinf=contract.LIDAR_MAX, neginf=0.0)
    r = np.clip(r, 0.0, contract.LIDAR_MAX)
    return r.reshape(contract.N_BEAMS, contract.BEAM_POOL).min(axis=1)


def assemble_observation(
    ranges,
    robot_xy: tuple[float, float],
    robot_yaw: float,
    goal_xy: tuple[float, float],
    prev_action,
    step_count: int,
    *,
    pooled: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 26-vector defined in CONTRACTS.md.

    ``pooled`` lets a caller that already ran :func:`pool_ranges` (the env, for
    its collision test) pass the result in rather than pooling twice. When it is
    given, ``ranges`` is ignored.

    All poses are in one consistent frame -- ``/odom`` during training and
    deployment alike. The goal must already be expressed in that same frame.
    """
    if pooled is None:
        pooled = pool_ranges(ranges)

    obs = np.empty(contract.OBS_DIM, dtype=np.float32)

    # 0..19 -- pooled beams, [0, LIDAR_MAX] -> [-1, 1]
    obs[0 : contract.N_BEAMS] = 2.0 * (pooled / contract.LIDAR_MAX) - 1.0

    dx = goal_xy[0] - robot_xy[0]
    dy = goal_xy[1] - robot_xy[1]
    distance = math.hypot(dx, dy)

    # 20 -- goal distance, clipped at the arena diagonal
    obs[contract.N_BEAMS] = 2.0 * min(distance / contract.D_MAX, 1.0) - 1.0

    # 21, 22 -- bearing to goal in the robot body frame, as sin/cos so the
    # representation stays continuous across the +/-pi wrap.
    bearing = wrap_angle(math.atan2(dy, dx) - robot_yaw)
    obs[contract.N_BEAMS + 1] = math.sin(bearing)
    obs[contract.N_BEAMS + 2] = math.cos(bearing)

    # 23, 24 -- previous action, already in [-1, 1]
    prev = np.clip(np.asarray(prev_action, dtype=np.float32), -1.0, 1.0)
    if prev.shape != (contract.ACT_DIM,):
        raise ValueError(f"prev_action must have shape ({contract.ACT_DIM},), got {prev.shape}")
    obs[contract.N_BEAMS + 3 : contract.N_BEAMS + 5] = prev

    # 25 -- how much of the episode budget is spent. Without it the task is
    # not Markovian: the same state truncates or does not depending on
    # history the policy cannot see.
    fraction = min(step_count / contract.MAX_EPISODE_STEPS, 1.0)
    obs[contract.OBS_DIM - 1] = 2.0 * fraction - 1.0

    return obs
