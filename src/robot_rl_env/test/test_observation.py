"""The pure observation pipeline. No simulator, no ROS.

These tests are the reason ``observation.py`` has no ROS import. Every one of
the failures they cover -- a nan reaching the network, mean-pooling hiding a
thin obstacle, a bearing discontinuity at +/-pi -- is silent at runtime: no
exception, no log line, just a policy that is worse than it should be. Cheap
tests are the only thing that catches them.
"""

import math

import numpy as np
import pytest

from robot_rl_env import contract
from robot_rl_env.observation import (
    assemble_observation,
    from_body_frame,
    pool_ranges,
    to_body_frame,
    wrap_angle,
    yaw_from_quaternion,
)

FULL_RANGE = np.full(contract.N_RAW_BEAMS, contract.LIDAR_MAX, dtype=np.float32)
ZERO_ACTION = np.zeros(contract.ACT_DIM, dtype=np.float32)


def obs_at(robot_xy, robot_yaw, goal_xy, ranges=None, prev=None, step=0):
    return assemble_observation(
        FULL_RANGE if ranges is None else ranges,
        robot_xy,
        robot_yaw,
        goal_xy,
        ZERO_ACTION if prev is None else prev,
        step,
    )


# --- pooling ------------------------------------------------------------------

def test_pooling_takes_the_minimum_not_the_mean():
    """The single most consequential line in the module.

    One near beam inside a block of 18 must survive pooling. Under mean-pooling
    a 0.2 m chair leg among 17 open beams reports ~9.5 m and the policy drives
    into it.
    """
    ranges = FULL_RANGE.copy()
    ranges[5] = 0.2  # block 0 spans raw beams 0..17
    pooled = pool_ranges(ranges)
    assert pooled[0] == pytest.approx(0.2)
    assert pooled[1] == pytest.approx(contract.LIDAR_MAX)


def test_pooling_blocks_are_contiguous_and_ordered():
    ranges = np.arange(contract.N_RAW_BEAMS, dtype=np.float32) % contract.LIDAR_MAX
    pooled = pool_ranges(ranges)
    expected = ranges.reshape(contract.N_BEAMS, contract.BEAM_POOL).min(axis=1)
    assert np.array_equal(pooled, expected)


def test_nan_and_inf_are_mapped_before_clipping():
    ranges = FULL_RANGE.copy()
    ranges[0] = np.nan          # no return -> max range
    ranges[18] = np.inf         # no return -> max range
    ranges[36] = -np.inf        # sensor fault -> 0
    ranges[54] = 999.0          # beyond max -> clipped
    pooled = pool_ranges(ranges)
    assert np.isfinite(pooled).all()
    assert pooled[0] == pytest.approx(contract.LIDAR_MAX)
    assert pooled[1] == pytest.approx(contract.LIDAR_MAX)
    assert pooled[2] == pytest.approx(0.0)
    assert pooled[3] == pytest.approx(contract.LIDAR_MAX)


def test_wrong_beam_count_raises():
    with pytest.raises(ValueError, match="raw beams"):
        pool_ranges(np.zeros(180, dtype=np.float32))


# --- shape, dtype, bounds -----------------------------------------------------

def test_observation_shape_dtype_and_bounds():
    obs = obs_at((0.0, 0.0), 0.0, (3.0, 4.0))
    assert obs.shape == (contract.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_no_input_can_push_the_observation_out_of_bounds():
    """Fuzz the extremes; an out-of-range value silently violates the Box space."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        ranges = rng.choice(
            np.array([np.nan, np.inf, -np.inf, -5.0, 0.0, 3.0, 1e9], dtype=np.float32),
            size=contract.N_RAW_BEAMS,
        )
        obs = assemble_observation(
            ranges,
            (float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))),
            float(rng.uniform(-20, 20)),
            (float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))),
            rng.uniform(-3, 3, size=contract.ACT_DIM),
            int(rng.integers(0, 5000)),
        )
        assert np.isfinite(obs).all()
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_bad_prev_action_shape_raises():
    with pytest.raises(ValueError, match="prev_action"):
        assemble_observation(FULL_RANGE, (0, 0), 0.0, (1, 1), np.zeros(3), 0)


# --- goal encoding ------------------------------------------------------------

def test_distance_channel_maps_zero_to_minus_one_and_dmax_to_plus_one():
    i = contract.N_BEAMS
    assert obs_at((0.0, 0.0), 0.0, (0.0, 0.0))[i] == pytest.approx(-1.0)
    assert obs_at((0.0, 0.0), 0.0, (contract.D_MAX, 0.0))[i] == pytest.approx(1.0)
    # Saturates rather than exceeding the space.
    assert obs_at((0.0, 0.0), 0.0, (100.0, 0.0))[i] == pytest.approx(1.0)


def test_bearing_is_in_the_body_frame_not_the_world_frame():
    """A goal dead ahead reads the same regardless of which way that is."""
    sin_i, cos_i = contract.N_BEAMS + 1, contract.N_BEAMS + 2
    for yaw in (0.0, math.pi / 2, -2.0, 3.0):
        goal = (math.cos(yaw), math.sin(yaw))  # 1 m directly in front
        obs = obs_at((0.0, 0.0), yaw, goal)
        assert obs[sin_i] == pytest.approx(0.0, abs=1e-6)
        assert obs[cos_i] == pytest.approx(1.0, abs=1e-6)


def test_bearing_is_continuous_across_the_pi_wrap():
    """The reason for sin/cos rather than a raw angle.

    A goal just either side of directly-behind must produce nearly identical
    observations. A raw angle would jump the full width of the space.
    """
    sin_i, cos_i = contract.N_BEAMS + 1, contract.N_BEAMS + 2
    eps = 1e-3
    a = obs_at((0.0, 0.0), 0.0, (-1.0, eps))
    b = obs_at((0.0, 0.0), 0.0, (-1.0, -eps))
    assert abs(a[sin_i] - b[sin_i]) < 0.01
    assert abs(a[cos_i] - b[cos_i]) < 0.01
    assert a[cos_i] == pytest.approx(-1.0, abs=0.01)


def test_goal_to_the_left_has_positive_sine():
    sin_i = contract.N_BEAMS + 1
    assert obs_at((0.0, 0.0), 0.0, (0.0, 1.0))[sin_i] == pytest.approx(1.0)
    assert obs_at((0.0, 0.0), 0.0, (0.0, -1.0))[sin_i] == pytest.approx(-1.0)


# --- action and step channels -------------------------------------------------

def test_previous_action_is_passed_through_and_clipped():
    lo, hi = contract.N_BEAMS + 3, contract.N_BEAMS + 5
    obs = obs_at((0, 0), 0.0, (1, 1), prev=np.array([0.3, -0.7], dtype=np.float32))
    assert obs[lo:hi] == pytest.approx([0.3, -0.7])
    obs = obs_at((0, 0), 0.0, (1, 1), prev=np.array([9.0, -9.0], dtype=np.float32))
    assert obs[lo:hi] == pytest.approx([1.0, -1.0])


def test_step_channel_spans_the_episode():
    i = contract.OBS_DIM - 1
    assert obs_at((0, 0), 0.0, (1, 1), step=0)[i] == pytest.approx(-1.0)
    half = contract.MAX_EPISODE_STEPS // 2
    assert obs_at((0, 0), 0.0, (1, 1), step=half)[i] == pytest.approx(0.0)
    assert obs_at((0, 0), 0.0, (1, 1), step=contract.MAX_EPISODE_STEPS)[i] == pytest.approx(1.0)


def test_pooled_argument_bypasses_repooling():
    """The env pools once and reuses it; both paths must agree exactly."""
    ranges = FULL_RANGE.copy()
    ranges[100] = 0.4
    pooled = pool_ranges(ranges)
    a = assemble_observation(ranges, (0, 0), 0.0, (1, 1), ZERO_ACTION, 0)
    b = assemble_observation(None, (0, 0), 0.0, (1, 1), ZERO_ACTION, 0, pooled=pooled)
    assert np.array_equal(a, b)


# --- geometry helpers ---------------------------------------------------------

def test_wrap_angle_folds_into_pi_range():
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * math.pi + 0.5) == pytest.approx(0.5)
    assert wrap_angle(-2 * math.pi - 0.5) == pytest.approx(-0.5)
    # Antipodal inputs land on +/-pi; the sign is float-rounding dependent and
    # irrelevant, since only sin/cos of the result are ever used.
    assert abs(wrap_angle(3 * math.pi)) == pytest.approx(math.pi)
    assert abs(wrap_angle(-3 * math.pi)) == pytest.approx(math.pi)


def test_yaw_from_quaternion_round_trips():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        assert yaw_from_quaternion(*q) == pytest.approx(yaw)


def test_body_frame_helpers_are_inverses():
    """These carry the world->odom goal transform in env.reset(); if they are
    wrong the goal lands somewhere else entirely and nothing reports it."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        p = (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)))
        o = (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)))
        yaw = float(rng.uniform(-math.pi, math.pi))
        back = from_body_frame(to_body_frame(p, o, yaw), o, yaw)
        assert back == pytest.approx(p)


def test_to_body_frame_known_case():
    # Origin at (1, 0) facing +y; the world point (1, 2) is 2 m dead ahead.
    assert to_body_frame((1.0, 2.0), (1.0, 0.0), math.pi / 2) == pytest.approx((2.0, 0.0))
