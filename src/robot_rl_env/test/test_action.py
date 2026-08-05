"""The action -> Twist mapping. Pure; no simulator, no ROS.

These assertions lived in ``test_env.py`` until Phase 4, where they ran only
inside the container behind ``pytest.importorskip("rclpy")`` -- which is the
wrong place for arithmetic that both the training env and the deployment node
depend on. The mapping is now the one thing standing between a policy trained
on ``[0, 0.4] m/s`` and a robot driven at something else, so it is tested where
it will actually be run: on the host, in milliseconds, on every commit.
"""

import numpy as np
import pytest

from robot_rl_env import contract
from robot_rl_env.action import STOP, clip_action, scale_action


def test_action_scaling_matches_the_contract():
    assert scale_action([-1.0, 0.0]) == (0.0, 0.0)
    assert scale_action([1.0, 0.0]) == (contract.MAX_LINEAR_VEL, 0.0)
    assert scale_action([0.0, 0.0]) == (contract.MAX_LINEAR_VEL / 2, 0.0)
    assert scale_action([-1.0, 1.0]) == (0.0, contract.MAX_ANGULAR_VEL)
    assert scale_action([-1.0, -1.0]) == (0.0, -contract.MAX_ANGULAR_VEL)
    # Out-of-range input is clipped, never scaled past the envelope.
    assert scale_action([5.0, 5.0]) == (contract.MAX_LINEAR_VEL, contract.MAX_ANGULAR_VEL)


def test_no_action_can_command_reverse():
    """``a[0] = -1`` is a stop. See CONTRACTS.md ("Action space")."""
    for a0 in np.linspace(-3.0, 3.0, 101):
        v, _ = scale_action([a0, 0.0])
        assert 0.0 <= v <= contract.MAX_LINEAR_VEL


def test_stop_is_expressible_as_an_action():
    """The safety layer and the watchdog stop the robot by *proposing an
    action*, not by bypassing the scaling with a hand-built zero Twist. If STOP
    ever stopped scaling to (0, 0), every one of those paths would fail at once
    rather than one of them drifting alone."""
    assert scale_action(STOP) == (0.0, 0.0)


def test_the_scaling_returns_floats_not_numpy_scalars():
    """``geometry_msgs/Twist`` field assignment rejects a numpy float32 with an
    AssertionError from deep inside rclpy that names neither the field nor the
    caller."""
    v, omega = scale_action([0.3, -0.2])
    assert type(v) is float
    assert type(omega) is float


def test_a_single_batch_row_is_accepted_and_flattened():
    """``model.predict`` returns ``(1, 2)`` when handed a batched observation,
    and both the env and the deployment node route that straight here. Accepting
    it is deliberate; what must not happen is the array surviving into a Twist
    field, so the result is asserted flat."""
    assert clip_action([[0.25, -0.5]]).shape == (contract.ACT_DIM,)
    assert scale_action([[-1.0, 0.0]]) == (0.0, 0.0)


def test_a_wrong_sized_action_raises_rather_than_broadcasting():
    """The real hazard is an action of the wrong *length* -- a policy exported
    for a different action space, say. Broadcasting one would command a
    plausible velocity forever; reshape refuses instead."""
    for wrong in ([0.0, 0.0, 0.0], [0.0], [[0.0, 0.0], [0.0, 0.0]]):
        with pytest.raises(ValueError):
            clip_action(wrong)


def test_clip_action_is_float32_and_bounded():
    clipped = clip_action([2.0, -7.0])
    assert clipped.dtype == np.float32
    assert clipped.tolist() == [1.0, -1.0]
