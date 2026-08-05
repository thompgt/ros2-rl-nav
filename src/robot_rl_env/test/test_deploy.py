"""The deployment controller. Pure; no ROS, no torch, no simulator.

The property under test throughout is that this agrees with ``env.step``. The
Phase 4 measurement subtracts a free-running success rate from a
step-synchronized one and attributes the difference to timing; every rule that
differs between the two loops for any *other* reason contaminates that number,
so the rules are pinned here one at a time.
"""

import math

import numpy as np
import pytest

from robot_rl_env import contract
from robot_rl_env.action import STOP
from robot_rl_env.deploy import DeploymentController, Outcome

CLEAR = np.full(contract.N_BEAMS, contract.LIDAR_MAX, dtype=np.float32)
"""A scan with nothing in it."""

GOAL = (5.0, 0.0)
ORIGIN = (0.0, 0.0)


def scan(min_range: float) -> np.ndarray:
    """A clear scan with one beam at ``min_range``."""
    pooled = CLEAR.copy()
    pooled[7] = min_range
    return pooled


def controller(action=(1.0, 0.0), **kwargs) -> DeploymentController:
    """A controller whose policy always proposes ``action``.

    A constant policy rather than a trained one: everything being tested is the
    logic *around* the policy, and a policy that reacts to its input would make
    every assertion below depend on a network's weights.
    """
    return DeploymentController(lambda obs: np.asarray(action, dtype=np.float32), **kwargs)


def tick(ctrl, *, pooled=None, xy=ORIGIN, yaw=0.0, age=0.0):
    return ctrl.tick(
        pooled=CLEAR if pooled is None else pooled, robot_xy=xy, robot_yaw=yaw, age=age
    )


# --- the ordinary case --------------------------------------------------------

def test_a_running_episode_drives_the_policy_action():
    ctrl = controller(action=(1.0, 0.0))
    ctrl.set_goal(GOAL)
    command = tick(ctrl)
    assert command.reason == "policy"
    assert command.outcome is Outcome.RUNNING
    assert command.linear == pytest.approx(contract.MAX_LINEAR_VEL)
    assert command.angular == pytest.approx(0.0)


def test_the_policy_sees_the_contract_observation():
    """Not a reimplementation of the assembler -- that is tested in
    test_observation.py. What is asserted here is that the controller feeds it
    the right *arguments*: the pooled scan it was handed, the goal in the odom
    frame, and its own step counter."""
    seen = {}

    def policy(obs):
        seen["obs"] = obs.copy()
        return np.array([0.0, 0.0], dtype=np.float32)

    ctrl = DeploymentController(policy)
    ctrl.set_goal(GOAL)
    tick(ctrl)

    obs = seen["obs"]
    assert obs.shape == (contract.OBS_DIM,)
    assert obs.dtype == np.float32
    # A clear scan is LIDAR_MAX everywhere, which normalizes to +1.
    assert obs[: contract.N_BEAMS] == pytest.approx(np.ones(contract.N_BEAMS))
    # Goal 5 m dead ahead: bearing zero, so sin=0 and cos=1.
    assert obs[contract.N_BEAMS + 1] == pytest.approx(0.0, abs=1e-6)
    assert obs[contract.N_BEAMS + 2] == pytest.approx(1.0, abs=1e-6)
    # First tick of an episode: no previous action, no elapsed budget.
    assert obs[contract.N_BEAMS + 3 : contract.N_BEAMS + 5] == pytest.approx(STOP)
    assert obs[-1] == pytest.approx(-1.0)


def test_the_previous_action_carries_across_ticks():
    seen = []
    ctrl = DeploymentController(
        lambda obs: seen.append(obs.copy()) or np.array([0.5, -0.25], dtype=np.float32)
    )
    ctrl.set_goal(GOAL)
    tick(ctrl)
    tick(ctrl)
    assert seen[1][contract.N_BEAMS + 3 : contract.N_BEAMS + 5] == pytest.approx([0.5, -0.25])


# --- the watchdog -------------------------------------------------------------

def test_no_data_at_all_stops_the_robot():
    ctrl = controller()
    ctrl.set_goal(GOAL)
    command = ctrl.tick(pooled=None, robot_xy=None, robot_yaw=None, age=0.0)
    assert command.reason == "watchdog"
    assert (command.linear, command.angular) == (0.0, 0.0)


def test_a_stale_sample_stops_the_robot():
    ctrl = controller()
    ctrl.set_goal(GOAL)
    fresh = tick(ctrl, age=contract.WATCHDOG_TIMEOUT)
    assert fresh.reason == "policy"
    stale = tick(ctrl, age=contract.WATCHDOG_TIMEOUT + 1e-6)
    assert stale.reason == "watchdog"
    assert (stale.linear, stale.angular) == (0.0, 0.0)


def test_a_watchdog_tick_does_not_spend_the_step_budget():
    """Index 25 is the fraction of the 500-step budget spent, and the policy
    learned it against a counter that advanced once per action issued. Counting
    stopped ticks would run the deployment clock fast exactly when the system
    was under load -- a gap manufactured by the instrument measuring it."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    tick(ctrl)
    assert ctrl.step == 1
    tick(ctrl, age=10.0)
    assert ctrl.step == 1


def test_a_stopped_tick_is_recorded_as_the_previous_action():
    """The wheels were told to stop, so that is what index 23/24 must say. The
    alternative leaves the last policy action in place and tells the next
    observation the robot is still driving at a speed it was ordered to
    abandon."""
    seen = []
    ctrl = DeploymentController(
        lambda obs: seen.append(obs.copy()) or np.array([1.0, 1.0], dtype=np.float32)
    )
    ctrl.set_goal(GOAL)
    tick(ctrl)
    tick(ctrl, age=10.0)  # watchdog: stops, issues no action
    tick(ctrl)
    assert seen[-1][contract.N_BEAMS + 3 : contract.N_BEAMS + 5] == pytest.approx(STOP)


# --- the safety layer ---------------------------------------------------------

def test_the_safety_gate_fires_below_its_threshold():
    ctrl = controller()
    ctrl.set_goal(GOAL)
    command = tick(ctrl, pooled=scan(contract.SAFETY_STOP_THRESHOLD - 0.01))
    assert command.reason == "safety"
    assert (command.linear, command.angular) == (0.0, 0.0)


def test_a_safety_trip_is_still_recorded_as_a_collision():
    """0.15 m is inside 0.18 m, so the episode *is* a collision however the
    wheels came to stop. If the gate swallowed the outcome, a deployment run
    would report collisions only in the 30 mm band between the thresholds and
    the measured collision rate would be nonsense."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    command = tick(ctrl, pooled=scan(contract.SAFETY_STOP_THRESHOLD - 0.01))
    assert command.outcome is Outcome.COLLISION
    assert ctrl.outcome is Outcome.COLLISION


def test_the_safety_gate_does_not_depend_on_the_episode_logic():
    """It has to stop the robot when the code around it is wrong, so it must
    not sit downstream of that code. With no goal set there is no episode at
    all, and the gate still fires."""
    ctrl = controller()
    command = tick(ctrl, pooled=scan(0.05))
    assert command.reason == "safety"
    assert ctrl.outcome is Outcome.IDLE


def test_clearance_between_the_thresholds_terminates_without_the_gate():
    """The band where the policy has failed but the safety layer has not had
    to act. Named separately so a future change to either threshold shows up
    here as a reason change rather than as a silent shift in what is
    reported."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    between = (contract.SAFETY_STOP_THRESHOLD + contract.COLLISION_THRESHOLD) / 2
    command = tick(ctrl, pooled=scan(between))
    assert command.reason == "collision"
    assert command.outcome is Outcome.COLLISION


# --- episode outcomes, matching env.step --------------------------------------

def test_reaching_the_goal_terminates_as_a_success():
    ctrl = controller()
    ctrl.set_goal(GOAL)
    near = (GOAL[0] - contract.GOAL_TOLERANCE / 2, GOAL[1])
    command = tick(ctrl, xy=near)
    assert command.outcome is Outcome.SUCCESS
    assert command.reason == "success"
    assert (command.linear, command.angular) == (0.0, 0.0)


def test_the_goal_tolerance_is_exclusive_exactly_as_in_the_env():
    """``env.step`` uses ``distance < GOAL_TOLERANCE``. A controller using
    ``<=`` would score a few extra successes per run, which is well inside the
    difference the Phase 4 measurement is trying to report."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    on_the_line = (GOAL[0] - contract.GOAL_TOLERANCE, GOAL[1])
    assert tick(ctrl, xy=on_the_line).outcome is Outcome.RUNNING


def test_the_step_limit_truncates_at_the_contract_value():
    ctrl = controller(action=(-1.0, 0.0))  # stand still; never reaches the goal
    ctrl.set_goal(GOAL)
    for _ in range(contract.MAX_EPISODE_STEPS):
        assert tick(ctrl).reason == "policy"
    assert ctrl.step == contract.MAX_EPISODE_STEPS
    command = tick(ctrl)
    assert command.outcome is Outcome.TIMEOUT
    assert command.reason == "timeout"


def test_a_terminal_outcome_latches_until_a_new_goal():
    """Without the latch a robot that stopped 0.2 m from the goal reports
    success on every subsequent tick, and a run of episodes counts one arrival
    many times."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    tick(ctrl, xy=GOAL)
    assert ctrl.outcome is Outcome.SUCCESS
    # Back at the origin, nowhere near the goal, and still terminal.
    assert tick(ctrl, xy=ORIGIN).outcome is Outcome.SUCCESS
    ctrl.set_goal((1.0, 1.0))
    assert ctrl.outcome is Outcome.RUNNING
    assert tick(ctrl).reason == "policy"


def test_the_three_terminal_outcomes_are_the_ones_evaluate_reports():
    """Success / collision / timeout, the same partition evaluate.py sums to
    one. IDLE and RUNNING are deliberately outside it so "waiting for a goal"
    can never be counted as an episode."""
    terminal = {o for o in Outcome if o.is_terminal}
    assert terminal == {Outcome.SUCCESS, Outcome.COLLISION, Outcome.TIMEOUT}


# --- no goal ------------------------------------------------------------------

def test_a_controller_without_a_goal_stands_still():
    ctrl = controller()
    command = tick(ctrl)
    assert command.reason == "no_goal"
    assert command.outcome is Outcome.IDLE
    assert (command.linear, command.angular) == (0.0, 0.0)


def test_a_new_goal_restarts_the_budget():
    ctrl = controller(action=(-1.0, 0.0))
    ctrl.set_goal(GOAL)
    for _ in range(10):
        tick(ctrl)
    assert ctrl.step == 10
    ctrl.set_goal((2.0, 2.0))
    assert ctrl.step == 0
    assert ctrl.goal == (2.0, 2.0)


def test_clearing_the_goal_returns_to_idle():
    ctrl = controller()
    ctrl.set_goal(GOAL)
    tick(ctrl)
    ctrl.clear_goal()
    assert ctrl.goal is None
    assert tick(ctrl).reason == "no_goal"


# --- reported fields ----------------------------------------------------------

def test_the_command_reports_the_distance_and_clearance_it_decided_on():
    """The node logs these. Recomputing them there would be a second
    implementation of the same arithmetic, one tick out of date."""
    ctrl = controller()
    ctrl.set_goal(GOAL)
    command = tick(ctrl, xy=(1.0, 0.0), pooled=scan(3.5))
    assert command.distance_to_goal == pytest.approx(math.dist((1.0, 0.0), GOAL))
    assert command.min_lidar == pytest.approx(3.5)


def test_a_policy_returning_a_batch_row_is_accepted():
    """TorchScript modules are traced on batched input and return ``(1, 2)``."""
    ctrl = DeploymentController(lambda obs: np.array([[1.0, 0.0]], dtype=np.float32))
    ctrl.set_goal(GOAL)
    assert tick(ctrl).linear == pytest.approx(contract.MAX_LINEAR_VEL)


def test_an_out_of_range_policy_output_is_clipped_not_scaled():
    ctrl = controller(action=(9.0, -9.0))
    ctrl.set_goal(GOAL)
    command = tick(ctrl)
    assert command.linear == pytest.approx(contract.MAX_LINEAR_VEL)
    assert command.angular == pytest.approx(-contract.MAX_ANGULAR_VEL)
