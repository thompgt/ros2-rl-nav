"""The gap comparison. Pure; the rollout loop needs an unpaused simulator.

This is the arithmetic that produces the most interesting paragraph in the
README, so it is tested where a wrong answer would otherwise be indistinguishable
from an interesting result.
"""

import math

import pytest

from robot_rl_env import contract
from robot_rl_env.deploy_eval import (
    compare,
    effective_control_hz,
    episode_deadline,
    format_comparison,
    parse_args,
)


def test_the_comparison_is_deployment_minus_step_synchronized():
    """Sign convention: a negative delta is a regression on deployment, which
    is the direction this measurement expects to find."""
    rows = compare({"success_rate": 0.78}, {"success_rate": 0.87})
    assert rows == [("success_rate", 0.87, 0.78, pytest.approx(-0.09))]


def test_metrics_missing_from_either_side_are_skipped():
    """The deployment loop reports no mean_reward, because computing one would
    put a second implementation of the contract's reward outside env.step. A
    comparison that treated the absence as zero would report a catastrophic
    regression in a number nobody measured."""
    rows = compare(
        {"success_rate": 0.5},
        {"success_rate": 0.6, "mean_reward": 8.4},
    )
    assert [row[0] for row in rows] == ["success_rate"]


def test_non_numeric_metrics_are_skipped():
    rows = compare({"n_episodes": 100, "model": "x.pt"}, {"n_episodes": 100, "model": "y.zip"})
    assert [row[0] for row in rows] == ["n_episodes"]


def test_booleans_are_not_compared_as_numbers():
    """``isinstance(True, int)`` is True in Python, and a flag differing between
    two runs would otherwise appear as a delta of 1.0000 in a table of rates."""
    assert compare({"deterministic": False}, {"deterministic": True}) == []


def test_no_baseline_yields_no_rows_rather_than_an_error():
    assert compare({"success_rate": 0.5}, None) == []
    assert "no baseline" in format_comparison([])


def test_the_table_signs_its_deltas():
    table = format_comparison(compare({"success_rate": 0.78}, {"success_rate": 0.87}))
    assert "-0.0900" in table
    improvement = format_comparison(compare({"collision_rate": 0.02}, {"collision_rate": 0.01}))
    assert "+0.0100" in improvement


def test_rows_that_are_nan_on_both_sides_are_left_out():
    """A policy that never succeeds reports nan path metrics on both sides. A
    row of `nan nan nan` is noise in a table read for its one real number."""
    rows = compare({"mean_path_length": math.nan}, {"mean_path_length": math.nan})
    assert "mean_path_length" not in format_comparison(rows)


def test_the_episode_deadline_is_a_multiple_of_the_nominal_duration():
    """500 steps at 20 Hz is 25 s of control time. The budget is generous
    because a container under Docker Desktop does not run at a real-time factor
    of 1 -- but it is finite, because a stalled control loop must be reported
    rather than waited on."""
    nominal = contract.MAX_EPISODE_STEPS / contract.CONTROL_HZ
    assert nominal == pytest.approx(25.0)
    assert episode_deadline(factor=1.0) == pytest.approx(nominal)
    assert episode_deadline(factor=4.0) == pytest.approx(100.0)


def test_a_slow_simulator_raises_the_control_rate_in_sim_time():
    """The confound nobody looks for, and the reason the real-time factor is
    reported beside the gap. The node ticks at 20 Hz on a wall clock like a
    robot; at a real-time factor of 0.5 that is 40 Hz of sim time, so the policy
    issues two actions per 50 ms of simulated world where training issued one.
    That is a different control problem, not staleness, and it lands in the
    measured gap indistinguishably."""
    assert effective_control_hz(1.0) == pytest.approx(contract.CONTROL_HZ)
    assert effective_control_hz(0.5) == pytest.approx(2 * contract.CONTROL_HZ)
    assert effective_control_hz(2.0) == pytest.approx(contract.CONTROL_HZ / 2)


def test_an_unmeasurable_real_time_factor_is_nan_not_a_division_error():
    assert math.isnan(effective_control_hz(0.0))
    assert math.isnan(effective_control_hz(math.nan))


def test_the_default_episode_count_is_the_full_held_out_set():
    """The gap must be measured on the same 100 episodes evaluate.py reports,
    or the two success rates differ by which episodes they drew as much as by
    the timing."""
    from robot_rl_env import eval_set

    assert parse_args(["--policy", "p.pt"]).episodes == eval_set.N_EVAL_EPISODES
