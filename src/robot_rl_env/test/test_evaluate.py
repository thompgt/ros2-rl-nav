"""Evaluation metrics and CLI plumbing. Pure; the rollout loop needs a sim.

These are the numbers that go in the README, so the arithmetic is tested
directly. The specific thing being pinned is *which episodes each average is
taken over* -- averaging path length over failures too is the natural
implementation, and it silently rewards a policy for failing early.
"""

import math

import pytest

from robot_rl_env.evaluate import infer_algorithm, parse_args, summarize_rollouts


def _record(success=False, collided=False, steps=100, reward=1.0, path=5.0, straight=4.0):
    return {
        "success": success,
        "collided": collided,
        "steps": steps,
        "reward": reward,
        "path_length": path,
        "straight_line": straight,
    }


def test_the_three_outcomes_partition_the_episodes():
    results = [
        _record(success=True),
        _record(collided=True),
        _record(),  # neither: truncated at the step limit
        _record(success=True),
    ]
    m = summarize_rollouts(results)
    assert m["success_rate"] == pytest.approx(0.5)
    assert m["collision_rate"] == pytest.approx(0.25)
    assert m["timeout_rate"] == pytest.approx(0.25)
    assert m["success_rate"] + m["collision_rate"] + m["timeout_rate"] == pytest.approx(1.0)


def test_path_metrics_average_over_successes_only():
    """A failed episode's path length is how far it wandered before hitting a
    wall. Including it makes giving up early look like efficiency."""
    results = [
        _record(success=True, steps=50, path=6.0, straight=4.0),
        _record(collided=True, steps=3, path=0.2, straight=4.0),
    ]
    m = summarize_rollouts(results)
    assert m["mean_path_length"] == pytest.approx(6.0)
    assert m["mean_steps"] == pytest.approx(50)
    assert m["mean_path_efficiency"] == pytest.approx(1.5)


def test_mean_reward_averages_over_everything():
    """Unlike the path metrics: reward is defined for every episode, and the
    -10 collision penalty is exactly what a mean reward should reflect."""
    m = summarize_rollouts(
        [_record(success=True, reward=10.0), _record(collided=True, reward=-10.0)]
    )
    assert m["mean_reward"] == pytest.approx(0.0)


def test_a_policy_that_never_succeeds_reports_nan_not_zero():
    """Zero mean path length reads as a perfect straight line to anyone
    skimming a results table."""
    m = summarize_rollouts([_record(collided=True), _record()])
    assert m["success_rate"] == 0.0
    assert math.isnan(m["mean_path_length"])
    assert math.isnan(m["mean_path_efficiency"])


def test_an_empty_run_raises():
    with pytest.raises(ValueError):
        summarize_rollouts([])


def test_algorithm_is_inferred_from_the_run_directory():
    """Loading a SAC zip with PPO.load fails deep in the policy constructor
    with a shape error that names neither algorithm."""
    assert infer_algorithm("runs/sac-seed0/best/best_model.zip") == "sac"
    assert infer_algorithm("runs/PPO-seed2/final.zip") == "ppo"


def test_an_ambiguous_path_demands_an_explicit_algorithm():
    with pytest.raises(ValueError, match="--algo"):
        infer_algorithm("runs/compare-sac-vs-ppo/final.zip")
    with pytest.raises(ValueError, match="--algo"):
        infer_algorithm("checkpoints/final.zip")
    assert infer_algorithm("checkpoints/final.zip", "ppo") == "ppo"


def test_an_unknown_explicit_algorithm_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        infer_algorithm("runs/sac-seed0/final.zip", "td3")


def test_reported_numbers_are_deterministic_by_default():
    """--stochastic exists for debugging. Reported numbers must not use it, or
    two evaluations of one policy disagree."""
    assert parse_args(["--model", "x.zip"]).stochastic is False
