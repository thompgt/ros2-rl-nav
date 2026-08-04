"""The held-out evaluation set. Pure; no simulator.

The set is the measuring instrument for all of Phase 3. A bug here does not
produce an error -- it produces numbers that are wrong in a direction nobody
can see, and it invalidates every comparison in the README.
"""

import math

import pytest

from robot_rl_env import arena, contract, eval_set


def test_the_set_is_identical_across_calls():
    """The whole point. Two runs, two processes, two machines, same episodes."""
    assert eval_set.episodes(25) == eval_set.episodes(25)


def test_the_callback_set_is_a_prefix_of_the_reported_set():
    """So the in-training curve and the final number are on one scale.

    If these were independent samples, an EvalCallback trend that flattened at
    80% and a final report of 88% would be indistinguishable from a policy that
    improved late.
    """
    full = eval_set.episodes(eval_set.N_EVAL_EPISODES)
    assert eval_set.episodes(eval_set.N_CALLBACK_EPISODES) == full[
        : eval_set.N_CALLBACK_EPISODES
    ]


def test_every_episode_is_legal_against_the_current_arena():
    for i, episode in enumerate(eval_set.episodes()):
        arena.validate_episode(episode.start[:2], episode.goal), f"episode {i}"
        assert arena.is_free(*episode.start[:2])
        assert arena.is_free(*episode.goal)
        assert -math.pi <= episode.start[2] <= math.pi


def test_no_episode_is_pre_solved_or_unreachable():
    """Both ends of the difficulty range are silent failures: a pre-solved
    episode inflates every success rate equally, and one that cannot be
    finished inside the truncation limit caps it below 100% for reasons that
    look like the policy."""
    for episode in eval_set.episodes():
        assert episode.straight_line_distance > contract.GOAL_TOLERANCE
        assert episode.straight_line_distance >= contract.MIN_START_GOAL_DISTANCE
        assert episode.straight_line_distance <= eval_set.MAX_EVAL_DISTANCE


def test_episodes_are_frozen():
    """A worker that mutated a shared episode would evaluate a different set
    from its siblings, and nothing would report the divergence."""
    episode = eval_set.episodes(1)[0]
    with pytest.raises(Exception):  # noqa: B017 -- dataclasses raise FrozenInstanceError
        episode.goal = (0.0, 0.0)


def test_reset_options_round_trip():
    episode = eval_set.episodes(1)[0]
    options = episode.as_reset_options()
    assert options == {"start": episode.start, "goal": episode.goal}
    # The keys RobotNavEnv.reset actually reads. Spelled wrong, the env would
    # silently sample a random episode instead and the eval set would be a
    # decoration.
    assert set(options) == {"start", "goal"}


def test_summary_describes_the_set():
    summary = eval_set.summarize(eval_set.episodes())
    assert summary["n"] == eval_set.N_EVAL_EPISODES
    assert summary["distance_min"] <= summary["distance_mean"] <= summary["distance_max"]
    assert summary["unreachable_at_full_speed"] == 0
    assert summary["no_detour_headroom"] == 0


def test_the_reachable_bound_comes_from_the_contract():
    """If the truncation limit or the speed limit changes, the eval cap must
    move with them -- a hard-coded 10.0 here would silently start admitting
    impossible episodes."""
    assert eval_set.REACHABLE_DISTANCE == pytest.approx(
        contract.MAX_LINEAR_VEL * contract.STEP_DURATION * contract.MAX_EPISODE_STEPS
    )
    # The arena is larger than one episode can cross. This is the fact the cap
    # exists for; if it ever stops being true, say so rather than assuming.
    assert contract.D_MAX > eval_set.REACHABLE_DISTANCE


def test_a_different_seed_gives_a_different_set():
    """Guards against a generator that is deterministic because it ignores its
    seed -- which would pass every test above."""
    assert eval_set.episodes(10) != eval_set.episodes(10, seed=eval_set.EVAL_SEED + 1)


def test_zero_episodes_raises():
    with pytest.raises(ValueError):
        eval_set.episodes(0)
