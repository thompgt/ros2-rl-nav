"""The held-out evaluation set: fixed start/goal pairs, generated once.

Why this exists as a module rather than as a call to ``env.reset(seed=k)``
-------------------------------------------------------------------------
Evaluating on freshly sampled episodes measures the sampler as much as the
policy. Two runs draw two different sets of episodes, one of them happens to
contain more short goals, and the resulting few-point difference in success
rate is indistinguishable from a real difference between the algorithms. With
three seeds per algorithm and two algorithms, that noise is exactly the size of
the effect being reported.

So the evaluation episodes are fixed, shared by every run, and generated from a
seed that training never uses.

Held out, specifically
----------------------
``EVAL_SEED`` is deliberately far from the small integers that training runs
use as seeds. It is not a guarantee of disjointness -- two ``Generator``
streams can coincide on a draw -- but training resets sample fresh episodes
continuously rather than replaying a fixed set, so overlap on individual
episodes is expected and harmless. What matters is that the *evaluation* set is
identical across runs, and that no training run ever adapts to it, which the
curriculum callback respects by leaving the eval env alone.

The set is generated, not stored as a literal table: the arena geometry is
already versioned in ``arena.py``, and a checked-in table of coordinates would
be a second copy of it that goes stale silently the first time an obstacle
moves. Regenerating from a fixed seed makes the set a pure function of the
arena, and ``episodes()`` validates every pair against the current geometry
before returning it -- so if an obstacle does move onto an eval start, the run
fails loudly instead of scoring the policy on an impossible episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot_rl_env import arena, contract

EVAL_SEED = 20_260_803
"""Seed for the held-out set. Never use it as a training seed."""

N_EVAL_EPISODES = 100
"""Episodes in the full set. WORKPLAN's Phase 3 exit criterion is a success
rate over 100 deterministic episodes; 100 also puts the standard error of a
success rate near 85% at about 3.6 points, which is small enough to rank two
algorithms and large enough that nobody should read a 2-point gap as real."""

N_CALLBACK_EPISODES = 20
"""Episodes for the in-training ``EvalCallback``. Deliberately fewer than the
full set: this one runs every ``EVAL_FREQ`` steps and its cost is paid out of
the training budget, in simulator time. It reports a trend, not the number that
goes in the README -- that comes from ``evaluate.py`` over the full set."""


REACHABLE_DISTANCE = (
    contract.MAX_LINEAR_VEL * contract.STEP_DURATION * contract.MAX_EPISODE_STEPS
)
"""Metres coverable in one episode at full speed in a straight line: 10.0 m.

The arena diagonal is 14.15 m, so **the sampler can produce episodes that no
policy can finish**. That is a property of the contract (500 steps at 0.4 m/s),
not of this module, and it applies to training resets too -- where it is
harmless, because an occasional impossible episode is just a hard negative.

On an evaluation set it is not harmless. It puts a ceiling below 100% on the
achievable success rate, and the ceiling moves whenever the set is regenerated,
so "85% success" stops being comparable to itself.
"""

DETOUR_ALLOWANCE = 1.3
"""Path-length multiplier assumed for routing around obstacles.

Straight-line distance understates the distance actually driven; a goal at
exactly ``REACHABLE_DISTANCE`` is unreachable the moment anything is in the
way. Eval goals are capped at ``REACHABLE_DISTANCE / DETOUR_ALLOWANCE`` so that
a policy taking a 30% longer route than optimal still arrives in time.

This makes the eval set *easier* than the training distribution, which is a
deliberate trade and is why ``summarize`` prints the distance range beside the
result: the alternative is a headline number whose ceiling is unknown.
"""

MAX_EVAL_DISTANCE = REACHABLE_DISTANCE / DETOUR_ALLOWANCE


@dataclass(frozen=True)
class Episode:
    """One evaluation episode, in **world** coordinates.

    Frozen because these are shared across every run and every process in a
    ``SubprocVecEnv``; a mutable episode that one worker nudged would produce
    an eval set that differs per worker, which is the exact failure this module
    exists to prevent.
    """

    start: tuple[float, float, float]  # x, y, yaw
    goal: tuple[float, float]

    @property
    def straight_line_distance(self) -> float:
        """Metres from start to goal, ignoring obstacles.

        The denominator for path efficiency: a policy that reaches the goal by
        a route twice this long has learned something worse than one that goes
        straight there, and success rate alone cannot tell them apart.
        """
        return math.dist(self.start[:2], self.goal)

    def as_reset_options(self) -> dict:
        """The ``options`` dict ``RobotNavEnv.reset`` expects."""
        return {"start": self.start, "goal": self.goal}


def episodes(n: int = N_EVAL_EPISODES, seed: int = EVAL_SEED) -> tuple[Episode, ...]:
    """Generate the held-out set: deterministic in ``n`` and ``seed``.

    Prefix-stable by construction -- ``episodes(20)`` is exactly the first 20
    of ``episodes(100)``, because the generator is drawn from in order and
    never rewound. The callback set is therefore a subset of the reported set
    rather than an independent sample, which keeps the in-training curve and
    the final number on the same scale instead of merely similar.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # goal_radius is the sampler's existing annulus cap -- the same hook
        # the curriculum uses. Capping at sample time rather than filtering
        # afterwards keeps the set prefix-stable: a rejected episode would
        # shift every subsequent one.
        start_xy, yaw, goal_xy = arena.sample_episode(rng, goal_radius=MAX_EVAL_DISTANCE)
        # The sampler cannot emit an illegal episode against the arena it was
        # run with -- but this set is regenerated against whatever arena.py
        # currently says, and this call is what turns a moved obstacle into a
        # failure here rather than a depressed success rate three hours later.
        arena.validate_episode(start_xy, goal_xy)
        out.append(Episode(start=(start_xy[0], start_xy[1], yaw), goal=goal_xy))
    return tuple(out)


def summarize(eval_episodes: tuple[Episode, ...]) -> dict:
    """Descriptive statistics for the set, for printing before a run.

    An eval set whose goals are all 2 m away makes any policy look good; one
    whose goals are all 12 m away makes every policy look broken. Print this
    beside the results so the number has a scale attached.
    """
    distances = [e.straight_line_distance for e in eval_episodes]
    return {
        "n": len(eval_episodes),
        "distance_min": min(distances),
        "distance_mean": sum(distances) / len(distances),
        "distance_max": max(distances),
        # Must be zero. Non-zero means the achievable success rate is capped
        # below 100% by geometry, and the headline number is not what it says.
        "unreachable_at_full_speed": sum(d > REACHABLE_DISTANCE for d in distances),
        # Zero too, and stricter: no route 30% longer than the straight line
        # runs out of steps either. See DETOUR_ALLOWANCE.
        "no_detour_headroom": sum(d > MAX_EVAL_DISTANCE for d in distances),
    }
