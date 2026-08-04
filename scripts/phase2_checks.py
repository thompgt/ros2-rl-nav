#!/usr/bin/env python3
"""Phase 2 exit criteria, as printed PASS/FAIL lines.

Run through ``scripts/verify_phase2.sh``, which launches the world first.

The unit tests in ``src/robot_rl_env/test/test_env.py`` prove the environment is
*correct*. This script proves it is *usable*: that it survives a hundred
episodes without hanging or leaking, and that it is fast enough for Phase 3 to
be worth starting. Those are the two ways a Gym environment passes its tests and
is still useless.

Usage:  scripts/verify_phase2.sh [n_episodes]
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time

import numpy as np

from robot_rl_env import contract
from robot_rl_env.env import RobotNavEnv

MIN_SPEEDUP = 5.0
"""sim seconds per wall second below which Phase 3 is impractical. From
WORKPLAN.md; below 1x, profile rather than adding parallel envs."""

MAX_RSS_GROWTH_MB = 50.0
"""Resident-set growth across the run that counts as a leak.

Generous on purpose: the allocator does not return freed pages promptly and
numpy's arena grows early. A real leak here is a subscription or a node created
per reset, and that grows by hundreds of megabytes, not tens."""

results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    results.append((ok, message))
    return ok


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def random_episodes(env: RobotNavEnv, n_episodes: int) -> dict:
    """Run ``n_episodes`` under a random policy. Any hang or crash propagates."""
    rng = np.random.default_rng(0)
    outcomes = {"success": 0, "collision": 0, "truncation": 0}
    steps = 0
    sim_seconds = 0.0
    wall_start = time.monotonic()

    for episode in range(n_episodes):
        env.reset(seed=episode)
        t0 = None
        while True:
            action = rng.uniform(-1.0, 1.0, size=contract.ACT_DIM).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            if t0 is None:
                t0 = info["sim_time"] - contract.STEP_DURATION
            if terminated or truncated:
                sim_seconds += info["sim_time"] - t0
                if info["is_success"]:
                    outcomes["success"] += 1
                elif info["collided"]:
                    outcomes["collision"] += 1
                else:
                    outcomes["truncation"] += 1
                break
        if (episode + 1) % 10 == 0:
            print(f"        {episode + 1}/{n_episodes} episodes, {steps} steps")

    wall = time.monotonic() - wall_start
    return {
        "steps": steps,
        "sim_seconds": sim_seconds,
        "wall_seconds": wall,
        "outcomes": outcomes,
    }


def measure_per_step_travel(env: RobotNavEnv, n_steps: int = 4) -> list[float]:
    """Metres travelled by each of the first few steps of full-forward command.

    Publishing ``/cmd_vel`` and calling the ``multi_step`` service take two
    different paths through ``ros_gz_bridge``, and nothing guarantees the
    velocity command is applied to the iterations it was meant to drive. If it
    consistently lands one env step late, the policy trains against an action
    delay that will not exist at deployment, and no other check would notice.

    Measured by counting leading steps that cover no ground, rather than
    against a closed-form dynamics model. The model version of this check was
    wrong in exactly the way such checks usually are: it assumed the DiffDrive
    ``max_wheel_acceleration`` limit shaped the ramp, reported a confident
    -399 ms, and the truth was that the plugin reaches commanded speed at once.
    Counting zero-displacement steps assumes nothing about the dynamics.
    """
    _, info = env.reset(seed=1234)
    previous = info["robot_pose"][:2]

    travel = []
    for _ in range(n_steps):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], np.float32))
        current = info["robot_pose"][:2]
        travel.append(math.dist(previous, current))
        previous = current
        if terminated or truncated:  # ran into something; the measurement is void
            return []
    return travel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n_episodes", nargs="?", type=int, default=100)
    args = parser.parse_args()

    print("=== Phase 2 verification ===")
    print()

    env = RobotNavEnv()
    try:
        print("--- check 1: command timing ---")
        travel = measure_per_step_travel(env)
        if not travel:
            check(False, "lag measurement aborted -- the robot terminated mid-run")
        else:
            print(
                "        travel per step under a full-forward command from rest: "
                + ", ".join(f"{t * 1000:.1f} mm" for t in travel)
            )
            # A step at full speed covers MAX_LINEAR_VEL * STEP_DURATION = 20 mm.
            # Half of that is far above any settling jitter and far below a step
            # that actually ran under the command.
            threshold = contract.MAX_LINEAR_VEL * contract.STEP_DURATION / 2
            late = next((i for i, t in enumerate(travel) if t > threshold), len(travel))
            check(
                late == 0,
                f"/cmd_vel drives the very first step after it is published "
                f"(first moving step: index {late}). An index of 1 would mean "
                f"every action is applied one env step late.",
            )
        print()

        print(f"--- check 2: {args.n_episodes} random-action episodes ---")
        rss_before = rss_mb()
        stats = random_episodes(env, args.n_episodes)
        rss_after = rss_mb()
        check(True, f"{args.n_episodes} episodes, {stats['steps']} steps, no hang or crash")
        print(f"        outcomes: {stats['outcomes']}")
        print()

        print("--- check 3: throughput ---")
        speedup = stats["sim_seconds"] / stats["wall_seconds"]
        steps_per_second = stats["steps"] / stats["wall_seconds"]
        print(
            f"        {stats['sim_seconds']:.1f} s of sim in "
            f"{stats['wall_seconds']:.1f} s of wall clock "
            f"({steps_per_second:.0f} env steps/s)"
        )
        check(
            speedup >= MIN_SPEEDUP,
            f"{speedup:.1f}x real time (need >= {MIN_SPEEDUP}x for Phase 3; "
            f"below 1x, profile before adding parallel envs)",
        )
        print()

        print("--- check 4: memory ---")
        growth = rss_after - rss_before
        print(f"        peak RSS {rss_before:.0f} MB -> {rss_after:.0f} MB")
        check(
            growth < MAX_RSS_GROWTH_MB,
            f"peak RSS grew {growth:.0f} MB over the run (limit {MAX_RSS_GROWTH_MB:.0f} MB)",
        )
        print()
    finally:
        env.close()

    failed = [m for ok, m in results if not ok]
    print("============================")
    print(f" PASS: {len(results) - len(failed)}    FAIL: {len(failed)}")
    print("============================")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
