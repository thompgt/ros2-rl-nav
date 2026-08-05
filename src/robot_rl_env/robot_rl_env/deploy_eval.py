"""Phase 4 -- the sim-to-deployment gap, measured.

    python3 -m robot_rl_env.deploy_eval --policy runs/sac-seed0/policy.pt \\
        --baseline runs/sac-seed0/eval.json --json runs/sac-seed0/gap.json

Scores a policy on the **same held-out episodes** ``evaluate.py`` uses, with
the same termination rules and the same observation assembly -- but against an
unpaused, free-running world, driven by the real ``policy_node`` at 20 Hz off a
wall clock. Everything that can be held constant is held constant, so the
difference between the two success rates is attributable to the one thing that
changed: nobody controls when the world advances any more.

What the gap is
---------------
``env.step`` publishes a velocity, advances the world by exactly 50 ms, and
blocks until the sensors carry a stamp at or beyond the new sim time. Every
observation the policy ever trained on was exactly one action old. Here there
is no barrier to block on: the timer fires, the freshest available sample is
whatever the bridge has delivered, and its staleness varies with bridge
latency, executor scheduling and CPU load. The policy is being asked to control
a system it never saw.

There will be a gap. Reporting only its size would be the uninteresting version
of this measurement, so the tick statistics come with it -- mean and p95
observation age, and the fraction of ticks lost to the watchdog. A 5-point drop
with a 12 ms p95 age is a different finding from a 5-point drop with a 300 ms
p95 and 4% watchdog ticks, and it is the second number that says which.

Why this drives the real node
-----------------------------
It builds a ``PolicyNode`` and gives it goals, rather than reimplementing the
control loop against the same controller. A harness with its own loop measures
the harness. The only things this file adds are the ones a robot's operator
would supply anyway: where to start, and where to go.

On ``time.sleep`` appearing in this file
----------------------------------------
CLAUDE.md forbids it *inside* ``step()``, because there it papers over a broken
step-synchronization handshake. There is no handshake here by construction --
that is the entire subject of the measurement -- and the world advances whether
or not this process is looking at it. Waiting for wall-clock time to pass is
the only correct way to wait for anything in a free-running system, and the
alternative is a busy loop that steals the CPU from the simulator being
measured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

from robot_rl_env import contract, eval_set

SETTLE_SECONDS = 1.2
"""Wall-clock seconds to let the robot brake before it is teleported.

The robot must be at rest when an episode starts or it inherits momentum from
the end of the last one. DiffDrive decays 8 rad/s to zero in 0.8 s under its
acceleration limit -- the same reasoning as ``contract.BRAKE_ITERATIONS``,
except that here the world runs in real time and the wait is real seconds
rather than physics iterations.
"""

TELEPORT_SETTLE_SECONDS = 0.5
"""Wall-clock seconds after the teleport before the first observation is taken.

``set_pose`` is applied on the following physics iteration, and the sensors
publish at 20 Hz. Reading immediately returns a scan from the *previous* pose,
and the episode would then begin with a goal anchored to the wrong place.
"""

DEADLINE_FACTOR = 4.0
"""Wall-clock budget per episode, as a multiple of its nominal duration.

500 steps at 20 Hz is 25 s of control time, so the default budget is 100 s.
Generous because a container under Docker Desktop does not run at a real-time
factor of 1, and an episode cut off by this deadline is reported separately
rather than folded into the timeout rate -- a deadline hit means the control
loop stopped ticking, which is a different failure from a policy that ran out
of steps.
"""

POLL_SECONDS = 1.0 / contract.CONTROL_HZ
"""How often the episode loop looks at the controller. One control period: the
path integral below wants a sample per control tick, matching the per-step
integration in ``evaluate.py``."""


RTF_TOLERANCE = 0.1
"""How far the real-time factor may sit from 1.0 before the result is qualified.

Not a pass/fail threshold -- the run is still reported -- but past this the gap
is no longer attributable to staleness alone. See :func:`effective_control_hz`.
"""


def episode_deadline(max_steps: int = contract.MAX_EPISODE_STEPS, factor: float = DEADLINE_FACTOR):
    """Wall-clock seconds allowed for one episode."""
    return max_steps / contract.CONTROL_HZ * factor


def effective_control_hz(real_time_factor: float) -> float:
    """The control rate this run achieved *in sim time*.

    The confound that has to be reported beside the gap, and the one nobody
    looks for. The node ticks at 20 Hz on a wall clock, as a robot would. If the
    simulator runs at a real-time factor of 0.5, then 20 wall-clock Hz is 40 Hz
    of sim time -- the policy issues two actions per 50 ms of simulated world
    where training issued one, and the robot travels half as far between
    decisions as it ever did in training.

    That is a difference in the *control problem*, not in observation
    staleness, and it would land in the measured gap indistinguishably. When
    the factor is near 1 the two coincide and the gap means what it says; when
    it is not, the number is qualified rather than quietly reported.

    Driving the timer from ``/clock`` instead would remove this confound and
    introduce a worse one -- a control loop that slows down when the world does,
    which no robot's does. See ``deploy.launch.py``.
    """
    if not real_time_factor or math.isnan(real_time_factor):
        return float("nan")
    return contract.CONTROL_HZ / real_time_factor


def compare(deployment: dict, baseline: dict | None) -> list[tuple[str, float, float, float]]:
    """``(metric, step_synchronized, free_running, delta)`` for shared metrics.

    Pure, because this is the arithmetic that produces the paragraph in the
    README and it should not need a simulator to check. Only metrics present in
    both are compared: the deployment loop reports no ``mean_reward``, and a
    comparison that silently treated a missing metric as zero would report a
    catastrophic regression in a number nobody measured.
    """
    if not baseline:
        return []
    rows = []
    for key, sync_value in baseline.items():
        deploy_value = deployment.get(key)
        if not isinstance(sync_value, (int, float)) or not isinstance(deploy_value, (int, float)):
            continue
        if isinstance(sync_value, bool) or isinstance(deploy_value, bool):
            continue
        rows.append((key, float(sync_value), float(deploy_value), float(deploy_value) - sync_value))
    return rows


def format_comparison(rows: list[tuple[str, float, float, float]]) -> str:
    """The side-by-side table. Deltas signed, so a regression reads as one."""
    if not rows:
        return "(no baseline given; pass --baseline runs/<run>/eval.json to see the gap)"
    header = f"{'metric':24s} {'step-sync':>12s} {'free-running':>14s} {'delta':>10s}"
    lines = [header, "-" * len(header)]
    for name, sync_value, deploy_value, delta in rows:
        if math.isnan(sync_value) and math.isnan(deploy_value):
            continue
        lines.append(f"{name:24s} {sync_value:12.4f} {deploy_value:14.4f} {delta:+10.4f}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", required=True, help="a .pt from robot_rl_env.export_policy")
    parser.add_argument("--episodes", type=int, default=eval_set.N_EVAL_EPISODES)
    parser.add_argument(
        "--baseline",
        default=None,
        help="eval.json written by robot_rl_env.evaluate, for the side-by-side",
    )
    parser.add_argument("--json", default=None, help="write the results here")
    parser.add_argument(
        "--deadline-factor",
        type=float,
        default=DEADLINE_FACTOR,
        help="wall-clock budget per episode, as a multiple of its nominal duration",
    )
    return parser.parse_args(argv)


def _integrate(previous, current) -> float:
    return math.dist(previous, current)


def run_episode(node, sim, episode, *, deadline: float) -> dict:
    """Place the robot, hand the node a goal, and watch until it stops.

    The goal is bridged from world to odom coordinates exactly as
    ``env.reset`` does it -- expressed relative to the sampled world pose, then
    re-planted at whatever odometry reports after the teleport. Odometry is
    dead-reckoned and a teleport does not move it, so anchoring on the measured
    odom pose rather than assuming it reads zero is what keeps the goal where
    the eval set put it.
    """
    from robot_rl_env.deploy import Outcome
    from robot_rl_env.observation import from_body_frame, to_body_frame

    rx, ry, ryaw = episode.start

    # Stop first: with no goal the node's timer publishes a zero velocity every
    # tick, so this brakes the robot through the same path deployment uses.
    node.controller.clear_goal()
    time.sleep(SETTLE_SECONDS)

    sim.set_entity_pose(contract.ROBOT_NAME, rx, ry, ryaw)
    time.sleep(TELEPORT_SETTLE_SECONDS)

    latest = node.observations.latest_sample()
    if latest is None:
        raise RuntimeError(
            "no observation after the teleport. The world is not running: "
            "deploy_eval needs paused:=false, and a paused Gazebo publishes "
            "nothing at all."
        )
    sample, _ = latest
    goal_odom = from_body_frame(to_body_frame(episode.goal, (rx, ry), ryaw), sample.xy, sample.yaw)

    path_length = 0.0
    previous = sample.xy
    last_stamp = sample.stamp_ns

    node.set_goal(goal_odom)
    started = time.monotonic()
    stalled = False

    while not node.controller.outcome.is_terminal:
        if time.monotonic() - started > deadline:
            stalled = True
            break
        # See the module docstring on sleeping here. The executor spins on its
        # own thread; this one only samples.
        time.sleep(POLL_SECONDS)
        latest = node.observations.latest_sample()
        if latest is None:
            continue
        sample, _ = latest
        # One integration step per new sensor stamp, which at 20 Hz is one per
        # control tick -- the same cadence evaluate.py integrates on. Counting
        # repeated reads of one sample would add nothing but would inflate
        # nothing either; counting them at a faster poll rate would.
        if sample.stamp_ns != last_stamp:
            path_length += _integrate(previous, sample.xy)
            previous = sample.xy
            last_stamp = sample.stamp_ns

    outcome = node.controller.outcome
    elapsed = time.monotonic() - started
    # Read before clearing: clear_goal() resets the step counter, and reading
    # it afterwards reported 0 for every episode -- which looks like a policy
    # that terminated instantly rather than one that ran its budget.
    steps = node.controller.step
    node.controller.clear_goal()

    return {
        "start": episode.start,
        "goal": episode.goal,
        "success": outcome is Outcome.SUCCESS,
        "collided": outcome is Outcome.COLLISION,
        "stalled": stalled,
        "outcome": outcome.value,
        "steps": steps,
        "path_length": path_length,
        "straight_line": episode.straight_line_distance,
        "wall_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from robot_rl_env import sim_launcher
    from robot_rl_env.evaluate import summarize_rollouts
    from robot_rl_env.policy_node import PolicyNode
    from robot_rl_env.sim_control import SimControl

    episodes = eval_set.episodes(args.episodes)
    deadline = episode_deadline(factor=args.deadline_factor)

    print(f"policy={args.policy} episodes={len(episodes)} deadline={deadline:.0f}s/episode")
    print(f"eval set: {eval_set.summarize(episodes)}", flush=True)
    print("world: UNPAUSED, real time. Control: 20 Hz off a wall clock.", flush=True)

    # paused=False is the whole point: this is the free-running counterpart of
    # the step-synchronized world evaluate.py scores against.
    with sim_launcher.Simulator(index=0, paused=False):
        context = rclpy.Context()
        rclpy.init(context=context)
        executor = MultiThreadedExecutor(num_threads=4, context=context)
        node = sim = spin_thread = None
        try:
            sim = SimControl("deploy_eval_sim_control", context=context)
            node = PolicyNode(context=context, policy_path=args.policy)
            for member in (sim, node, node.observations):
                executor.add_node(member)
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()

            sim.wait_for_services(timeout=60.0)
            node.observations.wait_for_first_message(timeout=60.0)

            sim_started = node.observations.sim_time_ns()
            wall_started = time.monotonic()

            results = []
            for index, episode in enumerate(episodes, start=1):
                record = run_episode(node, sim, episode, deadline=deadline)
                results.append(record)
                print(
                    f"[{index:3d}/{len(episodes)}] {record['outcome']:9s} "
                    f"steps={record['steps']:3d} path={record['path_length']:5.2f}m "
                    f"wall={record['wall_seconds']:5.1f}s",
                    flush=True,
                )

            sim_elapsed = (node.observations.sim_time_ns() - sim_started) / 1e9
            wall_elapsed = time.monotonic() - wall_started
            statistics = node.stats.summary()
        finally:
            if node is not None:
                node.brake()
            executor.shutdown(timeout_sec=2.0)
            for member in (node.observations, node, sim) if node is not None else (sim,):
                if member is not None:
                    member.destroy_node()
            if context.ok():
                rclpy.shutdown(context=context)
            if spin_thread is not None:
                spin_thread.join(timeout=5.0)

    metrics = summarize_rollouts(results)
    statistics["real_time_factor"] = sim_elapsed / wall_elapsed if wall_elapsed else float("nan")

    print()
    for key, value in metrics.items():
        print(f"{key:24s} {value:.4f}" if isinstance(value, float) else f"{key:24s} {value}")
    print()
    print("control loop, which is where the gap comes from:")
    for key, value in statistics.items():
        print(f"{key:24s} {value:.4f}" if isinstance(value, float) else f"{key:24s} {value}")

    rtf = statistics["real_time_factor"]
    if math.isfinite(rtf) and abs(rtf - 1.0) > RTF_TOLERANCE:
        print(
            f"\nWARNING: real-time factor {rtf:.2f}. The node ticks at "
            f"{contract.CONTROL_HZ:.0f} Hz on a wall clock, as a robot would, so "
            f"this run controlled the world at {effective_control_hz(rtf):.1f} Hz "
            f"of *sim* time against the {contract.CONTROL_HZ:.0f} Hz it was "
            f"trained at. Part of any gap below is that rate difference rather "
            f"than observation staleness, and the two are not separable from "
            f"these numbers. Take the measurement on a machine that sustains a "
            f"factor near 1, or report both figures together.",
            flush=True,
        )

    stalled = sum(r["stalled"] for r in results)
    if stalled:
        # Not folded into the timeout rate: a deadline hit means the control
        # loop stopped ticking, which is a different failure from a policy that
        # ran out of steps, and the numbers above are not trustworthy while it
        # is happening.
        print(
            f"\nWARNING: {stalled} episode(s) hit the wall-clock deadline rather "
            f"than terminating. The control loop stalled; treat the rates above "
            f"as unreliable and look at watchdog_fraction.",
            flush=True,
        )

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text()).get("metrics", {})
    print("\nsim-to-deployment gap:")
    print(format_comparison(compare(metrics, baseline)))

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "mode": "free_running",
                    "metrics": metrics,
                    "control_loop": statistics,
                    "baseline": args.baseline,
                    "episodes": results,
                },
                indent=2,
            )
        )
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
