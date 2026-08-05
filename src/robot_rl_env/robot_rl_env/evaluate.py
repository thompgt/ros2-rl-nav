"""Phase 3 -- deterministic evaluation on the held-out set.

    python3 -m robot_rl_env.evaluate --model runs/sac-seed0/best/best_model.zip
    python3 -m robot_rl_env.evaluate --model ... --episodes 100 --json out.json

Prints one line per metric and writes a JSON file. Text output, because the
result of this script is the thing that goes in the README and the thing an
agent can actually read -- nobody can see a Gazebo window.

What it reports, and why each one
---------------------------------
success rate      the headline. Fraction of episodes that reached the goal.
collision rate    the failure that matters; a policy can raise success by
                  driving recklessly, and this is where that shows up.
timeout rate      the third outcome. A policy that learns to stand still has a
                  low collision rate and looks safe on every other metric.
mean path length  metres actually driven, on successes only. Failures have no
                  meaningful path length and averaging them in rewards giving
                  up early.
path efficiency   driven / straight-line, on successes. 1.0 is a straight line;
                  2.0 means the policy takes twice the necessary route, which
                  success rate alone cannot distinguish from optimal.
mean episode steps ditto, on successes.

Deterministic actions (``deterministic=True``), fixed episodes, no exploration
noise: two runs of this script against one policy must agree exactly. If they
do not, something is still stochastic that should not be.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from robot_rl_env import eval_set


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="path to a .zip saved by train.py")
    parser.add_argument(
        "--algo", default=None, help="sac or ppo; inferred from the model path if omitted"
    )
    parser.add_argument("--episodes", type=int, default=eval_set.N_EVAL_EPISODES)
    parser.add_argument("--json", default=None, help="write the metrics here as JSON")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of taking the mean. Not for reported numbers.",
    )
    return parser.parse_args(argv)


def infer_algorithm(model_path: str, override: str | None = None) -> str:
    """``runs/sac-seed0/best/best_model.zip`` -> ``"sac"``.

    SB3 zips do not record which algorithm wrote them in any way that
    ``load`` can dispatch on, and loading a SAC policy with ``PPO.load``
    fails deep inside the policy constructor with a shape error that names
    neither. Guess from the path, let ``--algo`` override, and say so plainly
    when the guess fails.
    """
    if override:
        algorithm = override.strip().lower()
        if algorithm not in ("sac", "ppo"):
            raise ValueError(f"unknown algorithm {override!r}; expected sac or ppo")
        return algorithm

    lowered = str(model_path).lower()
    found = [name for name in ("sac", "ppo") if name in lowered]
    if len(found) != 1:
        raise ValueError(
            f"cannot tell which algorithm wrote {model_path!r} (matched {found}); "
            f"pass --algo explicitly"
        )
    return found[0]


def summarize_rollouts(results: list[dict]) -> dict:
    """Aggregate per-episode records into the reported metrics.

    Pure, so the arithmetic that produces the README's numbers is testable
    without a simulator -- including the part that is easy to get wrong, which
    is which episodes each average is taken over.
    """
    n = len(results)
    if n == 0:
        raise ValueError("no episodes to summarize")

    successes = [r for r in results if r["success"]]
    metrics = {
        "n_episodes": n,
        "success_rate": len(successes) / n,
        "collision_rate": sum(r["collided"] for r in results) / n,
        "timeout_rate": sum(not r["success"] and not r["collided"] for r in results) / n,
    }
    # Reward is optional, and only Phase 4 omits it. The deployment loop has no
    # reward: computing one there would mean a second implementation of the
    # contract's reward function living outside env.step, which is the kind of
    # duplication this project treats as a defect rather than a convenience.
    # Reporting nan instead would put a number-shaped hole in the results table.
    if all("reward" in r for r in results):
        metrics["mean_reward"] = sum(r["reward"] for r in results) / n
    if successes:
        # Successes only. A failed episode's path length measures how long it
        # wandered before hitting a wall, and averaging that in rewards a
        # policy for failing early.
        metrics["mean_path_length"] = sum(r["path_length"] for r in successes) / len(successes)
        metrics["mean_steps"] = sum(r["steps"] for r in successes) / len(successes)
        metrics["mean_path_efficiency"] = sum(
            r["path_length"] / r["straight_line"] for r in successes
        ) / len(successes)
    else:
        metrics["mean_path_length"] = float("nan")
        metrics["mean_steps"] = float("nan")
        metrics["mean_path_efficiency"] = float("nan")
    return metrics


def run_episode(model, env, episode, deterministic: bool = True, trace: list | None = None) -> dict:
    """Play one fixed episode to termination and record what happened.

    Path length is integrated from ``info["robot_pose"]``, which is the odom
    frame -- the only pose a real robot has, and the same one Phase 4 will
    measure against. Straight-line distance comes from the episode's world
    coordinates. Mixing frames is safe here and only here: both are metres,
    and the quantity is a *difference* of positions within one frame.
    """
    obs, info = env.reset(options=episode.as_reset_options())
    previous = info["robot_pose"][:2]
    # ``trace`` is how record.py gets the frames for a GIF. It is an optional
    # out-parameter rather than a second rollout loop in that module, because
    # a GIF rendered from a loop that differs from this one would show an
    # episode the results table never scored.
    if trace is not None:
        trace.append({"obs": obs, "info": dict(info)})

    path_length, total_reward, steps = 0.0, 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        if trace is not None:
            trace.append({"obs": obs, "info": dict(info)})
        current = info["robot_pose"][:2]
        path_length += math.dist(previous, current)
        previous = current
        total_reward += float(reward)
        steps += 1

    return {
        "start": episode.start,
        "goal": episode.goal,
        "success": bool(info["is_success"]),
        "collided": bool(info["collided"]),
        "steps": steps,
        "reward": total_reward,
        "path_length": path_length,
        "straight_line": episode.straight_line_distance,
        "final_distance": float(info["distance_to_goal"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from stable_baselines3 import PPO, SAC

    from robot_rl_env import sim_launcher
    from robot_rl_env.env import RobotNavEnv

    algorithm = infer_algorithm(args.model, args.algo)
    model_class = {"sac": SAC, "ppo": PPO}[algorithm]
    episodes = eval_set.episodes(args.episodes)

    print(f"model={args.model} algo={algorithm}")
    print(f"eval set: {eval_set.summarize(episodes)}", flush=True)

    # The env is built directly rather than through make_eval_env: this script
    # drives the episodes itself, one at a time, so it can record per-episode
    # results. A VecEnv would auto-reset on termination and hide the boundary.
    model = model_class.load(args.model, device="cpu")
    with sim_launcher.Simulator(index=0):
        env = RobotNavEnv()
        try:
            results = [
                run_episode(model, env, episode, deterministic=not args.stochastic)
                for episode in episodes
            ]
        finally:
            env.close()

    metrics = summarize_rollouts(results)
    print()
    for key, value in metrics.items():
        print(f"{key:24s} {value:.4f}" if isinstance(value, float) else f"{key:24s} {value}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Episodes as well as metrics: a success rate with no record of which
        # episodes failed cannot be debugged, and re-running to find out costs
        # another full evaluation.
        path.write_text(
            json.dumps(
                {"model": args.model, "algo": algorithm, "metrics": metrics, "episodes": results},
                indent=2,
            )
        )
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
