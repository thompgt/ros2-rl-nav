"""Phase 5 -- render an episode as a GIF, from the policy's point of view.

    python3 -m robot_rl_env.record --model runs/sac-seed0/best/best_model.zip
    python3 -m robot_rl_env.record --model ... --episode 7 --out docs/nav.gif

Why a top-down plot rather than a screen capture of Gazebo
----------------------------------------------------------
The obvious way to get a GIF is to run the GUI and record the window. That
needs a GPU for ogre2, which the container does not have and CI certainly does
not, so the recording would only ever be reproducible on one machine.

More importantly, a Gazebo screenshot shows the *world*. This plot shows the
**observation**: the 20 min-pooled LiDAR sectors the policy actually receives,
drawn as the wedges they are, alongside the goal bearing it is given. A viewer
can see the policy squeeze past an obstacle and simultaneously see that it had
18 degrees of angular resolution to do it with. That is the more honest picture
of what was learned, and it is the picture this project is about.

The frames are recorded through the same ``evaluate.run_episode`` the results
table comes from, so the episode being animated is scored exactly as the
reported episodes are -- no separate rollout loop that might diverge from it.

Frames, and the one transform in here
-------------------------------------
Observations are in the odom frame; the arena is in the world frame. The two
differ by whatever odom read when the episode was reset, since ``reset_world``
plants odom's zero and teleporting does not move it. So the trace is drawn in
the world frame by composing the rigid transform between the two, taken once at
reset. That is dead reckoning: if the wheels slip, the drawn robot drifts off
the drawn obstacles. That is a true statement about the odometry the policy
navigates on, and it is left visible rather than corrected with ground truth.

matplotlib is imported inside the drawing functions, not at module scope, so
the geometry below stays importable -- and testable -- on a host that has no
plotting stack at all.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_rl_env import arena, contract
from robot_rl_env.observation import from_body_frame, to_body_frame, wrap_angle

#: Real time. The environment steps at 20 Hz, so the GIF plays at 1x.
FRAME_INTERVAL_MS = int(1000 * contract.STEP_DURATION)

#: Half the angular width of one pooled sector, in radians. A pooled beam is
#: the minimum over ``BEAM_POOL`` raw beams, so it describes a wedge rather
#: than a ray, and drawing it as a ray would overstate the sensor's resolution.
SECTOR_HALF_WIDTH = contract.BEAM_POOL * contract.LIDAR_ANGLE_INCREMENT / 2.0


@dataclass(frozen=True)
class Frame:
    """One step, in world coordinates, ready to draw."""

    x: float
    y: float
    yaw: float
    pooled: np.ndarray  # metres, length N_BEAMS
    goal: tuple[float, float]
    step: int
    distance: float


def rigid_transform(
    odom_xy: tuple[float, float], odom_yaw: float, world_xy: tuple[float, float], world_yaw: float
) -> tuple[tuple[float, float], float]:
    """The odom -> world transform implied by one pose known in both frames.

    Returned as ``(translation, rotation)`` to be applied with
    :func:`apply_transform`. Taken once, at reset, from the pose the reset
    teleported to and the odom pose that was read back.
    """
    rotation = wrap_angle(world_yaw - odom_yaw)
    # Where the odom origin lands in the world: the world pose, walked back
    # along the odom position rotated into world orientation.
    c, s = math.cos(rotation), math.sin(rotation)
    tx = world_xy[0] - (c * odom_xy[0] - s * odom_xy[1])
    ty = world_xy[1] - (s * odom_xy[0] + c * odom_xy[1])
    return (tx, ty), rotation


def apply_transform(
    transform: tuple[tuple[float, float], float], xy: tuple[float, float], yaw: float
) -> tuple[tuple[float, float], float]:
    """Map an odom pose into the world with a transform from above."""
    (tx, ty), rotation = transform
    c, s = math.cos(rotation), math.sin(rotation)
    return (tx + c * xy[0] - s * xy[1], ty + s * xy[0] + c * xy[1]), wrap_angle(yaw + rotation)


def unnormalize_beams(obs) -> np.ndarray:
    """Observation beams (``[-1, 1]``) back to metres.

    The inverse of the one line in ``assemble_observation`` that scales them.
    Reading the metric ranges off the info dict instead would work for the env
    but not for a trace loaded from disk, and the round trip is exact.
    """
    beams = np.asarray(obs, dtype=np.float32)[: contract.N_BEAMS]
    return (beams + 1.0) / 2.0 * contract.LIDAR_MAX


def sector_endpoints(frame: Frame) -> list[tuple[float, float]]:
    """World-frame midpoints of each pooled sector's arc, for drawing.

    The sensor sits ``LIDAR_OFFSET_X`` ahead of the robot origin, and ranges
    are measured from the sensor. Drawing them from the origin would put every
    beam 15 cm long, which at a 18 cm collision threshold is the difference
    between a plot that explains a collision and one that contradicts it.
    """
    sensor = from_body_frame((contract.LIDAR_OFFSET_X, 0.0), (frame.x, frame.y), frame.yaw)
    points = []
    for index, distance in enumerate(frame.pooled):
        # The pooled block spans BEAM_POOL raw beams; its centre is half a
        # block past its first beam, less half an increment.
        block = index * contract.BEAM_POOL * contract.LIDAR_ANGLE_INCREMENT
        first = contract.LIDAR_MIN_ANGLE + block
        centre = first + SECTOR_HALF_WIDTH - contract.LIDAR_ANGLE_INCREMENT / 2.0
        angle = frame.yaw + centre
        points.append(
            (sensor[0] + distance * math.cos(angle), sensor[1] + distance * math.sin(angle))
        )
    return points


def goal_bearing(frame: Frame) -> float:
    """Bearing to the goal in the robot's body frame, in radians."""
    body = to_body_frame(frame.goal, (frame.x, frame.y), frame.yaw)
    return math.atan2(body[1], body[0])


def frames_from_trace(trace: list[dict]) -> list[Frame]:
    """Turn the raw per-step records into world-frame frames.

    ``trace`` entries carry the observation and the info dict as the env
    produced them; the odom -> world transform is derived from the first,
    which is the reset.
    """
    if not trace:
        raise ValueError("empty trace: the episode recorded no steps")

    first = trace[0]["info"]
    start = first["start_world"]
    pose = first["robot_pose"]
    transform = rigid_transform(
        (pose[0], pose[1]), float(pose[2]), (start[0], start[1]), float(start[2])
    )
    goal = tuple(first["goal_world"])

    frames = []
    for entry in trace:
        info = entry["info"]
        pose = info["robot_pose"]
        (x, y), yaw = apply_transform(transform, (pose[0], pose[1]), float(pose[2]))
        frames.append(
            Frame(
                x=x,
                y=y,
                yaw=yaw,
                pooled=unnormalize_beams(entry["obs"]),
                goal=goal,
                step=int(info.get("step", 0)),
                distance=float(info["distance_to_goal"]),
            )
        )
    return frames


def outcome_label(result: dict) -> str:
    """The caption's last word, from an ``evaluate.run_episode`` record.

    Three outcomes, not two: an episode that neither reached the goal nor hit
    anything ran out of steps, and a GIF captioned "collision" for a policy
    that simply stalled would misrepresent it.
    """
    if result.get("success"):
        return "goal reached"
    if result.get("collided"):
        return "collision"
    return "timeout"


# --- drawing ---------------------------------------------------------------


def _draw_arena(axis) -> None:
    """Walls and obstacles, from arena.py -- the same geometry reset sampling
    rejects against, so an obstacle moved in the SDF cannot be missing here
    without also being missing there."""
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.transforms import Affine2D

    half = contract.ARENA_SIZE / 2.0
    axis.add_patch(
        Rectangle((-half, -half), contract.ARENA_SIZE, contract.ARENA_SIZE,
                  fill=False, edgecolor="#444", linewidth=2)
    )
    for obstacle in arena.OBSTACLES:
        if isinstance(obstacle, arena.Cylinder):
            axis.add_patch(
                Circle((obstacle.cx, obstacle.cy), obstacle.radius, color="#8a8a8a")
            )
        else:
            rect = Rectangle(
                (-obstacle.sx / 2.0, -obstacle.sy / 2.0), obstacle.sx, obstacle.sy,
                color="#8a8a8a",
            )
            # Rotate about the box's own centre, then place it: the same order
            # the SDF pose applies, and the reason the rectangle is built at
            # the origin rather than at (cx, cy).
            placement = Affine2D().rotate(obstacle.yaw).translate(obstacle.cx, obstacle.cy)
            rect.set_transform(placement + axis.transData)
            axis.add_patch(rect)


def animate(frames: list[Frame], outcome: str, output: Path, fps: int | None = None) -> Path:
    """Write the GIF. Requires matplotlib; imported here, not at module scope."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    half = contract.ARENA_SIZE / 2.0
    figure, axis = plt.subplots(figsize=(6, 6), dpi=100)
    axis.set_xlim(-half - 0.3, half + 0.3)
    axis.set_ylim(-half - 0.3, half + 0.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    _draw_arena(axis)

    goal = frames[0].goal
    axis.add_patch(Circle(goal, contract.GOAL_TOLERANCE, color="#2e8b57", alpha=0.35))
    axis.plot(*goal, marker="*", color="#2e8b57", markersize=14, zorder=5)

    (path_line,) = axis.plot([], [], color="#1f77b4", linewidth=1.5, zorder=4)
    beam_lines = [
        axis.plot([], [], color="#d62728", linewidth=1.0, alpha=0.5, zorder=3)[0]
        for _ in range(contract.N_BEAMS)
    ]
    robot = Circle((0.0, 0.0), 0.15, color="#1f77b4", zorder=6)
    axis.add_patch(robot)
    (heading,) = axis.plot([], [], color="#111", linewidth=2.0, zorder=7)
    caption = axis.set_title("", fontsize=10, family="monospace")

    def update(index: int):
        frame = frames[index]
        path_line.set_data([f.x for f in frames[: index + 1]], [f.y for f in frames[: index + 1]])
        for line, (ex, ey) in zip(beam_lines, sector_endpoints(frame), strict=True):
            line.set_data([frame.x, ex], [frame.y, ey])
        robot.center = (frame.x, frame.y)
        heading.set_data(
            [frame.x, frame.x + 0.35 * math.cos(frame.yaw)],
            [frame.y, frame.y + 0.35 * math.sin(frame.yaw)],
        )
        label = outcome if index == len(frames) - 1 else "running"
        caption.set_text(
            f"step {frame.step:3d}/{contract.MAX_EPISODE_STEPS}   "
            f"goal {frame.distance:4.2f} m   {label}"
        )
        return [path_line, robot, heading, caption, *beam_lines]

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        figure, update, frames=len(frames), interval=FRAME_INTERVAL_MS, blit=False
    )
    animation.save(
        str(output), writer=PillowWriter(fps=fps or round(contract.CONTROL_HZ))
    )
    plt.close(figure)
    return output


# --- CLI -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="a .zip saved by train.py")
    parser.add_argument("--algo", default=None, help="sac or ppo; inferred from the path")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="index into the held-out set. The GIF shows a *scored* episode.",
    )
    parser.add_argument("--out", default="docs/nav.gif", help="where to write the GIF")
    parser.add_argument("--fps", type=int, default=None, help="default: real time, 20 fps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from stable_baselines3 import PPO, SAC

    from robot_rl_env import eval_set, evaluate, sim_launcher
    from robot_rl_env.env import RobotNavEnv

    algorithm = evaluate.infer_algorithm(args.model, args.algo)
    episodes = eval_set.episodes()
    if not 0 <= args.episode < len(episodes):
        raise SystemExit(f"--episode must be in [0, {len(episodes)}), got {args.episode}")
    episode = episodes[args.episode]

    model = {"sac": SAC, "ppo": PPO}[algorithm].load(args.model, device="cpu")
    trace: list[dict] = []
    with sim_launcher.Simulator(index=0):
        env = RobotNavEnv()
        try:
            result = evaluate.run_episode(model, env, episode, trace=trace)
        finally:
            env.close()

    frames = frames_from_trace(trace)
    outcome = outcome_label(result)
    print(f"episode {args.episode}: {outcome} in {len(frames)} steps")
    output = animate(frames, outcome, Path(args.out), fps=args.fps)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
