"""Phase 6 -- the payloads behind the live web monitor. PURE.

The browser is a renderer and nothing more: it receives world-frame points and
draws them. Every piece of arithmetic between a ROS message and a pixel lives
here, for the same reason ``policy_node.py`` contains none -- a second
implementation of the observation, this one in JavaScript, would diverge from
the one the policy sees and the drawing would quietly stop being evidence.

So there is no geometry in ``web/app.js``: no pooling, no beam angles, no
quaternions, no odom -> world transform. It plots line segments it was handed.

Frames, again
-------------
``/scan`` and ``/odom`` are in the odom frame; the arena is in the world frame.
A deployed robot has no ground truth to relate them, which is exactly the
condition Phase 4 measures -- so the monitor does what a real one would: it
seeds the transform from the pose odometry started at
(``contract.ROBOT_SPAWN_POSE``) and dead-reckons from there. If the wheels
slip, the drawn robot drifts off the drawn obstacles. That drift is the
odometry error the policy is navigating on, and it is left visible rather than
corrected against Gazebo's pose feed. The same stance, and the same two
functions, as ``record.py``.

Clicks make the round trip: the browser reports a world-frame point, this
module maps it back through the inverse of that transform, and the goal is
published in ``odom`` -- the only frame ``policy_node`` accepts.
"""

from __future__ import annotations

import json
import math

from robot_rl_env import arena, contract
from robot_rl_env.record import Frame, apply_transform, rigid_transform, sector_endpoints

#: How often the node samples its state for the event stream. Matched to the
#: control rate: a faster stream would send duplicate frames, a slower one
#: would hide single-tick watchdog stops, which are the interesting event.
STREAM_HZ = contract.CONTROL_HZ


def spawn_transform(
    spawn: tuple[float, float, float] = contract.ROBOT_SPAWN_POSE,
) -> tuple[tuple[float, float], float]:
    """The odom -> world transform for a robot that has not been teleported.

    Odometry is zeroed where the robot spawned, so odom's origin *is* the spawn
    pose and the transform is that pose. Derived through ``rigid_transform``
    rather than written out, so it stays correct if the odom origin ever stops
    being zero.
    """
    return rigid_transform((0.0, 0.0), 0.0, (spawn[0], spawn[1]), spawn[2])


def invert(
    transform: tuple[tuple[float, float], float], xy: tuple[float, float]
) -> tuple[float, float]:
    """World point -> odom point. The inverse of :func:`record.apply_transform`."""
    (tx, ty), rotation = transform
    dx, dy = xy[0] - tx, xy[1] - ty
    c, s = math.cos(-rotation), math.sin(-rotation)
    return (c * dx - s * dy, s * dx + c * dy)


def arena_payload() -> dict:
    """The static scene: walls, obstacles, and the constants the legend needs.

    Sent once when a browser connects. Built from ``arena.OBSTACLES`` -- the
    same table reset sampling rejects against and ``test_arena.py`` checks
    against the SDF -- so the drawing cannot show an arena the robot is not in.
    """
    obstacles = []
    for obstacle in arena.OBSTACLES:
        if isinstance(obstacle, arena.Cylinder):
            obstacles.append(
                {
                    "kind": "circle",
                    "name": obstacle.name,
                    "cx": obstacle.cx,
                    "cy": obstacle.cy,
                    "radius": obstacle.radius,
                }
            )
        else:
            obstacles.append(
                {
                    "kind": "box",
                    "name": obstacle.name,
                    "cx": obstacle.cx,
                    "cy": obstacle.cy,
                    "sx": obstacle.sx,
                    "sy": obstacle.sy,
                    "yaw": obstacle.yaw,
                }
            )
    return {
        "size": contract.ARENA_SIZE,
        "half": contract.ARENA_SIZE / 2.0,
        "obstacles": obstacles,
        "goal_tolerance": contract.GOAL_TOLERANCE,
        "collision_threshold": contract.COLLISION_THRESHOLD,
        "safety_threshold": contract.SAFETY_STOP_THRESHOLD,
        "max_steps": contract.MAX_EPISODE_STEPS,
        "max_linear": contract.MAX_LINEAR_VEL,
        "max_angular": contract.MAX_ANGULAR_VEL,
        "watchdog_timeout": contract.WATCHDOG_TIMEOUT,
        "control_hz": contract.CONTROL_HZ,
        "n_beams": contract.N_BEAMS,
        "lidar_max": contract.LIDAR_MAX,
        "spawn": list(contract.ROBOT_SPAWN_POSE),
    }


def telemetry_payload(
    *,
    transform: tuple[tuple[float, float], float],
    pooled=None,
    robot_xy: tuple[float, float] | None = None,
    robot_yaw: float | None = None,
    goal_odom: tuple[float, float] | None = None,
    age: float | None = None,
    linear: float | None = None,
    angular: float | None = None,
    status: dict | None = None,
) -> dict:
    """One frame for the browser, in world coordinates.

    ``pooled`` is the 20-sector min-pooled scan in metres, taken straight from
    the assembler the policy node reads -- not a re-pooling of ``/scan``, which
    would be a second implementation of the one preprocessing step this project
    is most careful about.

    Everything is optional because the monitor starts before the simulator
    publishes anything, and a monitor that crashed on the first empty frame
    would be useless in exactly the situation you open it for.
    """
    payload: dict = {"connected": robot_xy is not None}

    if robot_xy is not None and robot_yaw is not None:
        (x, y), yaw = apply_transform(transform, robot_xy, robot_yaw)
        payload["robot"] = {"x": x, "y": y, "yaw": yaw}

        if pooled is not None:
            frame = Frame(
                x=x, y=y, yaw=yaw, pooled=pooled, goal=(0.0, 0.0), step=0, distance=0.0
            )
            payload["beams"] = [list(point) for point in sector_endpoints(frame)]
            payload["min_range"] = float(min(pooled))

    if goal_odom is not None:
        (gx, gy), _ = apply_transform(transform, goal_odom, 0.0)
        payload["goal"] = [gx, gy]
        if robot_xy is not None:
            payload["distance_to_goal"] = math.hypot(
                goal_odom[0] - robot_xy[0], goal_odom[1] - robot_xy[1]
            )

    if age is not None:
        payload["age"] = age
        payload["stale"] = age > contract.WATCHDOG_TIMEOUT
    if linear is not None:
        payload["linear"] = linear
    if angular is not None:
        payload["angular"] = angular
    if status:
        payload["status"] = status

    return payload


def parse_goal_request(body: bytes, transform: tuple[tuple[float, float], float]):
    """A browser click -> a goal in the odom frame.

    Raises ``ValueError`` with a message meant to be shown to the user. The
    click arrives in world coordinates because that is the frame drawn; the
    node publishes in odom because that is the frame ``policy_node`` accepts,
    and the mapping between them happens once, here.
    """
    try:
        request = json.loads(body or b"{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON: {error}") from error

    if not isinstance(request, dict) or "x" not in request or "y" not in request:
        raise ValueError("expected an object with 'x' and 'y' in world metres")

    try:
        world = (float(request["x"]), float(request["y"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"'x' and 'y' must be numbers: {error}") from error

    half = contract.ARENA_SIZE / 2.0
    if abs(world[0]) > half or abs(world[1]) > half:
        raise ValueError(
            f"({world[0]:.2f}, {world[1]:.2f}) is outside the {contract.ARENA_SIZE:g} m "
            f"arena; the robot cannot leave it and would drive into a wall trying"
        )

    return invert(transform, world), world


def sse_frame(payload: dict, event: str | None = None) -> bytes:
    """One server-sent event. ``data:`` on one line, terminated by a blank one.

    Server-sent events rather than a WebSocket: the telemetry is one-way at
    20 Hz, EventSource reconnects on its own, and ``http.server`` from the
    standard library can serve it -- so the monitor adds no dependency to an
    image that already takes a long time to build. Goals go the other way as
    an ordinary POST.
    """
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
