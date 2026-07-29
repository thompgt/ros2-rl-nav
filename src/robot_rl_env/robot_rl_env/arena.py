"""Arena geometry and free-space rejection sampling.

Pure python and numpy. No ROS, no simulator, no Gymnasium -- which is the whole
point: reset sampling is the part of the environment most likely to silently
produce degenerate episodes (robot inside a wall, goal inside an obstacle,
start already within the goal tolerance), and none of those failures raise. They
just make the policy look untrainable.

Being pure, every one of those cases is a sub-millisecond unit test.

``OBSTACLES`` below duplicates the geometry in ``worlds/arena.sdf``. That
duplication is deliberate -- parsing SDF at runtime to recover collision shapes
would be considerably more code and would fail at a worse time -- and
``test_arena.py`` parses the SDF and asserts the two agree, so the copy cannot
drift.
"""

from __future__ import annotations

import math

import numpy as np

from robot_rl_env import contract

HALF_EXTENT = contract.ARENA_SIZE / 2.0
"""Metres from the origin to the inner face of a perimeter wall.

The walls in arena.sdf are centred at +/-5.1 and are 0.2 m thick, so their inner
faces sit at exactly +/-5.0 = ARENA_SIZE / 2.
"""


class Box:
    """An axis-aligned-in-its-own-frame rectangle, possibly yawed in the world."""

    __slots__ = ("name", "cx", "cy", "sx", "sy", "yaw")

    def __init__(self, name: str, cx: float, cy: float, sx: float, sy: float, yaw: float = 0.0):
        self.name = name
        self.cx, self.cy = cx, cy
        self.sx, self.sy = sx, sy
        self.yaw = yaw

    def distance(self, x: float, y: float) -> float:
        """Euclidean distance from ``(x, y)`` to the rectangle. 0 if inside."""
        dx, dy = x - self.cx, y - self.cy
        # Rotate the query point into the box frame rather than rotating the box.
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        ox = max(abs(lx) - self.sx / 2.0, 0.0)
        oy = max(abs(ly) - self.sy / 2.0, 0.0)
        return math.hypot(ox, oy)


class Cylinder:
    """A circle in the plane."""

    __slots__ = ("name", "cx", "cy", "radius")

    def __init__(self, name: str, cx: float, cy: float, radius: float):
        self.name = name
        self.cx, self.cy = cx, cy
        self.radius = radius

    def distance(self, x: float, y: float) -> float:
        return max(math.hypot(x - self.cx, y - self.cy) - self.radius, 0.0)


OBSTACLES: tuple[Box | Cylinder, ...] = (
    Box("obstacle_box_1", -2.8, 2.6, 0.8, 0.8, 0.0),
    Box("obstacle_box_2", 2.4, 3.2, 1.2, 0.6, 0.0),
    Box("obstacle_box_3", 0.2, -0.4, 1.0, 1.0, 0.4),
    Box("obstacle_box_4", -3.1, -2.2, 0.6, 1.4, 0.0),
    Cylinder("obstacle_cyl_1", 3.0, -1.0, 0.45),
    Cylinder("obstacle_cyl_2", 1.4, -3.4, 0.35),
    Cylinder("obstacle_cyl_3", -1.2, -3.9, 0.4),
)


def clearance(x: float, y: float) -> float:
    """Distance from ``(x, y)`` to the nearest obstacle *or wall*.

    Negative outside the arena, so a caller comparing against a positive
    clearance rejects out-of-bounds samples without a separate bounds check.
    """
    to_wall = HALF_EXTENT - max(abs(x), abs(y))
    to_obstacle = min(o.distance(x, y) for o in OBSTACLES)
    return min(to_wall, to_obstacle)


def is_free(x: float, y: float, margin: float = contract.MIN_OBSTACLE_CLEARANCE) -> bool:
    return clearance(x, y) >= margin


class SamplingFailure(RuntimeError):
    """Rejection sampling exhausted its attempt budget.

    Raised rather than silently relaxing the constraint. See
    ``contract.MAX_SAMPLE_ATTEMPTS``.
    """


def sample_free_point(
    rng: np.random.Generator,
    margin: float = contract.MIN_OBSTACLE_CLEARANCE,
    *,
    center: tuple[float, float] | None = None,
    min_radius: float = 0.0,
    max_radius: float | None = None,
) -> tuple[float, float]:
    """Uniformly sample a point in the arena with at least ``margin`` clearance.

    ``center`` / ``min_radius`` / ``max_radius`` add an annulus constraint,
    which is how both the "goal must be at least 1 m from the start" rule and
    the ``options={"goal_radius": r}`` curriculum hook are expressed.

    Sampling is uniform over the *arena*, then rejected -- not uniform over the
    annulus. That keeps the distribution over goal positions flat in space,
    which matters because uniform-in-polar-coordinates would concentrate goals
    near the robot and quietly make the task easier than intended.
    """
    lo = -HALF_EXTENT + margin
    hi = HALF_EXTENT - margin
    for _ in range(contract.MAX_SAMPLE_ATTEMPTS):
        x = float(rng.uniform(lo, hi))
        y = float(rng.uniform(lo, hi))
        if not is_free(x, y, margin):
            continue
        if center is not None:
            d = math.hypot(x - center[0], y - center[1])
            if d < min_radius:
                continue
            if max_radius is not None and d > max_radius:
                continue
        return x, y
    raise SamplingFailure(
        f"no free point after {contract.MAX_SAMPLE_ATTEMPTS} attempts "
        f"(margin={margin}, center={center}, min_radius={min_radius}, "
        f"max_radius={max_radius}). The arena geometry and the sampling "
        f"constraints are inconsistent; do not relax the margin to hide this."
    )


def sample_episode(
    rng: np.random.Generator,
    goal_radius: float | None = None,
) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """Sample ``(robot_xy, robot_yaw, goal_xy)`` for one episode.

    Both points clear obstacles by ``contract.MIN_OBSTACLE_CLEARANCE`` and are
    at least ``contract.MIN_START_GOAL_DISTANCE`` apart. ``goal_radius`` caps
    the start-goal distance for curriculum learning.
    """
    robot_xy = sample_free_point(rng)
    robot_yaw = float(rng.uniform(-math.pi, math.pi))
    goal_xy = sample_free_point(
        rng,
        center=robot_xy,
        min_radius=contract.MIN_START_GOAL_DISTANCE,
        max_radius=goal_radius,
    )
    return robot_xy, robot_yaw, goal_xy
