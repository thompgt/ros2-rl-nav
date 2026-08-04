"""Arena geometry and reset sampling. No simulator; runs in milliseconds.

Two jobs:

1. Prove ``arena.OBSTACLES`` still agrees with ``worlds/arena.sdf``. The python
   copy exists so sampling does not have to parse SDF at runtime; this test is
   the thing that stops the copy from drifting.
2. Prove reset sampling cannot emit a degenerate episode. Every failure mode
   here -- start inside an obstacle, goal inside a wall, start already at the
   goal -- is silent at runtime and shows up only as "the policy will not
   learn".
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robot_rl_env import arena, contract

PKG_ROOT = Path(__file__).resolve().parents[1]
WORLD_SDF = PKG_ROOT / "worlds" / "arena.sdf"


# --- the python copy matches the SDF -----------------------------------------

def _sdf_obstacles():
    """Extract obstacle_* models from arena.sdf as (name, cx, cy, yaw, shape)."""
    root = ET.parse(WORLD_SDF).getroot()
    world = root.find("world")
    out = {}
    for model in world.iter("model"):
        name = model.get("name")
        if not name.startswith("obstacle_"):
            continue
        pose = [float(v) for v in model.find("pose").text.split()]
        cx, cy, yaw = pose[0], pose[1], pose[5]
        collision = next(model.iter("collision"))
        box = collision.find(".//box")
        cyl = collision.find(".//cylinder")
        if box is not None:
            sx, sy, _ = (float(v) for v in box.find("size").text.split())
            shape = ("box", sx, sy)
        else:
            shape = ("cylinder", float(cyl.find("radius").text))
        out[name] = (cx, cy, yaw, shape)
    return out


def test_python_obstacles_match_the_sdf():
    sdf = _sdf_obstacles()
    py = {o.name: o for o in arena.OBSTACLES}
    assert set(sdf) == set(py), (
        "arena.OBSTACLES and arena.sdf disagree on which obstacles exist; "
        "reset sampling would place the robot inside something"
    )
    for name, (cx, cy, yaw, shape) in sdf.items():
        o = py[name]
        assert (o.cx, o.cy) == pytest.approx((cx, cy)), name
        if shape[0] == "box":
            assert isinstance(o, arena.Box), name
            assert (o.sx, o.sy) == pytest.approx((shape[1], shape[2])), name
            assert o.yaw == pytest.approx(yaw), name
        else:
            assert isinstance(o, arena.Cylinder), name
            assert o.radius == pytest.approx(shape[1]), name


def test_wall_half_extent_matches_the_sdf():
    """Inner wall faces must sit at exactly ARENA_SIZE / 2."""
    root = ET.parse(WORLD_SDF).getroot()
    world = root.find("world")
    for model in world.iter("model"):
        name = model.get("name")
        if not name.startswith("wall_"):
            continue
        pose = [float(v) for v in model.find("pose").text.split()]
        size = [float(v) for v in next(model.iter("collision")).find(".//box/size").text.split()]
        # The wall's inner face is its centre pulled back by half its thickness,
        # along whichever axis it is thin in.
        axis = 0 if size[0] < size[1] else 1
        inner = abs(pose[axis]) - size[axis] / 2.0
        assert inner == pytest.approx(arena.HALF_EXTENT), (
            f"{name} inner face at {inner}, not {arena.HALF_EXTENT}"
        )


# --- distance functions -------------------------------------------------------

def test_distance_is_zero_inside_a_shape():
    box = arena.Box("b", 0.0, 0.0, 2.0, 1.0)
    assert box.distance(0.0, 0.0) == pytest.approx(0.0)
    assert box.distance(0.9, 0.4) == pytest.approx(0.0)
    cyl = arena.Cylinder("c", 1.0, 1.0, 0.5)
    assert cyl.distance(1.2, 1.0) == pytest.approx(0.0)


def test_box_distance_along_face_and_corner():
    box = arena.Box("b", 0.0, 0.0, 2.0, 2.0)
    assert box.distance(2.0, 0.0) == pytest.approx(1.0)       # face
    assert box.distance(2.0, 2.0) == pytest.approx(math.sqrt(2))  # corner


def test_yawed_box_distance_uses_the_rotation():
    """A 45-degree square points a corner at +x where an aligned one shows a face.

    So the turned box reaches *closer* to a point on the +x axis, by exactly
    the difference between its half-diagonal and its half-width.
    """
    aligned = arena.Box("a", 0.0, 0.0, 2.0, 2.0, 0.0)
    turned = arena.Box("t", 0.0, 0.0, 2.0, 2.0, math.pi / 4)
    p = (2.0, 0.0)
    assert turned.distance(*p) < aligned.distance(*p)
    assert aligned.distance(*p) == pytest.approx(1.0)
    assert turned.distance(*p) == pytest.approx(2.0 - math.sqrt(2))


def test_clearance_is_negative_outside_the_arena():
    assert arena.clearance(-1.0, 1.0) > 0.0  # open floor, north-west of centre
    assert arena.clearance(arena.HALF_EXTENT + 1.0, 0.0) < 0.0
    assert arena.clearance(0.0, -arena.HALF_EXTENT - 0.5) < 0.0


def test_obstacle_centres_are_not_free_space():
    for o in arena.OBSTACLES:
        assert not arena.is_free(o.cx, o.cy), f"{o.name} centre reported as free"


# --- reset sampling -----------------------------------------------------------

def test_sampled_episodes_always_satisfy_the_contract():
    rng = np.random.default_rng(0)
    for _ in range(300):
        robot_xy, robot_yaw, goal_xy = arena.sample_episode(rng)
        assert arena.is_free(*robot_xy)
        assert arena.is_free(*goal_xy)
        assert -math.pi <= robot_yaw <= math.pi
        d = math.dist(robot_xy, goal_xy)
        assert d >= contract.MIN_START_GOAL_DISTANCE
        assert d > contract.GOAL_TOLERANCE  # never pre-solved


def test_goal_radius_option_caps_the_goal_distance():
    rng = np.random.default_rng(1)
    for _ in range(200):
        robot_xy, _, goal_xy = arena.sample_episode(rng, goal_radius=2.0)
        assert math.dist(robot_xy, goal_xy) <= 2.0


def test_sampling_is_deterministic_for_a_seed():
    a = arena.sample_episode(np.random.default_rng(7))
    b = arena.sample_episode(np.random.default_rng(7))
    assert a == b


def test_impossible_constraints_raise_rather_than_relax():
    """A goal_radius below MIN_START_GOAL_DISTANCE has no solution."""
    rng = np.random.default_rng(2)
    with pytest.raises(arena.SamplingFailure):
        arena.sample_episode(rng, goal_radius=contract.MIN_START_GOAL_DISTANCE / 2)


def test_reset_clearance_cannot_start_an_episode_in_collision():
    """The load-bearing consistency check between sampling and termination.

    Clearance is measured from the robot origin, but the collision test reads
    the LiDAR, which sits LIDAR_OFFSET_X forward. Facing an obstacle head-on,
    the sensor is that much closer to it. If the resulting range could fall
    below COLLISION_THRESHOLD, reset() would emit episodes that terminate on
    step 1 with -10 reward, through no fault of the policy.
    """
    worst_case = contract.MIN_OBSTACLE_CLEARANCE - contract.LIDAR_OFFSET_X
    assert worst_case > contract.COLLISION_THRESHOLD, (
        f"a freshly reset robot can read {worst_case:.3f} m, at or below the "
        f"{contract.COLLISION_THRESHOLD} m collision threshold"
    )


# --- explicitly specified episodes -------------------------------------------

def test_validate_accepts_what_the_sampler_produces():
    """The sampler and the validator must agree on what "legal" means.

    If they drift apart, every fixed eval episode generated from the sampler
    starts raising -- and the natural fix, loosening the validator, is the
    wrong one.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        robot_xy, _, goal_xy = arena.sample_episode(rng)
        arena.validate_episode(robot_xy, goal_xy)


@pytest.mark.parametrize(
    "robot_xy, goal_xy, expected",
    [
        ((-2.8, 2.6), (2.0, 2.0), "start"),        # inside obstacle_box_1
        ((0.0, 4.0), (3.0, -1.0), "goal"),         # inside obstacle_cyl_1
        ((0.0, 4.95), (2.0, 2.0), "start"),        # inside the north wall
        ((4.0, 4.0), (12.0, 0.0), "goal"),         # outside the arena entirely
    ],
)
def test_validate_rejects_unreachable_or_in_collision(robot_xy, goal_xy, expected):
    with pytest.raises(ValueError, match=expected):
        arena.validate_episode(robot_xy, goal_xy)


def test_validate_rejects_a_pre_solved_episode():
    with pytest.raises(ValueError, match="apart"):
        arena.validate_episode((4.0, 4.0), (4.1, 4.0))
