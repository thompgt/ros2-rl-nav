"""Phase 5 -- the geometry behind the GIF.

Everything here is pure. The one test that actually renders skips if matplotlib
is absent, which it is on the development host; the drawing is exercised in the
image, where the CI slow path runs.

The cases worth testing are the ones where a wrong answer still looks like a
picture: beams drawn from the robot origin instead of the sensor, beam 0 drawn
forwards instead of backwards, and the odom -> world transform dropped so the
robot walks through obstacles that are 4 m from where it thinks they are.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_rl_env import contract, record
from robot_rl_env.observation import assemble_observation

# --- the odom -> world transform -------------------------------------------


def test_identity_when_the_frames_coincide():
    transform = record.rigid_transform((1.0, 2.0), 0.5, (1.0, 2.0), 0.5)

    xy, yaw = record.apply_transform(transform, (3.0, -1.0), 0.25)

    assert xy == pytest.approx((3.0, -1.0))
    assert yaw == pytest.approx(0.25)


def test_the_reference_pose_maps_onto_its_world_pose():
    transform = record.rigid_transform((0.4, -0.2), 1.1, (-3.0, 2.5), -0.6)

    xy, yaw = record.apply_transform(transform, (0.4, -0.2), 1.1)

    assert xy == pytest.approx((-3.0, 2.5))
    assert yaw == pytest.approx(-0.6)


def test_the_transform_is_rigid():
    """Distances survive it. A transform that scaled would draw a path that
    fits the arena while describing a different one."""
    transform = record.rigid_transform((0.4, -0.2), 1.1, (-3.0, 2.5), -0.6)

    a, _ = record.apply_transform(transform, (0.0, 0.0), 0.0)
    b, _ = record.apply_transform(transform, (1.0, 2.0), 0.0)

    assert math.dist(a, b) == pytest.approx(math.hypot(1.0, 2.0))


def test_pure_rotation_of_the_odom_frame():
    """Odom read yaw 0 where the robot was actually facing +90 degrees."""
    transform = record.rigid_transform((0.0, 0.0), 0.0, (0.0, 0.0), math.pi / 2)

    xy, yaw = record.apply_transform(transform, (1.0, 0.0), 0.0)

    assert xy == pytest.approx((0.0, 1.0), abs=1e-9)
    assert yaw == pytest.approx(math.pi / 2)


# --- beams -----------------------------------------------------------------


def test_unnormalize_inverts_the_observation_scaling():
    pooled = np.linspace(0.0, contract.LIDAR_MAX, contract.N_BEAMS, dtype=np.float32)
    obs = assemble_observation(None, (0.0, 0.0), 0.0, (1.0, 0.0), [0.0, 0.0], 0, pooled=pooled)

    assert record.unnormalize_beams(obs) == pytest.approx(pooled, abs=1e-4)


def test_unnormalize_ignores_everything_past_the_beams():
    obs = np.full(contract.OBS_DIM, 1.0, dtype=np.float32)

    assert record.unnormalize_beams(obs).shape == (contract.N_BEAMS,)


def _frame(**kwargs):
    defaults = dict(
        x=0.0,
        y=0.0,
        yaw=0.0,
        pooled=np.full(contract.N_BEAMS, 1.0, dtype=np.float32),
        goal=(1.0, 0.0),
        step=0,
        distance=1.0,
    )
    return record.Frame(**{**defaults, **kwargs})


def test_beams_start_at_the_sensor_not_the_robot_origin():
    """15 cm at an 18 cm collision threshold is the difference between a plot
    that explains a collision and one that contradicts it."""
    points = record.sector_endpoints(_frame(pooled=np.zeros(contract.N_BEAMS, dtype=np.float32)))

    for point in points:
        assert point == pytest.approx((contract.LIDAR_OFFSET_X, 0.0))


def test_beam_zero_points_backwards():
    """min_angle is -pi. Drawing beam 0 forwards is wrong and looks fine."""
    points = record.sector_endpoints(_frame())

    assert points[0][0] < contract.LIDAR_OFFSET_X


def test_the_sectors_span_a_full_turn():
    frame = _frame()
    points = record.sector_endpoints(frame)
    sensor = (contract.LIDAR_OFFSET_X, 0.0)

    angles = sorted(math.atan2(y - sensor[1], x - sensor[0]) for x, y in points)
    gaps = [b - a for a, b in zip(angles, angles[1:], strict=False)]

    assert len(points) == contract.N_BEAMS
    # Every neighbouring pair is one pooled sector apart, uniformly.
    expected = 2 * math.pi / contract.N_BEAMS
    assert gaps == pytest.approx([expected] * (contract.N_BEAMS - 1), abs=1e-6)


#: Straight ahead is the *first* raw beam of block 10, not its centre: the
#: blocks start at -pi, so their boundaries fall on multiples of 18 degrees and
#: 0 is one of them. The drawn sector is therefore centred half a block minus
#: half an increment past forward -- 8.5 degrees.
FORWARD_SECTOR_OFFSET = math.radians(8.5)


def test_the_forward_sector_is_drawn_at_its_own_centre():
    """Straight ahead falls on a block boundary, so the block that contains it
    is drawn 8.5 degrees off. Drawing it dead ahead would claim a resolution
    the sensor pooling does not have."""
    pooled = np.full(contract.N_BEAMS, contract.LIDAR_MAX, dtype=np.float32)
    pooled[contract.N_BEAMS // 2] = 0.5
    sensor = (contract.LIDAR_OFFSET_X, 0.0)

    points = record.sector_endpoints(_frame(pooled=pooled))
    nearest = min(points, key=lambda p: math.dist(p, sensor))

    assert math.dist(nearest, sensor) == pytest.approx(0.5, abs=1e-6)
    angle = math.atan2(nearest[1] - sensor[1], nearest[0] - sensor[0])
    assert angle == pytest.approx(FORWARD_SECTOR_OFFSET, abs=1e-6)


def test_beams_rotate_and_translate_with_the_robot():
    pooled = np.full(contract.N_BEAMS, contract.LIDAR_MAX, dtype=np.float32)
    pooled[contract.N_BEAMS // 2] = 1.0
    frame = _frame(x=2.0, y=-1.0, yaw=math.pi / 2, pooled=pooled)
    # The sensor is 15 cm ahead of the robot, which now faces +y.
    sensor = (2.0, -1.0 + contract.LIDAR_OFFSET_X)

    points = record.sector_endpoints(frame)
    nearest = min(points, key=lambda p: math.dist(p, sensor))

    assert math.dist(nearest, sensor) == pytest.approx(1.0, abs=1e-6)
    angle = math.atan2(nearest[1] - sensor[1], nearest[0] - sensor[0])
    assert angle == pytest.approx(math.pi / 2 + FORWARD_SECTOR_OFFSET, abs=1e-6)


def test_goal_bearing_is_zero_when_the_goal_is_ahead():
    assert record.goal_bearing(_frame(goal=(5.0, 0.0))) == pytest.approx(0.0)


def test_goal_bearing_is_signed_in_the_body_frame():
    assert record.goal_bearing(_frame(yaw=math.pi / 2, goal=(5.0, 0.0))) == pytest.approx(
        -math.pi / 2
    )


# --- traces ----------------------------------------------------------------


def _record(x, y, yaw, *, start=(1.0, 1.0, 0.0), goal=(3.0, 1.0), step=0, distance=2.0):
    return {
        "obs": assemble_observation(
            None,
            (x, y),
            yaw,
            goal,
            [0.0, 0.0],
            step,
            pooled=np.full(contract.N_BEAMS, 2.0, dtype=np.float32),
        ),
        "info": {
            "robot_pose": (x, y, yaw),
            "start_world": start,
            "goal_world": goal,
            "distance_to_goal": distance,
            "step": step,
        },
    }


def test_frames_place_the_first_step_at_the_episodes_world_start():
    """Odom reads zero at reset while the robot was teleported to (1, 1)."""
    trace = [_record(0.0, 0.0, 0.0), _record(0.5, 0.0, 0.0, step=1)]

    frames = record.frames_from_trace(trace)

    assert (frames[0].x, frames[0].y) == pytest.approx((1.0, 1.0))
    assert (frames[1].x, frames[1].y) == pytest.approx((1.5, 1.0))


def test_frames_carry_the_world_goal_not_the_odom_one():
    frames = record.frames_from_trace([_record(0.0, 0.0, 0.0)])

    assert frames[0].goal == (3.0, 1.0)


def test_frames_recover_the_metric_beams():
    frames = record.frames_from_trace([_record(0.0, 0.0, 0.0)])

    assert frames[0].pooled == pytest.approx(np.full(contract.N_BEAMS, 2.0), abs=1e-4)


def test_an_empty_trace_raises_rather_than_writing_a_blank_gif():
    with pytest.raises(ValueError, match="empty trace"):
        record.frames_from_trace([])


def test_a_rotated_start_rotates_the_whole_path():
    trace = [
        _record(0.0, 0.0, 0.0, start=(0.0, 0.0, math.pi / 2)),
        _record(1.0, 0.0, 0.0, start=(0.0, 0.0, math.pi / 2), step=1),
    ]

    frames = record.frames_from_trace(trace)

    # One metre "forward" in odom is one metre along +y in the world.
    assert (frames[1].x, frames[1].y) == pytest.approx((0.0, 1.0), abs=1e-9)


# --- captions --------------------------------------------------------------


def test_outcome_labels_distinguish_all_three_endings():
    assert record.outcome_label({"success": True, "collided": False}) == "goal reached"
    assert record.outcome_label({"success": False, "collided": True}) == "collision"
    assert record.outcome_label({"success": False, "collided": False}) == "timeout"


def test_the_gif_plays_at_real_time():
    assert record.FRAME_INTERVAL_MS == 50


# --- rendering -------------------------------------------------------------


def test_a_gif_is_actually_written(tmp_path):
    pytest.importorskip("matplotlib")

    frames = record.frames_from_trace(
        [_record(i * 0.1, 0.0, 0.0, step=i, distance=2.0 - i * 0.1) for i in range(5)]
    )

    output = record.animate(frames, "goal reached", tmp_path / "out" / "nav.gif")

    assert output.exists()
    assert output.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
