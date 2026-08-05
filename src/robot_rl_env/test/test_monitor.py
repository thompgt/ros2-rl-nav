"""The monitor's payloads, without ROS, a browser or a simulator.

What these guard is the claim the monitor makes by existing: that what the page
draws is the observation the policy receives, in the arena the robot is in. Two
failure modes matter and neither raises -- beams drawn from the wrong origin or
at the wrong angles, and a scene drawn in the wrong frame. Both produce a
plausible picture.
"""

import json
import math

import numpy as np
import pytest

from robot_rl_env import arena, contract, monitor
from robot_rl_env.record import apply_transform

# --- frames ------------------------------------------------------------------

def test_spawn_transform_maps_the_odom_origin_onto_the_spawn_pose():
    transform = monitor.spawn_transform()
    (x, y), yaw = apply_transform(transform, (0.0, 0.0), 0.0)

    assert (x, y) == pytest.approx(contract.ROBOT_SPAWN_POSE[:2])
    assert yaw == pytest.approx(contract.ROBOT_SPAWN_POSE[2])


def test_a_metre_ahead_in_odom_is_a_metre_along_the_spawn_heading():
    """Rotation, not just translation. A transform that only translated would
    pass the origin test above and put every drawn beam in the wrong place."""
    transform = monitor.spawn_transform()
    (x, y), _ = apply_transform(transform, (1.0, 0.0), 0.0)

    sx, sy, syaw = contract.ROBOT_SPAWN_POSE
    assert (x, y) == pytest.approx((sx + math.cos(syaw), sy + math.sin(syaw)))


@pytest.mark.parametrize("point", [(0.0, 0.0), (1.5, -2.25), (-4.0, 4.0)])
def test_invert_round_trips_apply_transform(point):
    transform = monitor.spawn_transform()
    (wx, wy), _ = apply_transform(transform, point, 0.0)
    assert monitor.invert(transform, (wx, wy)) == pytest.approx(point)


# --- the static scene ---------------------------------------------------------

def test_arena_payload_carries_every_obstacle_exactly_once():
    payload = monitor.arena_payload()
    names = [o["name"] for o in payload["obstacles"]]

    assert names == [o.name for o in arena.OBSTACLES]
    assert len(set(names)) == len(names)


def test_arena_payload_distinguishes_boxes_from_cylinders():
    payload = monitor.arena_payload()
    kinds = {o["name"]: o["kind"] for o in payload["obstacles"]}

    for obstacle in arena.OBSTACLES:
        expected = "circle" if isinstance(obstacle, arena.Cylinder) else "box"
        assert kinds[obstacle.name] == expected


def test_arena_payload_keeps_box_yaw():
    """One box is yawed in the SDF. Dropping the rotation draws it
    axis-aligned, which is a picture in which the robot clips a corner that
    is not there."""
    payload = monitor.arena_payload()
    yawed = [o for o in payload["obstacles"] if o["kind"] == "box" and o["yaw"] != 0.0]

    assert yawed, "arena.py has a rotated box; the payload lost the rotation"


def test_arena_payload_is_json_serializable():
    json.dumps(monitor.arena_payload())


# --- telemetry ----------------------------------------------------------------

def _pooled(value: float = 4.0) -> np.ndarray:
    return np.full(contract.N_BEAMS, value, dtype=np.float32)


def test_telemetry_before_any_data_is_a_frame_not_a_crash():
    payload = monitor.telemetry_payload(transform=monitor.spawn_transform())

    assert payload["connected"] is False
    assert "robot" not in payload
    json.dumps(payload)


def test_telemetry_reports_the_robot_in_world_coordinates():
    payload = monitor.telemetry_payload(
        transform=monitor.spawn_transform(),
        pooled=_pooled(),
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )

    assert payload["connected"] is True
    assert (payload["robot"]["x"], payload["robot"]["y"]) == pytest.approx(
        contract.ROBOT_SPAWN_POSE[:2]
    )


def test_beams_start_at_the_sensor_not_the_robot_origin():
    """The LiDAR sits 15 cm ahead of the robot origin. Drawing a 4 m beam from
    the origin misplaces its far end by that much -- which at an 18 cm
    collision threshold is the difference between a picture that explains a
    collision and one that contradicts it. ``sector_endpoints`` handles the
    offset; this asserts the monitor did not lose it."""
    payload = monitor.telemetry_payload(
        transform=monitor.spawn_transform(),
        pooled=_pooled(4.0),
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )
    robot = (payload["robot"]["x"], payload["robot"]["y"])

    reaches = [math.dist(robot, point) for point in payload["beams"]]
    assert len(reaches) == contract.N_BEAMS
    # A beam forward of the robot reaches past its own range by the offset, one
    # behind falls short by it, and none is exactly 4 m -- no sector centre
    # lands on the beam axis, since 20 sectors of 18 degrees straddle it.
    assert max(reaches) > 4.0
    assert min(reaches) < 4.0
    assert max(reaches) == pytest.approx(4.0 + contract.LIDAR_OFFSET_X, abs=0.01)
    assert min(reaches) == pytest.approx(4.0 - contract.LIDAR_OFFSET_X, abs=0.01)


def test_min_range_is_the_closest_pooled_sector():
    pooled = _pooled(4.0)
    pooled[7] = 0.21
    payload = monitor.telemetry_payload(
        transform=monitor.spawn_transform(),
        pooled=pooled,
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )

    assert payload["min_range"] == pytest.approx(0.21)


def test_distance_to_goal_is_measured_in_odom_not_in_the_drawing():
    """The two frames are a rigid transform apart, so the distance is the same
    either way -- and it is computed in odom because that is where the
    controller computes it. If these ever disagree, the transform is not
    rigid and the drawing is wrong, not the number."""
    transform = monitor.spawn_transform()
    payload = monitor.telemetry_payload(
        transform=transform, robot_xy=(1.0, 0.0), robot_yaw=0.0, goal_odom=(4.0, 4.0)
    )

    assert payload["distance_to_goal"] == pytest.approx(math.hypot(3.0, 4.0))
    drawn = math.dist((payload["robot"]["x"], payload["robot"]["y"]), payload["goal"])
    assert drawn == pytest.approx(payload["distance_to_goal"])


def test_staleness_is_flagged_against_the_watchdog_timeout():
    transform = monitor.spawn_transform()
    fresh = monitor.telemetry_payload(transform=transform, age=contract.WATCHDOG_TIMEOUT / 2)
    stale = monitor.telemetry_payload(transform=transform, age=contract.WATCHDOG_TIMEOUT * 2)

    assert fresh["stale"] is False
    assert stale["stale"] is True


# --- goals in --------------------------------------------------------------

def test_a_click_becomes_a_goal_in_odom():
    transform = monitor.spawn_transform()
    world = (2.0, -1.0)
    odom, echoed = monitor.parse_goal_request(json.dumps({"x": 2.0, "y": -1.0}).encode(), transform)

    assert echoed == pytest.approx(world)
    # The round trip is what policy_node will act on.
    (back_x, back_y), _ = apply_transform(transform, odom, 0.0)
    assert (back_x, back_y) == pytest.approx(world)


def test_a_click_outside_the_arena_is_refused():
    transform = monitor.spawn_transform()
    outside = contract.ARENA_SIZE / 2.0 + 1.0

    with pytest.raises(ValueError, match="outside"):
        monitor.parse_goal_request(json.dumps({"x": outside, "y": 0.0}).encode(), transform)


@pytest.mark.parametrize(
    "body", [b"", b"not json", b"[]", b'{"x": 1.0}', b'{"x": "over there", "y": 0.0}']
)
def test_malformed_goal_requests_raise_rather_than_publish_something(body):
    with pytest.raises(ValueError):
        monitor.parse_goal_request(body, monitor.spawn_transform())


# --- the stream ---------------------------------------------------------------

def test_sse_frame_is_one_event_terminated_by_a_blank_line():
    frame = monitor.sse_frame({"a": 1})

    assert frame.startswith(b"data: ")
    assert frame.endswith(b"\n\n")
    assert json.loads(frame[len(b"data: ") :].decode()) == {"a": 1}


def test_sse_payloads_never_span_lines():
    """A newline inside the data would split the event and the browser would
    parse half a JSON document."""
    frame = monitor.sse_frame(monitor.telemetry_payload(transform=monitor.spawn_transform()))

    assert frame.count(b"\n") == 2


def test_sse_frame_can_name_an_event():
    frame = monitor.sse_frame({"size": 10.0}, event="arena")

    assert frame.startswith(b"event: arena\ndata: ")
