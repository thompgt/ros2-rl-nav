"""The monitor's HTTP surface, against a real socket on an ephemeral port.

No ROS and no browser: the server is constructed over two plain callables, so
the routes, the error codes and the shape of the event stream can be exercised
on the Windows host in a fraction of a second.
"""

import json
import urllib.error
import urllib.request

import pytest

from robot_rl_env import contract, monitor, monitor_server


@pytest.fixture
def server():
    """A server on port 0 -- the OS picks a free one, so a stale container
    holding 8080 cannot make this suite fail for an unrelated reason."""
    goals = []
    state = {"payload": {"connected": False}}

    running = monitor_server.MonitorServer(
        state=lambda: state["payload"],
        on_goal=lambda odom, world: goals.append((odom, world)),
        transform=monitor.spawn_transform(),
        host="127.0.0.1",
        port=0,
        stream_hz=50.0,
    )
    running.start()
    running.goals = goals  # what the node would have published
    running.state_slot = state  # what the node would be reporting
    try:
        yield running
    finally:
        running.stop()


def get(server, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=5) as response:
        return response.status, response.read()


def post(server, path: str, payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}", data=body, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


# --- the static scene ---------------------------------------------------------

def test_arena_route_serves_the_scene(server):
    status, body = get(server, "/arena.json")

    assert status == 200
    scene = json.loads(body)
    assert scene["size"] == contract.ARENA_SIZE
    assert len(scene["obstacles"]) == len(monitor.arena_payload()["obstacles"])


def test_the_page_itself_is_served(server):
    status, body = get(server, "/")

    assert status == 200
    assert b"<canvas" in body, "index.html should be the page, not a directory listing"


def test_an_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(server, "/nope.html")

    assert caught.value.code == 404


def test_path_traversal_does_not_escape_the_web_root(server):
    """The server runs beside a policy file and a workspace. It serves one
    directory."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(server, "/../../../robot_rl_env/contract.py")

    assert caught.value.code in (400, 403, 404)


# --- goals --------------------------------------------------------------------

def test_a_posted_goal_reaches_the_sink_in_odom(server):
    status, response = post(server, "/goal", {"x": 2.0, "y": -1.0})

    assert status == 200
    assert len(server.goals) == 1
    odom, world = server.goals[0]
    assert world == pytest.approx((2.0, -1.0))
    assert response["odom"] == pytest.approx(list(odom))
    # The spawn pose is not the odom origin, so a goal that came back unchanged
    # would mean the transform was skipped.
    assert odom != pytest.approx(world)


def test_a_goal_outside_the_arena_is_400_with_a_reason_and_is_not_published(server):
    outside = contract.ARENA_SIZE
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, "/goal", {"x": outside, "y": 0.0})

    assert caught.value.code == 400
    assert b"outside" in caught.value.read()
    assert server.goals == [], "a rejected goal must not reach the robot"


def test_a_malformed_goal_is_400_not_500(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, "/goal", b"{not json")

    assert caught.value.code == 400
    assert server.goals == []


def test_posting_anywhere_else_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, "/cmd_vel", {"x": 1.0})

    assert caught.value.code == 404


# --- the event stream ---------------------------------------------------------

def read_events(response, count: int) -> list[tuple[str | None, dict]]:
    """Read ``count`` complete server-sent events off an open response."""
    events, name, buffer = [], None, b""
    while len(events) < count:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith(b"event:"):
            name = line[len(b"event:") :].strip().decode()
        elif line.startswith(b"data:"):
            buffer = line[len(b"data:") :].strip()
        elif not line and buffer:
            events.append((name, json.loads(buffer)))
            name, buffer = None, b""
    return events


def test_the_stream_opens_with_the_arena_then_sends_telemetry(server):
    server.state_slot["payload"] = {"connected": True, "robot": {"x": 1.0, "y": 2.0, "yaw": 0.0}}

    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/stream", timeout=5) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        events = read_events(response, 3)

    assert events[0][0] == "arena", "a client that reconnects needs the scene again"
    assert events[0][1]["size"] == contract.ARENA_SIZE
    assert [name for name, _ in events[1:]] == [None, None]
    assert events[1][1]["robot"]["x"] == 1.0


def test_the_stream_reflects_state_as_it_changes(server):
    """The frames are sampled at send time, not captured when the stream
    opened -- the failure that makes a live view show a frozen robot."""
    server.state_slot["payload"] = {"tick": 0}

    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/stream", timeout=5) as response:
        read_events(response, 2)
        server.state_slot["payload"] = {"tick": 1}
        later = read_events(response, 4)

    assert any(payload.get("tick") == 1 for _, payload in later)


def test_stopping_the_server_ends_an_open_stream(server):
    """Otherwise Ctrl-C on the launch hangs on whatever tab is still open."""
    response = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/stream", timeout=5)
    read_events(response, 2)

    server.stop()

    # The socket closes, so this returns rather than blocking until timeout.
    assert read_events(response, 100) is not None
    response.close()
