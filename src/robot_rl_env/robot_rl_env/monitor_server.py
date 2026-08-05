"""Phase 6 -- the monitor's HTTP surface. No ROS, no simulator.

Three routes and a static directory:

===================  ==========================================================
``GET  /``           ``web/index.html`` and its assets
``GET  /arena.json`` the static scene, once
``GET  /stream``     server-sent events, one telemetry frame per control tick
``POST /goal``       ``{"x": ..., "y": ...}`` in world metres -> ``/goal_pose``
===================  ==========================================================

It takes two callables and knows nothing else: one that returns the current
telemetry payload, one that accepts a goal. ``monitor_node.py`` supplies the
ROS ends of both. That split is what lets this file be tested against a real
socket on the Windows host, with no rclpy anywhere -- the same reason the rest
of the project keeps a pure core behind a ROS shell.

The standard library's ``ThreadingHTTPServer`` is enough because the load is
one browser and 20 small frames a second. Anything heavier here would be a
dependency added to a 13 GB image for a debug view.

**Bound to localhost by default.** There is no authentication, and the POST
route drives a robot. Publishing it on 0.0.0.0 means anyone who can reach the
port can steer; the container publishes the port to the host explicitly, which
is the deliberate act.
"""

from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from robot_rl_env import monitor

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
"""The page, beside the package in the source tree. ``setup.py`` installs it to
``share/robot_rl_env/web``; ``monitor_node`` prefers the installed copy and
falls back here, so the page can be edited without a colcon build."""

DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"


class MonitorServer:
    """A background HTTP server over a telemetry source and a goal sink."""

    def __init__(
        self,
        *,
        state: Callable[[], dict],
        on_goal: Callable[[tuple[float, float], tuple[float, float]], None],
        transform,
        web_root: Path = WEB_ROOT,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        stream_hz: float = monitor.STREAM_HZ,
        log: Callable[[str], None] = lambda message: None,
    ):
        self._stopping = threading.Event()
        server = ThreadingHTTPServer((host, port), _make_handler(self, web_root))
        # Otherwise shutdown blocks on browser tabs still holding a stream
        # open, and Ctrl-C on the launch does not return.
        server.daemon_threads = True

        self.state = state
        self.on_goal = on_goal
        self.transform = transform
        self.stream_interval = 1.0 / stream_hz
        self.log = log
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        """The bound port, which is the requested one unless it was 0."""
        return self._server.server_address[1]

    @property
    def stopping(self) -> threading.Event:
        return self._stopping

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # Set first: open streams poll this between frames, so they finish the
        # frame they are on and return rather than being cut mid-JSON.
        self._stopping.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def _make_handler(server: MonitorServer, web_root: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)

        # --- routes ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's name
            if self.path.split("?")[0] == "/arena.json":
                self._send_json(200, monitor.arena_payload())
            elif self.path.split("?")[0] == "/stream":
                self._stream()
            else:
                # SimpleHTTPRequestHandler resolves against `directory` and
                # rejects traversal itself; reimplementing that check is how
                # you get it wrong.
                super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/goal":
                self._send_json(404, {"error": f"no route {self.path}"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                odom, world = monitor.parse_goal_request(body, server.transform)
            except ValueError as error:
                # 400 and the reason, which the page shows verbatim. A goal
                # silently dropped looks identical to a policy ignoring it.
                self._send_json(400, {"error": str(error)})
                return

            server.on_goal(odom, world)
            self._send_json(200, {"odom": list(odom), "world": list(world)})

        # --- the event stream -----------------------------------------------

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # Proxies and Chrome both buffer otherwise, and the page arrives in
            # bursts of a second, which is worse than useless for judging
            # latency -- the one thing this view exists to show.
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                self.wfile.write(monitor.sse_frame(monitor.arena_payload(), event="arena"))
                self.wfile.flush()
                while not server.stopping.is_set():
                    self.wfile.write(monitor.sse_frame(server.state()))
                    self.wfile.flush()
                    # Waiting on the stop event rather than sleeping: shutdown
                    # then takes one frame, not one frame plus the sleep.
                    server.stopping.wait(server.stream_interval)
            except (BrokenPipeError, ConnectionResetError):
                # The tab was closed. Normal, and not worth a traceback --
                # EventSource reconnects on its own when it comes back.
                pass

        # --- plumbing ---------------------------------------------------------

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            """Route the access log through the caller's logger.

            The default writes to stderr unconditionally, which at 20 Hz would
            bury the policy node's own output -- the reason you have the
            terminal open at all.
            """
            server.log(fmt % args)

    return Handler
