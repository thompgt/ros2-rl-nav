"""Launch and tear down an isolated simulator, one per training worker.

Phase 3 runs N environments in parallel, and each needs its own Gazebo server,
its own pair of bridge nodes, and no visibility of its neighbours whatsoever.
Two isolation mechanisms are required, because there are two buses:

- ``ROS_DOMAIN_ID`` partitions DDS. Without it, worker 2's ``/cmd_vel``
  publisher drives worker 1's robot, and both read a ``/scan`` from whichever
  bridge answered first.
- ``GZ_PARTITION`` partitions gz-transport, which is a *separate* transport
  that the ROS domain does not touch. Without it, the ``/world/arena/control``
  service call that advances one worker's simulator can be served by another
  worker's Gazebo — so a step that should advance 50 ms advances someone
  else's world by 50 ms and the caller then blocks on a stamp that will never
  arrive.

Setting only one of the two is the failure worth naming: it *mostly* works.
Discovery is racy, so a two-worker run may be correct for ten minutes and then
silently cross-wire, and the only symptom is an observation timeout in a worker
that has done nothing wrong.

This module is also what ``test_env.py``'s session fixture uses, so the launch
path the tests exercise is the launch path training uses.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from robot_rl_env import contract

LAUNCH_TIMEOUT = 90.0
"""Seconds to wait for gz plus both bridges. Generous on purpose: a cold
container under Docker Desktop is slow, and a tight timeout here reads as a bug
in the environment rather than as the machine being busy."""

DOMAIN_ID_BASE = 42
"""Worker ``i`` gets ``DOMAIN_ID_BASE + i``. 42 is the value docker-compose sets
for the single-simulator services, so a one-worker training run shares a domain
with an interactive shell and can be inspected with ``ros2 topic echo``."""

DOMAIN_ID_MAX = 101
"""ROS 2 caps the portable range at 101. Beyond it, DDS port arithmetic
overflows into ports the OS may already have allocated, and the failure is a
discovery timeout rather than an error naming the domain."""


def worker_environment(index: int, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables isolating worker ``index``. Pure; no side effects.

    Returned as a fresh dict rather than applied to ``os.environ``: a
    ``SubprocVecEnv`` worker inherits the parent's environment, and mutating the
    parent's would leave whichever worker started last dictating the domain for
    every subsequent one.
    """
    if index < 0:
        raise ValueError(f"worker index must be non-negative, got {index}")

    domain = DOMAIN_ID_BASE + index
    if domain > DOMAIN_ID_MAX:
        raise ValueError(
            f"worker {index} would need ROS_DOMAIN_ID {domain}, above the "
            f"portable maximum of {DOMAIN_ID_MAX}. Lower DOMAIN_ID_BASE or run "
            f"fewer workers; do not let it wrap, which silently pairs two "
            f"workers on one domain."
        )

    env = dict(os.environ if base is None else base)
    env["ROS_DOMAIN_ID"] = str(domain)
    env["GZ_PARTITION"] = f"{contract.WORLD_NAME}_{index}"
    return env


class Simulator:
    """A launched world: ``gz sim -s`` plus the topic and service bridges.

    Owns a process *group*, not a process. ``ros2 launch`` spawns three
    children; killing only the parent leaves gz and both bridges alive holding
    their DDS ports, and the next run fails during discovery for reasons that
    point at the wrong thing entirely.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        headless: bool = True,
        paused: bool = True,
        timeout: float = LAUNCH_TIMEOUT,
    ):
        self.index = index
        self.env = worker_environment(index)
        self._proc: subprocess.Popen | None = None
        self._headless = headless
        self._paused = paused
        self._timeout = timeout

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> Simulator:
        if self._proc is not None:
            raise RuntimeError("already started")

        self._proc = subprocess.Popen(
            [
                "ros2", "launch", "robot_rl_env", "world.launch.py",
                f"headless:={str(self._headless).lower()}",
                f"paused:={str(self._paused).lower()}",
            ],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_ready(self) -> None:
        """Block until the services and topics the env needs actually exist.

        Polling the ROS graph rather than sleeping a fixed interval, because
        the thing being waited for is not time: it is the service bridge
        advertising ``/world/arena/control``. A sleep long enough to be safe on
        a cold container is wasted on every subsequent launch, and one tuned to
        a warm container fails intermittently on a cold one.
        """
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                output = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(f"launch exited early:\n{output}")
            if contract.WORLD_CONTROL_SERVICE in self._graph("service") and "/scan" in self._graph(
                "topic"
            ):
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"simulator {self.index} did not advertise "
            f"{contract.WORLD_CONTROL_SERVICE} and /scan within {self._timeout}s "
            f"on ROS_DOMAIN_ID={self.env['ROS_DOMAIN_ID']} "
            f"GZ_PARTITION={self.env['GZ_PARTITION']}"
        )

    def _graph(self, kind: str) -> str:
        """``ros2 <kind> list``, run in *this* worker's domain.

        Inheriting the caller's environment here would list the parent's graph
        and declare a worker ready that has not started -- the readiness check
        must ask the same bus the env will use.
        """
        try:
            return subprocess.run(
                ["ros2", kind, "list"],
                env=self.env,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        except subprocess.TimeoutExpired:
            # A daemon that is still indexing a new domain, not a failure. The
            # outer loop retries until its own deadline.
            return ""

    def stop(self, grace: float = 15.0) -> None:
        """Terminate the whole process group. Idempotent."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            # gz occasionally ignores SIGTERM while a physics iteration is in
            # flight. A surviving server holds the partition and the next run
            # attaches to the wrong world.
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=grace)

    # --- context manager ------------------------------------------------------

    def __enter__(self) -> Simulator:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
