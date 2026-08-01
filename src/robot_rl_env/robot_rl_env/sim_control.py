"""Client for the Gazebo Harmonic world-control services.

Harmonic, not Classic. The services are ``ros_gz_interfaces/srv/ControlWorld``
on ``/world/<name>/control`` and ``ros_gz_interfaces/srv/SetEntityPose`` on
``/world/<name>/set_pose``, both provided by the ``gz-sim-user-commands-system``
plugin loaded in ``worlds/arena.sdf``. Anything under ``/gazebo/`` belongs to a
different, end-of-life simulator.

Concurrency
-----------
This node is spun by a ``MultiThreadedExecutor`` owned by the caller (the env).
Nothing here calls ``rclpy.spin_once`` -- spinning inside a call that is itself
running on the executor deadlocks against the very future it is waiting for.

Service futures are awaited with a ``threading.Event`` fired from the future's
done-callback rather than by polling. Callbacks land on a dedicated
``MutuallyExclusiveCallbackGroup``, distinct from the subscription group in
``observation_node``, so a service response can be delivered while a scan
callback is still executing.
"""

from __future__ import annotations

import math
import threading

from geometry_msgs.msg import Pose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity, WorldControl, WorldReset
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose

from robot_rl_env import contract

SERVICE_TIMEOUT = 10.0
"""Seconds of wall clock to wait for a service response before raising."""

DISCOVERY_TIMEOUT = 30.0
"""Seconds to wait for the bridged services to appear at construction."""


class SimControlError(RuntimeError):
    """A world-control service call failed, timed out, or returned success=False.

    Always raised, never logged-and-swallowed. A dropped step desynchronizes sim
    time from the env's step counter, and every observation after it is wrong in
    a way that nothing downstream will report.
    """


class SimControl(Node):
    """Pause, step, reset, and teleport, via the bridged Gazebo services."""

    def __init__(self, node_name: str = "sim_control", *, context=None):
        super().__init__(node_name, context=context)
        self._group = MutuallyExclusiveCallbackGroup()
        self._control = self.create_client(
            ControlWorld, contract.WORLD_CONTROL_SERVICE, callback_group=self._group
        )
        self._set_pose = self.create_client(
            SetEntityPose, contract.SET_POSE_SERVICE, callback_group=self._group
        )

    def wait_for_services(self, timeout: float = DISCOVERY_TIMEOUT) -> None:
        """Block until both services are available, or raise.

        Called explicitly rather than from ``__init__`` because the caller has
        to start the executor first. A missing service here is the single most
        common Phase 1 misconfiguration and it otherwise presents as every
        step() hanging for OBS_TIMEOUT.
        """
        for client, name in (
            (self._control, contract.WORLD_CONTROL_SERVICE),
            (self._set_pose, contract.SET_POSE_SERVICE),
        ):
            if not client.wait_for_service(timeout_sec=timeout):
                raise SimControlError(
                    f"{name} did not appear within {timeout}s. Either the "
                    f"ros_gz_bridge service bridge is not running, or "
                    f"gz-sim-user-commands-system is missing from the world."
                )

    # --- world control --------------------------------------------------------

    def pause(self) -> None:
        self._call_control(WorldControl(pause=True))

    def unpause(self) -> None:
        self._call_control(WorldControl(pause=False))

    def step(self, iterations: int = contract.SIM_STEPS_PER_ACTION) -> None:
        """Advance exactly ``iterations`` physics iterations, then stay paused.

        ``pause=True`` alongside ``multi_step`` is what makes this exact:
        Gazebo runs the requested iterations and returns to the paused state.
        It is emphatically not unpause-sleep-pause, which reintroduces the
        wall-clock coupling the whole architecture exists to remove.
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self._call_control(WorldControl(pause=True, multi_step=iterations))

    def reset_world(self) -> None:
        """Reset poses, velocities, and sim time to their initial values."""
        self._call_control(WorldControl(pause=True, reset=WorldReset(all=True)))

    def set_entity_pose(
        self,
        name: str,
        x: float,
        y: float,
        yaw: float,
        z: float = contract.ROBOT_SPAWN_Z,
    ) -> None:
        """Teleport a model. Takes effect on the next physics iteration.

        UserCommands queues the pose and applies it during the following
        update, so a caller that teleports and then reads sensors without
        stepping gets the *old* pose back.
        """
        request = SetEntityPose.Request()
        request.entity = Entity(name=name, type=Entity.MODEL)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)
        request.pose = pose
        self._call(self._set_pose, request, contract.SET_POSE_SERVICE)

    # --- plumbing -------------------------------------------------------------

    def _call_control(self, world_control: WorldControl) -> None:
        request = ControlWorld.Request()
        request.world_control = world_control
        self._call(self._control, request, contract.WORLD_CONTROL_SERVICE)

    def _call(self, client, request, name: str) -> None:
        if not client.service_is_ready():
            raise SimControlError(f"{name} is not available")

        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())

        if not done.wait(SERVICE_TIMEOUT):
            future.cancel()
            raise SimControlError(
                f"{name} did not respond within {SERVICE_TIMEOUT}s. The "
                f"simulator is wedged or the executor is not spinning this node."
            )

        exc = future.exception()
        if exc is not None:
            raise SimControlError(f"{name} raised: {exc!r}") from exc

        response = future.result()
        if response is None or not response.success:
            raise SimControlError(f"{name} returned success=False for {request}")
