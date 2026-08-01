"""Phase 2c -- ``RobotNavEnv(gymnasium.Env)``.

Wires ``sim_control`` (2a) and ``observation_node`` (2b) into the interface
specified by ``CONTRACTS.md``. Every number here comes from ``contract``; none
is invented locally.

The step loop
-------------
::

    publish cmd_vel  ->  advance exactly 50 sim iterations  ->  block for a
    sensor stamp at or beyond the new sim time  ->  assemble

No ``time.sleep``, no ``spin_once``, no cached-observation fallback on timeout.
Step 3 either produces data at or after the target sim time or it raises.

Frames
------
Sampling happens in **world** coordinates -- ``arena.py`` knows where the walls
and obstacles are, and the ``set_pose`` service takes world poses. The
observation, however, is built in the **odom** frame, because that is the only
frame ``policy_node.py`` will have at deployment: there is no ground-truth pose
on a real robot.

The two are related by wheel odometry, which is dead-reckoned from a zero that
``reset_world`` restores and that teleporting does *not* move. So on reset the
sampled goal is expressed relative to the robot's world pose and that same
body-frame offset is then re-planted at the robot's odom pose::

    goal_body = to_body_frame(goal_world, robot_world_xy, robot_world_yaw)
    goal_odom = from_body_frame(goal_body, odom_xy, odom_yaw)

This is exact whatever odom happens to read after the reset, which is the point
-- assuming it reads (0, 0, 0) would be an unchecked assumption about plugin
internals, and it would fail silently, as a goal quietly displaced by however
much odometry drifted.

Known race
----------
``/cmd_vel`` is published a moment before the ``multi_step`` service call, and
the two travel different paths through ``ros_gz_bridge``. Nothing in the
protocol *guarantees* the velocity command is applied to the first of the 50
iterations rather than to the second. The effect, if it happened, would be a
uniform one-iteration (1 ms) actuation lag, not jitter -- and
``scripts/verify_phase2.py`` measures it directly by commanding a known
velocity from rest and comparing displacement against the analytic prediction.
"""

from __future__ import annotations

import contextlib
import itertools
import threading

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from robot_rl_env import arena, contract
from robot_rl_env.observation import from_body_frame, to_body_frame
from robot_rl_env.observation_node import ObservationAssembler
from robot_rl_env.sim_control import SimControl

CMD_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
"""Depth 1: only the most recent command has any meaning. A queue of stale
velocity commands is worse than a dropped one."""

_INSTANCE = itertools.count()
"""Node-name suffix source. Two envs in one process with identical node names
produce a graph in which neither can be addressed unambiguously; ROS warns
about it and then carries on, which is the worst of the options."""


class RobotNavEnv(gym.Env):
    """Step-synchronized navigation environment. See ``CONTRACTS.md``."""

    metadata = {"render_modes": []}

    observation_space = spaces.Box(
        low=-1.0, high=1.0, shape=(contract.OBS_DIM,), dtype=np.float32
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(contract.ACT_DIM,), dtype=np.float32
    )

    def __init__(
        self,
        *,
        obs_timeout: float = contract.OBS_TIMEOUT,
        service_timeout: float = 30.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError(
                "render_mode is not supported: the world runs headless and the "
                "GUI is a separate `gz sim -g` process. Launch with "
                "headless:=false to watch."
            )
        self.render_mode = None
        self._obs_timeout = obs_timeout

        suffix = next(_INSTANCE)
        # A private context rather than the global one, so several envs can
        # share a process (and so an env can be closed and rebuilt inside one
        # pytest session) without fighting over rclpy's global state.
        self._context = rclpy.Context()
        rclpy.init(context=self._context)

        self._sim = SimControl(f"sim_control_{suffix}", context=self._context)
        self._obs = ObservationAssembler(
            f"observation_assembler_{suffix}", context=self._context
        )
        self._node = Node(f"robot_nav_env_{suffix}", context=self._context)
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", CMD_QOS)

        self._executor = MultiThreadedExecutor(num_threads=4, context=self._context)
        for node in (self._sim, self._obs, self._node):
            self._executor.add_node(node)
        self._spin_thread = threading.Thread(
            target=self._spin, name=f"rclpy-executor-{suffix}", daemon=True
        )
        self._spin_thread.start()
        self._closed = False

        self._sim.wait_for_services(timeout=service_timeout)
        # Paused is the resting state of this world. Assert it rather than
        # assume the launch file was given paused:=true.
        self._sim.pause()
        self._obs.wait_for_first_message(timeout=service_timeout)

        # --- episode state, all set properly in reset() ---
        self._goal_odom: tuple[float, float] = (0.0, 0.0)
        self._goal_world: tuple[float, float] = (0.0, 0.0)
        self._start_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._prev_action = np.zeros(contract.ACT_DIM, dtype=np.float32)
        self._prev_distance = 0.0
        self._step_count = 0
        self._target_ns = 0
        self._needs_reset = True

    # --- gymnasium API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """See CONTRACTS.md ("Reset"). ``options={"goal_radius": r}`` caps the
        start-goal distance for the Phase 3 curriculum."""
        super().reset(seed=seed)
        goal_radius = None if options is None else options.get("goal_radius")

        (rx, ry), ryaw, goal_world = arena.sample_episode(self.np_random, goal_radius)

        # Zero the command *before* the world resets, so the queued velocity
        # from the last step of the previous episode cannot carry over into the
        # first iteration of this one.
        self._publish_velocity(0.0, 0.0)
        self._sim.reset_world()
        self._sim.set_entity_pose(contract.ROBOT_NAME, rx, ry, ryaw)

        # Snapshot after reset_world() returns: everything the simulator sent
        # before the reset has been delivered by then, so "strictly newer than
        # this" means "generated after the reset". See ObservationAssembler.sequence.
        seq = self._obs.sequence()
        # set_pose is queued by UserCommands and applied on the next iteration,
        # so this step both applies the teleport and repopulates the sensors.
        self._sim.step(contract.SIM_STEPS_PER_ACTION)

        sample = self._obs.wait_for_sample(0, min_seq=seq, timeout=self._obs_timeout)

        # Re-baseline sim time on the stamp actually observed rather than
        # assuming reset_world zeroed the clock. Every subsequent step target is
        # this plus an exact integer number of STEP_DURATION_NS.
        self._target_ns = sample.stamp_ns

        self._start_world = (rx, ry, ryaw)
        self._goal_world = goal_world
        self._goal_odom = from_body_frame(
            to_body_frame(goal_world, (rx, ry), ryaw), sample.xy, sample.yaw
        )

        self._prev_action = np.zeros(contract.ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._needs_reset = False

        obs, info = self._obs.assemble(sample, self._goal_odom, self._prev_action, 0)
        self._prev_distance = info["distance_to_goal"]
        return obs, self._finish_info(info, reached=False, collided=False)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError(
                "step() before reset(), or after a terminal step. Gymnasium "
                "requires reset() to start an episode."
            )
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(contract.ACT_DIM), -1.0, 1.0
        )

        # 1. act
        self._publish_velocity(*self.scale_action(action))

        # 2. advance exactly one step's worth of sim time. Integer nanoseconds:
        #    accumulating 0.05 in float 500 times drifts past a sensor stamp by
        #    an ULP and hangs the last step of long episodes.
        self._target_ns += contract.STEP_DURATION_NS
        self._sim.step(contract.SIM_STEPS_PER_ACTION)

        # 3. block until the sensors have caught up, or raise
        self._step_count += 1
        obs, info = self._obs.get_obs(
            self._target_ns,
            self._goal_odom,
            action,
            self._step_count,
            timeout=self._obs_timeout,
        )

        # 4. reward and termination, from CONTRACTS.md
        distance = info["distance_to_goal"]
        reached = distance < contract.GOAL_TOLERANCE
        collided = info["min_lidar"] < contract.COLLISION_THRESHOLD

        reward = self._prev_distance - distance          # progress, metres
        reward += contract.STEP_COST                     # dithering
        reward += contract.ANGULAR_PENALTY * abs(float(action[1]))  # spinning
        if reached:
            reward += contract.GOAL_BONUS
        if collided:
            reward += contract.COLLISION_PENALTY

        self._prev_distance = distance
        self._prev_action = action

        terminated = bool(reached or collided)
        # Truncation is only reported when the episode did not also terminate:
        # SB3 bootstraps on truncated and does not on terminated, and reporting
        # both leaves the value target ambiguous.
        truncated = bool(not terminated and self._step_count >= contract.MAX_EPISODE_STEPS)
        self._needs_reset = terminated or truncated

        return obs, float(reward), terminated, truncated, self._finish_info(
            info, reached=reached, collided=collided
        )

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._executor.shutdown(timeout_sec=2.0)
        finally:
            for node in (self._sim, self._obs, self._node):
                node.destroy_node()
            if self._context.ok():
                rclpy.shutdown(context=self._context)
            self._spin_thread.join(timeout=5.0)

    def __del__(self):
        # Envs are routinely dropped without close() (SubprocVecEnv workers,
        # aborted notebooks). Leaking a spinning executor per drop exhausts
        # DDS participants long before it exhausts memory.
        #
        # Suppressed rather than reported: this runs during interpreter
        # teardown, where the logging machinery may already be gone.
        with contextlib.suppress(Exception):
            self.close()

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def scale_action(action) -> tuple[float, float]:
        """``[-1, 1]^2`` -> ``(v, omega)``. See CONTRACTS.md ("Action space").

        ``a[0] = -1`` is a full stop, not reverse: a policy that can back out of
        a concave obstacle never has to learn to avoid entering it.
        """
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v = contract.MAX_LINEAR_VEL * (float(a[0]) + 1.0) / 2.0
        omega = contract.MAX_ANGULAR_VEL * float(a[1])
        return v, omega

    def _publish_velocity(self, v: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    def _finish_info(self, info: dict, *, reached: bool, collided: bool) -> dict:
        info["is_success"] = bool(reached)  # exact spelling SB3's EvalCallback wants
        info["collided"] = bool(collided)
        info["goal_odom"] = self._goal_odom
        info["goal_world"] = self._goal_world
        info["start_world"] = self._start_world
        info["step"] = self._step_count
        return info

    @property
    def sim_time_ns(self) -> int:
        """Sim time from ``/clock``, for the step-exactness test."""
        return self._obs.sim_time_ns()

    def _spin(self) -> None:
        # An executor being shut down from close() raises out of spin(); that is
        # the normal exit path for this thread, not an error worth surfacing.
        with contextlib.suppress(Exception):
            self._executor.spin()
