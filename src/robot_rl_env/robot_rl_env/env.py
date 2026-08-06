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

Reset
-----
CONTRACTS.md ("Why reset does not restore the world") specifies braking rather
than a world reset. This does not call ``ControlWorld(reset.all)``: measured
against Harmonic, that service drops
the ``set_pose`` that has to follow it about half the time, and wedges the
server outright under repeated use. An episode brakes the robot to a stop and
teleports instead. The cost is that episodes are reproducible in their initial
state but not bit-identical over a full rollout. See
``contract.BRAKE_ITERATIONS``.

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
from robot_rl_env.action import clip_action, scale_action
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

        # Per-instance rather than class-level: a Box carries its own RNG, and
        # two envs in one process (DummyVecEnv) sharing one space would sample
        # from a single stream, so seeding either would silently steer both.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(contract.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(contract.ACT_DIM,), dtype=np.float32
        )

        # Named early so close() can run against a half-built env: anything
        # below here can raise, and __del__ will call close() regardless.
        self._closed = False
        self._context = None
        self._executor = None
        self._spin_thread = None
        self._nodes: tuple = ()

        try:
            suffix = next(_INSTANCE)
            # A private context rather than the global one, so several envs can
            # share a process (and so an env can be closed and rebuilt inside
            # one pytest session) without fighting over rclpy's global state.
            self._context = rclpy.Context()
            rclpy.init(context=self._context)

            self._sim = SimControl(f"sim_control_{suffix}", context=self._context)
            self._obs = ObservationAssembler(
                f"observation_assembler_{suffix}", context=self._context
            )
            self._node = Node(f"robot_nav_env_{suffix}", context=self._context)
            self._nodes = (self._sim, self._obs, self._node)
            self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", CMD_QOS)

            self._executor = MultiThreadedExecutor(num_threads=4, context=self._context)
            for node in self._nodes:
                self._executor.add_node(node)
            self._spin_thread = threading.Thread(
                target=self._spin, name=f"rclpy-executor-{suffix}", daemon=True
            )
            self._spin_thread.start()

            self._sim.wait_for_services(timeout=service_timeout)
            # Paused is the resting state of this world. Assert it rather than
            # assume the launch file was given paused:=true.
            self._sim.pause()
            # ... and then step it once, because a paused Gazebo publishes
            # nothing at all: sensors update in PostUpdate, which a paused
            # world never runs. Waiting for a first message without this hangs
            # until the timeout on a simulator that is working perfectly.
            #
            # Doing it here rather than leaving it to the first reset() means
            # the constructor exercises the full loop -- services, stepping,
            # both sensor topics, the QoS match -- and reports a broken bridge
            # at construction rather than as a mystery hang mid-episode.
            self._sim_ns = 0
            self._advance(contract.SIM_STEPS_PER_ACTION)
            self._obs.wait_for_first_message(timeout=service_timeout)
            # Baseline against the simulator's own clock. The launch file starts
            # the world paused at t = 0 and nothing else advances it, so this
            # normally agrees with the count above -- but attaching to a world
            # that has already been stepped is a legitimate thing to do, and an
            # env that assumed t = 0 would wait on stamps already in the past
            # and read a stale scan on every step without ever timing out.
            self._sim_ns = max(self._sim_ns, self._obs.sim_time_ns())
        except BaseException:
            # A constructor that raises after starting the executor leaves a
            # spinning thread and a DDS participant behind, and the usual cause
            # -- the simulator is not up -- is one a caller retries in a loop.
            self.close()
            raise

        # --- episode state, all set properly in reset() ---
        self._goal_odom: tuple[float, float] = (0.0, 0.0)
        self._goal_world: tuple[float, float] = (0.0, 0.0)
        self._start_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._prev_action = np.zeros(contract.ACT_DIM, dtype=np.float32)
        self._prev_distance = 0.0
        self._step_count = 0
        self._needs_reset = True
        self._curriculum_radius: float | None = None

    def set_curriculum_radius(self, radius: float | None) -> None:
        """Cap the sampled start-goal distance for every subsequent reset.

        The curriculum has to be driven from outside the episode loop, and SB3's
        ``VecEnv.reset()`` takes no ``options`` -- there is no way to pass
        ``goal_radius`` through a vectorized reset. So the cap is state on the
        env, set through ``VecEnv.env_method("set_curriculum_radius", r)``.

        ``None`` restores full-arena sampling. An explicit ``options`` on a
        single reset still wins, which is what keeps evaluation independent of
        whatever stage training has reached.
        """
        if radius is not None and radius < contract.MIN_START_GOAL_DISTANCE:
            raise ValueError(
                f"curriculum radius {radius} is below the "
                f"{contract.MIN_START_GOAL_DISTANCE} m minimum start-goal "
                f"distance; sampling would exhaust its attempts and raise on "
                f"every reset."
            )
        self._curriculum_radius = radius

    # --- gymnasium API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """See CONTRACTS.md ("Reset", "Reset options").

        ``options`` -- both keys are specified, and mutually exclusive:

        - ``{"goal_radius": r}`` caps the start-goal distance. The Phase 3
          curriculum hook.
        - ``{"start": (x, y, yaw), "goal": (x, y)}`` replaces sampling with a
          fixed episode, in **world** coordinates. Phase 3 needs a held-out
          evaluation set that is identical across algorithms, seeds and
          checkpoints: comparing a SAC run against a PPO run on two different
          sets of random episodes measures the sampler as much as the policies.
        """
        super().reset(seed=seed)
        options = options or {}
        explicit_radius = options.get("goal_radius")
        start, goal = options.get("start"), options.get("goal")

        if (start is None) != (goal is None):
            raise ValueError(
                "options 'start' and 'goal' must be given together. Fixing one "
                "and sampling the other yields an episode that is neither "
                "reproducible nor uniformly random."
            )
        if start is not None and explicit_radius is not None:
            raise ValueError(
                "options 'goal_radius' and 'start'/'goal' are mutually "
                "exclusive: the curriculum cap has no meaning for an episode "
                "whose goal is already fixed."
            )

        # An explicit cap beats the curriculum, and a fixed episode ignores
        # both -- so an evaluation reset returns the same episode whatever
        # stage training has reached.
        goal_radius = explicit_radius if explicit_radius is not None else self._curriculum_radius

        if start is None:
            (rx, ry), ryaw, goal_world = arena.sample_episode(self.np_random, goal_radius)
        else:
            rx, ry, ryaw = (float(v) for v in start)
            goal_world = (float(goal[0]), float(goal[1]))
            # Validated on every fixed reset, not once at load: an eval set can
            # arrive from a file, a CLI flag or a callback, and there is no one
            # place upstream that all of them pass through.
            arena.validate_episode((rx, ry), goal_world)

        # Brake to a stop, then teleport, as CONTRACTS.md specifies. Not
        # ControlWorld(reset.all): it drops the set_pose that follows it about
        # half the time and wedges the server outright under repeated use. See
        # contract.BRAKE_ITERATIONS for both measurements and what this costs.
        self._publish_velocity(0.0, 0.0)
        self._advance(contract.BRAKE_ITERATIONS)

        self._sim.set_entity_pose(contract.ROBOT_NAME, rx, ry, ryaw)

        # set_pose is queued by UserCommands and applied on the next iteration,
        # so this advance both lands the teleport and produces the first scan
        # from the new pose.
        self._advance(contract.SIM_STEPS_PER_ACTION)

        sample = self._obs.wait_for_sample(self._sim_ns, timeout=self._obs_timeout)

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
        action = clip_action(action)

        # 1. act
        self._publish_velocity(*self.scale_action(action))

        # 2. advance exactly one step's worth of sim time
        self._advance(contract.SIM_STEPS_PER_ACTION)

        # 3. block until the sensors have caught up, or raise
        self._step_count += 1
        obs, info = self._obs.get_obs(
            self._sim_ns,
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
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=2.0)
        finally:
            for node in self._nodes:
                node.destroy_node()
            if self._context is not None and self._context.ok():
                rclpy.shutdown(context=self._context)
            if self._spin_thread is not None:
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

    scale_action = staticmethod(scale_action)
    """``[-1, 1]^2`` -> ``(v, omega)``. See ``action.py``.

    Re-exported rather than reimplemented: ``policy_node.py`` needs the same
    mapping and cannot import this module, so the implementation lives in the
    pure ``action`` module and both callers share it. The alias is kept because
    the env reads better for it and because the scaling is genuinely part of
    this class's contract with CONTRACTS.md.
    """

    def _advance(self, iterations: int) -> None:
        """Step the simulator and track the sim time it now holds.

        Sim time is accumulated here as an integer count of physics iterations
        rather than read back from ``/clock``, because it has to be known
        *before* the data it labels arrives: it is the barrier ``get_obs``
        waits on. Integer nanoseconds, never float seconds -- ``target += 0.05``
        five hundred times accumulates error, and the error only has to exceed a
        sensor stamp by one ULP for the wait to hang on the last step of a long
        episode and look like flakiness.
        """
        self._sim.step(iterations)
        self._sim_ns += iterations * contract.PHYSICS_STEP_NS

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
