"""Phase 4 -- the deployment decision logic. PURE: no ROS, no torch, no clock.

``policy_node.py`` is subscriptions, a timer and a publisher. Everything it
*decides* lives here, for the same reason ``observation.py`` and ``action.py``
are pure: the interesting failures are arithmetic and ordering, and both are
testable in milliseconds on a host with no simulator.

What this replaces
------------------
The obvious deployment node is a callback that runs the policy on whatever
arrived last and publishes the result. That node has no episode, so it cannot
report success, cannot report failure, and cannot be compared against
``evaluate.py`` -- which makes the Phase 4 measurement, the point of the whole
project, impossible to take. So the controller mirrors ``env.step`` exactly:
same goal tolerance, same collision threshold, same 500-step limit, same
step-count bookkeeping. The **only** intended difference between a step here
and a step there is that sim time is no longer under anyone's control.

That is what makes the gap measurable. If this file also changed the
termination rules, the measured difference would be part architecture and part
bookkeeping, with no way to separate them.

The three ways a tick does not run the policy
---------------------------------------------
1. **Watchdog.** No sample, or a sample older than ``WATCHDOG_TIMEOUT``. Stop.
   A free-running robot has no barrier to block on -- ``get_obs`` raising is
   the training-time equivalent, and here the equivalent of raising is
   stopping, because there is nobody to catch the exception and the wheels are
   already turning.
2. **Safety.** Pooled minimum below ``SAFETY_STOP_THRESHOLD``. Stop,
   independently of the policy and of the episode logic.
3. **Terminal.** The episode reached the goal, collided, or ran out of steps.
   Stop and latch until a new goal arrives.

Why the safety gate is checked before the episode logic, not after
------------------------------------------------------------------
``SAFETY_STOP_THRESHOLD`` (0.15 m) is *tighter* than ``COLLISION_THRESHOLD``
(0.18 m) -- contract.py explains that choice: a looser safety threshold would
shadow the avoidance behaviour the policy learned, and then the deployment
numbers would measure the safety layer rather than the policy.

The consequence is that a safety trip is always also a collision, so ordering
the checks the obvious way -- terminate first, then gate -- makes the gate
unreachable code. The point of a safety layer is that it stops the robot when
the rest of the system is wrong, which it cannot do if reaching it requires the
rest of the system to be right. So the episode bookkeeping runs first and
records the collision, and then the gate is evaluated ahead of it: below
0.15 m the outcome is ``COLLISION`` and the reason is ``safety``, naming the
check that would have stopped the wheels even if the bookkeeping above it had
been deleted.

Step counting, and why a watchdog tick does not count
-----------------------------------------------------
Observation index 25 is the fraction of the 500-step budget spent, and the
policy was trained on a counter that advanced once per action *issued*. A tick
that stops the robot issues no action, so it does not advance the counter.
Counting it would make the deployment clock run faster than the training clock
whenever data was late -- the policy would believe it was further into the
episode than it was, and it would believe it most strongly exactly when the
system was under load. That is a sim-to-deployment gap manufactured by the
measuring instrument.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from robot_rl_env import contract
from robot_rl_env.action import STOP, clip_action, scale_action
from robot_rl_env.observation import assemble_observation


class Outcome(StrEnum):
    """How the current episode stands.

    The same partition ``evaluate.py`` reports over: a finished episode is
    exactly one of success, collision or timeout, so the three rates sum to one
    and a policy that stands still cannot hide in the gap between them.
    """

    RUNNING = "running"
    SUCCESS = "success"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    IDLE = "idle"
    """No goal. Distinct from RUNNING so "waiting for a goal" is never reported
    as an episode in progress, and from the terminal outcomes so it is never
    counted as one in a results table."""

    @property
    def is_terminal(self) -> bool:
        return self in (Outcome.SUCCESS, Outcome.COLLISION, Outcome.TIMEOUT)


@dataclass(frozen=True)
class Command:
    """One tick's decision: what to drive, why, and what the episode is doing.

    Carries the *reason* rather than only the velocities because a stopped
    robot is otherwise indistinguishable from a policy that chose to stop, and
    at deployment those two want opposite responses from whoever is watching.
    """

    linear: float
    angular: float
    reason: str
    outcome: Outcome
    step: int
    distance_to_goal: float | None = None
    min_lidar: float | None = None

    @property
    def is_policy(self) -> bool:
        return self.reason == "policy"


class TickStatistics:
    """What the control loop actually did, accumulated over a run.

    The Phase 4 headline is a difference in success rate, and a difference in
    success rate on its own is an observation, not an explanation. These are
    the numbers that turn it into one: how stale the observations were, how
    often the watchdog fired, how often the safety layer had to act. A gap with
    a p95 observation age of 12 ms means something entirely different from the
    same gap with a p95 of 300 ms and 4% watchdog ticks, and no amount of
    staring at the success rates distinguishes them.

    Pure and cumulative. ``policy_node`` feeds it; ``deploy_eval`` reads it.
    """

    def __init__(self):
        self.ticks = 0
        self.reasons: Counter[str] = Counter()
        self._ages: list[float] = []

    def record(self, reason: str, age: float) -> None:
        self.ticks += 1
        self.reasons[reason] += 1
        # An infinite age -- no sample has ever arrived -- is counted as a
        # watchdog tick but not as an age. One inf makes the mean inf and the
        # percentiles meaningless, which would hide the very distribution this
        # exists to show.
        if math.isfinite(age):
            self._ages.append(float(age))

    def fraction(self, reason: str) -> float:
        return self.reasons[reason] / self.ticks if self.ticks else 0.0

    def summary(self) -> dict:
        ages = np.asarray(self._ages, dtype=np.float64)
        return {
            "ticks": self.ticks,
            "policy_fraction": self.fraction("policy"),
            "watchdog_fraction": self.fraction("watchdog"),
            "safety_fraction": self.fraction("safety"),
            "obs_age_mean": float(ages.mean()) if ages.size else float("nan"),
            "obs_age_p95": float(np.percentile(ages, 95)) if ages.size else float("nan"),
            "obs_age_max": float(ages.max()) if ages.size else float("nan"),
        }


class DeploymentController:
    """Runs a policy against live sensor data, with the episode rules of ``env.step``.

    ``policy`` is any callable mapping a ``(26,)`` float32 observation to a
    ``(2,)`` action in ``[-1, 1]`` -- a TorchScript module, an ONNX session
    wrapper, or a lambda in a test. Nothing here knows which, which is what
    keeps ``stable_baselines3`` out of the deployment image.

    Not thread-safe: ``tick`` mutates the episode state and is expected to be
    called from one timer callback. ``policy_node`` gives it its own
    ``MutuallyExclusiveCallbackGroup`` for that reason.
    """

    def __init__(
        self,
        policy,
        *,
        watchdog_timeout: float = contract.WATCHDOG_TIMEOUT,
        max_steps: int = contract.MAX_EPISODE_STEPS,
    ):
        self._policy = policy
        self._watchdog_timeout = float(watchdog_timeout)
        self._max_steps = int(max_steps)

        self._goal: tuple[float, float] | None = None
        self._outcome = Outcome.IDLE
        self._step = 0
        self._prev_action = STOP.copy()

    # --- episode control ------------------------------------------------------

    def set_goal(self, goal_xy: tuple[float, float]) -> None:
        """Start a new episode toward ``goal_xy``, **in the odom frame**.

        Odom, not world and not map: it is the only frame the robot actually
        has, and it is the frame ``assemble_observation`` was trained on. A goal
        handed over in world coordinates would be silently wrong by however far
        odometry has drifted -- which is small at the start of a run and grows,
        so it presents as the policy degrading over time.
        """
        self._goal = (float(goal_xy[0]), float(goal_xy[1]))
        self._outcome = Outcome.RUNNING
        self._step = 0
        # Not carried over from the previous episode: index 23/24 mean "what
        # the wheels were last told", and across a goal change the honest
        # answer is the stop that ended the last episode.
        self._prev_action = STOP.copy()

    def clear_goal(self) -> None:
        """Drop the goal and stop. Used when a new goal is rejected."""
        self._goal = None
        self._outcome = Outcome.IDLE
        self._step = 0
        self._prev_action = STOP.copy()

    @property
    def goal(self) -> tuple[float, float] | None:
        return self._goal

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def step(self) -> int:
        """Actions issued in the current episode. See the module docstring."""
        return self._step

    # --- the tick -------------------------------------------------------------

    def tick(
        self,
        *,
        pooled: np.ndarray | None,
        robot_xy: tuple[float, float] | None,
        robot_yaw: float | None,
        age: float,
    ) -> Command:
        """Decide this control period's command.

        ``pooled`` is the min-pooled scan from ``observation.pool_ranges`` --
        the deployment node pools in its subscriber callback exactly as
        ``observation_node`` does, so the array here is the same one the
        training env would have built. ``age`` is seconds since that sample
        arrived, measured on whatever clock the caller trusts; the node uses a
        monotonic wall clock, because a sim clock that stops ticking is
        precisely the condition the watchdog exists to catch.
        """
        if pooled is None or robot_xy is None or robot_yaw is None:
            return self._stop("watchdog")
        if age > self._watchdog_timeout:
            return self._stop("watchdog", min_lidar=float(pooled.min()))

        min_lidar = float(pooled.min())

        distance = None if self._goal is None else math.dist(robot_xy, self._goal)

        # Episode bookkeeping, in the same order and against the same
        # thresholds as env.step. Done before any gate, so the outcome recorded
        # is the same one the training env would have recorded no matter which
        # check ends up stopping the wheels.
        if self._outcome is Outcome.RUNNING:
            if distance < contract.GOAL_TOLERANCE:
                self._outcome = Outcome.SUCCESS
            elif min_lidar < contract.COLLISION_THRESHOLD:
                self._outcome = Outcome.COLLISION
            elif self._step >= self._max_steps:
                self._outcome = Outcome.TIMEOUT

        # The safety layer, ahead of everything except the watchdog and
        # deliberately *not* downstream of the episode logic -- that is the
        # whole of its job. It has to stop the robot when the code above is
        # wrong, so it cannot be reached through the code above.
        if min_lidar < contract.SAFETY_STOP_THRESHOLD:
            return self._stop("safety", distance=distance, min_lidar=min_lidar)
        if self._outcome.is_terminal:
            return self._stop(self._outcome.value, distance=distance, min_lidar=min_lidar)
        if self._goal is None:
            return self._stop("no_goal", min_lidar=min_lidar)

        obs = assemble_observation(
            None,
            robot_xy,
            robot_yaw,
            self._goal,
            self._prev_action,
            self._step,
            pooled=pooled,
        )
        action = clip_action(self._policy(obs))
        self._prev_action = action
        self._step += 1

        linear, angular = scale_action(action)
        return Command(
            linear=linear,
            angular=angular,
            reason="policy",
            outcome=self._outcome,
            step=self._step,
            distance_to_goal=distance,
            min_lidar=min_lidar,
        )

    def _stop(
        self,
        reason: str,
        *,
        distance: float | None = None,
        min_lidar: float | None = None,
    ) -> Command:
        # The observation must describe what the wheels were actually told, so a
        # stop is recorded as the previous action even though no policy ran. The
        # alternative -- leaving the last policy action in place -- tells the
        # next observation the robot is still driving at a speed it was ordered
        # to abandon.
        self._prev_action = STOP.copy()
        linear, angular = scale_action(STOP)
        return Command(
            linear=linear,
            angular=angular,
            reason=reason,
            outcome=self._outcome,
            step=self._step,
            distance_to_goal=distance,
            min_lidar=min_lidar,
        )
