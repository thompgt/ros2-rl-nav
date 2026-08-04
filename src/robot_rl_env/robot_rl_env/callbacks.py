"""Training callbacks: the goal-distance curriculum, and success logging.

WORKPLAN names "no exploration signal" as a top-three cause of a failed first
run: with goals sampled across a 10 m arena, a randomly initialized policy may
never once reach a goal, and neither SAC nor PPO can learn from a terminal
bonus it has never observed. The curriculum starts goals close enough that
random driving succeeds sometimes, and expands as the policy earns it.

Kept out of ``env.py`` deliberately. The environment implements CONTRACTS.md;
the schedule for exercising it is a training decision, and mixing the two makes
the env's behaviour depend on how far along a training run happens to be.
"""

from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from robot_rl_env import hyperparams


class CurriculumCallback(BaseCallback):
    """Expand the sampled goal radius as the rolling success rate crosses a
    threshold.

    Success is read from ``info["is_success"]`` on episode boundaries, which is
    the same flag ``EvalCallback`` uses -- so the curriculum and the reported
    metric cannot disagree about what a success is.

    The window is cleared on every expansion. Without that, the episodes that
    triggered one expansion remain in the window and trigger the next one
    immediately, walking the radius to its maximum in a few hundred steps and
    reproducing the very distribution the curriculum exists to avoid. That
    failure is invisible in the logs unless the radius itself is logged, which
    is why it is.
    """

    def __init__(
        self,
        *,
        start_radius: float = hyperparams.CURRICULUM_START_RADIUS,
        max_radius: float = hyperparams.CURRICULUM_MAX_RADIUS,
        increment: float = hyperparams.CURRICULUM_INCREMENT,
        threshold: float = hyperparams.CURRICULUM_SUCCESS_THRESHOLD,
        window: int = hyperparams.CURRICULUM_WINDOW,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_radius = start_radius
        self.max_radius = max_radius
        self.increment = increment
        self.threshold = threshold
        self.radius = start_radius
        self._outcomes: deque[bool] = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self._apply(self.start_radius)

    def _apply(self, radius: float) -> None:
        self.radius = radius
        # Every worker, not just the one that finished an episode: a curriculum
        # that advanced per-worker would have four different task
        # distributions feeding one replay buffer.
        self.training_env.env_method("set_curriculum_radius", radius)
        if self.verbose:
            print(f"[curriculum] goal radius -> {radius:.2f} m")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # "episode" is Monitor's end-of-episode key. Reading is_success on
            # every step instead would count a 500-step failure 500 times and
            # a 40-step success once, which biases the rate toward zero by an
            # order of magnitude and stalls the curriculum permanently.
            if "episode" not in info:
                continue
            self._outcomes.append(bool(info.get("is_success", False)))

        if len(self._outcomes) == self._outcomes.maxlen:
            rate = sum(self._outcomes) / len(self._outcomes)
            self.logger.record("curriculum/success_rate", rate)
            if rate >= self.threshold and self.radius < self.max_radius:
                self._apply(min(self.radius + self.increment, self.max_radius))
                self._outcomes.clear()

        self.logger.record("curriculum/goal_radius", self.radius)
        return True


class SuccessRateCallback(BaseCallback):
    """Log the training-time success and collision rates over a rolling window.

    ``ep_rew_mean`` is the only outcome metric SB3 logs by default, and on this
    reward it is close to unreadable: progress is bounded by the start-goal
    distance, so a run whose episodes got *harder* under the curriculum shows a
    falling mean reward while improving. Success and collision rates are the
    two curves that mean what they appear to mean.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._successes: deque[bool] = deque(maxlen=window)
        self._collisions: deque[bool] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._successes.append(bool(info.get("is_success", False)))
            self._collisions.append(bool(info.get("collided", False)))

        if self._successes:
            n = len(self._successes)
            self.logger.record("rollout/success_rate", sum(self._successes) / n)
            self.logger.record("rollout/collision_rate", sum(self._collisions) / n)
            # Everything else is a success, a collision, or a truncation. The
            # third is the one nobody thinks to plot, and a run that is 90%
            # truncations is a policy that has learned to survive by not
            # moving -- which reads as "not colliding" on every other curve.
            outcomes = zip(self._successes, self._collisions, strict=True)
            self.logger.record(
                "rollout/timeout_rate",
                sum(not s and not c for s, c in outcomes) / n,
            )
        return True
