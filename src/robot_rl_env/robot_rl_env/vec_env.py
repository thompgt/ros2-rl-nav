"""Constructing the vectorized training env and the evaluation env.

Each worker owns a whole simulator: a ``gz sim -s`` process and two bridge
nodes, isolated by ``sim_launcher.worker_environment``. That is why the
simulator is launched *inside* the worker factory rather than beside it — under
``SubprocVecEnv`` the factory runs in the worker's own process, so the
environment variables it sets are process-local and the simulator it starts is
the one that process talks to.

The lifetime rule that follows is easy to get wrong: the simulator must die
with the env. SB3 closes envs; it knows nothing about the processes behind
them. An orphaned ``gz sim`` holds its partition, and the next run's worker
attaches to a world already carrying the previous run's robot — which produces
a training curve, just not one of anything.
"""

from __future__ import annotations

import gymnasium as gym

from robot_rl_env import eval_set, hyperparams, sim_launcher

# RobotNavEnv is imported inside make_env, not here. It pulls in rclpy, which
# does not exist on the Windows host or on the CI fast path -- and the wrappers
# below are exactly the logic worth testing there, since neither of them needs
# a simulator to be wrong.


class SimulatorLifetime(gym.Wrapper):
    """Ties a launched simulator to the env that uses it.

    A wrapper rather than a ``finally`` in the factory: the factory returns long
    before the env is closed, and nothing else in the SB3 call graph has a
    reference to the process.
    """

    def __init__(self, env: gym.Env, simulator: sim_launcher.Simulator):
        super().__init__(env)
        self.simulator = simulator

    def close(self):
        try:
            super().close()
        finally:
            # Ordered: the env's rclpy context is torn down first, so the
            # bridges are not killed out from under an in-flight service call.
            self.simulator.stop()


class FixedEpisodeCycle(gym.Wrapper):
    """Replays a fixed list of episodes, one per ``reset()``, in order.

    Evaluation has to be identical across runs, and SB3's ``EvalCallback``
    calls ``reset()`` with no ``options`` — so the episode cannot be selected
    from outside. This wrapper supplies it.

    Cycling rather than sampling, and in order rather than shuffled: an
    evaluation of ``n_eval_episodes`` equal to ``len(episodes)`` then covers the
    set exactly once, and two evaluations at different points in training see
    the same episodes in the same sequence. A shuffled or sampled variant makes
    consecutive eval points differ by which episodes they drew, which is
    indistinguishable in the logs from the policy changing.
    """

    def __init__(self, env: gym.Env, episodes: tuple[eval_set.Episode, ...]):
        super().__init__(env)
        if not episodes:
            raise ValueError("FixedEpisodeCycle needs at least one episode")
        self.episodes = episodes
        self._next = 0

    def reset(self, *, seed=None, options=None):
        episode = self.episodes[self._next % len(self.episodes)]
        self._next += 1
        # The caller's options are ignored rather than merged: a merge would
        # let a stray goal_radius from elsewhere in SB3 collide with the fixed
        # episode and raise mid-evaluation.
        return self.env.reset(seed=seed, options=episode.as_reset_options())

    @property
    def episode_index(self) -> int:
        """Index of the episode the *next* reset will play. Lets evaluate.py
        label per-episode results without duplicating the cycling logic."""
        return self._next % len(self.episodes)


def make_env(index: int = 0, *, episodes: tuple[eval_set.Episode, ...] | None = None):
    """Return a zero-argument factory: its own simulator, then an env on it.

    ``episodes`` makes it an evaluation env replaying that fixed set;
    ``None`` makes it a training env that samples.

    Deferred into a closure because that is what ``SubprocVecEnv`` requires,
    and because launching four simulators eagerly in the parent process would
    put all four on the parent's ROS domain.
    """

    def _init() -> gym.Env:
        from robot_rl_env.env import RobotNavEnv

        simulator = sim_launcher.Simulator(index=index).start()
        try:
            env: gym.Env = RobotNavEnv()
            if episodes is not None:
                env = FixedEpisodeCycle(env, episodes)
            return SimulatorLifetime(env, simulator)
        except BaseException:
            # RobotNavEnv's constructor can fail on a bridge that came up
            # incompletely. Without this the simulator survives its env and
            # holds the partition for the rest of the session.
            simulator.stop()
            raise

    return _init


def make_training_env(n_envs: int = hyperparams.N_ENVS, *, monitor_dir: str | None = None):
    """The vectorized training env, one simulator per worker.

    ``n_envs=1`` uses ``DummyVecEnv``: a single worker in the parent process is
    easier to debug, keeps ``ROS_DOMAIN_ID=42`` (so ``make shell`` can watch
    it), and avoids paying for a subprocess that isolates nothing.
    """
    # Imported here rather than at module scope so this module stays importable
    # -- and testable -- on a host with no stable-baselines3.
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if n_envs < 1:
        raise ValueError(f"n_envs must be at least 1, got {n_envs}")

    def factory(i: int):
        build = make_env(i)

        def _init():
            # Monitor inside the vec env, so ep_rew_mean and ep_len_mean are
            # recorded per episode rather than per rollout segment.
            return Monitor(build(), filename=None if monitor_dir is None else f"{monitor_dir}/{i}")

        return _init

    factories = [factory(i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    # "spawn" rather than the default fork: forking a process that already
    # holds an rclpy context copies DDS participants into the child, which
    # then fight over the same GUIDs. The parent has no env yet at this point,
    # but train.py imports torch, and forking a torch process is its own
    # well-known source of hangs.
    return SubprocVecEnv(factories, start_method="spawn")


def make_eval_env(episodes: tuple[eval_set.Episode, ...], *, index: int | None = None):
    """The evaluation env: one simulator, fixed episodes, never curriculum-capped.

    Given its own worker index past the training workers by default, so it does
    not share a domain with a training simulator -- an eval env on the same bus
    as worker 0 would read worker 0's scans.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    slot = hyperparams.N_ENVS if index is None else index
    return DummyVecEnv([lambda: Monitor(make_env(slot, episodes=episodes)())])
