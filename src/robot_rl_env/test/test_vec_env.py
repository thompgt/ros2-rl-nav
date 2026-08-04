"""Env construction: the fixed-episode wrapper and simulator lifetime.

No Gazebo. What is checked here is the wiring that decides *which* episode an
evaluation plays and *when* a simulator process dies -- both of which are
invisible in a training log until they are wrong.
"""

import numpy as np
import pytest

from robot_rl_env import contract, eval_set

# The wrappers need gymnasium but not rclpy and not SB3, which is the point:
# they are the part of env construction that can be wrong without a simulator.
gym = pytest.importorskip("gymnasium", reason="gymnasium is not installed on this host")

from robot_rl_env.vec_env import FixedEpisodeCycle, SimulatorLifetime  # noqa: E402


class _RecordingEnv(gym.Env):
    """Records the options each reset was given."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (contract.OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACT_DIM,), dtype=np.float32)

    def __init__(self):
        self.resets = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.resets.append(options)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        self.closed = True


class _FakeSimulator:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_the_cycle_plays_every_episode_once_in_order():
    """So one evaluation of len(episodes) covers the set exactly, and two
    evaluations at different points in training see the same sequence."""
    episodes = eval_set.episodes(5)
    inner = _RecordingEnv()
    env = FixedEpisodeCycle(inner, episodes)

    for _ in range(5):
        env.reset()

    played = [(o["start"], o["goal"]) for o in inner.resets]
    assert played == [(e.start, e.goal) for e in episodes]


def test_the_cycle_wraps():
    episodes = eval_set.episodes(3)
    inner = _RecordingEnv()
    env = FixedEpisodeCycle(inner, episodes)
    for _ in range(7):
        env.reset()
    assert inner.resets[3] == inner.resets[0]
    assert inner.resets[6] == inner.resets[0]


def test_the_cycle_ignores_caller_options():
    """EvalCallback resets with no options; anything that did arrive would be
    a stray goal_radius, which collides with a fixed episode and raises in the
    middle of an evaluation."""
    inner = _RecordingEnv()
    env = FixedEpisodeCycle(inner, eval_set.episodes(1))
    env.reset(options={"goal_radius": 2.0})
    assert "goal_radius" not in inner.resets[0]


def test_an_empty_episode_list_raises():
    with pytest.raises(ValueError):
        FixedEpisodeCycle(_RecordingEnv(), ())


def test_closing_the_env_stops_the_simulator():
    """SB3 closes envs and knows nothing about the processes behind them. An
    orphaned gz sim holds its partition, and the next run's worker attaches to
    a world still carrying the previous run's robot."""
    inner, sim = _RecordingEnv(), _FakeSimulator()
    SimulatorLifetime(inner, sim).close()
    assert inner.closed and sim.stopped


def test_the_simulator_is_stopped_even_if_the_env_close_raises():
    """A leaked simulator is worse than a noisy shutdown: it survives the run
    and breaks the next one."""

    class _Angry(_RecordingEnv):
        def close(self):
            raise RuntimeError("rclpy context already destroyed")

    sim = _FakeSimulator()
    with pytest.raises(RuntimeError):
        SimulatorLifetime(_Angry(), sim).close()
    assert sim.stopped


def test_the_wrapper_stack_exposes_the_curriculum_hook():
    """The curriculum reaches the env through VecEnv.env_method, which resolves
    the name through the wrapper stack. A wrapper that shadowed or blocked it
    would leave every worker sampling the full arena, silently."""

    class _WithHook(_RecordingEnv):
        def set_curriculum_radius(self, radius):
            self.radius = radius

    inner = _WithHook()
    env = SimulatorLifetime(FixedEpisodeCycle(inner, eval_set.episodes(1)), _FakeSimulator())
    env.get_wrapper_attr("set_curriculum_radius")(3.0)
    assert inner.radius == 3.0
