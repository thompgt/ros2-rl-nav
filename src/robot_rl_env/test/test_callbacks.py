"""Curriculum and logging callbacks, against a stub env. No simulator.

The curriculum is the piece most likely to fail *quietly*: every wrong version
of it still produces a training run, a curve, and a saved policy. It just
trains on the wrong task distribution. So the schedule is tested directly --
what the radius does in response to a sequence of episode outcomes -- rather
than by watching a run and forming an impression.
"""

import numpy as np
import pytest

from robot_rl_env import contract, hyperparams

gym = pytest.importorskip("gymnasium", reason="gymnasium is not installed on this host")
pytest.importorskip("stable_baselines3", reason="SB3 is not installed on this host")

from robot_rl_env.callbacks import CurriculumCallback, SuccessRateCallback  # noqa: E402


class _StubEnv(gym.Env):
    """The contract's spaces and the curriculum hook. Nothing else.

    Records every radius it is handed, which is what the callback is judged on.
    """

    observation_space = gym.spaces.Box(-1.0, 1.0, (contract.OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACT_DIM,), dtype=np.float32)

    def __init__(self):
        self.radii = []

    def set_curriculum_radius(self, radius):
        self.radii.append(radius)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


class _StubModel:
    """Enough of a BaseAlgorithm for a callback to bind to.

    ``training_env`` and ``logger`` are read-only properties on ``BaseCallback``
    that delegate to the model, so a callback cannot be wired up by assigning
    to them -- it has to be given something model-shaped, exactly as
    ``init_callback`` does.
    """

    def __init__(self, env, logger):
        self._env = env
        self.logger = logger

    def get_env(self):
        return self._env


def _bind(callback, n_envs=1):
    """Attach a callback to a vec env of stubs, as SB3 would."""
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.vec_env import DummyVecEnv

    envs = [_StubEnv() for _ in range(n_envs)]
    vec = DummyVecEnv([lambda e=e: e for e in envs])
    callback.init_callback(_StubModel(vec, Logger(folder=None, output_formats=[])))
    callback.locals = {"infos": []}
    return callback, envs


def _finish_episodes(callback, successes):
    """Feed end-of-episode infos, exactly as Monitor emits them."""
    for success in successes:
        callback.locals = {
            "infos": [{"episode": {"r": 0.0, "l": 1}, "is_success": success, "collided": False}]
        }
        callback._on_step()


def test_the_curriculum_starts_narrow_on_every_worker():
    """A worker that never received a radius samples the full arena, and one
    worker's episodes are then drawn from a different distribution than the
    other three -- into a single shared replay buffer."""
    callback, envs = _bind(CurriculumCallback(verbose=0), n_envs=4)
    callback._on_training_start()
    for env in envs:
        assert env.radii == [hyperparams.CURRICULUM_START_RADIUS]


def test_the_radius_expands_only_once_the_window_is_full():
    """Three lucky episodes are not evidence. Expanding on a partial window is
    how a curriculum reaches maximum radius in its first hundred steps."""
    window = 20
    callback, envs = _bind(CurriculumCallback(window=window, verbose=0))
    callback._on_training_start()

    _finish_episodes(callback, [True] * (window - 1))
    assert callback.radius == hyperparams.CURRICULUM_START_RADIUS

    _finish_episodes(callback, [True])
    assert callback.radius == pytest.approx(
        hyperparams.CURRICULUM_START_RADIUS + hyperparams.CURRICULUM_INCREMENT
    )


def test_expansion_clears_the_window():
    """Otherwise the episodes that earned one expansion earn the next one on
    the very next step, and the radius walks straight to its maximum."""
    window = 10
    callback, envs = _bind(CurriculumCallback(window=window, increment=1.0, verbose=0))
    callback._on_training_start()

    _finish_episodes(callback, [True] * window)
    after_first = callback.radius

    _finish_episodes(callback, [True])  # one episode: window is nowhere near full
    assert callback.radius == after_first


def test_a_failing_policy_does_not_advance():
    window = 10
    callback, _ = _bind(CurriculumCallback(window=window, verbose=0))
    callback._on_training_start()
    _finish_episodes(callback, [False, True] * window)  # 50%, below the threshold
    assert callback.radius == hyperparams.CURRICULUM_START_RADIUS


def test_the_radius_stops_at_the_maximum():
    """And stops calling env_method once there, so a long run does not spend
    the rest of itself broadcasting the same number."""
    callback, envs = _bind(
        CurriculumCallback(start_radius=2.0, max_radius=3.0, increment=1.0, window=5, verbose=0)
    )
    callback._on_training_start()
    for _ in range(20):
        _finish_episodes(callback, [True] * 5)
    assert callback.radius == 3.0
    assert envs[0].radii == [2.0, 3.0]


def test_steps_inside_an_episode_are_not_counted_as_outcomes():
    """Reading is_success on every step counts a 500-step failure 500 times
    and a 40-step success once, which pins the measured rate near zero and
    stalls the curriculum for the whole run."""
    callback, _ = _bind(CurriculumCallback(window=5, verbose=0))
    callback._on_training_start()
    for _ in range(50):
        callback.locals = {"infos": [{"is_success": False}]}  # no "episode" key
        callback._on_step()
    _finish_episodes(callback, [True] * 5)
    assert callback.radius > hyperparams.CURRICULUM_START_RADIUS


def test_success_rate_callback_separates_the_three_outcomes():
    """A run that is mostly truncations is a policy that learned to stand
    still, and it looks identical to a safe one on the collision curve."""
    callback, _ = _bind(SuccessRateCallback(window=10))
    recorded = {}
    callback.logger.record = lambda key, value, *a, **k: recorded.__setitem__(key, value)

    callback.locals = {
        "infos": [
            {"episode": {}, "is_success": True, "collided": False},
            {"episode": {}, "is_success": False, "collided": True},
            {"episode": {}, "is_success": False, "collided": False},
            {"episode": {}, "is_success": False, "collided": False},
        ]
    }
    callback._on_step()

    assert recorded["rollout/success_rate"] == pytest.approx(0.25)
    assert recorded["rollout/collision_rate"] == pytest.approx(0.25)
    assert recorded["rollout/timeout_rate"] == pytest.approx(0.50)
