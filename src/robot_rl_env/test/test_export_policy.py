"""Policy export. Needs torch; the round-trip cases also need stable-baselines3.

The question these answer is the one nothing downstream can: *is the file the
robot runs the policy that was trained?* A wrong answer here looks exactly like
a policy that deploys badly, which is the measurement Phase 4 exists to take.
"""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this host")

from robot_rl_env import contract  # noqa: E402
from robot_rl_env.export_policy import (  # noqa: E402
    TOLERANCE,
    ExportedPolicy,
    default_output,
    find_vecnormalize,
    load_policy,
    observation_statistics,
    trace,
    verify,
)


class FirstTwo(torch.nn.Module):
    """A "policy" that hands back the first two elements of its observation.

    Lets the normalization and clipping be asserted against arithmetic done by
    hand, which a trained network could never support.
    """

    def _predict(self, obs, deterministic: bool = True):
        return obs[:, : contract.ACT_DIM]


class Amplify(torch.nn.Module):
    """A policy that leaves the action envelope, like an untrained PPO Gaussian."""

    def _predict(self, obs, deterministic: bool = True):
        return obs[:, : contract.ACT_DIM] * 10.0


class FakeRunningMeanStd:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var


class FakeVecNormalize:
    """Enough of a ``VecNormalize`` to be pickled and read back.

    Module-level so pickle can find it, and hand-rolled rather than a real
    ``VecNormalize`` so this test does not require SB3 just to check a
    ``getattr``.
    """

    def __init__(self, norm_obs, mean=None, var=None, clip_obs=10.0, epsilon=1e-8):
        self.norm_obs = norm_obs
        self.obs_rms = FakeRunningMeanStd(mean, var)
        self.clip_obs = clip_obs
        self.epsilon = epsilon


# --- finding the statistics ---------------------------------------------------

def test_vecnormalize_is_found_at_the_run_root_from_a_checkpoint(tmp_path):
    """train.py writes vecnormalize.pkl at the run root while the model lands in
    best/. A sibling-only lookup misses it for exactly the checkpoint anyone
    would deploy."""
    run = tmp_path / "runs" / "sac-seed0"
    (run / "best").mkdir(parents=True)
    pkl = run / "vecnormalize.pkl"
    pkl.write_bytes(b"")
    assert find_vecnormalize(run / "best" / "best_model.zip") == pkl


def test_vecnormalize_is_found_beside_the_model_too(tmp_path):
    tmp_path.joinpath("vecnormalize.pkl").write_bytes(b"")
    assert find_vecnormalize(tmp_path / "final.zip") == tmp_path / "vecnormalize.pkl"


def test_a_missing_vecnormalize_is_reported_as_missing(tmp_path):
    (tmp_path / "best").mkdir()
    assert find_vecnormalize(tmp_path / "best" / "best_model.zip") is None


def test_the_search_does_not_wander_into_another_run(tmp_path):
    """Walking up until something is found would happily attach run A's
    statistics to run B's policy, and produce a plausible-looking export."""
    runs = tmp_path / "runs"
    (runs / "sac-seed0" / "best").mkdir(parents=True)
    (runs / "vecnormalize.pkl").write_bytes(b"")  # two levels up: too far
    assert find_vecnormalize(runs / "sac-seed0" / "best" / "best_model.zip") is None


def test_statistics_are_none_when_training_did_not_normalize(tmp_path):
    """This project's configuration -- hyperparams.NORMALIZE_OBS is False."""
    pkl = tmp_path / "vecnormalize.pkl"
    pkl.write_bytes(pickle.dumps(FakeVecNormalize(norm_obs=False)))
    assert observation_statistics(pkl) is None


def test_statistics_are_read_when_training_did_normalize(tmp_path):
    mean = np.arange(contract.OBS_DIM, dtype=np.float64)
    var = np.ones(contract.OBS_DIM) * 4.0
    pkl = tmp_path / "vecnormalize.pkl"
    pkl.write_bytes(pickle.dumps(FakeVecNormalize(True, mean, var, clip_obs=7.0)))

    read_mean, read_var, clip_obs, _ = observation_statistics(pkl)
    assert read_mean.dtype == np.float32
    assert read_mean == pytest.approx(mean)
    assert read_var == pytest.approx(var)
    assert clip_obs == 7.0


# --- what the exported module computes ----------------------------------------

def test_an_unnormalized_export_passes_the_observation_through():
    exported = ExportedPolicy(FirstTwo())
    obs = torch.full((1, contract.OBS_DIM), 0.5)
    assert exported(obs).numpy()[0] == pytest.approx([0.5, 0.5])


def test_normalization_is_baked_into_the_module():
    """The failure this prevents: a policy trained on normalized observations,
    deployed on raw ones. It loads, it runs, and it is several standard
    deviations outside anything it ever saw."""
    mean = np.full(contract.OBS_DIM, 0.25, dtype=np.float32)
    var = np.full(contract.OBS_DIM, 4.0, dtype=np.float32)
    exported = ExportedPolicy(FirstTwo(), obs_mean=mean, obs_var=var, clip_obs=10.0)

    obs = torch.full((1, contract.OBS_DIM), 0.75)
    expected = (0.75 - 0.25) / np.sqrt(4.0 + 1e-8)  # 0.25
    assert exported(obs).numpy()[0] == pytest.approx([expected, expected], abs=1e-6)


def test_the_normalization_clip_is_applied():
    mean = np.zeros(contract.OBS_DIM, dtype=np.float32)
    var = np.full(contract.OBS_DIM, 1e-6, dtype=np.float32)  # tiny: huge z-scores
    exported = ExportedPolicy(FirstTwo(), obs_mean=mean, obs_var=var, clip_obs=0.5)
    obs = torch.full((1, contract.OBS_DIM), 1.0)
    # Clipped to 0.5 by clip_obs, then within the action envelope anyway.
    assert exported(obs).numpy()[0] == pytest.approx([0.5, 0.5])


def test_the_action_is_clipped_to_the_contract_envelope():
    """SAC's tanh makes this a no-op; PPO's Gaussian mean does not, and an
    unclipped export would command velocities outside the contract."""
    exported = ExportedPolicy(Amplify())
    obs = torch.full((1, contract.OBS_DIM), 0.5)  # -> 5.0 before clipping
    assert exported(obs).numpy()[0] == pytest.approx([1.0, 1.0])
    assert exported(-obs).numpy()[0] == pytest.approx([-1.0, -1.0])


# --- tracing and loading ------------------------------------------------------

def test_a_traced_module_survives_a_save_load_round_trip(tmp_path):
    traced = trace(ExportedPolicy(FirstTwo()))
    path = tmp_path / "policy.pt"
    traced.save(str(path))

    policy = load_policy(path)
    obs = np.full(contract.OBS_DIM, 0.5, dtype=np.float32)
    action = policy(obs)
    assert action.shape == (contract.ACT_DIM,)
    assert action == pytest.approx([0.5, 0.5])


def test_load_policy_accepts_the_flat_observation_the_controller_produces():
    """``assemble_observation`` returns ``(26,)``; the traced graph wants a
    batch dimension. If the loader did not add one, every deployment tick would
    fail on a shape error -- which is at least loud, but only reachable with a
    simulator running."""
    with_batch = trace(ExportedPolicy(FirstTwo()))
    obs = np.zeros(contract.OBS_DIM, dtype=np.float32)
    assert with_batch(torch.from_numpy(obs).reshape(1, -1)).shape == (1, contract.ACT_DIM)


def test_the_traced_graph_is_not_pinned_to_the_batch_size_it_saw():
    """Traced with a batch of 1. ``verify`` then runs it on 512 at once, so a
    graph that had baked in the batch dimension would fail the verification
    step rather than the deployment."""
    traced = trace(ExportedPolicy(FirstTwo()))
    batch = torch.zeros(37, contract.OBS_DIM)
    assert traced(batch).shape == (37, contract.ACT_DIM)


# --- agreement with the trained policy ----------------------------------------

@pytest.mark.parametrize("algorithm", ["sac", "ppo"])
def test_the_export_agrees_with_the_policy_it_came_from(algorithm):
    """The check the CLI refuses to write without. Tracing records whatever the
    example input happened to exercise, so agreement is asserted over the whole
    observation box rather than assumed from the fact that it ran."""
    sb3 = pytest.importorskip("stable_baselines3", reason="SB3 is not installed on this host")
    gym = pytest.importorskip("gymnasium")

    class Fake(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (contract.OBS_DIM,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (contract.ACT_DIM,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    model = {"sac": sb3.SAC, "ppo": sb3.PPO}[algorithm]("MlpPolicy", Fake(), device="cpu", seed=0)
    assert verify(trace(ExportedPolicy(model.policy)), model) <= TOLERANCE


def test_verification_actually_fails_on_a_policy_that_is_not_the_model():
    """A tolerance check nothing can fail is decoration. This confirms the
    comparison has teeth before it is trusted to gate a write."""
    sb3 = pytest.importorskip("stable_baselines3", reason="SB3 is not installed on this host")
    gym = pytest.importorskip("gymnasium")

    class Fake(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (contract.OBS_DIM,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (contract.ACT_DIM,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    model = sb3.SAC("MlpPolicy", Fake(), device="cpu", seed=0)
    assert verify(trace(ExportedPolicy(FirstTwo())), model) > TOLERANCE


# --- output paths -------------------------------------------------------------

def test_the_export_lands_beside_the_run_not_beside_the_checkpoint():
    """best/ is an implementation detail of EvalCallback; the exported policy is
    the run's deliverable."""
    assert default_output("runs/sac-seed0/best/best_model.zip").as_posix().endswith(
        "runs/sac-seed0/policy.pt"
    )
    assert default_output("runs/ppo-seed1/checkpoints/ppo_50000_steps.zip").as_posix().endswith(
        "runs/ppo-seed1/policy.pt"
    )
    assert default_output("runs/sac-seed0/final.zip").as_posix().endswith(
        "runs/sac-seed0/policy.pt"
    )
