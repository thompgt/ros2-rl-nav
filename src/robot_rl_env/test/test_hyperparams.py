"""Training hyperparameters. Pure, except where SB3 is available to check against.

A typo in a hyperparameter name does not raise anywhere useful: SB3 accepts
``**kwargs`` at several layers, and the usual symptom is a run that trains for
three hours with the default value of whatever was misspelled. These tests turn
that into a collection error.
"""

import inspect

import pytest

from robot_rl_env import contract, eval_set, hyperparams


def test_for_algorithm_resolves_both_names_case_insensitively():
    assert hyperparams.for_algorithm("sac") == hyperparams.for_algorithm("SAC")
    assert hyperparams.for_algorithm("  Ppo ") == hyperparams.for_algorithm("ppo")


def test_unknown_algorithm_raises():
    for bad in ("td3", "", "sacc"):
        with pytest.raises(ValueError, match="unknown algorithm"):
            hyperparams.for_algorithm(bad)
        with pytest.raises(ValueError, match="unknown algorithm"):
            hyperparams.timesteps_for(bad)


def test_the_returned_kwargs_are_a_copy_all_the_way_down():
    """Training SAC then PPO in one process must not let the first run edit the
    second's configuration. SB3 mutates policy_kwargs in place, so a shallow
    copy is not enough -- and the failure would appear only in runs that train
    both."""
    first = hyperparams.for_algorithm("sac")
    first["learning_rate"] = 999.0
    first["policy_kwargs"]["net_arch"].append(4096)
    first["policy_kwargs"]["extra"] = True

    second = hyperparams.for_algorithm("sac")
    assert second["learning_rate"] == hyperparams.SAC_KWARGS["learning_rate"]
    assert second["policy_kwargs"]["net_arch"] == hyperparams.NET_ARCH
    assert "extra" not in second["policy_kwargs"]
    assert hyperparams.NET_ARCH == [256, 256]  # the module constant itself


def test_ppo_gets_a_larger_budget_than_sac():
    """On-policy PPO discards every transition after one rollout. Equal budgets
    would measure sample efficiency and report it as algorithm quality."""
    assert hyperparams.timesteps_for("ppo") > hyperparams.timesteps_for("sac")


def test_both_algorithms_share_the_discount_and_the_network():
    """Otherwise the SAC-vs-PPO comparison is confounded by capacity and
    effective horizon, and the results table means nothing."""
    assert hyperparams.SAC_KWARGS["gamma"] == hyperparams.PPO_KWARGS["gamma"]
    assert hyperparams.SAC_KWARGS["policy_kwargs"]["net_arch"] == hyperparams.NET_ARCH
    assert hyperparams.PPO_KWARGS["policy_kwargs"]["net_arch"]["pi"] == hyperparams.NET_ARCH
    assert hyperparams.PPO_KWARGS["policy_kwargs"]["net_arch"]["vf"] == hyperparams.NET_ARCH


def test_observation_normalization_stays_off():
    """Guards a load-bearing decision, not a preference.

    norm_obs=True puts a running mean/variance into the policy's input
    pipeline, which policy_node.py would have to reproduce at deployment --
    a second preprocessing path outside assemble_observation, which is exactly
    what CONTRACTS.md's shared-code requirement forbids. It fails silently.
    """
    assert hyperparams.NORMALIZE_OBS is False


def test_the_replay_buffer_outlives_the_training_budget():
    """So the earliest collision transitions are never evicted. A policy that
    has learned not to collide stops producing them, and a FIFO buffer would
    then forget why it stopped."""
    assert hyperparams.SAC_KWARGS["buffer_size"] >= hyperparams.timesteps_for("sac")


def test_learning_starts_covers_several_full_episodes():
    assert hyperparams.SAC_KWARGS["learning_starts"] >= 5 * contract.MAX_EPISODE_STEPS


def test_the_curriculum_ends_where_the_evaluation_set_ends():
    """Expanding past the eval cap trains on episodes the evaluation never
    measures, and eventually on ones the truncation limit makes impossible."""
    assert hyperparams.CURRICULUM_START_RADIUS >= contract.MIN_START_GOAL_DISTANCE
    assert hyperparams.CURRICULUM_MAX_RADIUS == eval_set.MAX_EVAL_DISTANCE
    assert hyperparams.CURRICULUM_START_RADIUS < hyperparams.CURRICULUM_MAX_RADIUS
    assert 0.0 < hyperparams.CURRICULUM_SUCCESS_THRESHOLD < 1.0


def test_describe_names_every_run_shaping_value():
    """The run log has to be enough to reproduce the run."""
    text = hyperparams.describe()
    for expected in ("sac:", "ppo:", "eval_freq", "normalize_reward", "curriculum", "net_arch"):
        assert expected in text


def test_describe_does_not_claim_a_worker_count():
    """n_envs is a per-run choice that train.py prints from its own arguments.
    Printing the module default here made a one-worker run's config.txt claim
    n_envs=4, two lines below the line that said 1."""
    assert "n_envs" not in hyperparams.describe()


# --- checked against the real SB3 signatures, where it is installed ----------

@pytest.mark.parametrize("algorithm", ["sac", "ppo"])
def test_every_key_is_a_real_constructor_parameter(algorithm):
    """A misspelled key is otherwise absorbed and silently defaulted."""
    sb3 = pytest.importorskip("stable_baselines3", reason="SB3 is not installed on this host")

    cls = {"sac": sb3.SAC, "ppo": sb3.PPO}[algorithm]
    accepted = set(inspect.signature(cls.__init__).parameters)
    unknown = set(hyperparams.for_algorithm(algorithm)) - accepted
    assert not unknown, f"{cls.__name__} does not accept {sorted(unknown)}"


@pytest.mark.parametrize("algorithm", ["sac", "ppo"])
def test_the_kwargs_actually_construct_a_model(algorithm):
    """The signature check cannot catch a value of the wrong type or an
    out-of-range one; constructing against a stub env can."""
    sb3 = pytest.importorskip("stable_baselines3", reason="SB3 is not installed on this host")
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")

    class _Stub(gym.Env):
        """The contract's spaces, no simulator. Never stepped."""

        observation_space = gym.spaces.Box(-1.0, 1.0, (contract.OBS_DIM,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACT_DIM,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    cls = {"sac": sb3.SAC, "ppo": sb3.PPO}[algorithm]
    kwargs = hyperparams.for_algorithm(algorithm)
    kwargs["verbose"] = 0
    model = cls(hyperparams.POLICY, _Stub(), **kwargs)
    assert model.policy is not None
