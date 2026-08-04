"""Phase 3 training configuration: SAC and PPO hyperparameters, in one place.

Pure. No ROS, no simulator, no stable-baselines3 import — so the values can be
inspected and tested on a machine that has none of them, and so importing this
module never costs a DDS participant.

What shapes the choices here
----------------------------
The environment is **simulator-bound**. A step costs a Gazebo advance plus a
service round-trip; a gradient step on a 26→256→256→2 MLP costs microseconds of
CPU. That inverts the usual trade: sample efficiency is nearly the only thing
that matters, and update cost is nearly free. Hence SAC first, a large batch,
and more gradient steps than SB3's defaults — spending compute to extract more
from each transition is a good deal when transitions are the scarce resource.

Nothing here duplicates a number from ``contract.py``. Episode length, control
rate and the action envelope are contract facts; these are training knobs.
"""

from __future__ import annotations

from robot_rl_env import contract, eval_set

POLICY = "MlpPolicy"
"""The observation is a 26-vector, already normalized to [-1, 1] by contract.
There is no image, so no CNN, and nothing to be gained from a recurrent policy:
the observation includes the previous action and the step fraction, which is
what a memory would otherwise have to reconstruct."""

NET_ARCH = [256, 256]
"""Two hidden layers of 256.

Larger than SB3's SAC default ([256, 256] is in fact the SAC default; PPO's is
[64, 64]) and deliberately the same for both algorithms, because the point of
the comparison is the algorithm, not the capacity. On CPU this width is still
far below the simulator's cost per step."""

# --- SAC ---------------------------------------------------------------------

SAC_KWARGS: dict = {
    "learning_rate": 3e-4,
    "buffer_size": 200_000,
    # Larger than the 150k training budget on purpose: nothing is ever evicted,
    # so the critic keeps the early collision transitions. Those are the rarest
    # and most informative samples in the run -- a policy that has stopped
    # colliding stops generating them, and a FIFO buffer would then forget why.
    "learning_starts": 5_000,
    # 10 episodes of random actions before the first gradient step. Below about
    # this, the critic fits a handful of correlated transitions and the entropy
    # coefficient collapses early.
    "batch_size": 512,
    "tau": 0.005,
    "gamma": 0.99,
    # 0.99 at 20 Hz is a ~5 s effective horizon. Adequate because the reward is
    # dense: progress is paid every step, so the goal bonus does not have to
    # propagate 500 steps back through the value function to be learned.
    "train_freq": 1,
    "gradient_steps": 2,
    # Two updates per environment step. The usual reason to keep this at 1 is
    # wall-clock; here the wall-clock is Gazebo's, and the extra update is
    # essentially free.
    "ent_coef": "auto",
    # Auto-tuned temperature. A fixed coefficient has to be retuned whenever the
    # reward scale changes, and the reward scale here is metres of progress,
    # which changes the moment the arena or MAX_LINEAR_VEL does.
    "policy_kwargs": {"net_arch": NET_ARCH},
    "verbose": 1,
}

SAC_TIMESTEPS = 150_000
"""WORKPLAN's figure. At 5x real time and 20 Hz control this is a few hours."""

# --- PPO ---------------------------------------------------------------------

PPO_KWARGS: dict = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    # Per environment. With N_ENVS=4 that is a 2048-transition rollout, and 512
    # steps is 25.6 s of sim time -- long enough that most rollouts contain a
    # terminal event, which is what the advantage estimate needs to see.
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,          # same as SAC, or the comparison measures the horizon
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    # PPO on a continuous action space explores through the policy's own
    # log-std. An entropy bonus here fights the angular penalty in the reward,
    # which exists specifically to stop the policy spinning; raising it is the
    # wrong lever for a policy that will not explore, and CONTRACTS.md's
    # escalation ladder is the right one.
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": NET_ARCH, "vf": NET_ARCH}},
    "verbose": 1,
}

PPO_TIMESTEPS = 500_000
"""More than triple SAC's budget, for a fair comparison rather than a flattering
one: PPO is on-policy and discards every transition after one rollout. Reporting
PPO at 150k steps would measure sample efficiency and call it algorithm quality.
"""

_ALGORITHMS = {
    "sac": (SAC_KWARGS, SAC_TIMESTEPS),
    "ppo": (PPO_KWARGS, PPO_TIMESTEPS),
}


def for_algorithm(name: str) -> dict:
    """Constructor kwargs for ``name``. Returns a **copy**.

    A copy because SB3 mutates what it is handed (``policy_kwargs`` especially),
    and a caller that trained SAC and then PPO in one process would otherwise
    train the second one on hyperparameters the first had edited — silently,
    and only in the run where both appear.
    """
    key = name.strip().lower()
    if key not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {name!r}; expected one of {sorted(_ALGORITHMS)}"
        )
    kwargs, _ = _ALGORITHMS[key]
    # deepcopy-by-hand: the nested policy_kwargs dict is the one that matters.
    copied = dict(kwargs)
    copied["policy_kwargs"] = {
        k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
        for k, v in kwargs["policy_kwargs"].items()
    }
    return copied


def timesteps_for(name: str) -> int:
    """Training budget for ``name``. See ``PPO_TIMESTEPS`` for why they differ."""
    key = name.strip().lower()
    if key not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {name!r}; expected one of {sorted(_ALGORITHMS)}"
        )
    return _ALGORITHMS[key][1]


# --- run plumbing ------------------------------------------------------------

N_ENVS = 4
"""Parallel Gazebo instances. Each is a full simulator process plus two bridge
nodes, so this is bounded by RAM and cores, not by the RL side. Four fits the
development machine; ``--n-envs`` overrides it."""

EVAL_FREQ = 10_000
"""Environment steps between in-training evaluations, counted **across all
envs**. SB3's ``EvalCallback`` counts callback invocations, which happen once
per vectorized step — so train.py divides this by ``n_envs``. Getting that
wrong makes evaluation N times rarer than intended, which looks like a flat
learning curve."""

CHECKPOINT_FREQ = 25_000
"""Also in total environment steps, and divided the same way.

Frequent enough that a run killed at hour three is not a total loss: this
project's runs are launched by hand outside the agent loop, and there is no
scheduler to restart them."""

# --- VecNormalize ------------------------------------------------------------

NORMALIZE_OBS = False
"""**Load-bearing.** Do not turn this on.

The observation is already in [-1, 1] by contract, so there is little to gain —
and a great deal to lose. ``VecNormalize`` with ``norm_obs=True`` learns a
running mean and variance that become part of the policy's input pipeline. At
deployment ``policy_node.py`` would then have to load and apply those statistics
to reproduce what the policy was trained on, which is a second preprocessing
path outside ``assemble_observation`` — precisely the divergence CONTRACTS.md's
shared-code requirement exists to forbid. It fails silently: the node runs, the
policy acts, and it is merely worse.
"""

NORMALIZE_REWARD = True
"""This one does matter, and costs nothing at deployment.

The per-step reward is metres of progress (order 0.02) while the terminal
bonuses are ±10 — a 500:1 range that puts the value target's scale entirely at
the mercy of how often episodes terminate. Reward normalization is applied to
the learning signal only; the deployed policy never sees a reward.
"""

CLIP_REWARD = 10.0
"""Matches the terminal bonus magnitude: clip the tail, keep the events."""

# --- Curriculum --------------------------------------------------------------

CURRICULUM_START_RADIUS = 2.0
"""Metres. WORKPLAN: start goals within 2 m, so a random policy sees successes
at all. With goals sampled across a 10 m arena, an untrained policy may never
reach one, and SAC cannot learn from a reward signal it has never observed."""

CURRICULUM_SUCCESS_THRESHOLD = 0.7
"""Expand the radius once the rolling success rate crosses this."""

CURRICULUM_INCREMENT = 1.0
"""Metres added per expansion. Small enough that the policy is never handed a
task distribution it has no chance on, which would undo the curriculum."""

CURRICULUM_MAX_RADIUS = eval_set.MAX_EVAL_DISTANCE
"""Stop expanding where the evaluation set stops.

Training past the eval cap is not wasted, but it is unmeasured -- and a
curriculum that ran to the arena diagonal would spend its last stage on
episodes that cannot be completed inside the truncation limit at all. See
``eval_set.REACHABLE_DISTANCE``.
"""

CURRICULUM_WINDOW = 20
"""Episodes in the rolling success-rate window.

Short enough to react, long enough that the threshold is not crossed by three
lucky episodes: at a true 50% success rate, 20 episodes still cross 70% about
once in 40 windows, and 5 would do it constantly.
"""


def describe() -> str:
    """One text block naming every training-relevant value, for the run log.

    A run whose hyperparameters are not in its own log cannot be reproduced
    from the artifacts it leaves behind, and TensorBoard records scalars, not
    configuration.
    """
    lines = [
        f"policy={POLICY} net_arch={NET_ARCH}",
        f"n_envs={N_ENVS} eval_freq={EVAL_FREQ} checkpoint_freq={CHECKPOINT_FREQ}",
        f"normalize_obs={NORMALIZE_OBS} normalize_reward={NORMALIZE_REWARD} "
        f"clip_reward={CLIP_REWARD}",
        f"curriculum={CURRICULUM_START_RADIUS}m -> {CURRICULUM_MAX_RADIUS:.2f}m "
        f"step {CURRICULUM_INCREMENT}m at {CURRICULUM_SUCCESS_THRESHOLD:.0%} "
        f"over {CURRICULUM_WINDOW} episodes",
        f"episode={contract.MAX_EPISODE_STEPS} steps @ {contract.CONTROL_HZ:.0f} Hz "
        f"({contract.MAX_EPISODE_STEPS * contract.STEP_DURATION:.0f} s sim time)",
    ]
    for name, (kwargs, steps) in sorted(_ALGORITHMS.items()):
        lines.append(f"{name}: timesteps={steps} " + " ".join(
            f"{k}={v}" for k, v in kwargs.items() if k not in ("verbose", "policy_kwargs")
        ))
    return "\n".join(lines)
