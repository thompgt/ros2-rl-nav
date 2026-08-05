"""Phase 4 -- export a trained policy to TorchScript, and load it back.

    python3 -m robot_rl_env.export_policy --model runs/sac-seed0/best/best_model.zip
    python3 -m robot_rl_env.export_policy --model ... --out policies/sac-seed0.pt

Writes a ``.pt`` that maps a ``(B, 26)`` observation to a ``(B, 2)`` action in
``[-1, 1]``, and nothing else. ``load_policy`` reads it back as a plain
callable for ``deploy.DeploymentController``.

Why the deployment artifact is not the .zip
-------------------------------------------
An SB3 zip is a pickle of a Python object graph. Loading one requires
stable-baselines3, which requires the exact class layout it was saved with, and
a robot then carries a training framework and its version constraints for the
sake of two matrix multiplies. Worse, it is a *silent* coupling: the zip loads
fine under a different SB3 version right up until the policy class gains a
field, and the failure surfaces as a shape error deep inside a constructor.

TorchScript is the smallest artifact that still contains the whole
computation. It is also the one that can be inspected: ``torch.jit.load(p).code``
prints the forward pass, so what the robot runs can be read rather than
trusted.

The one thing this file must never do quietly
---------------------------------------------
Export a policy whose observations were normalized during training, without the
statistics. ``VecNormalize`` with ``norm_obs=True`` means the policy learned
against ``(obs - mean) / sqrt(var)``; hand it a raw contract observation and it
receives inputs several standard deviations from anything it ever saw, and
produces confident, wrong actions. Nothing errors. The measured
sim-to-deployment gap would be enormous and would be attributed to timing.

So the statistics are looked for, baked into the exported module when they
exist, and their *absence* is checked against ``hyperparams.NORMALIZE_OBS``
rather than assumed benign. This project trains with ``NORMALIZE_OBS = False``,
so the normal path bakes in nothing -- but the guard is what makes that a fact
rather than a hope.

Verification is not optional
----------------------------
Tracing silently records whatever the example input happened to exercise. So
the export is checked against ``model.predict(deterministic=True)`` over random
observations before the file is written, and a mismatch aborts rather than
warns. An exported policy that is subtly not the trained policy is
indistinguishable from a policy that deploys badly, which is the exact question
Phase 4 exists to answer.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from robot_rl_env import contract

TOLERANCE = 1e-5
"""Maximum permitted ``|traced - predict|`` over the verification batch.

Float32 arithmetic reordered by tracing accounts for ~1e-7; 1e-5 is loose
enough not to fail on that and tight enough that a genuinely different
computation -- a missed tanh, a skipped normalization -- cannot pass. A
difference of 1e-3 in an action is 0.4 mm/s and harmless; it is also not
something a correct trace produces, so it is treated as a bug.
"""

VERIFY_SAMPLES = 512


class ExportedPolicy(torch.nn.Module):
    """Deterministic policy plus, if training used them, the input statistics.

    The normalization is baked into the module rather than shipped beside it as
    a second file. A ``.pt`` and a ``.pkl`` that must travel together is a
    thing that eventually travels apart, and the failure mode of arriving
    without the statistics is silent.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        obs_mean: np.ndarray | None = None,
        obs_var: np.ndarray | None = None,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.policy = policy
        self.normalize = obs_mean is not None
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)

        zeros = torch.zeros(contract.OBS_DIM, dtype=torch.float32)
        ones = torch.ones(contract.OBS_DIM, dtype=torch.float32)
        # Registered as buffers even when unused, so the traced graph has the
        # same shape either way and a normalized and an unnormalized export can
        # be diffed against each other.
        self.register_buffer(
            "obs_mean",
            zeros if obs_mean is None else torch.as_tensor(obs_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "obs_var",
            ones if obs_var is None else torch.as_tensor(obs_var, dtype=torch.float32),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            obs = torch.clamp(
                (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        action = self.policy._predict(obs, deterministic=True)
        # SB3's own predict() clips to the action space for policies that do
        # not squash, and env.step clips again on the way in. SAC's tanh makes
        # this a no-op; PPO's Gaussian mean does not, and an unclipped export
        # would command velocities outside the contract envelope.
        return torch.clamp(action, -1.0, 1.0)


def find_vecnormalize(model_path: str | Path) -> Path | None:
    """Locate the ``vecnormalize.pkl`` belonging to a saved model.

    ``train.py`` writes it at the run root while the model lands in ``best/``
    or ``checkpoints/``, so a naive sibling lookup misses it for exactly the
    checkpoint anyone would actually deploy. Searches the model's own directory
    and then exactly one level up -- far enough to reach the run root from
    ``runs/<algo>-seed<N>/best/best_model.zip``, and no further. A search that
    kept walking would find ``runs/vecnormalize.pkl`` left behind by some other
    run and attach one run's statistics to another run's policy, which produces
    an export that loads, runs, and is wrong.
    """
    directory = Path(model_path).resolve().parent
    for candidate in (directory, *list(directory.parents)[:1]):
        pkl = candidate / "vecnormalize.pkl"
        if pkl.is_file():
            return pkl
    return None


def observation_statistics(pkl: Path) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Read ``(mean, var, clip_obs, epsilon)`` out of a pickled ``VecNormalize``.

    Returns ``None`` when the wrapper was not normalizing observations, which
    is this project's configuration -- see ``hyperparams.NORMALIZE_OBS``. Reads
    the pickle directly rather than through ``VecNormalize.load``, which wants
    a live ``venv`` to attach to and there is none here.
    """
    with open(pkl, "rb") as handle:
        vec_normalize = pickle.load(handle)

    if not getattr(vec_normalize, "norm_obs", False):
        return None

    rms = vec_normalize.obs_rms
    mean = np.asarray(rms.mean, dtype=np.float32).reshape(contract.OBS_DIM)
    var = np.asarray(rms.var, dtype=np.float32).reshape(contract.OBS_DIM)
    return mean, var, float(vec_normalize.clip_obs), float(vec_normalize.epsilon)


def trace(exportable: ExportedPolicy) -> torch.jit.ScriptModule:
    """Trace to TorchScript on a single-observation batch.

    ``eval()`` and ``no_grad`` are not cosmetic: a module traced in training
    mode records dropout and batch-norm behaviour into the graph, and this
    project's MLP policy has neither -- which is exactly the kind of thing that
    stays true until someone changes ``NET_ARCH``.
    """
    exportable.eval()
    example = torch.zeros(1, contract.OBS_DIM, dtype=torch.float32)
    with torch.no_grad():
        return torch.jit.trace(exportable, example)


def verify(traced: torch.jit.ScriptModule, model, *, samples: int = VERIFY_SAMPLES) -> float:
    """Largest disagreement with ``model.predict(deterministic=True)``.

    Observations are drawn uniformly from the whole ``[-1, 1]`` box rather than
    from a rollout. A rollout would only cover the states the policy already
    steers itself into, and the interesting failure -- a missed normalization,
    a squashing function dropped by the trace -- is largest away from them.
    """
    rng = np.random.default_rng(0)
    observations = rng.uniform(
        -1.0, 1.0, size=(samples, contract.OBS_DIM)
    ).astype(np.float32)

    expected, _ = model.predict(observations, deterministic=True)
    with torch.no_grad():
        actual = traced(torch.from_numpy(observations)).numpy()

    return float(np.abs(np.asarray(expected, dtype=np.float32) - actual).max())


def load_policy(path: str | Path, device: str = "cpu"):
    """Load an exported ``.pt`` as a ``obs -> action`` callable. No SB3.

    This is the deployment entry point, and the whole reason the export exists:
    ``policy_node`` imports this module, torch, and nothing else from the
    training stack.
    """
    module = torch.jit.load(str(path), map_location=device)
    module.eval()

    def policy(obs) -> np.ndarray:
        tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, contract.OBS_DIM)
        )
        with torch.no_grad():
            return module(tensor).numpy().reshape(contract.ACT_DIM)

    return policy


def default_output(model_path: str | Path) -> Path:
    """``runs/sac-seed0/best/best_model.zip`` -> ``runs/sac-seed0/policy.pt``.

    Beside the run rather than beside the checkpoint, because the exported
    policy is the run's deliverable and ``best/`` is an implementation detail
    of ``EvalCallback``.
    """
    model_path = Path(model_path)
    directory = model_path.parent
    if directory.name in ("best", "checkpoints"):
        directory = directory.parent
    return directory / "policy.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="path to a .zip saved by train.py")
    parser.add_argument(
        "--algo", default=None, help="sac or ppo; inferred from the model path if omitted"
    )
    parser.add_argument("--out", default=None, help="default: <run dir>/policy.pt")
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="path to vecnormalize.pkl; found automatically beside the run if omitted",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help="maximum permitted disagreement with model.predict",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from stable_baselines3 import PPO, SAC

    from robot_rl_env import hyperparams
    from robot_rl_env.evaluate import infer_algorithm

    algorithm = infer_algorithm(args.model, args.algo)
    model = {"sac": SAC, "ppo": PPO}[algorithm].load(args.model, device="cpu")
    print(f"model={args.model} algo={algorithm}")

    pkl = Path(args.vecnormalize) if args.vecnormalize else find_vecnormalize(args.model)
    statistics = observation_statistics(pkl) if pkl else None

    if statistics is None:
        # The guard the module docstring is about. Exporting a policy trained on
        # normalized observations without its statistics produces a file that
        # loads, runs, and is wrong.
        if hyperparams.NORMALIZE_OBS:
            print(
                f"ERROR: hyperparams.NORMALIZE_OBS is True, so this policy was "
                f"trained on normalized observations, but no statistics were "
                f"found ({pkl or 'no vecnormalize.pkl beside the run'}). "
                f"Exporting would produce a policy that runs and is silently "
                f"wrong. Pass --vecnormalize explicitly.",
                file=sys.stderr,
            )
            return 1
        print(f"observation normalization: none ({pkl or 'no vecnormalize.pkl'})")
        exportable = ExportedPolicy(model.policy)
    else:
        mean, var, clip_obs, epsilon = statistics
        print(f"observation normalization: baked in from {pkl} (clip_obs={clip_obs})")
        exportable = ExportedPolicy(
            model.policy, obs_mean=mean, obs_var=var, clip_obs=clip_obs, epsilon=epsilon
        )

    traced = trace(exportable)
    error = verify(traced, model)
    print(f"max |traced - predict| over {VERIFY_SAMPLES} observations: {error:.3e}")

    if error > args.tolerance:
        print(
            f"ERROR: the exported policy disagrees with the trained one by "
            f"{error:.3e}, above the {args.tolerance:.0e} tolerance. Nothing "
            f"written. This is a bug in the export, not in the policy -- "
            f"suspect a squashing function or a normalization the trace did "
            f"not record.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out) if args.out else default_output(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    print(f"\nwrote {out}")
    print(f"next: ros2 run robot_rl_env policy_node --ros-args -p policy:={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
