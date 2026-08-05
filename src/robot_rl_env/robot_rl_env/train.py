"""Phase 3 -- training entry point.

    python3 -m robot_rl_env.train --algo sac --seed 0
    python3 -m robot_rl_env.train --algo ppo --seed 0 --n-envs 4

Launch this yourself; do not run it inside an agent loop. A run is hours of
mostly waiting, and the useful artifacts are the TensorBoard scalars and the
eval JSON, both of which can be read afterwards as text.

Every run writes to ``runs/<algo>-seed<N>/``:

    tensorboard/     scalars
    checkpoints/     periodic saves, so a killed run is not a total loss
    best/            the highest-scoring policy on the held-out set
    final.zip        the policy at the end of the budget
    vecnormalize.pkl reward statistics, required to resume
    config.txt       every hyperparameter, so the run is reproducible from
                     its own artifacts rather than from git history

Three seeds per algorithm. A single-seed RL result reports the seed as much as
the algorithm -- see WORKPLAN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_rl_env import eval_set, hyperparams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", default="sac", help="sac or ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=hyperparams.N_ENVS,
        help="parallel simulators; each is a gz server plus two bridge nodes",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="override the per-algorithm budget (SAC 150k, PPO 500k)",
    )
    parser.add_argument("--run-dir", default=None, help="default: runs/<algo>-seed<N>")
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="sample the full arena from step 0. Expect no exploration signal.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="path to a .zip to continue from. Its vecnormalize.pkl must sit beside it.",
    )
    return parser.parse_args(argv)


def build_run_dir(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir or f"runs/{args.algo.lower()}-seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Imported inside main so that --help, and the tests that exercise argument
    # parsing, do not pay for torch or start a DDS participant.
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecNormalize

    from robot_rl_env.callbacks import CurriculumCallback, SuccessRateCallback
    from robot_rl_env.vec_env import make_eval_env, make_training_env

    algo = args.algo.strip().lower()
    kwargs = hyperparams.for_algorithm(algo)          # raises on an unknown name
    timesteps = args.timesteps or hyperparams.timesteps_for(algo)
    model_class = {"sac": SAC, "ppo": PPO}[algo]

    run_dir = build_run_dir(args)
    set_random_seed(args.seed)

    # Written before anything is launched: a run that dies during startup should
    # still leave behind what it was trying to do.
    config = "\n".join(
        [
            f"algo={algo} seed={args.seed} n_envs={args.n_envs} timesteps={timesteps}",
            f"curriculum={'off' if args.no_curriculum else 'on'} resume={args.resume}",
            hyperparams.describe(),
            "eval_set=" + repr(eval_set.summarize(eval_set.episodes())),
        ]
    )
    (run_dir / "config.txt").write_text(config + "\n")
    print(config, flush=True)

    train_env = make_training_env(args.n_envs, monitor_dir=str(run_dir / "monitor"))
    train_env = VecNormalize(
        train_env,
        norm_obs=hyperparams.NORMALIZE_OBS,
        norm_reward=hyperparams.NORMALIZE_REWARD,
        clip_reward=hyperparams.CLIP_REWARD,
    )
    # The evaluation env is deliberately *not* wrapped in VecNormalize. That is
    # safe only because NORMALIZE_OBS is False: the policy sees raw contract
    # observations in both envs. If observation normalization is ever turned
    # on, this env must share the training env's statistics or every reported
    # number is measured on inputs the policy was not trained on.
    #
    # SB3 warns about this ("Training and eval env are not of the same type").
    # The warning is expected and the mismatch is the point: it fires because
    # EvalCallback would otherwise sync normalization statistics between the
    # two, and here there are no observation statistics to sync. Wrapping the
    # eval env to silence it would normalize *rewards* during evaluation,
    # making the reported mean_reward incomparable across runs.
    eval_env = make_eval_env(
        eval_set.episodes(eval_set.N_CALLBACK_EPISODES), index=args.n_envs
    )

    callbacks = [
        SuccessRateCallback(),
        # eval_freq and save_freq are counted in *vectorized* steps by SB3 --
        # one call per VecEnv step, not per transition. Dividing by n_envs is
        # what keeps "every 10k steps" meaning the same thing at n_envs=1 and
        # n_envs=8; without it, evaluation gets N times rarer and the curve
        # merely looks flat.
        EvalCallback(
            eval_env,
            n_eval_episodes=eval_set.N_CALLBACK_EPISODES,
            eval_freq=max(hyperparams.EVAL_FREQ // args.n_envs, 1),
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "best"),
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(hyperparams.CHECKPOINT_FREQ // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=algo,
            save_vecnormalize=True,
        ),
    ]
    if not args.no_curriculum:
        callbacks.insert(0, CurriculumCallback())

    if args.resume:
        model = model_class.load(args.resume, env=train_env, tensorboard_log=str(run_dir / "tb"))
    else:
        model = model_class(
            hyperparams.POLICY,
            train_env,
            seed=args.seed,
            tensorboard_log=str(run_dir / "tb"),
            **kwargs,
        )

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            # False on resume, so the step counter continues rather than
            # restarting -- otherwise the resumed segment overwrites the
            # original run's scalars from step 0 and the curve doubles back.
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
    finally:
        # Even on KeyboardInterrupt. Hours of training that exist only in a
        # process being torn down is the worst outcome available here, and
        # closing the envs matters just as much: four orphaned gz servers hold
        # their partitions and break the next run.
        model.save(run_dir / "final")
        train_env.save(str(run_dir / "vecnormalize.pkl"))
        train_env.close()
        eval_env.close()

    print(f"\nsaved {run_dir / 'final.zip'}", flush=True)
    print(
        "next: python3 -m robot_rl_env.evaluate "
        f"--model {run_dir / 'best' / 'best_model.zip'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
