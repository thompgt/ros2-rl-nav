# ros2-rl-nav

A differential-drive robot in Gazebo Harmonic, exposed as a Gymnasium
environment over ROS 2 Jazzy, trained with SAC/PPO, and redeployed as a
standalone ROS 2 inference node — then measured to find out what redeployment
cost.

> **Status: Phases 0–2 and 4 built and verified; Phase 3 built but not yet
> run.** Simulation, bridge, the Gym environment, the training stack, the
> TorchScript export, the deployment node and the gap harness all exist, are
> tested, and have been smoke-run end to end against a live Gazebo. What has
> **not** happened is a real training run — so the results tables below are
> deliberately empty rather than aspirational.

## The design decision this project is built around

Gym's `step()` is synchronous and blocking. ROS 2 and Gazebo are asynchronous
and free-running. Subscribing to `/scan`, publishing `/cmd_vel`, and sleeping
gives you observations that are stale by a *variable* amount and an effective
action duration that jitters with CPU load. The policy learns a controller for a
simulator that doesn't exist; training appears to work and then fails on
redeployment for reasons that can't be diagnosed from logs.

So the world is **paused by default**, and each `step()`:

1. publishes the scaled action to `/cmd_vel`
2. calls `/world/arena/control` to advance **exactly 50 iterations × 1 ms**
3. **blocks** until `/scan` and `/odom` report a stamp ≥ the target sim time
4. assembles and returns the observation

Episodes are bit-reproducible for a fixed seed, and throughput is decoupled
from wall-clock. `scripts/verify_phase1.sh` proves the mechanism empirically —
that 50 sim steps advance `/clock` by exactly 50 ms — before any RL code exists.

The trade-off is the interesting part, and Phase 4 measures it rather than
asserting it.

## Architecture

```
                    ┌──────────────────────────── training ────────────────────────────┐
                    │                                                                  │
  SB3 SAC/PPO ──────┤  VecNormalize → SubprocVecEnv ×N                                 │
                    │        │                                                         │
                    │        └── RobotNavEnv (gymnasium.Env)                           │
                    │              │                                                   │
                    │              │  ┌─── STEP-SYNC LOOP ────────────────────────┐    │
                    │              │  │ 1. publish /cmd_vel                       │    │
                    │              ├──┤ 2. ControlWorld(multi_step=50)  ← 50 ms   │    │
                    │              │  │ 3. BLOCK until stamp ≥ target sim time    │    │
                    │              │  │ 4. assemble_observation()                 │    │
                    │              │  └───────────────────────────────────────────┘    │
                    └──────────────┼───────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │              ros_gz_bridge              │   /scan /odom /clock /cmd_vel
              │      (topics + world control srvs)      │   /world/arena/{control,set_pose}
              └────────────────────┬────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │   gz sim -s arena   │   10×10 m, 7 obstacles, 1 ms physics
                        │  diffbot + 360 LiDAR│
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼───────────────────────────────────────────────────┐
                    │              │                              deployment           │
                    │              │  ┌─── FREE-RUNNING LOOP ─────────────────────┐    │
                    │              │  │ 20 Hz wall-clock timer                    │    │
                    │  policy_node ├──┤ read NEWEST sample, whatever age          │    │
                    │              │  │ watchdog / safety gate                    │    │
                    │  policy.pt   │  │ assemble_observation()   ← same function  │    │
                    │  (TorchScript)│ └───────────────────────────────────────────┘    │
                    └──────────────────────────────────────────────────────────────────┘
```

The two loops differ in **exactly one** respect: who controls when the world
advances. Same observation assembly, same action scaling, same goal tolerance,
same collision threshold, same 500-step limit — all of it shared code, not
parallel implementations. That is what makes the difference between them
attributable.

## Stack

ROS 2 Jazzy · Gazebo Harmonic (`gz sim`) · `ros_gz_bridge` · Python 3.12 ·
Gymnasium · Stable-Baselines3 ≥ 2.3 · PyTorch (CPU)

Everything runs in a container. Development host here is Windows 11 +
Docker Desktop, headless.

## Quick start

```bash
make build     # build the ROS 2 Jazzy + Gazebo Harmonic image
make verify    # Phase 1 bridge verification — PASS/FAIL per check
make verify2   # Phase 2 exit criteria — episodes, throughput, memory, timing
make verify4   # Phase 4 smoke — train tiny, export, free-running eval
make test      # colcon build + pytest
make shell     # interactive container shell, repo mounted at /ws
```

The full pipeline, in order:

```bash
make train ALGO=sac SEED=0 ENVS=4                       # hours; run it yourself
make evaluate MODEL=runs/sac-seed0/best/best_model.zip  # step-synchronized score
make export-policy MODEL=runs/sac-seed0/best/best_model.zip   # -> policy.pt
make gap                                                # free-running score + delta
make deploy POLICY=runs/sac-seed0/policy.pt             # drive it interactively
make board                                              # TensorBoard on :6006
```

One command to reproduce a training run end to end: `docker compose -f
docker/docker-compose.yml up train`.

## Layout

| Path | What |
|---|---|
| `CONTRACTS.md` | **Authoritative** RL interface spec — obs, action, reward, termination, reset |
| `CLAUDE.md` | Stack pins, the step-sync rationale, and the list of ways this project fails |
| `WORKPLAN.md` | The five phases and their exit criteria |
| `contract.py` | Every contract number, transcribed once |
| `observation.py` | The 26-vector assembly — **pure, shared by training and deployment** |
| `action.py` | The action → `(v, ω)` mapping — pure, shared for the same reason |
| `arena.py` | Obstacle geometry and reset rejection sampling — pure |
| `env.py` | `RobotNavEnv(gymnasium.Env)`, the step-synchronized loop |
| `deploy.py` | The deployment controller — watchdog, safety gate, episode rules. Pure |
| `policy_node.py` | Phase 4 ROS 2 inference node: subscriptions, a timer, a publisher |
| `export_policy.py` | SB3 `.zip` → TorchScript `.pt`, verified against `model.predict` |
| `deploy_eval.py` | The gap measurement |
| `eval_set.py` | The held-out episodes, generated from a seed training never uses |
| `hyperparams.py` | SAC/PPO configuration and every run-shaping constant |
| `scripts/verify_phase*.sh` | Turn silence into a non-zero exit code, per phase |

## Environment contract

| | |
|---|---|
| Observation | `Box(-1, 1, (26,))` — 20 min-pooled LiDAR beams, goal distance, bearing sin/cos, previous action, step fraction |
| Action | `Box(-1, 1, (2,))` — linear → [0, 0.4] m/s, angular → [−1.5, 1.5] rad/s |
| Reward | progress `(d_prev − d_curr)` − 0.01 step − 0.05·\|ω\| + 10 goal − 10 collision |
| Termination | goal within 0.25 m, or min LiDAR < 0.18 m |
| Truncation | 500 steps |
| Step | 50 ms sim time, exactly |

Full detail, including the exact downsampling arithmetic and the
anti-reward-hacking escalation ladder, is in [`CONTRACTS.md`](CONTRACTS.md).

## Results

No training run has been executed yet. This table gets filled with mean ± std
over **three seeds per algorithm** — single-seed RL numbers aren't worth
reporting.

Every run is scored on the same 100 fixed episodes, generated from a seed
training never uses, so two runs differ by the policy rather than by which
episodes they happened to draw. Alongside success rate, evaluation reports
collision rate and *timeout* rate: a policy that learns to stand still has an
excellent collision rate and is worthless, and only the third number shows it.

| Algorithm | Success rate | Mean path length | Path efficiency | Collision rate |
|---|---|---|---|---|
| SAC | — | — | — | — |
| PPO | — | — | — | — |

## Sim-to-deployment gap

The harness is built and smoke-tested; the numbers wait on Phase 3.

`deploy_eval.py` scores an exported policy on the *same* held-out episodes as
`evaluate.py`, with the same termination rules and the same observation
assembly, against an unpaused world driven by the real `policy_node` at 20 Hz
off a wall clock. It reports the two side by side.

| | Step-synchronized | Free-running | Δ |
|---|---|---|---|
| Success rate | — | — | — |
| Collision rate | — | — | — |
| Timeout rate | — | — | — |

**The mechanism is already measurable without a trained policy**, and it is the
more interesting half. From a smoke run in this container:

| | Training | Deployment (measured) |
|---|---|---|
| Observation age | 50 ms, fixed, by construction | 121 ms mean, 183 ms p95, 243 ms max |
| Ticks losing the observation entirely | 0 | 2.1% (past the 200 ms watchdog) |

The policy trained on a world that handed it a 50 ms-old observation every
single step. Deployed, it gets one two to four times older, varying tick to
tick, and a fiftieth of the time gets nothing usable at all and is stopped by
the watchdog. That is the gap's mechanism, and it is a property of the
architecture rather than of any policy.

**One caveat, reported rather than buried.** This container sustains a
real-time factor of ~0.3–0.5. The node ticks at 20 Hz on a wall clock, as a
robot would, so a run at RTF 0.3 controls the world at ~65 Hz of *sim* time
against the 20 Hz it trained at — a difference in the control problem, not in
staleness, and the two are not separable from the numbers above.
`deploy_eval.py` computes this and says so in as many words whenever the factor
strays more than 10% from 1. A headline gap taken on this machine would be part
architecture and part hardware; taking it on a machine that sustains RTF ≈ 1 is
the remaining work.

## What's left

- Run training. Three seeds × two algorithms; fill both tables above.
- Take the gap measurement on a host that sustains a real-time factor near 1.
- A GIF of a trained policy in the first 100 px of this README. There is no
  trained policy yet, so there is nothing honest to record.

## License

MIT
