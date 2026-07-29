# ros2-rl-nav

A differential-drive robot in Gazebo Harmonic, exposed as a Gymnasium
environment over ROS 2 Jazzy, trained with SAC/PPO, and redeployed as a
standalone ROS 2 inference node.

> **Status: Phase 1 of 5.** Simulation and bridge are up and verified. The Gym
> environment (Phase 2) is scaffolded but not implemented. There are no
> training results yet — the results table below is deliberately empty rather
> than aspirational.

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

The trade-off is the interesting part, and Phase 4 measures it: a policy trained
under perfect synchronization meets real sensor latency and timing jitter at
deployment. Quantifying that gap is the point of the project.

## Stack

ROS 2 Jazzy · Gazebo Harmonic (`gz sim`) · `ros_gz_bridge` · Python 3.12 ·
Gymnasium · Stable-Baselines3 ≥ 2.3 · PyTorch (CPU)

Everything runs in a container. Development host here is Windows 11 +
Docker Desktop, headless.

## Quick start

```bash
make build     # build the ROS 2 Jazzy + Gazebo Harmonic image
make verify    # Phase 1 bridge verification — PASS/FAIL per check
make test      # colcon build + pytest
make shell     # interactive container shell, repo mounted at /ws
```

## Layout

| Path | What |
|---|---|
| `CONTRACTS.md` | **Authoritative** RL interface spec — obs, action, reward, termination, reset |
| `CLAUDE.md` | Stack pins, the step-sync rationale, and the list of ways this project fails |
| `WORKPLAN.md` | The five phases and their exit criteria |
| `src/robot_rl_env/robot_rl_env/contract.py` | Every contract number, transcribed once |
| `src/robot_rl_env/worlds/arena.sdf` | 10×10 m arena, 7 obstacles, 1 ms physics |
| `src/robot_rl_env/models/diffbot/` | Chassis, diff drive, 360-beam LiDAR |
| `src/robot_rl_env/config/bridge.yaml` | `ros_gz_bridge` topic mapping |
| `scripts/verify_phase1.sh` | Turns bridge silence into a non-zero exit code |

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

Phase 3 not started. This table gets filled with mean ± std over **three seeds
per algorithm** — single-seed RL numbers aren't worth reporting.

| Algorithm | Success rate | Mean path length | Collision rate |
|---|---|---|---|
| SAC | — | — | — |
| PPO | — | — | — |

## Sim-to-deployment gap

Phase 4 not started. Will report success rate in step-synchronized mode versus
free-running real-time mode on identical held-out start/goal pairs, and account
for the difference.

## License

MIT
