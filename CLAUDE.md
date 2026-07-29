# CLAUDE.md — read this first, every session

`ros2-rl-nav`: a Gazebo differential-drive robot exposed as a Gymnasium env over
ROS 2, trained with SAC/PPO, redeployed as a standalone ROS 2 inference node.

Read `CONTRACTS.md` alongside this file. It specifies the observation, action,
reward, termination, and reset semantics. It is authoritative.

## Stack — pinned, do not substitute

| Component | Version |
|---|---|
| ROS 2 | **Jazzy** (Ubuntu 24.04) |
| Gazebo | **Harmonic** (`gz sim`) — *not* Gazebo Classic, which is EOL |
| Bridge | `ros_gz_bridge` / `ros_gz_sim` |
| Python | 3.12 |
| RL | Gymnasium, Stable-Baselines3 ≥ 2.3, PyTorch **CPU** wheels |
| Host | Windows 11 + Docker Desktop. **Headless.** No GPU on this machine. |

CPU-only is intentional and not a limitation here: a 26-dim observation with an
MLP policy is simulator-bound. Throughput comes from parallel envs, not CUDA.

## The load-bearing decision: step-synchronized simulation

Gym's `step()` is synchronous and blocking. ROS 2 and Gazebo are asynchronous
and free-running. Naively subscribing to `/scan`, publishing `/cmd_vel`, and
sleeping gives you observations that are stale by a *variable* amount and an
effective action duration that jitters with CPU load. The policy then learns a
controller for a simulator that does not exist. Training appears to work and
fails on redeployment for reasons that cannot be diagnosed from logs.

**Therefore: the world is paused by default.** Each `step()`:

1. Publishes the scaled action to `/cmd_vel`.
2. Calls `/world/arena/control` to advance **exactly 50 iterations × 1 ms = 50 ms**
   of sim time.
3. **Blocks** until `/scan` and `/odom` report a stamp ≥ the expected sim time.
4. Assembles and returns the observation.

Episodes are then bit-reproducible for a fixed seed, and training throughput is
decoupled from wall-clock. Every other architectural choice in this project is
negotiable. This one is not.

## Never do these

These are the specific ways this project fails. Each has been chosen
deliberately against.

- **Never use Gazebo Classic APIs.** No `gazebo_msgs`, no
  `/gazebo/reset_simulation`, no `/gazebo/set_model_state`, no `gazebo_ros`.
  They are heavily represented in training data and entirely wrong for
  Harmonic. The correct interfaces are `ros_gz_interfaces/srv/ControlWorld` on
  `/world/arena/control` and `ros_gz_interfaces/srv/SetEntityPose` on
  `/world/arena/set_pose`.
- **Never fall back to a cached observation on timeout.** `get_obs(min_stamp)`
  must **raise** when data does not arrive. A `return self._last_obs` fallback
  silently reintroduces exactly the staleness bug the whole architecture exists
  to eliminate, and it will not show up in any metric until deployment.
- **Never call `time.sleep()` inside `step()`.** If you think you need it, the
  step-sync handshake is broken; fix that instead.
- **Never use `rclpy.spin_once()` in the step loop.** Use a
  `MultiThreadedExecutor` spun on a background thread with a
  `ReentrantCallbackGroup` for subscriptions and a separate group for service
  clients. `spin_once()` in a blocking call deadlocks against service futures.
- **Never reimplement observation preprocessing** in `policy_node.py`. Import
  `assemble_observation` from `robot_rl_env.observation`. Training/inference
  preprocessing divergence is a silent, unloggable failure.
- **Never mean-pool the LiDAR.** Min-pool. See `CONTRACTS.md`.

## Layout

```
CONTRACTS.md              authoritative RL interface spec
WORKPLAN.md               the phased plan; which phase you are in
docker/                   Dockerfile, compose, entrypoint
src/robot_rl_env/
  robot_rl_env/
    sim_control.py        Phase 2a — world control service client
    observation.py        Phase 2b — SHARED obs assembly (train + deploy)
    env.py                Phase 2c — RobotNavEnv(gymnasium.Env)
    train.py              Phase 3
    evaluate.py           Phase 3
    policy_node.py        Phase 4 — ROS 2 inference node
  worlds/arena.sdf        10x10 m arena, 7 obstacles, 1 ms physics step
  models/diffbot/         chassis, diff drive, 360-beam LiDAR
  config/bridge.yaml      ros_gz_bridge topic + service mapping
  launch/world.launch.py  headless:=true default
  test/
scripts/verify_phase1.sh  text-output verification of the bridge
```

## Commands

Everything runs inside the container; the host is Windows.

```bash
make build          # docker build
make shell          # interactive container shell, workspace mounted at /ws
make test           # colcon build + pytest, headless
make verify         # Phase 1 bridge verification, prints pass/fail per check
make train          # Phase 3
make deploy         # Phase 4
```

Inside the container the workspace is `/ws`; build with
`colcon build --symlink-install && source install/setup.bash`.

## Working rules

- **Every task ends with a command whose text output verifies it.** An agent
  cannot see Gazebo. A bridge misconfiguration produces *silence*, not an
  error. Convert silence into a failing assertion.
- **Write the test first** for anything in Phase 2.
- **One phase per session.** Long sessions drift from `CONTRACTS.md` and start
  inventing helper abstractions nobody asked for.
- Commit and push after each small logical unit, not batched at the end.

## Current phase

**Phase 0–1 in progress.** Update this line when a phase closes; it is how the
next session knows where to start.
