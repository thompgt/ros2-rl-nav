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
- **Never call `ControlWorld(reset.all)` in the episode loop.** It is
  asynchronous in Harmonic: it answers immediately and then swallows the
  `set_pose` that follows it, about one reset in three, with no barrier that
  closes the window — `/clock` returns to zero well before the entity manager
  will accept a pose. `reset()` brakes the robot to a stop and teleports
  instead. See `contract.BRAKE_ITERATIONS`.
- **Never drive the deployment node's control timer from `/clock`.** It is
  tempting: `use_sim_time:=true` would hold the control period at exactly the
  50 ms the policy trained on, however slowly the simulator ran. It would also
  quietly subtract part of the effect Phase 4 exists to measure, because a real
  robot's control loop does not get to slow down when the world does. Same for
  the watchdog, which has to fire when the clock stops and therefore cannot be
  measured against that clock. The real-time factor is reported beside the gap
  instead — see `deploy_eval.effective_control_hz`.
- **Never let the deployment node compute a reward.** It has no `d_prev`
  bookkeeping and adding one would be a second implementation of the reward in
  `CONTRACTS.md`, living outside `env.step`. `summarize_rollouts` omits
  `mean_reward` when the records lack it rather than reporting a nan.
- **Never treat "a message arrived" as "the world has advanced".** The barrier
  is the *stamp*. Advancing many iterations at once queues a burst of samples,
  and the first one delivered afterwards is the **oldest** of that burst — so a
  sequence-counter barrier reads a scan from the middle of the advance while
  believing it has the one from the end. That is the staleness bug, wearing the
  costume of the fix for it.
- **Never wait for sensor data from a paused world.** A paused Gazebo publishes
  nothing at all — sensors update in `PostUpdate`, which a paused world never
  runs. Step it first, or block until the timeout on a simulator that is
  working perfectly.

## Frames: world in, odom out

Reset sampling and `set_pose` work in **world** coordinates — `arena.py` is
where the walls and obstacles are. The observation is built in the **odom**
frame, because that is the only frame `policy_node.py` will have on a robot.

The two are bridged once, in `reset()`: the goal is expressed relative to the
robot's sampled world pose, then re-planted at whatever pose odometry reports
after the teleport. Odometry is dead-reckoned and a teleport does **not** move
it, so anchoring on the measured odom pose rather than assuming it reads zero
is what keeps the goal where it was sampled.

The practical consequence, which surprises everyone once: teleporting the robot
mid-episode does not change `distance_to_goal`. Tests that need the robot at
the goal move the goal.

## Layout

```
CONTRACTS.md              authoritative RL interface spec
WORKPLAN.md               the phased plan; which phase you are in
docker/                   Dockerfile, compose, entrypoint
src/robot_rl_env/
  robot_rl_env/
    contract.py           every contract number, transcribed once
    arena.py              PURE — arena geometry + free-space rejection sampling
    observation.py        PURE, SHARED obs assembly (train + deploy)
    action.py             PURE, SHARED action -> (v, omega) mapping
    sim_control.py        Phase 2a — world control / set_pose service client
    observation_node.py   Phase 2b — ROS subscriber layer around observation.py
    env.py                Phase 2c — RobotNavEnv(gymnasium.Env)
    eval_set.py           Phase 3 — PURE, the held-out evaluation episodes
    hyperparams.py        Phase 3 — PURE, SAC/PPO config and run constants
    sim_launcher.py       Phase 3 — one isolated simulator per worker
    vec_env.py            Phase 3 — vec env + fixed-episode evaluation env
    callbacks.py          Phase 3 — curriculum, success/collision/timeout logs
    train.py              Phase 3 — entry point
    evaluate.py           Phase 3 — deterministic scoring, prints + JSON
    deploy.py             Phase 4 — PURE, the deployment decision logic
    export_policy.py      Phase 4 — SB3 zip -> TorchScript, and the loader
    policy_node.py        Phase 4 — ROS 2 inference node
    deploy_eval.py        Phase 4 — the sim-to-deployment gap measurement
  worlds/arena.sdf        10x10 m arena, 7 obstacles, 1 ms physics step
  models/diffbot/         chassis, diff drive, 360-beam LiDAR
  config/bridge.yaml      ros_gz_bridge topic + service mapping
  launch/world.launch.py  headless:=true, paused:=true — training defaults
  launch/deploy.launch.py Phase 4 — the same world with paused:=false
  test/                   test_env.py is @pytest.mark.sim; the rest are pure
scripts/verify_phase1.sh  text-output verification of the bridge
scripts/verify_phase2.sh  launches the world, then runs phase2_checks.py
scripts/phase2_checks.py  Phase 2 exit criteria as PASS/FAIL lines
scripts/verify_phase4.sh  train tiny -> export -> free-running eval, end to end
```


## Pure core, ROS shell

The parts that are hard to get right are deliberately ROS-free, so their tests
run in milliseconds on the Windows host with no simulator:

- `contract.py` — every number in `CONTRACTS.md`, once. Nothing downstream
  re-declares a threshold, a rate, or a limit. Change a value here *and* in
  `CONTRACTS.md`; never only in a consumer.
- `arena.py` — obstacle geometry and reset rejection sampling. Its `OBSTACLES`
  table intentionally duplicates `worlds/arena.sdf`; `test_arena.py` parses the
  SDF and asserts the two agree, so the copy cannot drift. Do not "fix" the
  duplication by parsing SDF at runtime.
- `observation.py` — the 26-vector assembly. `observation_node.py` is only
  subscriptions, QoS, and the stamp barrier; all arithmetic lives in the pure
  module, which is also what `policy_node.py` must import.

Similar constants also appear in `models/diffbot/model.sdf` (beam count, LiDAR
range, sensor pose) and `worlds/arena.sdf` (`max_step_size`). `contract.py`'s
docstrings name the SDF element each one must match, and `test_contract.py`
checks them.

## Commands

Everything runs inside the container; the host is Windows.

```bash
make build          # docker build
make shell          # interactive container shell, workspace mounted at /ws
make test           # colcon build + pytest, headless
make lint           # ruff check src scripts
make verify         # Phase 1 bridge verification, prints pass/fail per check
make verify2        # Phase 2 exit criteria: episodes, throughput, memory, timing
make verify2 EPISODES=25   # shorter run; the count cannot be appended to the
                           # compose service command, which is why it is a var
make world          # launch the arena headless and paused, for manual poking
make train ALGO=sac SEED=0 ENVS=4    # Phase 3 training � hours; launch it yourself
make evaluate MODEL=runs/sac-seed0/best/best_model.zip
make export-policy MODEL=runs/sac-seed0/best/best_model.zip  # -> <run>/policy.pt
make board          # TensorBoard over runs/ on :6006
make verify4        # Phase 4 smoke: train tiny -> export -> free-running eval
make deploy POLICY=runs/sac-seed0/policy.pt   # unpaused world + inference node
make gap            # Phase 4 measurement: free-running vs step-synchronized
make clean          # rm -rf build install log
```

Inside the container the workspace is `/ws`; build with
`colcon build --symlink-install && source install/setup.bash`.

To run one test, or one file, from a container shell — the world is launched by
a session fixture in `test_env.py`, so no separate `ros2 launch` is needed:

```bash
colcon build --symlink-install && source install/setup.bash
pytest src/robot_rl_env/test/test_env.py -k collision -x
pytest src/robot_rl_env/test -m "not sim"      # pure tests only, seconds
```

The pure tests also run on the Windows host with no container — but **only from
the package root**, because there is no colcon install to put `robot_rl_env` on
the path:

```powershell
cd src/robot_rl_env; python -m pytest test -q   # 100 passed, test_env skipped
```

From the repo root the same command dies in collection with
`ModuleNotFoundError: robot_rl_env`. CI's fast job sets
`working-directory: src/robot_rl_env` for exactly this reason, and runs
`ruff check src scripts` from the repo root — the two working directories are
also why `known-first-party` is pinned in `pyproject.toml`. CI's second job only
proves the image still builds; no runner executes the simulator.

`test_env.py` skips itself via `pytest.importorskip("rclpy")`, so a host run
reports it as skipped rather than failing.

Two gotchas that cost real time if you meet them cold:

- ROS ships pytest plugins written against pytest 7, and the image installs
  pytest 9. Every invocation dies before collection with an INTERNALERROR that
  names pluggy, not ROS. `pyproject.toml` disables them by entry-point name
  (`launch_testing`, `launch_ros` — *not* the distribution names).
- ROS setup scripts read unset variables, so `set -u` in a shell script must be
  lifted around `source /opt/ros/jazzy/setup.bash`.

## Phase 3: parallel workers, and what evaluation is measured against

Two rules specific to training, both of which produce a plausible-looking run
when broken:

- **Isolate both buses, not one.** Each worker needs `ROS_DOMAIN_ID` *and*
  `GZ_PARTITION`. gz-transport is a separate bus from DDS: with a shared
  partition, one worker's `/world/arena/control` call can be served by another
  worker's Gazebo, so a step advances someone else's world and the caller
  blocks on a stamp nobody will publish. It mostly works, then cross-wires
  under a discovery race. `sim_launcher.worker_environment` is the only place
  that should set either.
- **Evaluate on `eval_set.episodes()`, never on fresh random episodes.**
  Two runs scored on two different draws differ by the sampler as much as by
  the policy, and that difference is the same size as the effect being
  reported. The set is fixed, generated from a seed training never uses, and
  the callback's 20 episodes are the first 20 of the reported 100.

One contract-level fact that surfaced while building the eval set: 500 steps at
0.4 m/s covers 10.0 m, but the arena diagonal is 14.15 m — **the sampler can
emit episodes that no policy can finish**. Harmless in training (a hard
negative), so nothing was changed in `CONTRACTS.md`; on an evaluation set it
would cap the achievable success rate below 100% at an unknown and moving
value, so eval goals are capped at `eval_set.MAX_EVAL_DISTANCE`.

## Phase 4: what makes the gap a measurement rather than an anecdote

The point of Phase 4 is a *number*, and a number is only worth reporting if the
two things being subtracted differ in one respect. So:

- **Everything except the timing is shared code.** `deploy.DeploymentController`
  mirrors `env.step` — same goal tolerance, same collision threshold, same
  500-step limit, same step-count bookkeeping — and both call the same
  `assemble_observation` and the same `scale_action`. If the deployment loop
  also changed a termination rule, the measured difference would be part
  architecture and part bookkeeping with no way to separate them.
- **`deploy_eval` drives a real `PolicyNode`**, rather than running the same
  controller through its own loop. A harness with its own loop measures the
  harness.
- **A watchdog tick does not spend the step budget.** Observation index 25 is
  the fraction of the 500 steps used, learned against a counter that advanced
  once per action *issued*. Counting stopped ticks would run the deployment
  clock fast exactly when the system was under load — a gap manufactured by the
  instrument.
- **The safety gate is checked before the episode logic, not after.** 0.15 m is
  inside 0.18 m, so a safety trip is always also a collision; terminating first
  makes the gate unreachable code. A safety layer has to stop the robot when the
  surrounding logic is wrong, so it cannot sit downstream of it. The bookkeeping
  still records the collision — only the reported reason changes.

Two things about this container specifically, both measured rather than
assumed: observations reach the node ~121 ms stale on average (p95 183 ms,
max 243 ms) against training's fixed 50 ms, and 2% of ticks exceed the 200 ms
watchdog outright. And it sustains a real-time factor of only 0.3–0.5, which
means a 20 Hz wall-clock control loop is 40–65 Hz in *sim* time — a confound in
the headline gap that `deploy_eval` names in its output. The gap number itself
wants a host that holds RTF near 1.

## Working rules

- **Every task ends with a command whose text output verifies it.** An agent
  cannot see Gazebo. A bridge misconfiguration produces *silence*, not an
  error. Convert silence into a failing assertion.
- **Write the test first** for anything in Phase 2.
- **One phase per session.** Long sessions drift from `CONTRACTS.md` and start
  inventing helper abstractions nobody asked for.
- Commit and push after each small logical unit, not batched at the end.

## Current phase

**Phases 0–2 and 4 complete. Phase 3 is built but not yet run.** Update this
line when a phase closes; it is how the next session knows where to start.

Every Phase 3 and Phase 4 module exists and is tested, and both paths have been
smoke-run end to end in the container. `scripts/verify_phase4.sh` trains a
300-step throwaway policy, exports it to TorchScript, loads it in the real
deployment node, and scores it free-running — export agreement, `.pt` loading,
the world→odom goal bridge, episode termination and the gap report all pass.
**No real training run has happened yet**, so there are no results, and both of
the README's tables are still empty on purpose.

Next session: launch `make train ALGO=sac SEED=0` yourself, outside the agent
loop, and hand back the TensorBoard scalars or `runs/<algo>-seed<N>/eval.json`
as text. Three seeds per algorithm before anything is reported. Then
`make export-policy` and `make gap` fill the sim-to-deployment table.

Four deviations from `CONTRACTS.md` are outstanding and want a human ruling —
each is documented at the point of deviation, none is a silent divergence:

1. Reset does not restore the world (`contract.BRAKE_ITERATIONS`).
2. Episodes are therefore reproducible but not bit-identical
   (`test_same_seed_and_actions_give_identical_observations`).
3. `reset(options={"start": ..., "goal": ...})` is an addition to the specified
   options, needed for a held-out evaluation set (`env.reset`).
4. `CONTRACTS.md`'s shared-code requirement names only
   `assemble_observation`. Phase 4 extended the same treatment to the action
   mapping (`action.scale_action`), on the grounds that the argument for one is
   verbatim the argument for the other. The document has not been amended —
   it is authoritative and hand-reviewed, and this is a widening of a rule
   rather than a departure from one, but it is still a change to what the
   document says and wants signing off.
