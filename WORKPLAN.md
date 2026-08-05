# WORKPLAN

Goal: a Gazebo-simulated differential-drive robot exposed as a Gymnasium
environment over ROS 2, trained with SAC/PPO, and redeployed as a standalone
ROS 2 inference node.

Realistic timeline: 3–5 weeks part-time. Phases 0–2 are where projects die.

---

## Phase 0 — Contracts and scaffolding ✅

`CONTRACTS.md`, `CLAUDE.md`, repo scaffold, Dockerfile, `pyproject.toml`,
pytest config, empty modules, Makefile.

**Exit criterion:** `make build` succeeds and
`ros2 launch robot_rl_env world.launch.py` shows a robot in Gazebo.

## Phase 1 — Simulation and bridge ✅

- SDF world: 10×10 m arena, walls, 7 static box/cylinder obstacles
- Robot: diff-drive plugin, 360-beam 2D LiDAR (10 m), odometry
- `ros_gz_bridge` config for `/scan`, `/odom`, `/cmd_vel`, `/clock`, and the
  world control + set-pose services
- Launch file with `headless:=true` (default) and `gui:=true`

**Verify by hand — do not delegate:**

- `ros2 topic hz /scan` matches the configured sensor rate
- `ros2 topic echo /clock` freezes when paused, advances when stepped
- `/cmd_vel` driven manually produces sane odometry (`teleop_twist_keyboard`)
- headless launches no GUI process (`pgrep -a gz`)

Bridge misconfigurations produce **silence, not errors**. An agent cannot see
silence. This is the single most common place to lose three days — hence
`scripts/verify_phase1.sh`, which turns each check into a printed PASS/FAIL.

## Phase 2 — The Gym environment (the real work) ✅

Build in this order, testing each layer:

**2a. Sim control client** — `pause()`, `unpause()`, `step(n)`, `reset_world()`,
`set_entity_pose(name, pose)`. Test standalone, no Gym involved.

**2b. Observation assembler** — subscribes to `/scan` and `/odom` on a dedicated
callback group with a `MultiThreadedExecutor`. `get_obs(min_stamp)` blocks until
`stamp >= min_stamp`, with a timeout that **raises** rather than returning stale
data. See the "never do these" list in `CLAUDE.md`.

**2c. `RobotNavEnv(gymnasium.Env)`** — wires 2a and 2b to `CONTRACTS.md`.

Tests to write first:

- `check_env(RobotNavEnv())` passes
- same seed → identical observation sequence under a fixed action sequence
- every `step()` advances `/clock` by exactly 50 ms
- collision termination fires when the robot is teleported into a wall
- reward is positive when the robot is teleported closer to the goal

**Exit criterion:** a random-action agent runs 100 episodes without hanging,
crashing, or leaking memory. Log `sim_time / wall_time`; need ≥ 5× for training
to be practical. Below 1×, profile before proceeding.

## Phase 3 — Training (mostly waiting) — built, not yet run

Everything below exists and is tested; what remains is executing the runs.

- `train.py` on SB3 with configs, TensorBoard, checkpointing, `VecNormalize` ✅
- `SubprocVecEnv` with N parallel Gazebo instances on separate `GZ_PARTITION`
  values and ROS domain IDs ✅ (`sim_launcher.py`, `vec_env.py`)
- `EvalCallback` on a held-out set of fixed start/goal pairs ✅ (`eval_set.py`;
  goals are capped at `MAX_EVAL_DISTANCE` because 500 steps at 0.4 m/s covers
  10 m and the arena diagonal is 14.15 m)
- `evaluate.py` → success rate, collision rate, timeout rate, mean episode
  length, mean path length and path efficiency over 100 deterministic
  episodes ✅

SAC first (~150k steps), then PPO for comparison. **Three seeds minimum per
algorithm.** Reporting single-seed RL results is a tell that you haven't done
RL before; reporting seed variance is a tell that you have.

Expect the first run to fail. Usual causes, in frequency order:

1. **Reward hacking** — spinning or oscillating to farm progress reward. See the
   escalation ladder in `CONTRACTS.md`.
2. **No exploration signal** — goals too far, agent never sees a success. Use
   the `goal_radius` curriculum hook: start at 2 m, expand as success crosses
   70%.
3. **Observation scaling** — `VecNormalize` usually saves you, but check.

**Exit criterion:** ≥ 85% success on the held-out set, and a TensorBoard curve
you'd show someone.

Keep training runs out of the agent loop. Launch them yourself; hand back
TensorBoard scalars or the eval JSON as text.

## Phase 4 — Deployment node ✅ built and smoke-tested

The part that separates this from a Gym tutorial, and the part interviewers ask
about.

- `export_policy.py`: SB3 `.zip` → TorchScript, with the trace verified against
  `model.predict` over the whole observation box before anything is written ✅
- `policy_node.py`: loads the `.pt` (no SB3 at runtime), subscribes
  `/scan` + `/odom`, publishes `/cmd_vel` on a 20 Hz timer ✅
- Reuses `assemble_observation` and `scale_action` — reimplements neither ✅
- `geometry_msgs/PoseStamped` goals on `/goal_pose`, frame-checked rather than
  transformed ✅
- Safety layer: hard stop below 0.15 m, watchdog zeroing `/cmd_vel` after
  200 ms without an observation ✅ (`deploy.py`, pure and tested)
- World runs **unpaused, real time** ✅ (`deploy.launch.py`)
- `deploy_eval.py`: the same held-out episodes, free-running, side by side with
  the step-synchronized numbers ✅

**Exit criterion (open):** the gap itself, which needs a trained policy. The
harness is verified by `scripts/verify_phase4.sh`.

Two things the smoke run already established, before any policy exists to be
blamed for them:

- Observations reach the node ~121 ms stale (p95 183 ms) against training's
  fixed 50 ms, and 2% of ticks exceed the watchdog outright. That is the gap's
  mechanism.
- This container sustains a real-time factor of 0.3–0.5, so a 20 Hz wall-clock
  control loop is 40–65 Hz in *sim* time. That is a confound in the headline
  number, not a detail: take the measurement on a host that holds RTF near 1.

## Phase 5 — Packaging (partly done)

- ✅ Architecture diagram: both loops drawn, the step-sync one marked
- ✅ One command to reproduce: `docker compose -f docker/docker-compose.yml up train`
- ✅ CI: fast contract tests on the runner, plus every non-simulator test inside
  the image (which is the only place with rclpy, torch and SB3 together)
- ✅ The Phase 4 write-up — mechanism and caveat; the numbers wait on Phase 3
- ✅ Results table: SAC vs PPO, mean ± std over seeds — `report.py` aggregates
  `runs/<algo>-seed<N>/{eval,gap}.json` and splices both tables into the README
  between markers, so the numbers are generated rather than transcribed.
  Waiting only on runs to aggregate.
- ⬜ A GIF in the first 100 px. The recorder is built and tested (`record.py`,
  `make gif`) and draws the *observation* — the 20 pooled LiDAR sectors, not a
  Gazebo screenshot, which would need a GPU for ogre2 and would hide what the
  policy actually sees. There is no trained policy yet, so there is nothing
  honest to record.

---

## Reduced-scope fallback

If Phase 1 or 2 stalls badly: cut Gazebo, wrap a PyBullet differential-drive
robot instead, keep the ROS 2 deployment node. You lose bridge complexity but
keep the training loop, the policy export, and the sim-to-deployment analysis —
most of the value. **Do not cut Phase 4.** A training-only project is one of
thousands on GitHub.

## Agent failure modes to watch for

- Reaching for Gazebo Classic APIs — heavily represented in training data,
  entirely wrong for Harmonic
- Silent fallbacks on observation timeout
- Reimplementing observation preprocessing in the deployment node
- `rclpy.spin_once()` in the step loop instead of a properly configured
  multithreaded executor with callback groups
- Suggesting `time.sleep()` anywhere in `step()`
- Driving the deployment node's control timer from `/clock`, which looks like
  a fix and quietly subtracts the effect Phase 4 measures
