# CONTRACTS.md

**Authoritative specification of the RL interface.** This file is the single
source of truth. Code serves this document, not the other way around. If an
implementation disagrees with this file, the implementation is wrong.

This document is hand-authored and reviewed by a human. Do not edit it to match
code that already exists.

---

## Observation space

```python
gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32)
```

| Index | Content | Source | Normalization |
|---|---|---|---|
| 0–19 | 20 downsampled LiDAR beams | `/scan`, 360 raw beams | see below |
| 20 | Goal distance | `/odom` + goal pose | `2·clip(d / D_MAX, 0, 1) − 1`, `D_MAX = 14.15` m (arena diagonal) |
| 21 | Goal bearing sine | `/odom` + goal pose | `sin(θ_goal − θ_robot)`, already in [−1, 1] |
| 22 | Goal bearing cosine | `/odom` + goal pose | `cos(θ_goal − θ_robot)`, already in [−1, 1] |
| 23 | Previous linear action | last action sent | already in [−1, 1] |
| 24 | Previous angular action | last action sent | already in [−1, 1] |
| 25 | Normalized step count | env counter | `2·(t / 500) − 1` |

### LiDAR downsampling — exact arithmetic

The raw scan has `N_RAW = 360` beams. It is reduced to `N_BEAMS = 20` by
**min-pooling** over contiguous blocks of `360 / 20 = 18` beams:

```python
ranges = np.asarray(scan.ranges, dtype=np.float32)          # shape (360,)
ranges = np.nan_to_num(ranges, nan=LIDAR_MAX, posinf=LIDAR_MAX, neginf=0.0)
ranges = np.clip(ranges, 0.0, LIDAR_MAX)                    # LIDAR_MAX = 10.0 m
pooled = ranges.reshape(N_BEAMS, 18).min(axis=1)            # shape (20,)
obs[0:20] = 2.0 * (pooled / LIDAR_MAX) - 1.0                # -> [-1, 1]
```

**Min-pooling, not mean-pooling.** Mean-pooling averages away a thin obstacle —
a chair leg between two open beams becomes free space, and the policy drives
into it. Minimum is the conservative choice and is what a collision-avoidance
observation must report.

`nan`/`+inf` (no return) maps to max range. `-inf` maps to 0. This ordering
matters: an unmapped `nan` propagates through the network and produces `nan`
actions with no error message.

### Bearing convention

`θ_goal` is `atan2(goal_y − robot_y, goal_x − robot_x)` in the **odom frame**.
`θ_robot` is the robot yaw extracted from the `/odom` quaternion. The angle fed
to sin/cos is the wrapped difference `θ_goal − θ_robot`, i.e. bearing to goal in
the **robot body frame**. Sine/cosine rather than a raw angle so the
representation is continuous across the ±π wrap.

---

## Action space

```python
gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

| Index | Meaning | Scaling to `geometry_msgs/Twist` |
|---|---|---|
| 0 | Linear velocity | `v = 0.4 · (a[0] + 1) / 2` → **[0, 0.4] m/s** |
| 1 | Angular velocity | `ω = 1.5 · a[1]` → **[−1.5, 1.5] rad/s** |

The robot **cannot reverse**. `a[0] = −1` is a full stop. This is deliberate: a
reversing policy escapes concave obstacles by backing out, which masks a
failure to learn forward obstacle avoidance and does not transfer to a robot
with forward-only LiDAR coverage.

Actions are clipped to `[−1, 1]` before scaling. Anything outside is a bug in
the caller, but clipping is cheap insurance against an exploding policy.

---

## Reward

Computed **after** the sim advances, using distance to goal before and after:

```python
r  = (d_prev - d_curr)          # progress, metres. Dominant term.
r += -0.01                      # step cost — penalize dithering
r += -0.05 * abs(a[1])          # angular penalty — penalize spinning
r += +10.0  if reached_goal     # terminal bonus
r += -10.0  if collided         # terminal penalty
```

### Anti-reward-hacking note

`d_prev − d_curr` is symmetric: a robot that oscillates toward and away from the
goal nets zero, not positive, so plain oscillation does not farm reward. The
failure mode that *does* appear is a slow orbit that trades a little progress
for a lot of time. The `−0.01` step cost and `−0.05·|ω|` angular penalty exist
to close that.

If reward hacking still appears in Phase 3, the escalation — **in this order** —
is:

1. Replace progress with the decrease in **best-ever** distance:
   `r_prog = max(0, d_best − d_curr)`, then `d_best = min(d_best, d_curr)`.
   This makes backtracking free but un-farmable.
2. Raise the angular penalty coefficient.
3. Only then touch the terminal bonuses.

Do not tune all three at once; you will not know which one worked.

---

## Termination and truncation

| Condition | Type | Threshold |
|---|---|---|
| Goal reached | `terminated = True` | Euclidean distance to goal < **0.25 m** |
| Collision | `terminated = True` | `min(pooled_lidar) < 0.18 m` |
| Time limit | `truncated = True` | **500** steps |

Termination and truncation are distinct per the Gymnasium API. Bootstrapping on
a truncated episode is correct; bootstrapping on a terminated one is not. SB3
handles this only if the flags are reported correctly.

The collision threshold (0.18 m) is checked against the **pooled** ranges, i.e.
after min-pooling — consistent with what the policy sees.

---

## Step timing

| Property | Value |
|---|---|
| Physics `max_step_size` | **1 ms** |
| Sim iterations per `step()` | **50** |
| Sim time per `step()` | **50 ms**, exactly, always |
| Control rate (implied) | 20 Hz |
| Wall-clock per `step()` | irrelevant and unbounded |

Every `step()` advances `/clock` by exactly 50 ms. This is a **testable
invariant**, not an aspiration — see `test_step_advances_clock_exactly`.

The world is **paused** except during the explicit stepping window inside
`step()`. See `CLAUDE.md` for the rationale; it is the load-bearing decision of
the whole project.

---

## Reset

`reset(seed=None, options=None)` must:

1. Seed a `np.random.Generator` from `seed` via `super().reset(seed=seed)`.
2. Bring the robot to rest: command zero velocity and advance
   **1000 physics iterations** (`contract.BRAKE_ITERATIONS`) before teleporting.
   See "Why reset does not restore the world" below.
3. Sample a robot pose: `(x, y)` uniform in the arena, yaw uniform in [−π, π],
   rejecting samples closer than **0.4 m** to any obstacle or wall.
4. Sample a goal `(x, y)` under the same free-space rejection, additionally
   rejecting samples closer than **1.0 m** to the sampled robot position (an
   already-solved episode teaches nothing).
5. Teleport the robot via the sim `set_pose` service, zero its velocity,
   advance **one** sim step so sensors repopulate, and assemble the first
   observation.
6. Return `(obs, info)`.

### Why reset does not restore the world

The robot must be at rest when an episode starts, or it inherits momentum from
the end of the previous one and a fixed seed stops reproducing. The obvious way
to get that is `ControlWorld(reset.all)` — restore the world, then teleport.
**Do not use it.** Both failure modes below were measured against Harmonic in
this project's container, not inferred from documentation:

1. **It is asynchronous.** It answers immediately and then swallows the
   `set_pose` that has to follow it — never answering that request at all —
   roughly half the time. No ordering barrier closes the window: `/clock`
   returns to zero well before the entity manager will accept a pose.
   Resending the teleport does work, at about 0.4 s per reset.
2. **Under repeated use it wedges the server.** In a full test run,
   `/world/arena/control` itself stopped answering after a handful of episodes,
   and every subsequent call failed on a 10 s service timeout.

(2) is disqualifying for training, which needs thousands of resets. Braking is
synchronous, needs no barrier, and settles to a measured 0.000 mm of residual
drift: DiffDrive decays 8 rad/s to zero in 0.8 s under its 10 rad/s²
acceleration limit, and 1000 iterations is that plus margin.

What this costs is stated rather than hidden. A world reset would also restore
wheel joint angles, so two identically-seeded episodes would begin in a
bit-identical simulator state. Braking leaves the wheels at whatever rotation
the last episode ended on, and the contact solver amplifies that into
millimetres of trajectory divergence over tens of steps — which is why the
determinism requirement below is a bound rather than an equality.

### Determinism requirement

Identical `seed` plus an identical action sequence must reproduce the episode.
This is the canary for accidental async leakage into the step loop, and it is
specified as a **bound**, not an equality, for the reason above.

| Quantity | Requirement |
|---|---|
| Goal encoding at reset (obs 20–24) | **Bit-identical.** `np.array_equal` |
| Distance to goal, per step | max abs difference **< 10 mm** |
| Goal distance/bearing channels (obs 20–22), per step | max abs difference **< 0.04** |

The goal encoding at reset is computed rather than simulated — pure arithmetic
over the sampled goal and the reset pose — so it has no excuse to vary, and it
is asserted exactly. Everything downstream of the physics is bounded instead.

The bounds are chosen against the error they exist to catch, not picked round:
a one-step timing error while turning at 1.5 rad/s moves the bearing channels
by ~0.075 per step, against a measured baseline divergence of 0.022 over 30
steps; one step of travel at full speed covers 20 mm, against a measured
sub-millimetre settling difference. The distance bound is the primary one — the
bearing bound is separated from what it catches by only a factor of three.

Two limits of this check, both by construction:

- **Absolute odom poses are not comparable across episodes at all.** Odometry
  is dead-reckoned and nothing zeroes it, so two episodes start from whatever
  the odom frame accumulated — measured 360 mm and 0.14 rad apart. That offset
  is constant for the rollout: the trajectories are the same trajectory,
  rigidly displaced. The requirement is stated on relative quantities because
  relative quantities are all the policy sees.
- **The LiDAR block is excluded.** A single beam grazing an obstacle edge turns
  a micrometre of pose difference into a full-scale swing in one channel. A
  bound over those channels would test obstacle-edge geometry, not the step
  loop.
- **A *systematic* lag is invisible here** — both rollouts carry it equally.
  `scripts/phase2_checks.py` measures that directly against an analytic
  prediction.

`check_env` asserts bit-equality and therefore fails on this deviation. It is
tolerated **by message** rather than skipped, so that every other conformance
check it performs — spaces, dtypes, reset signature, info contents, seeding —
still has to pass, and any new failure still fails.

### Reset options

Two keys, mutually exclusive. Passing both raises rather than resolving a
precedence — a caller that supplies a goal radius *and* a fixed goal has a
confused intent, and silently honouring one of them would produce an evaluation
set quietly narrower than the one asked for.

**`options={"goal_radius": r}` — the curriculum hook.** Caps the sampled
start-goal distance at `r` metres. Phase 3 starts goals within 2 m and expands
as success rate crosses 70%. Default (`None`) is the full arena. It may also be
set once as env state rather than per call, because SB3 cannot thread a
per-reset option through a vectorized reset.

**`options={"start": (x, y, yaw), "goal": (x, y)}` — the fixed episode.**
Replaces sampling entirely, in **world** coordinates. Both must be given
together; one without the other raises. The pair is validated against the same
free-space rejection the sampler uses (`arena.validate_episode`), so a fixed
episode cannot start inside an obstacle or specify an already-solved goal.

The fixed episode is what makes evaluation a measurement. `eval_set.episodes()`
generates a held-out set from a seed training never uses, and every run is
scored on that same set: two runs then differ by the policy rather than by
which episodes they happened to draw. Scoring on fresh random draws would let
the sampler move the number by as much as the policy does. Goals in the set are
additionally capped at `eval_set.MAX_EVAL_DISTANCE`, because 500 steps at
0.4 m/s covers 10.0 m and the arena diagonal is 14.15 m — the unrestricted
sampler can emit episodes no policy can finish, which is a harmless hard
negative in training and an unknown, moving ceiling on an evaluation set.

---

## Info dict

`step()` and `reset()` return an `info` dict containing at least:

```python
{
  "distance_to_goal": float,   # metres
  "min_lidar": float,          # metres, pooled
  "sim_time": float,           # seconds, from /clock
  "is_success": bool,          # required by SB3's EvalCallback for success rate
  "collided": bool,
}
```

`is_success` is the key SB3 looks for. Spell it exactly.

---

## Shared-code requirement

The mapping from `(LaserScan, Odometry, goal, prev_action, step_count)` to a
26-vector lives in **exactly one** function:

```python
robot_rl_env.observation.assemble_observation(...)
```

The training environment (`env.py`) and the deployment node (`policy_node.py`)
both call it. Neither reimplements it, partially or otherwise. Divergence
between training-time and inference-time preprocessing is the classic silent
failure of this kind of project: nothing errors, and the policy is simply worse
in deployment for reasons no log will show you.
