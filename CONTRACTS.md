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
2. Reset the world to its initial state.
3. Sample a robot pose: `(x, y)` uniform in the arena, yaw uniform in [−π, π],
   rejecting samples closer than **0.4 m** to any obstacle or wall.
4. Sample a goal `(x, y)` under the same free-space rejection, additionally
   rejecting samples closer than **1.0 m** to the sampled robot position (an
   already-solved episode teaches nothing).
5. Teleport the robot via the sim `set_pose` service, zero its velocity,
   advance **one** sim step so sensors repopulate, and assemble the first
   observation.
6. Return `(obs, info)`.

**Determinism requirement:** identical `seed` plus an identical action sequence
must produce a bit-identical observation sequence. This is tested, and it is the
canary for accidental async leakage into the step loop.

### Curriculum hook

`options={"goal_radius": r}` optionally caps the sampled goal distance at `r`
metres. Phase 3 uses this to start goals within 2 m and expand as success rate
crosses 70%. Default (`None`) is the full arena.

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
