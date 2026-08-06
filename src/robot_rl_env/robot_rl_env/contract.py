"""Numeric constants from ``CONTRACTS.md``.

Every number the RL interface depends on lives here, once. The SDF world, the
observation assembler, the environment, and the deployment node all read from
this module so that changing a contract value cannot leave one consumer behind.

If you change a value here, re-read ``CONTRACTS.md`` and change it there too --
that document is authoritative and this module is its transcription.
"""

import math

# --- LiDAR -------------------------------------------------------------------
N_RAW_BEAMS = 360
"""Beams published by the sensor. Must match ``<samples>`` in models/diffbot."""

N_BEAMS = 20
"""Beams in the observation, after min-pooling."""

BEAM_POOL = N_RAW_BEAMS // N_BEAMS  # 18
"""Raw beams collapsed into each observation beam."""

LIDAR_MAX = 10.0
"""Metres. Must match ``<range><max>`` in models/diffbot."""

LIDAR_MIN_ANGLE = -math.pi
"""Radians, in the sensor frame. Must match ``<min_angle>`` in models/diffbot."""

LIDAR_ANGLE_INCREMENT = 2.0 * math.pi / N_RAW_BEAMS
"""Radians between consecutive beams: exactly one degree. ``<max_angle>`` is
one increment short of ``+pi`` so the last beam does not duplicate the first,
which is what makes the pooling blocks even."""

LIDAR_OFFSET_X = 0.15
"""Metres forward of the robot origin. Must match the sensor ``<pose>`` in
models/diffbot. Reset sampling needs it: clearance is measured from the robot
*origin*, but collision is measured from the *sensor*, and the sensor is closer
to whatever is in front."""

# --- Observation -------------------------------------------------------------
OBS_DIM = 26
"""20 pooled beams + goal distance + bearing sin/cos + prev action (2) + step."""

ARENA_SIZE = 10.0
"""Metres, side length of the square arena."""

D_MAX = 14.15
"""Metres. Arena diagonal; normalizes goal distance. ~= sqrt(2) * ARENA_SIZE."""

# --- Action ------------------------------------------------------------------
ACT_DIM = 2

MAX_LINEAR_VEL = 0.4
"""m/s. Action index 0 maps [-1, 1] -> [0, MAX_LINEAR_VEL]. No reversing."""

MAX_ANGULAR_VEL = 1.5
"""rad/s. Action index 1 maps [-1, 1] -> [-MAX, +MAX]."""

# --- Timing ------------------------------------------------------------------
PHYSICS_STEP_SIZE = 0.001
"""Seconds. Must match ``<max_step_size>`` in worlds/arena.sdf."""

PHYSICS_STEP_NS = 1_000_000
"""``PHYSICS_STEP_SIZE`` in integer nanoseconds. The env accumulates sim time in
these, never in float seconds -- see ``STEP_DURATION_NS``."""

SIM_STEPS_PER_ACTION = 50
"""Iterations advanced per env step."""

STEP_DURATION = PHYSICS_STEP_SIZE * SIM_STEPS_PER_ACTION  # 0.05 s
"""Seconds of sim time per env step. Exactly this, always. Tested."""

STEP_DURATION_NS = 50_000_000
"""``STEP_DURATION`` in integer nanoseconds.

Sim-time targets are accumulated in integer nanoseconds, never in float
seconds. ``target += 0.05`` five hundred times accumulates float error, and the
error only has to exceed the sensor stamp by one ULP for ``get_obs`` to wait for
a sample that will never come and time out on the last step of long episodes --
a bug that reproduces once every few thousand steps and looks like flakiness."""

CONTROL_HZ = 1.0 / STEP_DURATION  # 20 Hz
"""Implied control rate; the deployment node's timer runs at this rate."""

# --- Episode -----------------------------------------------------------------
MAX_EPISODE_STEPS = 500
"""Truncation limit."""

GOAL_TOLERANCE = 0.25
"""Metres. Within this distance the episode terminates as a success."""

COLLISION_THRESHOLD = 0.18
"""Metres. min(pooled_lidar) below this terminates the episode as a collision."""

SAFETY_STOP_THRESHOLD = 0.15
"""Metres. Phase 4 deployment hard-stop. Deliberately tighter than the training
collision threshold so the safety layer fires only when the policy has already
failed, rather than shadowing the behaviour the policy learned."""

WATCHDOG_TIMEOUT = 0.2
"""Seconds. Phase 4: zero /cmd_vel if no observation arrives within this."""

# --- Reward ------------------------------------------------------------------
STEP_COST = -0.01
ANGULAR_PENALTY = -0.05
GOAL_BONUS = 10.0
COLLISION_PENALTY = -10.0

# --- Reset sampling ----------------------------------------------------------
MIN_OBSTACLE_CLEARANCE = 0.4
"""Metres. Rejection radius when sampling robot and goal positions."""

MIN_START_GOAL_DISTANCE = 1.0
"""Metres. An already-solved episode teaches nothing."""

MAX_SAMPLE_ATTEMPTS = 1000
"""Rejection-sampling attempts before raising.

It raises rather than relaxing the clearance, because exhausting 1000 attempts
means the arena or the clearance changed such that free space is gone -- and a
silently relaxed clearance produces episodes that start in collision, which
reads as "the policy cannot learn" rather than as the configuration error it
is."""

BRAKE_ITERATIONS = 1000
"""Physics iterations of zero-velocity command before the reset teleport.

The robot must be at rest when an episode starts, or the episode inherits
momentum from the end of the previous one and a fixed seed stops reproducing.
DiffDrive drives the wheels to zero under its 10 rad/s^2 acceleration limit, so
8 rad/s (full speed) decays in 0.8 s; 1000 iterations is that plus margin, and
it settles to a measured 0.000 mm of residual drift.

Why not ``ControlWorld(reset.all)``
-----------------------------------
CONTRACTS.md ("Why reset does not restore the world") specifies braking and
names reset.all as the thing not to reach for. Both failure modes below were
measured against Harmonic in this container, not inferred:

1. It is asynchronous. It answers immediately and then swallows the ``set_pose``
   that has to follow it -- never answering that request at all -- roughly half
   the time. No ordering barrier closes the window: ``/clock`` returns to zero
   well before the entity manager will accept a pose. Resending the teleport
   does work, at about 0.4 s per reset.
2. Under repeated use it wedges the server outright. In a full test run,
   ``/world/arena/control`` itself stopped answering after a handful of
   episodes, and every subsequent test failed on a 10 s service timeout.

(2) is disqualifying for training, which needs thousands of resets. Braking is
synchronous, needs no barrier, and has never failed in this codebase.

What it costs: a world reset also restores wheel joint angles, so two episodes
would begin in a bit-identical simulator state. Braking leaves the wheels at
whatever rotation the last episode ended on, and the contact solver amplifies
that into millimetres of trajectory divergence over tens of steps. Episodes are
reproducible in their initial state and in the short term, but not bit-identical
over a full rollout. That is why CONTRACTS.md's determinism requirement is a
table of bounds rather than an equality, and why
``test_same_seed_and_actions_give_identical_observations`` grades the two
failure modes separately."""

ROBOT_SPAWN_Z = 0.06
"""Metres. Teleport height, matching the ``<include>`` pose in worlds/arena.sdf.
Spawning at z=0 drops the wheels through the ground plane."""

ROBOT_SPAWN_POSE = (-4.0, -4.0, 0.785)
"""World-frame ``(x, y, yaw)`` the robot is placed at by the ``<include>`` pose
in worlds/arena.sdf, which ``test_contract.py`` checks.

Training never needs this -- ``reset()`` samples a start and teleports there.
It matters for the deployment monitor, which has only ``odom`` and therefore
has to seed the odom -> world transform with the pose odometry started from.
"""

# --- Simulation plumbing -----------------------------------------------------
WORLD_NAME = "arena"
ROBOT_NAME = "diffbot"

WORLD_CONTROL_SERVICE = f"/world/{WORLD_NAME}/control"
SET_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"

OBS_TIMEOUT = 5.0
"""Seconds of wall-clock to wait for a fresh observation before RAISING.

This must raise. It must never return a cached observation -- see the
"never do these" list in CLAUDE.md.
"""
