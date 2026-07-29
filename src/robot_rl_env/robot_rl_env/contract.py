"""Numeric constants from ``CONTRACTS.md``.

Every number the RL interface depends on lives here, once. The SDF world, the
observation assembler, the environment, and the deployment node all read from
this module so that changing a contract value cannot leave one consumer behind.

If you change a value here, re-read ``CONTRACTS.md`` and change it there too --
that document is authoritative and this module is its transcription.
"""

# --- LiDAR -------------------------------------------------------------------
N_RAW_BEAMS = 360
"""Beams published by the sensor. Must match ``<samples>`` in models/diffbot."""

N_BEAMS = 20
"""Beams in the observation, after min-pooling."""

BEAM_POOL = N_RAW_BEAMS // N_BEAMS  # 18
"""Raw beams collapsed into each observation beam."""

LIDAR_MAX = 10.0
"""Metres. Must match ``<range><max>`` in models/diffbot."""

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

SIM_STEPS_PER_ACTION = 50
"""Iterations advanced per env step."""

STEP_DURATION = PHYSICS_STEP_SIZE * SIM_STEPS_PER_ACTION  # 0.05 s
"""Seconds of sim time per env step. Exactly this, always. Tested."""

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
