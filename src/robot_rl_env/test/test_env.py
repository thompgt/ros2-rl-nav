"""Phase 2c integration tests. These need a live Gazebo.

Everything here launches the real world, the real bridge, and the real
simulator, because every failure these tests exist to catch lives in the seam
between them -- and a mocked service client would agree with any implementation
that type-checks.

They are skipped automatically where ``rclpy`` is not importable (the Windows
host), so ``pytest`` still runs the pure tests there. Inside the container they
run: ``make test``.

The world is launched once per session and each test calls ``reset()``, which
restores it. The two tests that deliberately drive the robot into a wall are
therefore harmless to whatever runs next.
"""

import math

import numpy as np
import pytest

pytest.importorskip("rclpy", reason="ROS 2 is not available on this host")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from robot_rl_env import arena, contract, sim_launcher  # noqa: E402
from robot_rl_env.env import RobotNavEnv  # noqa: E402
from robot_rl_env.observation import from_body_frame  # noqa: E402

pytestmark = pytest.mark.sim

LAUNCH_TIMEOUT = 90.0
"""Seconds to wait for gz + both bridges to come up. Generous: a cold container
under Docker Desktop is slow, and a flaky timeout here would read as a bug in
the environment rather than in the fixture."""


@pytest.fixture(scope="session")
def simulator():
    """Launch the arena headless and paused for the whole session.

    Through ``sim_launcher``, which is also what training uses to start its
    parallel workers -- so the launch and readiness path these tests exercise
    is the one Phase 3 depends on, rather than a second copy that agrees with
    it today.
    """
    sim = sim_launcher.Simulator(index=0, timeout=LAUNCH_TIMEOUT)
    try:
        sim.start()
    except (RuntimeError, TimeoutError) as exc:
        pytest.fail(str(exc))
    yield sim
    sim.stop()


@pytest.fixture(scope="module")
def env(simulator):
    e = RobotNavEnv()
    yield e
    e.close()


# --- the Gymnasium contract ---------------------------------------------------

def test_check_env_passes_except_for_step_determinism(simulator):
    """The API conformance suite, on its own env instance.

    One assertion inside ``check_env`` is expected to fail, and only one:
    ``check_step_determinism`` requires two identically-seeded resets followed
    by one identical action to produce *bit-equivalent* observations. They do
    not, because reset brakes and teleports rather than resetting the world --
    Harmonic's world reset drops the teleport that follows it and wedges the
    server under repeated use, so it is unusable for training. The wheels keep
    their rotation across a teleport, and the contact solver turns that into
    sub-millimetre pose differences that a LiDAR beam grazing an obstacle edge
    amplifies into a large single-channel divergence.

    See ``contract.BRAKE_ITERATIONS`` for the measurements, and
    ``test_same_seed_and_actions_give_identical_observations`` for the bound
    that replaces the exact check.

    Tolerating it *by message* rather than skipping the whole call is the point:
    every other conformance check -- spaces, dtypes, reset signature, info
    contents, seeding of ``np_random`` -- still has to pass, and any *new*
    failure still fails this test.
    """
    e = RobotNavEnv()
    try:
        check_env(e, skip_render_check=True)
    except AssertionError as exc:
        if "Deterministic step observations are not equivalent" not in str(exc):
            raise
        pytest.xfail(f"known deviation from CONTRACTS.md determinism: {exc}")
    finally:
        e.close()


def test_reset_returns_an_observation_inside_the_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32
    for key in ("distance_to_goal", "min_lidar", "sim_time", "is_success", "collided"):
        assert key in info, f"CONTRACTS.md requires info['{key}']"
    assert info["is_success"] is False


def test_step_before_reset_raises(simulator):
    e = RobotNavEnv()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            e.step(np.zeros(contract.ACT_DIM, dtype=np.float32))
    finally:
        e.close()


# --- the load-bearing invariant -----------------------------------------------

def test_step_advances_clock_by_exactly_one_step_duration(env):
    """The whole architecture in one assertion.

    If this drifts, the policy is being trained on a variable action duration
    and nothing else in the project means anything.
    """
    _, info = env.reset(seed=1)
    # The sensor stamp, not /clock. Advancing the world in bursts (the reset
    # brake is 1000 iterations) queues a thousand clock messages against a
    # depth-10 subscription, so /clock read immediately after a reset reports
    # whatever survived the queue. The stamp on the sample the env actually
    # blocked for is exact by construction.
    stamps = [int(round(info["sim_time"] * 1e9))]
    for _ in range(10):
        _, _, _, _, info = env.step(np.array([0.5, 0.2], dtype=np.float32))
        stamps.append(int(round(info["sim_time"] * 1e9)))

    deltas = np.diff(stamps)
    assert (deltas == contract.STEP_DURATION_NS).all(), (
        f"sim time advanced by {deltas.tolist()} ns per step, expected "
        f"{contract.STEP_DURATION_NS} exactly. Check <update_rate> in "
        f"models/diffbot/model.sdf against contract.CONTROL_HZ."
    )


def test_same_seed_and_actions_give_identical_observations(env):
    """The canary for async leakage into the step loop.

    Measured on *relative* geometry, for a reason worth understanding before
    changing this test.

    Absolute odom coordinates are not comparable across episodes at all.
    Odometry is dead-reckoned and nothing ever zeroes it, so two episodes start
    from whatever the odom frame had accumulated -- measured at 360 mm and
    0.14 rad apart here. That offset is constant for the whole rollout: the
    trajectories are the *same* trajectory, rigidly displaced. Asserting on raw
    poses reports a 360 mm "divergence" for a perfectly reproducible episode.

    Everything the policy sees is relative -- distance and bearing to a goal
    that reset placed relative to the robot, and LiDAR from the robot's own
    frame -- so relative quantities are what have to reproduce, and they do, to
    within the sub-millimetre settling difference left by teleporting a robot
    whose wheels are at an arbitrary rotation. See ``contract.BRAKE_ITERATIONS``.

    Bit-equality is out of reach even so, and not because of the millimetres: a
    single LiDAR beam grazing an obstacle edge turns a micrometre of pose
    difference into a full-scale swing in one channel. Measured max divergence
    over 30 steps is ~0.005 in the goal channels, against the ~0.075 per step a
    one-step timing error would introduce while turning.

    What this test cannot catch, by construction, is a *systematic* lag -- both
    rollouts would carry it equally. ``scripts/phase2_checks.py`` measures that
    directly against an analytic prediction.

    What *is* asserted exactly is the goal encoding at reset, which is computed
    rather than simulated and has no excuse to vary.
    """
    actions = [
        np.array([math.sin(i * 0.3), math.cos(i * 0.7)], dtype=np.float32)
        for i in range(30)
    ]

    def rollout():
        obs, info = env.reset(seed=99)
        observations, distances = [obs], [info["distance_to_goal"]]
        for a in actions:
            obs, _, terminated, truncated, info = env.step(a)
            observations.append(obs)
            distances.append(info["distance_to_goal"])
            if terminated or truncated:
                break
        return np.array(observations), np.array(distances)

    (obs_a, dist_a), (obs_b, dist_b) = rollout(), rollout()
    assert obs_a.shape == obs_b.shape, "the two rollouts terminated at different steps"

    # Channels 20-24 of the first observation: goal distance, bearing sin/cos,
    # and the (zero) previous action. Pure arithmetic over the sampled goal and
    # the reset pose -- if these differ, the seeding or the frame transform is
    # non-deterministic, which no amount of physics would explain.
    goal_channels = slice(contract.N_BEAMS, contract.N_BEAMS + 5)
    assert np.array_equal(obs_a[0][goal_channels], obs_b[0][goal_channels]), (
        f"the goal encoding differs between identically-seeded resets: "
        f"{obs_a[0][goal_channels]} vs {obs_b[0][goal_channels]}"
    )

    # Distance to goal: relative, in metres, and the quantity the reward is
    # computed from. 10 mm is well above the observed sub-millimetre settling
    # and well below the 20 mm a single step of travel covers at full speed.
    distance_error = np.abs(dist_a - dist_b).max()
    assert distance_error < 0.010, (
        f"identical rollouts diverged by {distance_error * 1000:.1f} mm of "
        f"goal distance. Something in the step loop depends on wall-clock "
        f"ordering."
    )

    # The goal channels of the observation itself, which is what the policy
    # actually consumes. Excludes the LiDAR block: one grazing beam swings a
    # full channel off a micrometre of pose difference and would make this a
    # test of obstacle-edge geometry rather than of the step loop.
    #
    # 0.04 is a deliberately modest margin: the measured divergence over these
    # 30 steps is 0.022 -- about a degree of accumulated heading -- against the
    # ~0.075 per step a one-step timing error produces while turning at
    # 1.5 rad/s. Only a factor of three separates them, so this assertion is
    # the weaker of the two here; the distance bound above is the primary one.
    bearing = slice(contract.N_BEAMS, contract.N_BEAMS + 3)
    observation_error = np.abs(obs_a[:, bearing] - obs_b[:, bearing]).max()
    assert observation_error < 0.04, (
        f"goal distance/bearing channels diverged by {observation_error:.4f} "
        f"between identical rollouts (measured baseline 0.022; a one-step "
        f"timing error while turning would be ~0.075)"
    )


# --- termination --------------------------------------------------------------

def test_collision_terminates_when_teleported_into_a_wall(env):
    """Teleport rather than drive: driving into a wall takes hundreds of steps
    and depends on the very controller under test."""
    env.reset(seed=2)
    # x = 4.7 puts the LiDAR (0.15 m forward) 0.15 m from the wall face at
    # x = 5.0, inside the 0.18 m collision threshold. y = 0 is clear of every
    # obstacle in arena.OBSTACLES.
    env._sim.set_entity_pose(contract.ROBOT_NAME, 4.7, 0.0, 0.0)

    _, reward, terminated, truncated, info = env.step(
        np.array([-1.0, 0.0], dtype=np.float32)  # a[0] = -1 is a full stop
    )

    assert info["min_lidar"] < contract.COLLISION_THRESHOLD
    assert terminated and not truncated
    assert info["collided"] is True
    assert info["is_success"] is False
    assert reward < contract.COLLISION_PENALTY / 2


def test_reaching_the_goal_terminates_as_a_success(env):
    """Move the goal onto the robot rather than the robot onto the goal.

    Teleporting the robot would not work: the goal lives in the odom frame,
    odometry is dead-reckoned, and a teleport moves the robot without moving
    odometry -- so the measured distance would not change at all. That is worth
    knowing about this environment, and this test is where it is written down.
    """
    _, info = env.reset(seed=3)
    x, y, _ = info["robot_pose"]
    env._goal_odom = (x + 0.05, y)
    env._prev_distance = 0.05

    _, reward, terminated, truncated, info = env.step(
        np.array([-1.0, 0.0], dtype=np.float32)
    )

    assert info["distance_to_goal"] < contract.GOAL_TOLERANCE
    assert terminated and not truncated
    assert info["is_success"] is True
    assert reward > contract.GOAL_BONUS / 2


def test_truncation_fires_at_the_step_limit(env):
    """500 steps of a full-left turn: it spins in place, so it neither reaches
    a goal nor hits anything, and the episode can only end on the limit."""
    env.reset(seed=4)
    spin = np.array([-1.0, 1.0], dtype=np.float32)
    for _ in range(contract.MAX_EPISODE_STEPS):
        _, _, terminated, truncated, info = env.step(spin)
        if terminated or truncated:
            break
    assert info["step"] == contract.MAX_EPISODE_STEPS, "episode ended early"
    assert truncated and not terminated, "the step limit must truncate, not terminate"


# --- reward -------------------------------------------------------------------

def test_reward_is_positive_when_driving_toward_the_goal(env):
    """Over a stretch long enough to out-earn the step cost.

    A single step from rest cannot: ``max_wheel_acceleration`` is 10 rad/s^2, so
    the first 50 ms covers well under a millimetre while the step cost is a
    flat -0.01. That is the intended shape of the reward -- dithering is meant
    to be unprofitable -- and a test asserting a positive single step would be
    asserting something the contract does not say.
    """
    _, info = env.reset(seed=6)
    x, y, yaw = info["robot_pose"]
    # 3 m dead ahead in the odom frame: whatever the robot is facing, driving
    # forward closes on it.
    env._goal_odom = from_body_frame((3.0, 0.0), (x, y), yaw)
    env._prev_distance = math.dist((x, y), env._goal_odom)
    start_distance = env._prev_distance

    total = 0.0
    for _ in range(40):  # 2 s of sim time
        _, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        total += reward
        if terminated or truncated:
            break

    assert info["distance_to_goal"] < start_distance, "the robot did not close on the goal"
    assert total > 0.0, f"driving straight at the goal netted {total:.3f}"


def test_spinning_in_place_is_penalized(env):
    """The anti-reward-hacking term. Spinning makes no progress and must cost."""
    env.reset(seed=7)
    total = 0.0
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(
            np.array([-1.0, 1.0], dtype=np.float32)
        )
        total += reward
        if terminated or truncated:
            break
    assert total < 0.0, f"spinning in place netted {total:.3f}, which is farmable"


# --- reset sampling -----------------------------------------------------------

def test_goal_radius_option_caps_the_goal_distance(env):
    """The Phase 3 curriculum hook. Checked in world coordinates, which is
    where the sampler works."""
    for seed in range(5):
        _, info = env.reset(seed=seed, options={"goal_radius": 2.0})
        start = info["start_world"][:2]
        assert math.dist(start, info["goal_world"]) <= 2.0 + 1e-9


def test_fixed_episode_option_overrides_sampling(env):
    """Phase 3's held-out evaluation set. Two resets with the same fixed
    episode and *different* seeds must place the robot identically -- if the
    seed still moved anything, the eval set would not be held constant across
    the runs it exists to compare."""
    start, goal = (3.5, 3.5, 0.0), (-3.0, -4.0)
    for seed in (0, 999):
        _, info = env.reset(seed=seed, options={"start": start, "goal": goal})
        assert info["start_world"] == pytest.approx(start)
        assert info["goal_world"] == pytest.approx(goal)
        assert info["distance_to_goal"] == pytest.approx(math.dist(start[:2], goal), abs=0.05)


def test_a_fixed_episode_inside_an_obstacle_raises(env):
    """Rejected before the teleport, not diagnosed afterwards from a zero
    success rate."""
    with pytest.raises(ValueError, match="clearance"):
        env.reset(options={"start": (-2.8, 2.6, 0.0), "goal": (3.5, 3.5)})


def test_half_specified_fixed_episodes_raise(env):
    with pytest.raises(ValueError, match="together"):
        env.reset(options={"start": (3.5, 3.5, 0.0)})
    with pytest.raises(ValueError, match="mutually exclusive"):
        env.reset(options={"start": (3.5, 3.5, 0.0), "goal": (-3.0, -4.0), "goal_radius": 2.0})


def test_reset_samples_a_goal_that_is_not_already_reached(env):
    for seed in range(5):
        _, info = env.reset(seed=seed)
        assert info["distance_to_goal"] > contract.GOAL_TOLERANCE
        assert info["is_success"] is False


def test_reset_starts_the_robot_clear_of_obstacles(env):
    """A robot spawned in a wall reports a collision on step 1 forever, and it
    reads as "the policy cannot learn" rather than as a sampling bug."""
    for seed in range(5):
        _, info = env.reset(seed=seed)
        assert info["min_lidar"] >= contract.COLLISION_THRESHOLD


def test_reset_actually_teleports_the_robot_to_the_sampled_pose(env):
    """The one thing about reset that nothing else would catch.

    There is no ground-truth pose topic -- only dead-reckoned odometry, which
    reports (0, 0, 0) after a reset whether or not the teleport landed. If
    ``set_pose`` were queued behind the world reset and silently clobbered by
    it, every episode would start at the world origin, the observation would
    still be well-formed, and the only symptom would be a policy that never
    learns the arena.

    So: check the LiDAR against the analytic clearance at the sampled pose. The
    sensor sits 0.15 m ahead of the robot origin and arena.clearance() measures
    from the origin, which bounds the disagreement; anything larger means the
    robot is not where the sampler put it (or arena.OBSTACLES has drifted from
    the SDF).
    """
    tolerance = contract.LIDAR_OFFSET_X + 0.05  # sensor offset + beam/noise slop
    for seed in range(5):
        _, info = env.reset(seed=seed)
        x, y, _ = info["start_world"]
        predicted = arena.clearance(x, y)
        assert abs(info["min_lidar"] - predicted) <= tolerance, (
            f"seed {seed}: LiDAR reports {info['min_lidar']:.3f} m of clearance "
            f"at the sampled pose ({x:.2f}, {y:.2f}), where the arena geometry "
            f"predicts {predicted:.3f} m. The robot is not where reset() put it."
        )


# --- action scaling (pure) ----------------------------------------------------

def test_action_scaling_matches_the_contract():
    assert RobotNavEnv.scale_action([-1.0, 0.0]) == (0.0, 0.0)
    assert RobotNavEnv.scale_action([1.0, 0.0]) == (contract.MAX_LINEAR_VEL, 0.0)
    assert RobotNavEnv.scale_action([0.0, 0.0]) == (contract.MAX_LINEAR_VEL / 2, 0.0)
    assert RobotNavEnv.scale_action([-1.0, 1.0]) == (0.0, contract.MAX_ANGULAR_VEL)
    assert RobotNavEnv.scale_action([-1.0, -1.0]) == (0.0, -contract.MAX_ANGULAR_VEL)
    # Out-of-range input is clipped, never scaled past the envelope.
    assert RobotNavEnv.scale_action([5.0, 5.0]) == (
        contract.MAX_LINEAR_VEL,
        contract.MAX_ANGULAR_VEL,
    )


def test_no_action_can_command_reverse():
    """`a[0] = -1` is a stop. See CONTRACTS.md ("Action space")."""
    for a0 in np.linspace(-3.0, 3.0, 101):
        v, _ = RobotNavEnv.scale_action([a0, 0.0])
        assert 0.0 <= v <= contract.MAX_LINEAR_VEL
