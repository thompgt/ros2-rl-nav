"""Worker isolation. Pure: no simulator, no ROS, runs on the Windows host.

The launch path itself is exercised by ``test_env.py``'s session fixture, which
uses this module. What is tested here is the part that cannot be observed by
running one worker: that N of them would not collide. A cross-wired pair is not
a crash — it is two workers quietly sharing a simulator, which shows up as an
observation timeout in whichever one lost the race.
"""

import os

import pytest

from robot_rl_env import sim_launcher


def test_workers_get_distinct_domains_and_partitions():
    envs = [sim_launcher.worker_environment(i) for i in range(8)]
    assert len({e["ROS_DOMAIN_ID"] for e in envs}) == 8
    assert len({e["GZ_PARTITION"] for e in envs}) == 8


def test_both_buses_are_partitioned_not_just_one():
    """Setting only ROS_DOMAIN_ID is the failure mode worth a test.

    gz-transport is a separate bus from DDS: with a shared partition, one
    worker's world-control service call can be served by another worker's
    Gazebo, and the step that should have advanced its own world blocks
    forever on a stamp nobody will publish.
    """
    a, b = sim_launcher.worker_environment(0), sim_launcher.worker_environment(1)
    assert a["ROS_DOMAIN_ID"] != b["ROS_DOMAIN_ID"]
    assert a["GZ_PARTITION"] != b["GZ_PARTITION"]


def test_worker_zero_shares_the_compose_domain():
    """So a single-worker run stays inspectable from `make shell`."""
    assert sim_launcher.worker_environment(0)["ROS_DOMAIN_ID"] == "42"


def test_the_parent_environment_is_not_mutated():
    """A worker that edited os.environ would set the domain for every worker
    started after it, and the last one to start would win."""
    before = dict(os.environ)
    sim_launcher.worker_environment(3)
    assert dict(os.environ) == before


def test_the_returned_environment_inherits_the_rest_of_the_parent():
    """It is passed to `ros2 launch` wholesale: dropping PATH or
    AMENT_PREFIX_PATH would fail in a way that looks like a broken install."""
    env = sim_launcher.worker_environment(1, base={"PATH": "/usr/bin", "FOO": "bar"})
    assert env["PATH"] == "/usr/bin"
    assert env["FOO"] == "bar"


def test_running_past_the_portable_domain_range_raises():
    """Rather than wrapping, which would silently pair two workers on one
    domain -- the exact condition this module exists to prevent."""
    over = sim_launcher.DOMAIN_ID_MAX - sim_launcher.DOMAIN_ID_BASE + 1
    with pytest.raises(ValueError, match="portable maximum"):
        sim_launcher.worker_environment(over)

    # The last legal index must still work, so the bound is off-by-one-proof.
    assert sim_launcher.worker_environment(over - 1)["ROS_DOMAIN_ID"] == str(
        sim_launcher.DOMAIN_ID_MAX
    )


def test_a_negative_index_raises():
    with pytest.raises(ValueError, match="non-negative"):
        sim_launcher.worker_environment(-1)
