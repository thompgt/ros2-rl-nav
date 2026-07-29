"""Guards the numbers in ``contract.py`` against the SDF and against each other.

These tests need no simulator and run in well under a second. Their job is to
catch the failure where someone edits ``arena.sdf`` to a 2 ms physics step, or
drops the LiDAR to 180 beams, and the observation pipeline silently starts
producing something that no longer matches ``CONTRACTS.md``.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from robot_rl_env import contract

PKG_ROOT = Path(__file__).resolve().parents[1]
WORLD_SDF = PKG_ROOT / "worlds" / "arena.sdf"
ROBOT_SDF = PKG_ROOT / "models" / "diffbot" / "model.sdf"


# --- internal consistency ----------------------------------------------------

def test_beam_pooling_is_exact():
    """Min-pooling requires the raw beam count to divide evenly."""
    assert contract.N_RAW_BEAMS % contract.N_BEAMS == 0
    assert contract.BEAM_POOL * contract.N_BEAMS == contract.N_RAW_BEAMS


def test_obs_dim_matches_its_components():
    # 20 beams + distance + sin + cos + 2 prev action + step fraction
    assert contract.OBS_DIM == contract.N_BEAMS + 1 + 2 + contract.ACT_DIM + 1


def test_step_duration_is_fifty_milliseconds():
    assert contract.STEP_DURATION == pytest.approx(0.05)
    assert contract.CONTROL_HZ == pytest.approx(20.0)


def test_safety_stop_is_tighter_than_training_collision_threshold():
    """The Phase 4 safety layer must not shadow the learned behaviour.

    If the hard stop fired *before* the distance the policy was trained to treat
    as a collision, the safety layer would be doing the navigating and the
    deployment gap measurement would be meaningless.
    """
    assert contract.SAFETY_STOP_THRESHOLD < contract.COLLISION_THRESHOLD


def test_arena_diagonal_normalizer_is_correct():
    expected = (2 * contract.ARENA_SIZE**2) ** 0.5
    assert contract.D_MAX == pytest.approx(expected, abs=0.05)


def test_goal_sampling_cannot_produce_a_pre_solved_episode():
    assert contract.MIN_START_GOAL_DISTANCE > contract.GOAL_TOLERANCE


# --- consistency with the SDF ------------------------------------------------

def test_world_physics_step_matches_contract():
    root = ET.parse(WORLD_SDF).getroot()
    step_sizes = [e.text for e in root.iter("max_step_size")]
    assert step_sizes, "arena.sdf declares no <max_step_size>"
    for value in step_sizes:
        assert float(value) == pytest.approx(contract.PHYSICS_STEP_SIZE), (
            "arena.sdf physics step disagrees with contract.PHYSICS_STEP_SIZE; "
            "every step() would advance the wrong amount of sim time"
        )


def test_world_name_matches_contract():
    root = ET.parse(WORLD_SDF).getroot()
    world = root.find("world")
    assert world is not None
    assert world.get("name") == contract.WORLD_NAME, (
        "the world control service path is built from contract.WORLD_NAME; "
        "a mismatch makes every service call silently target nothing"
    )


def test_lidar_beam_count_and_range_match_contract():
    root = ET.parse(ROBOT_SDF).getroot()
    samples = [e.text for e in root.iter("samples")]
    assert samples, "model.sdf declares no <samples>"
    assert int(samples[0]) == contract.N_RAW_BEAMS

    ray = next(root.iter("lidar"), None)
    if ray is None:
        ray = next(root.iter("ray"), None)
    assert ray is not None, "model.sdf has no lidar/ray sensor"
    max_range = next(ray.iter("max"))
    assert float(max_range.text) == pytest.approx(contract.LIDAR_MAX)


def test_robot_name_matches_contract():
    root = ET.parse(ROBOT_SDF).getroot()
    model = root.find("model")
    assert model is not None
    assert model.get("name") == contract.ROBOT_NAME
