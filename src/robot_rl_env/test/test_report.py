"""Phase 5 -- the seed aggregation behind the README's tables.

Pure arithmetic and file discovery, so all of it runs on the host. The cases
that matter are the ones where a naive implementation produces a plausible
wrong number: a NaN poisoning a mean, a single seed reported as if it had a
spread, and a gap computed between two different sets of runs.
"""

from __future__ import annotations

import json
import math

import pytest

from robot_rl_env import report


def write_run(runs_dir, algorithm, seed, evaluation=None, gap=None):
    directory = runs_dir / f"{algorithm}-seed{seed}"
    directory.mkdir(parents=True)
    if evaluation is not None:
        (directory / "eval.json").write_text(json.dumps({"metrics": evaluation}))
    if gap is not None:
        (directory / "gap.json").write_text(json.dumps({"metrics": gap}))
    return directory


# --- discovery -------------------------------------------------------------


def test_discovery_finds_scored_runs_and_sorts_them(tmp_path):
    write_run(tmp_path, "ppo", 1, evaluation={"success_rate": 0.5})
    write_run(tmp_path, "sac", 2, evaluation={"success_rate": 0.9})
    write_run(tmp_path, "sac", 0, evaluation={"success_rate": 0.8})

    runs = report.discover(tmp_path)

    assert [run.name for run in runs] == ["ppo-seed1", "sac-seed0", "sac-seed2"]


def test_a_run_directory_with_no_scores_is_not_reported(tmp_path):
    """Training has finished, evaluation has not. Not an error, not a row."""
    (tmp_path / "sac-seed0").mkdir()
    (tmp_path / "sac-seed0" / "best").mkdir()

    assert report.discover(tmp_path) == []


def test_directories_that_are_not_runs_are_ignored(tmp_path):
    (tmp_path / "tensorboard").mkdir()
    (tmp_path / "sac-seed0-copy").mkdir()
    (tmp_path / "notes.txt").write_text("hello")
    write_run(tmp_path, "sac", 0, evaluation={"success_rate": 0.8})

    assert [run.name for run in report.discover(tmp_path)] == ["sac-seed0"]


def test_a_missing_runs_directory_is_empty_rather_than_an_error(tmp_path):
    assert report.discover(tmp_path / "nope") == []


def test_a_truncated_eval_json_raises_rather_than_dropping_a_seed(tmp_path):
    """Silently skipping it looks identical to a run that was never launched."""
    directory = tmp_path / "sac-seed0"
    directory.mkdir()
    (directory / "eval.json").write_text(json.dumps({"model": "x"}))

    with pytest.raises(ValueError, match="no 'metrics'"):
        report.discover(tmp_path)


def test_a_gap_only_run_is_still_discovered(tmp_path):
    write_run(tmp_path, "sac", 0, gap={"success_rate": 0.4})

    (run,) = report.discover(tmp_path)
    assert run.evaluation is None
    assert run.gap == {"success_rate": 0.4}


# --- aggregation -----------------------------------------------------------


def test_mean_and_sample_std_over_seeds():
    value = report.aggregate([0.8, 0.9, 1.0])

    assert value.mean == pytest.approx(0.9)
    # Sample std (ddof=1): 0.1, not the population 0.0816.
    assert value.std == pytest.approx(0.1)
    assert (value.n, value.n_total) == (3, 3)


def test_a_single_seed_has_no_spread():
    value = report.aggregate([0.8])

    assert value.mean == pytest.approx(0.8)
    assert value.std is None
    assert not value.partial


def test_nan_seeds_are_skipped_rather_than_poisoning_the_mean():
    """A seed with zero successes reports NaN path length. Two good seeds
    should still produce a number, and the cell should admit it used two."""
    value = report.aggregate([2.0, float("nan"), 4.0])

    assert value.mean == pytest.approx(3.0)
    assert (value.n, value.n_total) == (2, 3)
    assert value.partial


def test_all_nan_aggregates_to_nothing_rather_than_zero():
    assert report.aggregate([float("nan"), float("nan")]) is None


def test_no_values_aggregates_to_nothing():
    assert report.aggregate([]) is None


def test_collect_groups_by_algorithm(tmp_path):
    write_run(tmp_path, "sac", 0, evaluation={"success_rate": 0.8})
    write_run(tmp_path, "sac", 1, evaluation={"success_rate": 0.9})
    write_run(tmp_path, "ppo", 0, evaluation={"success_rate": 0.5})

    collected = report.collect(report.discover(tmp_path), "success_rate")

    assert collected["sac"].mean == pytest.approx(0.85)
    assert collected["ppo"].mean == pytest.approx(0.5)
    assert collected["ppo"].std is None


def test_collect_skips_runs_missing_the_metric(tmp_path):
    write_run(tmp_path, "sac", 0, evaluation={"success_rate": 0.8})
    write_run(tmp_path, "sac", 1, evaluation={"success_rate": 0.9, "mean_reward": 3.0})

    collected = report.collect(report.discover(tmp_path), "mean_reward")

    assert collected["sac"].mean == pytest.approx(3.0)
    assert collected["sac"].n_total == 1


# --- formatting ------------------------------------------------------------


def test_absent_values_render_as_an_em_dash_not_zero():
    assert report.format_cell(None) == "—"


def test_percentages_are_scaled_and_suffixed():
    cell = report.format_cell(report.aggregate([0.8, 0.9, 1.0]), digits=1, percent=True)

    assert cell == "90.0 ± 10.0%"


def test_a_single_seed_cell_says_so():
    assert "(1 seed)" in report.format_cell(report.aggregate([0.8]))


def test_a_partial_cell_names_the_seeds_it_used():
    cell = report.format_cell(report.aggregate([2.0, float("nan"), 4.0]))

    assert "(2/3 seeds)" in cell


# --- tables ----------------------------------------------------------------


def _full_metrics(success, collided=0.0, timeout=None):
    return {
        "success_rate": success,
        "collision_rate": collided,
        "timeout_rate": 1.0 - success - collided if timeout is None else timeout,
        "mean_path_length": 5.0,
        "mean_path_efficiency": 1.2,
    }


def test_results_table_has_one_row_per_algorithm(tmp_path):
    for seed in range(3):
        write_run(tmp_path, "sac", seed, evaluation=_full_metrics(0.8 + seed * 0.05))
    write_run(tmp_path, "ppo", 0, evaluation=_full_metrics(0.6))

    table = report.results_table(report.discover(tmp_path))

    assert "| PPO (1 seed)" in table
    assert "| SAC (3 seeds)" in table
    assert "85.0 ± 5.0%" in table


def test_sac_is_reported_above_ppo_regardless_of_alphabet(tmp_path):
    """SAC is the primary run and PPO the comparison; sorted() reverses that."""
    write_run(tmp_path, "ppo", 0, evaluation=_full_metrics(0.6))
    write_run(tmp_path, "sac", 0, evaluation=_full_metrics(0.9))

    table = report.results_table(report.discover(tmp_path))

    assert table.index("| SAC") < table.index("| PPO")


def test_results_table_flags_a_short_seed_count(tmp_path):
    write_run(tmp_path, "sac", 0, evaluation=_full_metrics(0.8))

    table = report.results_table(report.discover(tmp_path))

    assert "Fewer than 3 seeds" in table
    assert "SAC: 1" in table


def test_a_full_seed_count_gets_no_caveat(tmp_path):
    for seed in range(3):
        write_run(tmp_path, "sac", seed, evaluation=_full_metrics(0.8))
    for seed in range(3):
        write_run(tmp_path, "ppo", seed, evaluation=_full_metrics(0.6))

    assert "Fewer than" not in report.results_table(report.discover(tmp_path))


def test_results_table_without_runs_is_the_empty_placeholder(tmp_path):
    table = report.results_table([])

    assert "No scored run found" in table
    assert table.count("—") >= 8


def test_gap_table_subtracts_within_the_same_runs(tmp_path):
    """A seed evaluated but never deployed must not enter the step-sync mean:
    the delta would then be between two different experiments."""
    write_run(tmp_path, "sac", 0, evaluation=_full_metrics(0.9), gap=_full_metrics(0.7))
    write_run(tmp_path, "sac", 1, evaluation=_full_metrics(0.1))  # no gap.json

    table = report.gap_table(report.discover(tmp_path))

    assert "90.0" in table  # not the 50.0 that averaging both seeds would give
    assert "-20.0%" in table
    assert "sac-seed0" in table and "sac-seed1" not in table


def test_gap_table_without_a_paired_run_is_the_empty_placeholder(tmp_path):
    write_run(tmp_path, "sac", 0, evaluation=_full_metrics(0.9))

    assert "No run has both" in report.gap_table(report.discover(tmp_path))


def test_gap_table_survives_a_missing_metric(tmp_path):
    write_run(tmp_path, "sac", 0, evaluation={"success_rate": 0.9}, gap={"success_rate": 0.7})

    table = report.gap_table(report.discover(tmp_path))

    assert "-20.0%" in table
    assert "| Collision rate | — | — | — |" in table


# --- splicing --------------------------------------------------------------


def test_splice_replaces_between_the_markers_and_keeps_them():
    text = "before\n<!-- BEGIN RESULTS -->\nold\n<!-- END RESULTS -->\nafter\n"

    spliced = report.splice(text, "RESULTS", "new")

    assert spliced == "before\n<!-- BEGIN RESULTS -->\nnew\n<!-- END RESULTS -->\nafter\n"


def test_splice_is_idempotent():
    text = "<!-- BEGIN RESULTS -->\nold\n<!-- END RESULTS -->\n"

    once = report.splice(text, "RESULTS", "new")
    assert report.splice(once, "RESULTS", "new") == once


def test_splice_leaves_other_sections_alone():
    text = (
        "<!-- BEGIN RESULTS -->\nA\n<!-- END RESULTS -->\n"
        "<!-- BEGIN GAP -->\nB\n<!-- END GAP -->\n"
    )

    spliced = report.splice(text, "GAP", "C")

    assert "\nA\n" in spliced
    assert "\nC\n" in spliced


def test_splice_can_emit_crlf_so_a_windows_checkout_is_not_rewritten():
    text = "<!-- BEGIN RESULTS -->\r\nold\r\n<!-- END RESULTS -->\r\n"

    spliced = report.splice(text, "RESULTS", "new", eol="\r\n")

    assert spliced == "<!-- BEGIN RESULTS -->\r\nnew\r\n<!-- END RESULTS -->\r\n"


def test_main_preserves_the_files_line_endings(tmp_path):
    """Otherwise a two-table edit arrives as a whole-file diff on Windows."""
    target = tmp_path / "README.md"
    target.write_bytes(
        b"# x\r\n<!-- BEGIN RESULTS -->\r\n<!-- END RESULTS -->\r\n"
        b"<!-- BEGIN GAP -->\r\n<!-- END GAP -->\r\n"
    )

    assert report.main(["--runs", str(tmp_path / "runs"), "--write", str(target)]) == 0

    written = target.read_bytes()
    assert b"\r\n" in written
    assert written.replace(b"\r\n", b"").count(b"\n") == 0


def test_a_missing_marker_raises_rather_than_appending():
    with pytest.raises(ValueError, match="not found"):
        report.splice("no markers here", "RESULTS", "new")


def test_out_of_order_markers_raise():
    text = "<!-- END RESULTS -->\n<!-- BEGIN RESULTS -->"

    with pytest.raises(ValueError, match="out of order"):
        report.splice(text, "RESULTS", "new")


# --- end to end ------------------------------------------------------------


def test_main_writes_both_tables_into_a_file(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    write_run(runs_dir, "sac", 0, evaluation=_full_metrics(0.9), gap=_full_metrics(0.7))

    target = tmp_path / "README.md"
    target.write_text(
        "# x\n<!-- BEGIN RESULTS -->\n<!-- END RESULTS -->\n"
        "<!-- BEGIN GAP -->\n<!-- END GAP -->\n",
        encoding="utf-8",
    )

    assert report.main(["--runs", str(runs_dir), "--write", str(target)]) == 0

    written = target.read_text(encoding="utf-8")
    assert "| SAC (1 seed)" in written
    assert "Step-synchronized" in written
    assert written.startswith("# x\n")


def test_main_prints_the_placeholders_when_there_is_nothing_to_report(tmp_path, capsys):
    assert report.main(["--runs", str(tmp_path / "runs")]) == 0

    out = capsys.readouterr().out
    assert "No scored run found" in out
    assert "No run has both" in out


def test_reported_numbers_are_finite_where_a_number_is_claimed(tmp_path):
    """Guards the whole chain: no cell should ever read 'nan'."""
    write_run(runs := tmp_path, "sac", 0, evaluation={**_full_metrics(0.0),
                                                      "mean_path_length": math.nan,
                                                      "mean_path_efficiency": math.nan})

    table = report.results_table(report.discover(runs))

    assert "nan" not in table.lower()
