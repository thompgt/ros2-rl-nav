"""Phase 5 -- aggregate the per-seed JSON into the README's two tables.

    python3 -m robot_rl_env.report                     # print the markdown
    python3 -m robot_rl_env.report --write README.md   # splice it in place

``evaluate.py`` scores one run and ``deploy_eval.py`` measures one run's gap.
Neither knows about seeds, and neither should: an evaluation that averaged over
whatever happened to be on disk would silently change its answer as runs
accumulate. This script is the one place that reads across runs, and it is the
only place seed aggregation happens.

Why mean ± std and not a single number
--------------------------------------
Three seeds per algorithm is the project's stated minimum, and the reason is
that the seed-to-seed spread in continuous-control RL is routinely larger than
the difference between the algorithms being compared. A table of single-seed
numbers invites a conclusion the data does not support. So the spread is
reported next to the mean, and a table built from fewer than
``MIN_SEEDS`` runs is annotated as such rather than quietly presented as if it
were the real thing.

The spread is the **sample** standard deviation (``ddof=1``). The seeds are a
sample from the population of seeds, not the population itself; dividing by
``n`` would report a spread that is biased low, which is the wrong direction to
be wrong in when the whole point of the column is honesty about noise.

NaN, and why it is dropped rather than propagated
-------------------------------------------------
``mean_path_length``, ``mean_steps`` and ``mean_path_efficiency`` are averaged
over *successful* episodes only, so a run with zero successes reports NaN for
all three. Averaging that in makes the whole cell NaN and destroys the two
seeds that did work. Non-finite values are therefore skipped, and the number of
seeds each cell actually rests on is carried alongside it -- a mean over two of
three seeds is a different claim from a mean over three, and the table says
which it is when they differ.

This module is pure. It reads JSON and returns strings; the simulator, torch
and SB3 are all absent, so the arithmetic behind the reported numbers is
testable on any host in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

#: Three seeds per algorithm, per WORKPLAN Phase 3. Below this the table is
#: still produced -- refusing to print anything would be useless while a run is
#: in progress -- but it is labelled.
MIN_SEEDS = 3

#: ``runs/sac-seed0`` -> ("sac", 0). Anything else in ``runs/`` (TensorBoard
#: directories, scratch copies, a run someone renamed) is ignored rather than
#: guessed at.
RUN_DIR_PATTERN = re.compile(r"^(sac|ppo)-seed(\d+)$")

BEGIN_MARKER = "<!-- BEGIN {section} -->"
END_MARKER = "<!-- END {section} -->"


@dataclass(frozen=True)
class Run:
    """One seed's results: whatever of ``eval.json`` / ``gap.json`` exists."""

    algorithm: str
    seed: int
    path: Path
    evaluation: dict | None = None
    gap: dict | None = None

    @property
    def name(self) -> str:
        return f"{self.algorithm}-seed{self.seed}"


def _read_metrics(path: Path) -> dict | None:
    """Load the ``metrics`` block, or None if the file is absent.

    A malformed file is an error rather than a None: a truncated ``eval.json``
    from an interrupted run must not silently drop a seed out of the average,
    which would look exactly like a run that was never launched.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} has no 'metrics' object; is it from an older evaluate.py?")
    return metrics


def discover(runs_dir: Path) -> list[Run]:
    """Every ``<algo>-seed<N>`` directory under ``runs/`` that has results.

    Sorted by algorithm then seed, so the table is stable across invocations
    and a diff of two reports shows a change in the numbers rather than a
    reshuffling of the rows.
    """
    if not runs_dir.is_dir():
        return []

    runs = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        evaluation = _read_metrics(child / "eval.json")
        gap = _read_metrics(child / "gap.json")
        if evaluation is None and gap is None:
            # A directory with checkpoints but no scores: training has run,
            # evaluation has not. Not an error -- just nothing to report yet.
            continue
        runs.append(
            Run(
                algorithm=match.group(1),
                seed=int(match.group(2)),
                path=child,
                evaluation=evaluation,
                gap=gap,
            )
        )
    return sorted(runs, key=lambda run: (run.algorithm, run.seed))


@dataclass(frozen=True)
class Aggregate:
    """A cell of the table: mean, spread, and what they rest on."""

    mean: float
    std: float | None  # None for a single seed -- no spread is defined
    n: int  # seeds that contributed a finite value
    n_total: int  # seeds that were asked

    @property
    def partial(self) -> bool:
        return self.n != self.n_total


def aggregate(values: list[float]) -> Aggregate | None:
    """Mean ± sample std over seeds, skipping non-finite entries.

    Returns None when nothing finite survives, which is a real outcome: every
    seed failed every episode, so there is no path length to report. The caller
    renders that as an em dash rather than as 0.0, which would read as a
    measured zero.
    """
    n_total = len(values)
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    return Aggregate(
        mean=statistics.fmean(finite),
        std=statistics.stdev(finite) if len(finite) > 1 else None,
        n=len(finite),
        n_total=n_total,
    )


def collect(
    runs: list[Run], metric: str, source: str = "evaluation"
) -> dict[str, Aggregate | None]:
    """``{algorithm: Aggregate}`` for one metric, over the seeds of each."""
    by_algorithm: dict[str, list[float]] = {}
    for run in runs:
        metrics = getattr(run, source)
        if metrics is None or metric not in metrics:
            continue
        by_algorithm.setdefault(run.algorithm, []).append(metrics[metric])
    return {algorithm: aggregate(values) for algorithm, values in by_algorithm.items()}


def format_cell(value: Aggregate | None, digits: int = 2, percent: bool = False) -> str:
    """One table cell. Em dash for absent, ``mean ± std`` otherwise.

    A cell built from fewer seeds than its neighbours is marked with the count
    inline, because a footnote applying to one cell in a table of twenty is a
    footnote nobody reads.
    """
    if value is None:
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    text = f"{value.mean * scale:.{digits}f}"
    if value.std is not None:
        text += f" ± {value.std * scale:.{digits}f}"
    text += suffix
    if value.std is None and value.n_total <= 1:
        text += " (1 seed)"
    elif value.partial:
        text += f" ({value.n}/{value.n_total} seeds)"
    return text


#: (metric key, column header, decimal places, render as a percentage)
RESULT_COLUMNS = [
    ("success_rate", "Success rate", 1, True),
    ("mean_path_length", "Mean path length", 2, False),
    ("mean_path_efficiency", "Path efficiency", 2, False),
    ("collision_rate", "Collision rate", 1, True),
    ("timeout_rate", "Timeout rate", 1, True),
]

#: The gap table's rows. Rates only: path length over a *different* set of
#: successful episodes is not comparable between the two loops, and putting it
#: in a table headed by a delta column would invite exactly that comparison.
GAP_ROWS = [
    ("success_rate", "Success rate", 1),
    ("collision_rate", "Collision rate", 1),
    ("timeout_rate", "Timeout rate", 1),
]

ALGORITHM_LABELS = {"sac": "SAC", "ppo": "PPO"}

#: Row order. SAC is the primary run and PPO the comparison, per WORKPLAN
#: Phase 3; alphabetical order would silently reverse that.
ALGORITHM_ORDER = ["sac", "ppo"]


def _ordered(algorithms: set[str]) -> list[str]:
    known = [name for name in ALGORITHM_ORDER if name in algorithms]
    return known + sorted(algorithms - set(ALGORITHM_ORDER))


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def results_table(runs: list[Run]) -> str:
    """SAC vs PPO on the held-out set, mean ± std over seeds."""
    scored = [run for run in runs if run.evaluation is not None]
    if not scored:
        return _EMPTY_RESULTS

    columns = {key: collect(scored, key) for key, _, _, _ in RESULT_COLUMNS}
    rows = []
    for algorithm in _ordered({run.algorithm for run in scored}):
        seeds = sorted(run.seed for run in scored if run.algorithm == algorithm)
        label = ALGORITHM_LABELS.get(algorithm, algorithm.upper())
        row = [f"{label} ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})"]
        row += [
            format_cell(columns[key].get(algorithm), digits, percent)
            for key, _, digits, percent in RESULT_COLUMNS
        ]
        rows.append(row)

    table = _table(["Algorithm"] + [name for _, name, _, _ in RESULT_COLUMNS], rows)
    return table + _seed_caveat(scored)


def _seed_caveat(runs: list[Run]) -> str:
    """Say it in the table's own caption when the seed count is short."""
    short = []
    for algorithm in _ordered({run.algorithm for run in runs}):
        count = sum(1 for run in runs if run.algorithm == algorithm)
        if count < MIN_SEEDS:
            short.append(f"{ALGORITHM_LABELS.get(algorithm, algorithm)}: {count}")
    if not short:
        return ""
    return (
        f"\n\n> Fewer than {MIN_SEEDS} seeds ({'; '.join(short)}). "
        f"Treat the spread as indicative, not as an error bar."
    )


def gap_table(runs: list[Run]) -> str:
    """Step-synchronized vs free-running, averaged over the same seeds.

    Only runs that have *both* numbers contribute. A delta between a mean over
    three step-synchronized seeds and a mean over the one seed that was also
    deployed is not a gap; it is two different experiments subtracted.
    """
    paired = [run for run in runs if run.evaluation is not None and run.gap is not None]
    if not paired:
        return _EMPTY_GAP

    rows = []
    for key, label, digits in GAP_ROWS:
        sync = aggregate([run.evaluation[key] for run in paired if key in run.evaluation])
        free = aggregate([run.gap[key] for run in paired if key in run.gap])
        sync_cell = format_cell(sync, digits, percent=True)
        free_cell = format_cell(free, digits, percent=True)
        if sync is None or free is None:
            rows.append([label, sync_cell, free_cell, "—"])
            continue
        delta = (free.mean - sync.mean) * 100.0
        rows.append([label, sync_cell, free_cell, f"{delta:+.{digits}f}%"])

    names = ", ".join(run.name for run in paired)
    table = _table(["", "Step-synchronized", "Free-running", "Δ"], rows)
    return f"{table}\n\nOver {len(paired)} run{'s' if len(paired) != 1 else ''}: {names}."


# The placeholders are built from the same column definitions as the real
# tables. Written out as literals they would be a second copy of the header,
# and the first added metric would leave a README whose two states have
# different columns depending on whether anything has been run.
_EMPTY_RESULTS = _table(
    ["Algorithm"] + [name for _, name, _, _ in RESULT_COLUMNS],
    [[label] + ["—"] * len(RESULT_COLUMNS) for label in ALGORITHM_LABELS.values()],
) + (
    "\n\n> No scored run found under `runs/`. Run `make train` and then "
    "`make evaluate`;\n> this table is regenerated by `make report`."
)

_EMPTY_GAP = _table(
    ["", "Step-synchronized", "Free-running", "Δ"],
    [[label, "—", "—", "—"] for _, label, _ in GAP_ROWS],
) + (
    "\n\n> No run has both `eval.json` and `gap.json`. Run `make evaluate` and "
    "then\n> `make gap` on the same run."
)


def splice(text: str, section: str, replacement: str, eol: str = "\n") -> str:
    """Replace the content between the section's markers, markers kept.

    Raises when a marker is missing or out of order rather than appending, so
    a typo'd section name fails visibly instead of silently writing the table
    somewhere nobody looks.
    """
    begin = BEGIN_MARKER.format(section=section)
    end = END_MARKER.format(section=section)
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0:
        missing = begin if start < 0 else end
        raise ValueError(f"marker {missing!r} not found; cannot splice section {section!r}")
    if stop < start:
        raise ValueError(f"markers for section {section!r} are out of order")
    return text[: start + len(begin)] + eol + replacement + eol + text[stop:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs", help="directory holding <algo>-seed<N>/")
    parser.add_argument(
        "--write",
        default=None,
        help="markdown file to splice the tables into, between its BEGIN/END markers",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # The tables contain an em dash and a delta. On a Windows console that is
    # cp1252 by default, and printing them raises UnicodeEncodeError -- which
    # would make this the one module in the project that cannot be run on the
    # development host, for no reason but a terminal codepage. The file itself
    # is written as UTF-8 regardless.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    runs = discover(Path(args.runs))

    if runs:
        print(f"runs: {', '.join(run.name for run in runs)}", file=sys.stderr)
    else:
        print(f"no scored runs under {args.runs}/", file=sys.stderr)

    results, gap = results_table(runs), gap_table(runs)

    if args.write:
        path = Path(args.write)
        # newline="" both ways: without it, Python translates on read and
        # re-translates on write, so running this on Windows rewrites every
        # line ending in the file and turns a two-table edit into a whole-file
        # diff. The tables are emitted in whatever ending the file already uses.
        text = path.read_text(encoding="utf-8", newline="")
        eol = "\r\n" if "\r\n" in text else "\n"
        text = splice(text, "RESULTS", results.replace("\n", eol), eol=eol)
        text = splice(text, "GAP", gap.replace("\n", eol), eol=eol)
        path.write_text(text, encoding="utf-8", newline="")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(results)
        print()
        print(gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
