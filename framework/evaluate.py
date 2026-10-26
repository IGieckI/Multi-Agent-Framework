"""
Evaluation of finished runs from their logs only: the task's own metrics, the engine's
usage, and bootstrap confidence intervals over replicates
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from framework import provenance
from framework.engine.budget import INTERRUPTED
from framework.engine.config import VERSIONS
from framework.engine.usage import logged_completions
from framework.log import read_events
from framework.metrics import generic, text
from framework.metrics.log_view import build_view
from framework.tasks.base import Task
from framework.tasks.registry import build_task, parse_task_config

# bump when a metric's definition changes: it is written next to every evaluation
METRICS_VERSION = "2"


def run_dirs(condition: Path) -> list[Path]:
    """The run folders of a condition, in name order"""
    return sorted(p.parent for p in condition.glob("*/events.jsonl") if (p.parent / "meta.json").exists())


def versions_of(run_folder: Path) -> dict[str, Path]:
    """The version folders of a run folder that hold runs, in the order of VERSIONS"""
    return {v: run_folder / v for v in VERSIONS if run_dirs(run_folder / v)}


def engine_metrics(events: list[dict]) -> dict[str, float | None]:
    """Usage and diagnostics, from the LLM calls (of every budget line) and the engine's events"""
    calls = [e["data"] for e in events if e["kind"] == "llm_call"]
    completions = logged_completions(events)
    ended = next(e["data"] for e in events if e["kind"] == "run_ended")
    # an answer the task could not read: no action, and an error to re-prompt with
    parse_failures = sum(c["action"] is None and c["error"] is not None for c in calls)
    return {
        "total_tokens": sum(c["prompt_tokens"] + c["generated_tokens"] for c in completions),
        "completion_tokens": sum(c["generated_tokens"] for c in completions),
        "calls": len(completions),
        "channel_calls": len(completions) - len(calls),
        "steps": ended["steps"],
        "parse_failure_rate": parse_failures / len(calls) if calls else None,
        "no_op_turns": sum(e["kind"] == "no_op" for e in events),
        "truncation_events": sum(e["kind"] == "truncation_event" for e in events),
    }


class Evaluator:
    """Rebuilds each run's task from the config in its metadata (once per distinct task config) and scores the run.
    With an encoder (framework/metrics/text.py) it also scores the diversity of the agents' text"""

    def __init__(self, encoder: text.Encoder | None = None) -> None:
        self.tasks: dict[str, Task] = {}
        self.encoder = encoder

    def task(self, task_config: dict) -> Task:
        key = json.dumps(task_config, sort_keys=True)
        if key not in self.tasks:
            self.tasks[key] = build_task(parse_task_config(task_config))
        return self.tasks[key]

    def analyse(self, run_dir: Path) -> tuple[dict, list[dict]] | None:
        """The metrics of one run and its per-step rows; None if it has not ended or was interrupted (stopped, or
        failed)"""
        events = read_events(run_dir / "events.jsonl")
        ended_by = next((e["data"]["ended_by"] for e in events if e["kind"] == "run_ended"), None)
        if ended_by is None or ended_by in INTERRUPTED:
            return None
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        task = self.task(meta["config"]["task"])
        view = build_view(meta, events)
        rows = generic.series(view)
        metrics = {"run": run_dir.name, "ended_by": ended_by, **task.evaluate(meta, events), **engine_metrics(events),
                   **generic.usage_metrics(view), **generic.action_metrics(rows), **text.cue_metrics(view)}
        by_step = {row["t"]: row for row in rows}
        for row in getattr(task, "series", lambda *_: [])(meta, events) or []:
            by_step[row["t"]] |= {k: v for k, v in row.items() if k != "t"}
        if self.encoder is not None:
            scalars, text_rows = text.text_metrics(view, self.encoder)
            metrics |= scalars
            for row in text_rows:
                by_step[row["t"]] |= {k: v for k, v in row.items() if k != "t"}
        return metrics, [by_step[t] for t in sorted(by_step)]

    def run(self, run_dir: Path) -> dict | None:
        """The metrics of one run, None if it has not ended or was interrupted (stopped, or failed)"""
        analysed = self.analyse(run_dir)
        return None if analysed is None else analysed[0]

    def save(self, run_dir: Path) -> dict | None:
        """Scores a run and writes metrics.json (the metrics, and what they were computed from and with) and
        series.csv (one row per step) into its folder. Returns the metrics, None for a run that did not end"""
        analysed = self.analyse(run_dir)
        if analysed is None:
            return None
        metrics, rows = analysed
        report = {
            "metrics": metrics,
            "evaluation": {
                "metrics_version": METRICS_VERSION, "cues_version": text.CUES_VERSION,
                "agreement_cues": text.AGREEMENT_CUES, "disagreement_cues": text.DISAGREEMENT_CUES,
                "encoder": None if self.encoder is None else self.encoder.info(),
                "events_sha256": provenance.file_sha256(run_dir / "events.jsonl"),
                "meta_sha256": provenance.file_sha256(run_dir / "meta.json"),
                "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "environment": {"git": provenance.git_state(), "libraries": provenance.library_versions()},
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with open(run_dir / "series.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return metrics

    def condition(self, condition: Path) -> tuple[list[dict], list[str]]:
        """The metrics of every run of a condition that ended on its own, and the names of the others"""
        results, unfinished = [], []
        for run_dir in run_dirs(condition):
            result = self.run(run_dir)
            if result is None:
                unfinished.append(run_dir.name)
            else:
                results.append(result)
        return results, unfinished


def numeric(results: list[dict], metric: str) -> np.ndarray:
    """The metric's values over runs, without the runs where it is undefined"""
    return np.array([r[metric] for r in results if isinstance(r.get(metric), (int, float))], dtype=float)


def bootstrap_ci(values: np.ndarray, seed: int = 0, resamples: int = 10_000,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval; NaNs when there are no values"""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 0, resamples: int = 10_000,
            level: float = 0.95) -> tuple[float, float, float]:
    """mean(a) - mean(b) and its bootstrap interval, resampling the two groups independently (unpaired)"""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(a, size=(resamples, len(a))).mean(axis=1)
             - rng.choice(b, size=(resamples, len(b))).mean(axis=1))
    tail = (1 - level) / 2 * 100
    return float(a.mean() - b.mean()), float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100 - tail))
