"""
Evaluation of finished runs from their logs only: the task's own metrics, the engine's
usage, and bootstrap confidence intervals over replicates
"""

import json
from pathlib import Path

import numpy as np

from framework.engine.budget import INTERRUPTED
from framework.engine.config import VERSIONS
from framework.log import read_events
from framework.tasks.base import Task
from framework.tasks.registry import build_task, parse_task_config


def run_dirs(condition: Path) -> list[Path]:
    """The run folders of a condition, in name order"""
    return sorted(p.parent for p in condition.glob("*/events.jsonl") if (p.parent / "meta.json").exists())


def versions_of(run_folder: Path) -> dict[str, Path]:
    """The version folders of a run folder that hold runs, in the order of VERSIONS"""
    return {v: run_folder / v for v in VERSIONS if run_dirs(run_folder / v)}


def engine_metrics(events: list[dict]) -> dict[str, float | None]:
    """Usage and diagnostics, from the LLM calls and the engine's events"""
    calls = [e["data"] for e in events if e["kind"] == "llm_call"]
    ended = next(e["data"] for e in events if e["kind"] == "run_ended")
    # an answer the task could not read: no action, and an error to re-prompt with
    parse_failures = sum(c["action"] is None and c["error"] is not None for c in calls)
    return {
        "total_tokens": sum(c["completion"]["prompt_tokens"] + c["completion"]["generated_tokens"] for c in calls),
        "completion_tokens": sum(c["completion"]["generated_tokens"] for c in calls),
        "calls": len(calls),
        "steps": ended["steps"],
        "parse_failure_rate": parse_failures / len(calls) if calls else None,
        "no_op_turns": sum(e["kind"] == "no_op" for e in events),
        "truncation_events": sum(e["kind"] == "truncation_event" for e in events),
    }


class Evaluator:
    """Rebuilds each run's task from the config in its metadata (once per distinct task config) and scores the run"""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def task(self, task_config: dict) -> Task:
        key = json.dumps(task_config, sort_keys=True)
        if key not in self.tasks:
            self.tasks[key] = build_task(parse_task_config(task_config))
        return self.tasks[key]

    def run(self, run_dir: Path) -> dict | None:
        """The metrics of one run, None if it has not ended or was interrupted (stopped, or failed)"""
        events = read_events(run_dir / "events.jsonl")
        ended_by = next((e["data"]["ended_by"] for e in events if e["kind"] == "run_ended"), None)
        if ended_by is None or ended_by in INTERRUPTED:
            return None
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        task = self.task(meta["config"]["task"])
        return {"run": run_dir.name, "ended_by": ended_by, **task.evaluate(meta, events), **engine_metrics(events)}

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
