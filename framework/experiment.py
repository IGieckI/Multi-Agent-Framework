"""
Experiment grids: one YAML file names the models, protocols and task settings to cross, the versions (base and
baselines) to run in each cell, and the number of replicates. It expands into units: one replicate of one cell,
with all its versions. Units are what a cluster job runs (shard), what the manifest lists, and what the sweep log
reports on (scripts/sweep.py)

    configs/models/models.yaml   name -> Hugging Face id, pinned revision, size, family, sampling policy of the model
    configs/experiments/*.yaml   the grid
    runs/<experiment>/<cell>/<version>/seed<s>_rep<r>   every run (framework/baselines.py)
"""

import asyncio
import copy
import csv
import gc
import hashlib
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from framework import provenance
from framework.baselines import version_runs
from framework.comm.compiler import Strict
from framework.engine.budget import INTERRUPTED, finished
from framework.engine.config import ROOT, VERSIONS, RunConfig
from framework.engine.engine import create_engine
from framework.evaluate import Evaluator, bootstrap_ci, numeric
from framework.log import EventLog

MODELS_FILE = ROOT / "configs" / "models" / "models.yaml"
# names the grid may use for the mock backend (no weights, no GPU): random agents, for tests of the pipeline and as a
# baseline (mock picks with replacement, mock_norepeat never repeats a tried pair)
MOCKS = ("mock", "mock_norepeat")
# the sampling parameters a model may set; temperature is the same for every model
MODEL_SAMPLING_KEYS = {"max_tokens", "reasoning", "top_p", "top_k"}


class ModelEntry(Strict):
    """One model of the registry: where its weights come from and what it is"""
    family: str
    model: str
    revision: str
    parameters: int
    # the model card on the Hugging Face hub
    link: str
    gated: bool = False
    # whether the chat template can switch a separate thinking phase on (otherwise the reasoning is inline)
    native_reasoning: bool = False
    # the model's sampling policy: replaces these keys of the grid's `sampling` for this model
    sampling: dict[str, Any] = {}

    @field_validator("sampling")
    @classmethod
    def sampling_keys_are_allowed(cls, sampling: dict[str, Any]) -> dict[str, Any]:
        """A model may set only MODEL_SAMPLING_KEYS (not the temperature)"""
        if extra := set(sampling) - MODEL_SAMPLING_KEYS:
            raise ValueError(f"a model's sampling may only set {sorted(MODEL_SAMPLING_KEYS)}, not {sorted(extra)}")
        return sampling

    @field_validator("link")
    @classmethod
    def link_is_the_model_card(cls, link: str, info) -> str:
        """The link must be the hub page of `model`"""
        if link != f"https://huggingface.co/{info.data.get('model')}":
            raise ValueError(f"link {link!r} is not the hub page of {info.data.get('model')!r}")
        return link


class ProtocolEntry(Strict):
    """A protocol of the grid, with the personas of its roles (a persona of a role the protocol lacks is refused)"""
    name: str
    file: str
    personas: dict[str, str] = {}


class Grid(Strict):
    name: str
    # a run config without `protocol`, `personas` and the engine's model: what every cell shares
    base: dict[str, Any]
    models: tuple[str, ...]
    protocols: tuple[ProtocolEntry, ...]
    # task setting -> the values to cross (e.g. distractors: [0, 3]), or a list of settings to run as they are
    # (e.g. [{depth: 1, distractors: 3}, {depth: 2, distractors: 0}]); it overrides the base task
    task: dict[str, tuple[Any, ...]] | tuple[dict[str, Any], ...] = {}
    versions: tuple[str, ...] = ("base",)
    replicates: int = Field(default=3, ge=1)
    seed: int = 0

    @field_validator("models", "protocols", "versions")
    @classmethod
    def not_empty_or_repeated(cls, values: tuple) -> tuple:
        """Reject an empty list and a repeated entry"""
        keys = [getattr(v, "name", v) for v in values]
        if not keys or len(set(keys)) < len(keys):
            raise ValueError(f"needs at least one entry, each named once: {keys}")
        return values

    @field_validator("versions")
    @classmethod
    def known_versions(cls, versions: tuple[str, ...]) -> tuple[str, ...]:
        if unknown := [v for v in versions if v not in VERSIONS]:
            raise ValueError(f"unknown versions {unknown}; known: {list(VERSIONS)}")
        return versions

    @field_validator("task")
    @classmethod
    def axes_have_values(cls, task: dict[str, tuple[Any, ...]] | tuple[dict[str, Any], ...]):
        if isinstance(task, tuple):
            if not task:
                raise ValueError("an empty list of task settings")
            return task
        if empty := [k for k, values in task.items() if not values]:
            raise ValueError(f"task settings with no values: {empty}")
        return task


@dataclass
class Unit:
    """One replicate of one cell: all its versions, run one after the other with base first"""
    cell: str
    replicate: int
    # the cell's coordinates in the grid, written to the result tables (model, family, protocol, task settings)
    attrs: dict[str, Any]
    cfg: RunConfig
    # runs/<experiment>/<cell>: holds <version>/seed<s>_rep<r>
    folder: Path
    # version -> output folder of the same run in an earlier unit, which is run once and shared (run_key)
    same_as: dict[str, Path] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.cell}/rep{self.replicate}"

    def runs(self) -> list[tuple[str, RunConfig, Path]]:
        return version_runs(self.cfg, self.folder)


def load_models(path: Path = MODELS_FILE) -> dict[str, ModelEntry]:
    """The registry. A model's sampling policy is the file's `defaults.sampling` with the model's own keys on top"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {}).get("sampling", {})
    return {name: ModelEntry.model_validate({**entry, "sampling": {**defaults, **entry.get("sampling", {})}})
            for name, entry in raw["models"].items()}


def load_grid(path: str | Path) -> Grid:
    return Grid.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def cell_name(model: str, protocol: str, task: dict[str, Any]) -> str:
    return "__".join([model, protocol, *(f"{k}{v}" for k, v in task.items())])


def run_key(cfg: RunConfig) -> str:
    """Two runs with the same key are the same run, down to every sampled token: their configs differ at most in the
    name of their cell, which only places the output (the seeds come from the seed, the replicate and the agent ids).
    E.g. one_agent drops the protocol, so it is the same run in every protocol of a model and task setting"""
    data = cfg.model_dump(mode="json")
    del data["run"]["name"]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def task_settings(grid: Grid) -> list[dict[str, Any]]:
    """The task settings of the grid's cells: the listed ones, or every combination of the axes' values"""
    if isinstance(grid.task, tuple):
        return [dict(setting) for setting in grid.task]
    return [dict(zip(grid.task, values)) for values in itertools.product(*grid.task.values())]


def expand(grid: Grid, models: dict[str, ModelEntry], runs_root: Path = ROOT / "runs") -> list[Unit]:
    """Every unit of the grid, in the order models, protocols, task settings, replicates. Each config is validated
    here, so a wrong grid fails before anything runs. A run that an earlier unit already has (run_key) is marked in
    `same_as`: it is run once"""
    if unknown := [m for m in grid.models if m not in MOCKS and m not in models]:
        raise ValueError(f"models not in the registry: {unknown}; known: {sorted(models)}")
    units = []
    first: dict[str, Path] = {}
    for model, protocol, task in itertools.product(grid.models, grid.protocols, task_settings(grid)):
        cell = cell_name(model, protocol.name, task)
        entry = models.get(model)
        for replicate in range(grid.replicates):
            data = copy.deepcopy(grid.base)
            data["run"] = {**data.get("run", {}), "name": cell, "seed": grid.seed, "replicate": replicate,
                           "versions": list(grid.versions)}
            data["task"] = {**data["task"], **task}
            if entry is not None:
                data["sampling"] = {**data.get("sampling", {}), **entry.sampling}
            data["engine"] = ({**data.get("engine", {}), "backend": "mock", "model": model} if entry is None else
                              {**data.get("engine", {}), "backend": "hf_local", "model": entry.model,
                               "revision": entry.revision})
            data["protocol"], data["personas"] = protocol.file, protocol.personas
            attrs = {"model": model, "family": entry.family if entry else "mock",
                     "parameters": entry.parameters if entry else 0, "protocol": protocol.name, **task}
            unit = Unit(cell, replicate, attrs, RunConfig.model_validate(data), runs_root / grid.name / cell)
            if unknown := set(protocol.personas) - set(unit.cfg.resolved_protocol().roles):
                raise ValueError(f"protocol {protocol.name!r} has personas for roles it does not have: {sorted(unknown)}")
            for version, cfg, out in unit.runs():  # refuses a version that cannot be derived from this cell
                if (same := first.setdefault(run_key(cfg), out)) != out:
                    unit.same_as[version] = same
            units.append(unit)
    return units


def shard(units: list[Unit], index: int, count: int) -> list[Unit]:
    """The index-th of 'count' contiguous blocks of units. The units are ordered by model, so a block mostly holds
    one model, which is loaded once"""
    if not 0 <= index < count:
        raise ValueError(f"shard {index}/{count}: the index must be in [0, {count})")
    return units[index * len(units) // count:(index + 1) * len(units) // count]


def manifest(grid: Grid, grid_text: str, units: list[Unit], models: dict[str, ModelEntry]) -> dict:
    """What a sweep is made of: the grid, the models with their revisions, every run's config hash, the code"""
    return {
        "experiment": grid.name,
        "grid": grid.model_dump(mode="json"),
        "grid_sha256": hashlib.sha256(grid_text.encode("utf-8")).hexdigest(),
        "models": {m: models[m].model_dump() for m in grid.models if m in models},
        "units": [{"id": u.id, "cell": u.cell, "replicate": u.replicate, "attrs": u.attrs,
                   "folder": u.folder.relative_to(ROOT).as_posix() if u.folder.is_relative_to(ROOT) else str(u.folder),
                   "runs": [{"version": v, "dir": out.name,
                             "config_sha256": hashlib.sha256(json.dumps(c.model_dump(mode="json"), sort_keys=True)
                                                             .encode("utf-8")).hexdigest(),
                             "same_as": str(u.same_as[v]) if v in u.same_as else None}
                            for v, c, out in u.runs()]}
                  for u in units],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": provenance.environment(),
    }


def free_gpu() -> None:
    """Frees the GPU memory of models nothing refers to any more"""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def engine_key(cfg: RunConfig) -> tuple:
    """Two configs with the same key can share one loaded model"""
    e, s = cfg.engine, cfg.summarizer
    return e.backend, e.model, e.revision, None if s is None else (s.backend, s.model, s.revision)


def run_units(units: list[Unit], log: EventLog) -> int:
    """Runs the units in order, reusing the loaded model while it stays the same, and returns how many runs failed.
    Every skip and failure is an event of 'log' (the runs have their own logs); a failure never stops the sweep. A
    run that already ended on its own is skipped, so a sweep can be resumed, and so is a run another unit has
    (`same_as`). A model that fails to load is not tried again for the units that follow"""
    backend = summarizer = key = None
    broken: dict[tuple, str] = {}
    failures = 0
    for unit in units:
        log.write("sweep", "unit_started", unit=unit.id, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        for version, cfg, out in unit.runs():
            if version in unit.same_as:
                log.write("sweep", "run_skipped", unit=unit.id, version=version, out=str(out), reason="same_as",
                          same_as=str(unit.same_as[version]))
                continue
            if finished(out):
                log.write("sweep", "run_skipped", unit=unit.id, version=version, out=str(out), reason="finished")
                continue
            if engine_key(cfg) in broken:
                failures += 1
                log.write("sweep", "run_skipped", unit=unit.id, version=version, out=str(out),
                          reason="model_failed_to_load", error=broken[engine_key(cfg)])
                continue
            if engine_key(cfg) != key:
                backend = summarizer = None
                free_gpu()
            if (out / "summary.json").exists():
                (out / "summary.json").unlink()  # interrupted before: run it again
            started = time.time()
            try:
                engine = create_engine(cfg, out, backend=backend, summarizer=summarizer)
            except Exception as e:
                broken[engine_key(cfg)] = repr(e)
                failures += 1
                backend = summarizer = key = None
                log.write("sweep", "run_failed", unit=unit.id, version=version, out=str(out), stage="load_model",
                          error=repr(e), traceback=traceback.format_exc())
                continue
            backend, summarizer, key = engine.backend, engine.summarizer, engine_key(cfg)
            log.write("sweep", "run_started", unit=unit.id, version=version, out=str(out), engine=engine.meta["engine"])
            try:
                summary = asyncio.run(engine.run())
            except Exception as e:
                failures += 1
                log.write("sweep", "run_failed", unit=unit.id, version=version, out=str(out), stage="run",
                          error=repr(e), traceback=traceback.format_exc())
                continue
            interrupted = summary["ended_by"] in INTERRUPTED
            failures += interrupted
            log.write("sweep", "run_failed" if interrupted else "run_finished", unit=unit.id, version=version,
                      out=str(out), stage="run", ended_by=summary["ended_by"], steps=summary["steps"],
                      usage=summary["usage"], wall_s=round(time.time() - started, 3))
    return failures


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_units(experiment: str, units: list[Unit], evaluator: Evaluator, out_dir: Path, seed: int = 0) -> dict:
    """Scores every run that ended on its own (metrics.json and series.csv in its folder) and writes, to 'out_dir',
    runs.csv (one row per run), results.csv (per cell and version: n, mean and 95% bootstrap interval of every metric)
    and unfinished.json (the runs that are missing, interrupted or failed). A run shared by several units (`same_as`)
    is a row of each of them, with its folder in the `run_dir` column: count it once when pooling over cells.
    Returns the counts"""
    run_rows, unfinished = [], []
    scored: dict[Path, dict | None] = {}
    for unit in units:
        for version, _, out in unit.runs():
            out = unit.same_as.get(version, out)
            if out not in scored:
                scored[out] = evaluator.save(out) if (out / "events.jsonl").exists() else None
            if scored[out] is None:
                unfinished.append({"unit": unit.id, "version": version, "dir": str(out)})
            else:
                run_rows.append({"experiment": experiment, "cell": unit.cell, "replicate": unit.replicate,
                                 "version": version, **unit.attrs, "run_dir": str(out), **scored[out]})
    groups: dict[tuple, list[dict]] = {}
    for row in run_rows:
        groups.setdefault((row["cell"], row["version"]), []).append(row)
    cell_attrs = {u.cell: u.attrs for u in units}
    result_rows = []
    for (cell, version), rows in groups.items():
        attrs = cell_attrs[cell]
        identifiers = {"experiment", "cell", "replicate", "version", "run", "run_dir", "ended_by", *attrs}
        metric_names = [m for m in rows[0] if m not in identifiers and len(numeric(rows, m))]
        for metric in metric_names:
            mean, low, high = bootstrap_ci(numeric(rows, metric), seed)
            result_rows.append({"experiment": experiment, "cell": cell, "version": version, **attrs, "metric": metric,
                                "n": len(numeric(rows, metric)), "mean": mean, "ci_low": low, "ci_high": high})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runs.csv", run_rows)
    write_csv(out_dir / "results.csv", result_rows)
    (out_dir / "unfinished.json").write_text(json.dumps(unfinished, indent=2), encoding="utf-8")
    return {"runs": len(run_rows), "unfinished": len(unfinished), "cells": len({r["cell"] for r in run_rows})}
