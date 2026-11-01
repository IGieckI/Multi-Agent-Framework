"""Plan, run and evaluate an experiment grid (framework/experiment.py)

    python scripts/sweep.py plan     configs/experiments/pilot.yaml             # validate, count, write the manifest
    python scripts/sweep.py run      configs/experiments/pilot.yaml --shard 0/10 # run one block of the grid
    python scripts/sweep.py evaluate configs/experiments/pilot.yaml              # metrics, runs.csv, results.csv

"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from framework import experiment, provenance
from framework.engine.config import ROOT
from framework.evaluate import Evaluator
from framework.log import EventLog

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["plan", "run", "evaluate"])
parser.add_argument("grid", type=Path, help="experiment grid, e.g. configs/experiments/pilot.yaml")
parser.add_argument("--shard", default="0/1", help="run: 'i/n', the i-th of n blocks of units (default 0/1)")
parser.add_argument("--runs", type=Path, default=ROOT / "runs", help="folder of all experiments (default runs/)")
parser.add_argument("--encoder", help="evaluate: sentence-transformers model for the text metrics (default: none; "
                                      "'default' = the one of framework/metrics/text.py)")
args = parser.parse_args()

grid_text = args.grid.read_text(encoding="utf-8")
grid = experiment.load_grid(args.grid)
models = experiment.load_models()
units = experiment.expand(grid, models, args.runs)
folder = args.runs / grid.name
versions = len(grid.versions)

if args.command == "plan":
    folder.mkdir(parents=True, exist_ok=True)
    plan = experiment.manifest(grid, grid_text, units, models)
    (folder / "manifest.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    cells = len({u.cell for u in units})
    shared = sum(len(u.same_as) for u in units)
    print(f"{grid.name}: {cells} cells x {grid.replicates} replicates = {len(units)} units, "
          f"{len(units) * versions} runs ({versions} versions each), {len(units) * versions - shared} distinct "
          f"({shared} are the same run as another unit's and run once)")
    print(f"models: {', '.join(grid.models)}; wrote {folder / 'manifest.json'}")

elif args.command == "run":
    index, count = (int(x) for x in args.shard.split("/"))
    mine = experiment.shard(units, index, count)
    log = EventLog(folder / "sweep" / f"shard{index}of{count}.jsonl", run_id=f"{grid.name}/shard{index}of{count}",
                   append=True)
    log.write("sweep", "shard_started", experiment=grid.name, shard=args.shard, units=[u.id for u in mine],
              grid_sha256=hashlib.sha256(grid_text.encode("utf-8")).hexdigest(),
              environment=provenance.environment(), at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    failures = experiment.run_units(mine, log)
    log.write("sweep", "shard_finished", failures=failures, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    log.close()
    print(f"shard {args.shard}: {len(mine)} units, {failures} failed runs (see {log.file.name})")
    sys.exit(1 if failures else 0)

else:
    encoder = None
    if args.encoder:
        from framework.metrics.text import DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION, SentenceEncoder
        encoder = (SentenceEncoder(DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION) if args.encoder == "default"
                   else SentenceEncoder(args.encoder))
    counts = experiment.evaluate_units(grid.name, units, Evaluator(encoder), folder)
    print(f"evaluated {counts['runs']} runs in {counts['cells']} cells; {counts['unfinished']} unfinished "
          f"(see {folder / 'unfinished.json'}); wrote runs.csv and results.csv in {folder}")
