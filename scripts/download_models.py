"""Download the pinned weights of the models of an experiment into the Hugging Face cache, so the compute nodes of
a cluster can run offline (HF_HUB_OFFLINE=1). Run it on a node with internet access, with the cache where the jobs
will look for it (HF_HOME).

    export HF_HOME=/path/on/shared/storage/hf
    export HF_TOKEN=...            # needed for the gated models (Llama, Gemma), whose licences you must have accepted
    python scripts/download_models.py configs/experiments/pilot.yaml
    python scripts/download_models.py --all

Each model is fetched at the revision pinned in configs/models/models.yaml, and so is the sentence encoder of the
text metrics. A model that fails is reported and the
others go on; the exit code is 1 if any failed.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from framework import experiment
from framework.metrics.text import DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION

parser = argparse.ArgumentParser()
parser.add_argument("grid", type=Path, nargs="?", help="experiment grid: download the models it names")
parser.add_argument("--all", action="store_true", help="download every model of the registry")
args = parser.parse_args()
if not args.all and args.grid is None:
    parser.error("give an experiment grid or --all")

models = experiment.load_models()
names = list(models) if args.all else [m for m in experiment.load_grid(args.grid).models if m not in experiment.MOCKS]
failed = []
for name in names:
    entry = models[name]
    try:
        path = snapshot_download(entry.model, revision=entry.revision, ignore_patterns=["*.gguf", "original/*"])
        print(f"{name}: {entry.model}@{entry.revision[:10]} -> {path}")
    except Exception as e:
        failed.append(name)
        print(f"{name}: FAILED ({e!r}){' (gated: accept the licence and set HF_TOKEN)' if entry.gated else ''}")
try:  # the sentence encoder of the text metrics (sweep.py evaluate --encoder default)
    path = snapshot_download(DEFAULT_ENCODER, revision=DEFAULT_ENCODER_REVISION)
    print(f"encoder: {DEFAULT_ENCODER}@{DEFAULT_ENCODER_REVISION[:10]} -> {path}")
except Exception as e:
    failed.append("encoder")
    print(f"encoder: FAILED ({e!r})")
sys.exit(1 if failed else 0)
