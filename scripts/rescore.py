"""Confidence of the answers of finished runs: every logged answer of one model scored again under that model
(teacher forcing), with the measures engine.log_confidence logs while sampling (framework/agents/backends/base.py
confidence_summary). For the runs that ran without it. Nothing in the run folders changes: one file per run goes to
--out (default runs_confidence/<grid>/<cell>/<version>/<run>.jsonl), one line per LLM call.

    python scripts/rescore.py qwen3-4b runs/followup runs/main            # every run of qwen3-4b in those grids
    python scripts/rescore.py qwen3-4b runs/followup --versions base,all_independent

The prompt is the logged one under the model's chat template, as the call had it; the answer is the logged raw text
tokenized again, so its tokens can differ slightly from the sampled ones (the end-of-turn token is not scored).
A run already scored is skipped, so a job can be resumed.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from framework import experiment
from framework.agents.backends.base import confidence_summary
from framework.agents.backends.hf_local import HFLocalBackend, token_ends
from framework.engine.config import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("model", help="a model of configs/models/models.yaml")
parser.add_argument("grids", nargs="+", type=Path, help="grid folders, e.g. runs/followup")
parser.add_argument("--versions", help="comma-separated versions to score (default: all)")
parser.add_argument("--limit", type=int, help="stop after this many runs (a quick test)")
parser.add_argument("--out", type=Path, default=ROOT / "runs_confidence", help="output root (default runs_confidence/)")
args = parser.parse_args()

entry = experiment.load_models()[args.model]
versions = set(args.versions.split(",")) if args.versions else None
backend = HFLocalBackend(entry.model, revision=entry.revision)
tok, model = backend.tokenizer, backend.model
print(f"{args.model}: {entry.model} on {backend.device} in {backend.dtype}", flush=True)


def score(prompt: str, raw: str, reasoning: bool) -> dict:
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=reasoning)
    prompt_ids = tok(chat, add_special_tokens=False)["input_ids"]
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    if not ids:
        return {**confidence_summary(raw, [], [], []), "rescored_tokens": 0}
    x = torch.tensor([prompt_ids + ids], device=backend.device)
    with torch.inference_mode():
        try:
            logits = model(input_ids=x, logits_to_keep=len(ids) + 1).logits[0, :-1]
        except TypeError:  # a model without logits_to_keep
            logits = model(input_ids=x).logits[0, len(prompt_ids) - 1:-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        lps = logp.gather(1, torch.tensor(ids, device=backend.device)[:, None])[:, 0].tolist()
        ents = (-torch.where(torch.isfinite(logp), logp.exp() * logp, 0.0).sum(-1)).tolist()
    return {**confidence_summary(raw, lps, ents, token_ends(tok, ids)), "rescored_tokens": len(ids)}


done = 0
started = time.time()
for grid in args.grids:
    for events in sorted(grid.glob("*/*/seed*_rep*/events.jsonl")):
        run = events.parent
        version, cell = run.parent.name, run.parent.parent.name
        if versions and version not in versions:
            continue
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        if meta["engine"].get("model_id") != entry.model or not (run / "summary.json").exists():
            continue
        out = args.out / grid.name / cell / version / f"{run.name}.jsonl"
        if out.exists():
            continue
        if args.limit is not None and done >= args.limit:
            break
        lines = []
        for line in events.open(encoding="utf-8"):
            e = json.loads(line)
            if e["kind"] != "llm_call":
                continue
            d = e["data"]
            raw = d["completion"].get("full_raw_answer") or d["completion"]["text"]
            lines.append({"seq": e["seq"], "t": e["t"], "agent": d["agent"], "attempt": d["attempt"],
                          "generated_tokens": d["completion"]["generated_tokens"], "action": d["action"],
                          "error": d["error"], **score(d["prompt"], raw, d["sampling"]["reasoning"])})
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        tmp.rename(out)
        done += 1
        if done % 100 == 0:
            print(f"{done} runs, {time.time() - started:.0f} s", flush=True)
print(f"{args.model}: {done} runs scored in {time.time() - started:.0f} s")
