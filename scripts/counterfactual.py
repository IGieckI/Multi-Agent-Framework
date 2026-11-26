"""What a message does to the agent that reads it: logged turns of finished runs scored again under their model with
the prompt edited. For every turn after round 0 with something in its inbox (first attempt), the log-probability of
answering "Combination: 'a' and 'b'\n" (the whole line with its newline; both orders summed) for a set of candidate pairs, under these prompts:

    orig        the prompt as logged
    no_msgs     every teammate's message removed (the round-0 layout: the header with an empty list)
    swap_msgs   the messages replaced by those the same agent received in the same round of another replicate of the
                same cell (another target: the content-pairing control); skipped when none has any
    no_results  the shared results (Other players' valid / invalid combinations) removed
    drop:<ref>  one message removed (every message of the turn, one at a time)

Candidates: the pair the agent answered (even if refused), the path's recipes (meta task path), pairs named in the
messages it received (both items in its inventory), and the pairs teammates played in the previous round. Nothing in
the run folders changes: one file per run goes to --out (default runs_counterfactual/<grid>/<cell>/<version>/), one
line per scored turn. A run already scored is skipped. --dry builds and checks the prompts without a model.

    python scripts/counterfactual.py qwen3-8b runs/renamed runs/renamed_share --versions base
    python scripts/counterfactual.py qwen3-4b runs/main --versions base --cells fully_connected --limit 2
"""

import argparse
import copy
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

from framework import experiment
from framework.engine.config import ROOT
from framework.tasks.wordcraft.recipes import RecipeBook

parser = argparse.ArgumentParser()
parser.add_argument("model", help="a model of configs/models/models.yaml")
parser.add_argument("grids", nargs="+", type=Path, help="grid folders, e.g. runs/renamed_share")
parser.add_argument("--versions", default="base", help="comma-separated versions to score (default: base)")
parser.add_argument("--cells", help="score only cells whose name contains this text")
parser.add_argument("--limit", type=int, help="stop after this many runs (a quick test)")
# next to this script's checkout, not ROOT: the venv's editable install points ROOT at the main checkout, and output
# there while a grid runs would make its runs "dirty" (F41); run it from a second clone (HANDOFF, section 3)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runs_counterfactual",
                    help="output root (default: runs_counterfactual/ of this checkout)")
parser.add_argument("--dry", action="store_true", help="build the prompts and candidates, score nothing")
args = parser.parse_args()

entry = experiment.load_models()[args.model]
versions = set(args.versions.split(","))
PAIR = re.compile(r"'([^']+)' and '([^']+)'")
QUOTED = re.compile(r"'([^']+)'")

if not args.dry:
    import torch
    from framework.agents.backends.hf_local import HFLocalBackend
    backend = HFLocalBackend(entry.model, revision=entry.revision)
    tok, model = backend.tokenizer, backend.model
    print(f"{args.model}: {entry.model} on {backend.device} in {backend.dtype}", flush=True)


def chat_ids(prompt: str, reasoning: bool) -> list[int]:
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=reasoning)
    return tok(chat, add_special_tokens=False)["input_ids"]


def forward(keep: int, **kw):
    """the model's output with logits for the last `keep` positions only (all of them for a model without it)"""
    try:
        return model(**kw, logits_to_keep=keep)
    except TypeError:
        out = model(**kw)
        out.logits = out.logits[:, -keep:]
        return out


def score(prompt: str, answers: list[str], reasoning: bool) -> list[float]:
    """log P(answer | prompt) for each answer, teacher-forced: the prompt runs once, each answer continues a copy of
    its cache (one sequence at a time, no padding: a right-padded batch gave Llama 3.2 wrong values, 2026-11-23)"""
    if args.dry:
        return [0.0] * len(answers)
    p = chat_ids(prompt, reasoning)
    with torch.inference_mode():
        first = forward(input_ids=torch.tensor([p], device=backend.device), use_cache=True, keep=1)
        cache, last = first.past_key_values, torch.log_softmax(first.logits[0, -1].float(), dim=-1)
        out = []
        for a in answers:
            c = tok(a, add_special_tokens=False)["input_ids"]
            total = float(last[c[0]])
            if len(c) > 1:
                more = model(input_ids=torch.tensor([c[:-1]], device=backend.device),
                             past_key_values=copy.deepcopy(cache), use_cache=True)
                logp = torch.log_softmax(more.logits[0].float(), dim=-1)
                total += float(logp.gather(1, torch.tensor(c[1:], device=backend.device)[:, None]).sum())
            out.append(total)
    return out


def full_forward(prompt: str, answer: str, reasoning: bool) -> float:
    """the same log-probability with prompt and answer in one sequence (rescore.py's way): the per-run check"""
    p, c = chat_ids(prompt, reasoning), tok(answer, add_special_tokens=False)["input_ids"]
    with torch.inference_mode():
        logits = forward(input_ids=torch.tensor([p + c], device=backend.device), keep=len(c) + 1).logits[0, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        return float(logp.gather(1, torch.tensor(c, device=backend.device)[:, None]).sum())


def spans(d: dict) -> list[dict]:
    """the logged prompt cut into its segments: source, ref and text"""
    pos, out = 0, []
    for s in d["segments"]:
        out.append({"source": s["source"], "ref": s["ref"], "text": d["prompt"][pos:pos + s["chars"]]})
        pos += s["chars"]
    assert pos == len(d["prompt"]), "segments do not cover the prompt"
    return out


def join(segs: list[dict]) -> str:
    """the prompt from segments; when a results list lost its first entry, the new first one's ', ' becomes ' '"""
    text, first = "", False
    for s in segs:
        t = s["text"]
        if s["source"] == "message" and first and t.startswith(", "):
            t = " " + t[2:]
        first = s["source"] == "template" and t.endswith("combinations:")
        text += t
    return text


def edit(segs: list[dict], kinds: dict, drop=(), drop_kind=None, replace_msgs=None) -> str:
    """drop: refs to remove; drop_kind: remove every segment of that kind ('message' or 'result');
    replace_msgs: segment texts that take the place of the text messages"""
    out, placed = [], False
    for s in segs:
        if s["source"] == "message":
            kind = kinds.get(s["ref"])
            if s["ref"] in drop or kind == drop_kind:
                continue
            if replace_msgs is not None and kind == "message":
                if not placed:
                    out += [{"source": "message", "ref": None, "text": t} for t in replace_msgs]
                    placed = True
                continue
        out.append(s)
    if replace_msgs is not None and not placed:
        raise ValueError("no text message to replace")
    return join(out)


_books = {}


def path_recipes(meta: dict) -> list[tuple]:
    """the path's recipes (a, b, result): logged since hints; for older runs (main) every recipe of the target or an
    intermediate made of the starting items and intermediates, from the recipe book"""
    task = meta["task"]
    if task.get("path"):
        return [tuple(r) for r in task["path"]]
    data = meta["config"]["task"]["data"]
    if data not in _books:
        _books[data] = RecipeBook.load(ROOT / data)
    book, items = _books[data], [*task["initial"], *task.get("intermediate", [])]
    wanted = {task["target"], *task.get("intermediate", [])}
    return [(a, b, r) for i, a in enumerate(items) for b in items[i:] for r in sorted(book.combine(a, b) & wanted)]


def pair(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


def answers(p: tuple) -> list[str]:
    """the answer line with its newline: Llama 3.2 and Qwen3 write the closing quote and the newline as one token
    ("'\\n"), so a line cut at the quote ends on a token they almost never sample there (-6.4 nats for Llama against
    rescore.py, 2026-11-26); Gemma 3 tokenizes them apart and only gains a near-certain '\\n'"""
    a, b = p
    return [f"Combination: '{a}' and '{b}'\n"] + ([f"Combination: '{b}' and '{a}'\n"] if a != b else [])


def load_run(run: Path) -> dict:
    """the first-attempt calls by (agent, t), the published messages and the env steps of a run"""
    calls, msgs, steps = {}, {}, defaultdict(dict)
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        d = e["data"]
        if e["kind"] == "llm_call" and d["attempt"] == 0 and e["t"] is not None:
            calls[(d["agent"], e["t"])] = d
        elif e["kind"] == "msg_published":
            msgs[d["message"]["id"]] = d["message"]
        elif e["kind"] == "env_step":
            steps[e["t"]][d["agent"]] = d
    return {"calls": calls, "msgs": msgs, "steps": steps,
            "meta": json.loads((run / "meta.json").read_text(encoding="utf-8"))}


def kind_of(m: dict) -> str:
    return "message" if m["type"] == "message" else "result"


def turn(run: dict, agent: str, t: int, swap_from: list[dict]) -> dict | None:
    d = run["calls"][(agent, t)]
    segs = spans(d)
    refs = [s["ref"] for s in segs if s["source"] == "message"]
    if not refs:
        return None
    kinds = {r: kind_of(run["msgs"][r]) for r in refs}
    texts = [s["text"] for s in segs if s["source"] == "message" and kinds[s["ref"]] == "message"]
    inventory = set(QUOTED.findall(next(s["text"] for s in segs if s["ref"] == "inventory")))
    cands = defaultdict(set)
    if m := PAIR.search(d.get("parts", {}).get("combination") or ""):
        cands[pair(*m.groups())].add("played")
    target = run["meta"]["task"].get("target")
    for a, b, r in path_recipes(run["meta"]):
        cands[pair(a, b)].add("path_target" if r == target else "path_inter")
    for text in texts:
        for a, b in PAIR.findall(text):
            if a in inventory and b in inventory:
                cands[pair(a, b)].add("told")
    for other, s in run["steps"].get(t - 1, {}).items():
        if other != agent:
            cands[pair(*s["combo"])].add("mate_prev")
    prompts, swap_source = {"orig": d["prompt"]}, None
    if texts:
        prompts["no_msgs"] = edit(segs, kinds, drop_kind="message")
        for src in swap_from:
            other = src["calls"].get((agent, t))
            if other is None:
                continue
            osegs = spans(other)
            okinds = {s["ref"]: kind_of(src["msgs"][s["ref"]]) for s in osegs if s["source"] == "message"}
            otexts = [s["text"] for s in osegs if s["source"] == "message" and okinds[s["ref"]] == "message"]
            if otexts:
                prompts["swap_msgs"] = edit(segs, kinds, replace_msgs=otexts)
                swap_source = src["meta"]["run_id"]
                break
        for r in refs:
            if kinds[r] == "message":
                prompts[f"drop:{r}"] = edit(segs, kinds, drop={r})
    if any(k == "result" for k in kinds.values()):
        prompts["no_results"] = edit(segs, kinds, drop_kind="result")
    assert prompts["orig"] == join(segs), "rebuilding the logged prompt changed it"
    pairs = sorted(cands)
    flat = [a for p in pairs for a in answers(p)]
    scores = {}
    for name, prompt in prompts.items():
        s = score(prompt, flat, d["sampling"]["reasoning"])
        it, scores[name] = iter(s), []
        for p in pairs:
            vals = [next(it) for _ in answers(p)]
            m = max(vals)
            scores[name].append(round(m + math.log(sum(math.exp(v - m) for v in vals)), 4))
    check = None
    if run.get("check") is None and not args.dry and pairs:  # once per run: the cached path against a full forward
        a = answers(pairs[0])[0]
        check = round(score(d["prompt"], [a], d["sampling"]["reasoning"])[0] - full_forward(d["prompt"], a,
                                                                                          d["sampling"]["reasoning"]), 4)
        run["check"] = check
        checks.append(abs(check))
    senders = {r: run["msgs"][r].get("sender") for r in refs}
    return {"t": t, "agent": agent, "action": d["action"], "error": d["error"],
            "pairs": [list(p) for p in pairs], "tags": [sorted(cands[p]) for p in pairs],
            "messages": [{"ref": r, "sender": senders[r], "kind": kinds[r]} for r in refs],
            "swap_source": swap_source, "prompt_chars": {k: len(v) for k, v in prompts.items()}, "check": check,
            "logp": scores}


done, started, checks = 0, time.time(), []
for grid in args.grids:
    cells = defaultdict(list)
    for events in sorted(grid.glob("*/*/seed*_rep*/events.jsonl")):
        run = events.parent
        version, cell = run.parent.name, run.parent.parent.name
        if version in versions and (not args.cells or args.cells in cell):
            cells[(cell, version)].append(run)
    for (cell, version), runs in sorted(cells.items()):
        runs.sort(key=lambda r: int(r.name.split("rep")[1]))
        loaded = {}
        for i, run in enumerate(runs):
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            if meta["engine"].get("model_id") != entry.model or not (run / "summary.json").exists():
                continue
            out = args.out / grid.name / cell / version / f"{run.name}.jsonl"
            if out.exists():
                continue
            if args.limit is not None and done >= args.limit:
                break
            for j in (i, *[(i + k) % len(runs) for k in range(1, 4)]):
                if runs[j] not in loaded:
                    loaded[runs[j]] = load_run(runs[j])
            here = loaded[run]
            swap_from = [loaded[runs[(i + k) % len(runs)]] for k in range(1, 4) if len(runs) > 1]
            lines = [x for (agent, t) in sorted(here["calls"], key=lambda k: (k[1], k[0])) if t >= 1
                     for x in [turn(here, agent, t, swap_from)] if x is not None]
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
            tmp.rename(out)
            done += 1
            for old in [r for r in loaded if r not in runs[i:i + 4]]:
                del loaded[old]
            if done % 20 == 0:
                print(f"{done} runs, {time.time() - started:.0f} s", flush=True)
print(f"{args.model}: {done} runs scored in {time.time() - started:.0f} s; cached vs full forward, largest |difference| "
      f"{max(checks, default=0):.4f} nats over {len(checks)} runs")
