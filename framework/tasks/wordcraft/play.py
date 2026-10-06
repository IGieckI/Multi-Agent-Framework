"""
One LLM agent plays one WordCraft task, logging every call and every step
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

from framework.agents.backends.base import Sampling
from framework.agents.prompt import render
from framework.log import EventLog
from framework.tasks.wordcraft.env import WordCraftEnv
from framework.tasks.wordcraft.parse import ParseError, parse_combination
from framework.tasks.wordcraft.prompt import build_prompt, templates_hash
from framework.tasks.wordcraft.recipes import RecipeBook
from framework.tasks.wordcraft.tasks import TaskSpec

MAX_PARSE_ATTEMPTS = 5
MAX_ATTEMPTS = 6


def call_seed(seed: int, step: int, attempt: int) -> int:
    """A different, reproducible sampling seed for every call of the run"""
    return seed * 10_000 + step * 10 + attempt


async def choose(backend, env: WordCraftEnv, sampling: Sampling, seed: int, repeat_consumes_step: bool, log: EventLog):
    """Ask the model for a combination, re-prompting with the error until it gives a usable one,
    returns None if it never does."""
    error = None
    parse_failures = 0
    for attempt in range(MAX_ATTEMPTS):
        prompt = build_prompt(env, error)
        completion = await backend.generate(prompt, replace(sampling, seed=call_seed(seed, env.steps, attempt)))
        pair, error = None, None
        try:
            pair = parse_combination(completion.text, env.inventory)
            if not repeat_consumes_step and env.tried(*pair):
                error = f"You already tried '{pair[0]}' and '{pair[1]}'. Choose a combination you have not tried."
        except ParseError as e:
            error = str(e)
            parse_failures += 1
            if completion.finish_reason == "length":
                error += " Your answer was cut off: keep it shorter."
        log.write(
            "llm_call",
            env_step=env.steps,
            attempt=attempt,
            prompt=render(prompt),
            segments=[{"source": s.source, "ref": s.ref, "chars": len(s.text)} for s in prompt],
            sampling=asdict(replace(sampling, seed=call_seed(seed, env.steps, attempt))),
            completion=asdict(completion),
            reasoning_source="native" if completion.reasoning else None,
            combination=pair,
            error=error,
        )
        if error is None:
            return pair
        if parse_failures >= MAX_PARSE_ATTEMPTS:
            break
    return None


async def play(
    backend,
    book: RecipeBook,
    task: TaskSpec,
    sampling: Sampling,
    out_dir: Path,
    seed: int = 0,
    repeat_consumes_step: bool = True,
    data_sha256: str | None = None,
) -> dict:
    env = WordCraftEnv(book, task)
    meta = {
        "task": asdict(task),
        "seed": seed,
        "agents": ["agent_0"],
        "repeat_consumes_step": repeat_consumes_step,
        "sampling": asdict(sampling),
        "engine": backend.engine_info(),
        "prompts_hash": templates_hash(),
        "data_sha256": data_sha256,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log = EventLog(out_dir / "events.jsonl")
    log.write("run_started", env_step=0, **meta)
    try:
        while not env.done:
            t = env.steps
            pair = await choose(backend, env, sampling, seed, repeat_consumes_step, log)
            if pair is None:
                env.no_op()
                log.write("no_op", env_step=t)
                print(f"step {t}: no usable combination (no_op)")
                continue
            result = env.step(*pair)
            log.write("env_step", env_step=t, **asdict(result), inventory_size=len(env.inventory))
            print(f"step {t}: {pair[0]} + {pair[1]} -> {', '.join(result.results) or 'nothing'}"
                  + (" (repeat)" if result.repeat else ""))
    except Exception as e:
        log.write("run_failed", env_step=env.steps, error=repr(e))
        log.close()
        raise

    initial = set(task.initial)
    reachable = book.reachable(initial) - initial
    found = set(env.inventory) - initial
    summary = {
        "steps": env.steps,
        "solved": env.solved if task.target else None,
        "new_items": len(found),
        "coverage": len(found) / len(reachable) if reachable else None,
        "invalid": len(env.invalid),
    }
    log.write("run_ended", env_step=env.steps, **summary)
    log.close()
    return summary
