"""
WordCraft prompts: versioned rules first, then the current state block
"""

import hashlib
from pathlib import Path

from framework.agents.prompt import Prompt, Segment
from framework.tasks.wordcraft.env import WordCraftEnv

TEMPLATES = Path(__file__).resolve().parents[3] / "prompts" / "v1" / "wordcraft"


def rules_file(targeted: bool) -> Path:
    return TEMPLATES / ("rules_targeted.txt" if targeted else "rules_open_ended.txt")


def templates_hash() -> str:
    """One hash over every template file, stored in the run metadata"""
    digest = hashlib.sha256()
    for path in sorted(TEMPLATES.glob("*.txt")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def quote(items) -> str:
    return ", ".join(f"'{item}'" for item in items)


def build_prompt(env: WordCraftEnv, error: str | None = None) -> Prompt:
    """The rules, the state block and, on a retry, the error found in the previous answer"""
    rules = rules_file(env.task.target is not None)
    segments = [
        Segment(rules.read_text(encoding="utf-8"), "template", rules.name),
        Segment("<Current task>\n", "template"),
        Segment(f"Inventory: {quote(env.inventory)}\n", "observation", "inventory"),
    ]
    if env.task.target is not None:
        segments.append(Segment(f"Target: '{env.task.target}'\n", "observation", "target"))
    segments.append(Segment(f"Remaining rounds: {env.steps_left}\n", "observation", "steps_left"))
    valid = ", ".join(f"'{a}' and '{b}' -> {quote(results)}" for (a, b), results in env.valid.items())
    invalid = ", ".join(f"'{a}' and '{b}'" for a, b in env.invalid)
    segments.append(Segment(f"Task valid combinations: {valid}\n", "observation", "own_valid"))
    segments.append(Segment(f"Task invalid combinations: {invalid}\n", "observation", "own_invalid"))
    if error:
        segments.append(Segment(f"\n{error}\n", "own_output", "retry_error"))
    return tuple(segments)
