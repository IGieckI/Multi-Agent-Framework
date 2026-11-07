"""
WordCraft prompts
"""

import hashlib
from pathlib import Path

from framework.agents.prompt import Prompt, Segment, View
from framework.comm.message import Message
from framework.tasks.wordcraft.env import WordCraftEnv

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"

# Observation fields a role can see (`visible_fields`)
FIELDS = ("inventory", "target", "notes", "steps_left", "own_valid", "own_invalid")

# State-block lines fed by the bus: (block, label, message types). A line is rendered only for an agent that can
# receive one of its types in the compiled structure, so it stays the same for the whole run
SOCIAL_BLOCKS = (
    ("others_valid", "Other players' valid combinations", ("result.success",)),
    ("others_invalid", "Other players' invalid combinations", ("result.failure",)),
    ("messages", "Messages from other players", ("message", "task", "verdict")),
    ("reasoning", "Other players' reasoning", ("reasoning",)),
)


# with task.share_notes, under the notes of an agent that can write messages
SHARE_NOTES = "The other players do not know your notes: use your message to tell them what your notes say.\n"
# with task.use_notes, under the messages of an agent that can receive them
USE_NOTES = "Recipes the other players tell you from their notes are true: when one helps make the target, use it.\n"


def full_view(env: WordCraftEnv) -> View:
    """Every observation field and the whole own history, no social blocks (a single agent)"""
    return View(env.observation())


def templates(version: str) -> Path:
    """The folder of the WordCraft templates of a prompt version"""
    return PROMPTS / version / "wordcraft"


def rules_file(targeted: bool, version: str = "v1") -> Path:
    return templates(version) / ("rules_targeted.txt" if targeted else "rules_open_ended.txt")


def templates_hash(version: str = "v1") -> str:
    """One hash over every template file of the version, stored in the run metadata"""
    digest = hashlib.sha256()
    for path in sorted(templates(version).glob("*.txt")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def quote(items) -> str:
    return ", ".join(f"'{item}'" for item in items)


def social_entry(message: Message) -> str:
    """How a delivered message reads in the state block"""
    if message.type == "result.success":
        a, b = message.payload["combo"]
        return f"'{a}' and '{b}' -> {quote(message.payload['result'])}"
    if message.type == "result.failure":
        a, b = message.payload["combo"]
        return f"'{a}' and '{b}'"
    sender = message.sender if message.type in ("message", "reasoning") else f"{message.sender} ({message.type})"
    return f"{sender}: {message.payload['text']}"


def social_segments(label: str, messages: list[Message], as_list: bool) -> list[Segment]:
    """One segment per message, so that each one keeps its message id"""
    segments = [Segment(f"{label}:", "template")]
    for i, m in enumerate(messages):
        separator = "\n- " if as_list else (" " if i == 0 else ", ")
        segments.append(Segment(separator + social_entry(m), "message", m.id))
    segments.append(Segment("\n", "template"))
    return segments


def build_prompt(view: View, targeted: bool, error: str | None = None, version: str = "v1",
                 share_notes: bool = False, use_notes: bool = False, made: list[str] | None = None) -> Prompt:
    """The rules, the role's persona if any, the message slot if any, the state block and, on a retry, the error found
    in the previous answer. The state block shows only the observation fields in the view. `made`: the items the
    agent made itself, listed under the inventory when there are any (task.mark_made)"""
    observation = view.observation
    rules = rules_file(targeted, version)
    message_slot = templates(version) / "message_slot.txt"
    segments = [Segment(rules.read_text(encoding="utf-8"), "template", rules.name)]
    if view.persona is not None:
        file, text = view.persona
        segments.append(Segment(text.rstrip("\n") + "\n\n", "role", file))
    if view.message_slot:
        segments.append(Segment(message_slot.read_text(encoding="utf-8"), "template", message_slot.name))
    segments.append(Segment("<Current task>\n", "template"))
    if "inventory" in observation:
        segments.append(Segment(f"Inventory: {quote(observation['inventory'])}\n", "observation", "inventory"))
        if made:
            segments.append(Segment(f"Items you made (already in your inventory): {quote(made)}\n", "observation",
                                    "made"))
    if observation.get("target") is not None:
        segments.append(Segment(f"Target: '{observation['target']}'\n", "observation", "target"))
    if "notes" in observation:
        notes = "; ".join(f"'{a}' and '{b}' make '{result}'" for a, b, result in observation["notes"])
        segments.append(Segment(f"Your notes (true, and only you were told them): {notes}\n", "observation", "notes"))
        if share_notes and view.message_slot:
            segments.append(Segment(SHARE_NOTES, "template"))
    if "steps_left" in observation:
        segments.append(Segment(f"Remaining rounds: {observation['steps_left']}\n", "observation", "steps_left"))
    if "budget_left" in observation:
        left, unit = observation["budget_left"]
        label = {"total_tokens": "tokens", "completion_tokens": "answer tokens"}.get(unit, unit)
        segments.append(Segment(f"Remaining budget: {left} {label} (the game ends when it runs out)\n", "observation",
                                "budget_left"))
    if "own_valid" in observation:
        valid = ", ".join(f"'{a}' and '{b}' -> {quote(results)}" for (a, b), results in observation["own_valid"])
        segments.append(Segment(f"Task valid combinations: {valid}\n", "observation", "own_valid"))
    if "own_invalid" in observation:
        invalid = ", ".join(f"'{a}' and '{b}'" for a, b in observation["own_invalid"])
        segments.append(Segment(f"Task invalid combinations: {invalid}\n", "observation", "own_invalid"))
    for block, label, _ in SOCIAL_BLOCKS:
        if block in view.social:
            segments += social_segments(label, view.social[block], as_list=block in ("messages", "reasoning"))
            if use_notes and block == "messages":
                segments.append(Segment(USE_NOTES, "template"))
    if error:
        segments.append(Segment(f"\n{error}\n", "own_output", "retry_error"))
    return tuple(segments)
