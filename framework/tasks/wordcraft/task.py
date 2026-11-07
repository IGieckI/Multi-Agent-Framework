"""
WordCraft behind the Task interface
"""

import hashlib
import random
from dataclasses import asdict
from typing import Sequence

from framework.agents.backends.base import Sampling
from framework.agents.prompt import Prompt, View
from framework.comm.message import AgentId, MessageDraft
from framework.tasks.base import Observation
from framework.tasks.wordcraft import metrics
from framework.tasks.wordcraft.config import ROOT, WordCraftConfig
from framework.tasks.wordcraft.env import StepResult, WordCraftEnv
from framework.tasks.wordcraft.parse import ANSWER_LABELS, parse_combination
from framework.tasks.wordcraft.prompt import FIELDS, SOCIAL_BLOCKS, build_prompt, templates_hash
from framework.tasks.wordcraft.recipes import Combo, RecipeBook
from framework.tasks.wordcraft.tasks import TaskSpec, deal_notes, open_ended, sample_targeted


class WordCraftTask:
    fields = FIELDS
    social_blocks = SOCIAL_BLOCKS
    answer_labels = ANSWER_LABELS
    # the outcome messages of a step (see outcome_message)
    message_types = {
        "result.success": frozenset({"combo", "result"}),
        "result.failure": frozenset({"combo"}),
    }

    def __init__(self, cfg: WordCraftConfig, book: RecipeBook, data_sha256: str | None = None) -> None:
        self.cfg = cfg
        # the oracle: never reaches a prompt
        self.book = book
        self.data_sha256 = data_sha256
        self.spec: TaskSpec | None = None
        self.envs: dict[AgentId, WordCraftEnv] = {}

    @classmethod
    def from_config(cls, cfg: WordCraftConfig) -> "WordCraftTask":
        """Loads the recipe file named in the config"""
        data = ROOT / cfg.data
        if not data.exists():
            raise FileNotFoundError(f"{data} is missing: run `python scripts/fetch_data.py` first")
        return cls(cfg, RecipeBook.load(data), hashlib.sha256(data.read_bytes()).hexdigest())

    def reset(self, seed: int, agents: Sequence[AgentId]) -> None:
        # the open-ended task is the same for every seed, a targeted one is sampled
        rng = random.Random(seed)
        if self.cfg.type == "open_ended":
            self.spec = open_ended(self.cfg.step_limit())
        else:
            self.spec = sample_targeted(self.book, rng, self.cfg.depth, self.cfg.distractors, self.cfg.step_limit(),
                                        self.cfg.allow_shortcuts)
        # dealt after the task is drawn, so a task does not depend on its hints
        notes = deal_notes(self.book, self.spec, len(agents), self.cfg.hints, self.cfg.notes, rng, self.cfg.decoys)
        self.envs = {a: WordCraftEnv(self.book, self.spec, notes=tuple(own)) for a, own in zip(agents, notes)}

    def observe(self, agent: AgentId) -> Observation:
        return self.envs[agent].observation()

    def build_prompt(self, view: View, error: str | None) -> Prompt:
        # the rules template depends on the task type, which is static for the run
        made = None
        if self.cfg.mark_made and "inventory" in view.observation:
            made = [item for item in view.observation["inventory"] if item not in self.spec.initial]
        return build_prompt(view, self.spec.target is not None, error, self.cfg.prompts, self.cfg.share_notes,
                            self.cfg.use_notes, made)

    def truncate(self, observation: Observation) -> str | None:
        """Invalid combinations go first, then valid ones, oldest first"""
        if observation.get("own_invalid"):
            a, b = observation["own_invalid"].pop(0)
            return f"own_invalid:{a}+{b}"
        if observation.get("own_valid"):
            (a, b), _ = observation["own_valid"].pop(0)
            return f"own_valid:{a}+{b}"
        return None

    def parse(self, agent: AgentId, text: str) -> Combo:
        return parse_combination(text, self.envs[agent].inventory)

    def check(self, agent: AgentId, action: Combo) -> str | None:
        if not self.cfg.repeat_consumes_step and self.envs[agent].tried(*action):
            return f"You already tried '{action[0]}' and '{action[1]}'. Choose a combination you have not tried."
        return None

    def step(self, agent: AgentId, action: Combo) -> StepResult:
        return self.envs[agent].step(*action)

    def step_record(self, agent: AgentId, result: StepResult) -> dict:
        return {**asdict(result), "inventory_size": len(self.envs[agent].inventory)}

    def no_op(self, agent: AgentId) -> None:
        self.envs[agent].no_op()

    def outcome_message(self, agent: AgentId, action: Combo, result: StepResult) -> MessageDraft:
        combo = list(result.combo)
        if result.results:
            return MessageDraft(agent, "result.success", {"combo": combo, "result": list(result.results)},
                                meta={"auto": True})
        return MessageDraft(agent, "result.failure", {"combo": combo}, meta={"auto": True})

    def agent_done(self, agent: AgentId) -> bool:
        return self.envs[agent].done

    def end_reason(self) -> str | None:
        envs = self.envs.values()
        if not all(env.done for env in envs):
            return None
        return "all_solved" if all(env.solved for env in envs) else "max_steps"

    def summary(self) -> dict:
        """Each agent's result and the team's"""
        initial = set(self.spec.initial)
        reachable = self.book.reachable(initial) - initial
        targeted = self.spec.target is not None
        agents, union = {}, set()
        for a, env in self.envs.items():
            found = set(env.inventory) - initial
            union |= found
            agents[a] = {
                "solved": env.solved if targeted else None,
                "steps": env.steps,
                "new_items": sorted(found),
                "coverage": round(len(found) / len(reachable), 4) if reachable else None,
                "invalid": len(env.invalid),
            }
        solved = [env.solved for env in self.envs.values()]
        return {
            "task": asdict(self.spec),
            "team": {
                "any_solved": any(solved) if targeted else None,
                "all_solved": all(solved) if targeted else None,
                "union_new_items": len(union),
                "union_coverage": round(len(union) / len(reachable), 4) if reachable else None,
            },
            "agents": agents,
        }

    def evaluate(self, meta: dict, events: list[dict]) -> dict[str, float | None]:
        return metrics.evaluate(self.book, meta, events)

    def series(self, meta: dict, events: list[dict]) -> list[dict]:
        return metrics.series(meta, events)

    def state(self, agent: AgentId) -> dict:
        env = self.envs[agent]
        return {"inventory": list(env.inventory), "solved": env.solved, "steps_left": env.steps_left,
                "done": env.done}

    def metadata(self) -> dict:
        return {"task": asdict(self.spec), "notes": {a: env.notes for a, env in self.envs.items()},
                "prompts": self.cfg.prompts, "prompts_hash": templates_hash(self.cfg.prompts),
                "data_sha256": self.data_sha256}

    def mock_answer(self, prompt: Prompt, sampling: Sampling, repeats: bool = True) -> str:
        return random_answer(prompt, sampling, repeats)


def inventory_in(prompt: Prompt) -> list[str]:
    """The items of the 'Inventory:' segment, empty if the prompt has none"""
    for segment in prompt:
        if segment.ref == "inventory":
            body = segment.text.removeprefix("Inventory:").strip()
            return body[1:-1].split("', '") if body else []
    return []


def tried_in(prompt: Prompt) -> str:
    """The text of the prompt's own valid and invalid combinations, and of other players' invalid ones"""
    others_invalid = next(label for block, label, _ in SOCIAL_BLOCKS if block == "others_invalid") + ":"
    texts, block = [], None
    for segment in prompt:
        if segment.source == "template":
            block = segment.text
        elif segment.ref in ("own_valid", "own_invalid") or (segment.source == "message" and block == others_invalid):
            texts.append(segment.text)
    return "".join(texts)


def random_answer(prompt: Prompt, sampling: Sampling, repeats: bool = True) -> str:
    """Plays like a random agent: two items of the prompt's inventory, picked with the call's seed, and a message
    when the prompt has a message slot. Without repeats it picks uniformly among the pairs the prompt does not list as
    tried by itself or as failed by others (any pair once all are listed)"""
    rng = random.Random(sampling.seed)
    inventory = inventory_in(prompt) or ["nothing"]
    if repeats:
        a, b = rng.choice(inventory), rng.choice(inventory)
    else:
        tried = tried_in(prompt)
        pairs = [(a, b) for i, a in enumerate(inventory) for b in inventory[i:]]
        fresh = [(a, b) for a, b in pairs if f"'{a}' and '{b}'" not in tried and f"'{b}' and '{a}'" not in tried]
        a, b = rng.choice(fresh or pairs)
    answer = f"Combination: '{a}' and '{b}'\nReasoning: a random pick."
    if any(segment.ref == "message_slot.txt" for segment in prompt):
        answer += f"\nMessage: I am trying '{a}' and '{b}'."
    return answer
