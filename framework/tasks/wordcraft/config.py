"""
The task section of a WordCraft run config
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from framework.tasks.base import TaskConfig

ROOT = Path(__file__).resolve().parents[3]


class WordCraftConfig(TaskConfig):
    name: Literal["wordcraft"]
    type: Literal["targeted", "open_ended"] = "targeted"
    max_steps: int | None = Field(default=None, ge=1)
    depth: Literal[1, 2] = 1
    distractors: int = Field(default=3, ge=0)
    repeat_consumes_step: bool = True
    # recipe file, path from the repo root
    data: str = "data/alchemy2.json"
    # private notes, true recipes shown to one agent only (FINDINGS.md F30): none; split = the target's recipes dealt to
    # different agents; all = every agent knows them. Each agent's other notes are recipes off the path
    hints: Literal["none", "split", "all"] = "none"
    # notes per agent, when there are hints
    notes: int = Field(default=2, ge=1)
    # fill each agent's notes up to `notes` with recipes off the path; without, an agent holds only its share of the path
    decoys: bool = True
    # with a message slot, ask the agent to tell the others what its notes say
    share_notes: bool = False
    # with messages from others, tell the agent that the notes they pass on are true and to use them
    use_notes: bool = False
    # targets whose path items can also be made another way from the starting items (e.g. a depth-2 target in one
    # step, F32). The grids up to `hints` ran with them; new grids set false
    allow_shortcuts: bool = True
    # prompt version: the templates in prompts/<version>/wordcraft
    prompts: Literal["v1", "v2"] = "v1"

    def step_limit(self) -> int:
        """Crafting steps per agent: `max_steps`, or SPEC §3.2's 6 (targeted) and 200 (open-ended)"""
        return self.max_steps or (6 if self.type == "targeted" else 200)

    def scale_horizon(self, factor: int) -> "WordCraftConfig":
        return self.model_copy(update={"max_steps": self.step_limit() * factor})
