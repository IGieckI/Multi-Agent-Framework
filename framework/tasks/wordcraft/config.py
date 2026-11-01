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
    # prompt version: the templates in prompts/<version>/wordcraft
    prompts: Literal["v1", "v2"] = "v1"

    def step_limit(self) -> int:
        """Crafting steps per agent: `max_steps`, or SPEC §3.2's 6 (targeted) and 200 (open-ended)"""
        return self.max_steps or (6 if self.type == "targeted" else 200)

    def scale_horizon(self, factor: int) -> "WordCraftConfig":
        return self.model_copy(update={"max_steps": self.step_limit() * factor})
