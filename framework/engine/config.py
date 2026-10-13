"""
Run configuration
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from framework.comm.compiler import ProtocolConfig, Strict, load_protocol

ROOT = Path(__file__).resolve().parents[2]


class RunSection(Strict):
    name: str = "run"
    seed: int = 0
    replicate: int = 0


class TaskConfig(Strict):
    type: Literal["targeted", "open_ended"] = "targeted"
    max_steps: int | None = Field(default=None, ge=1)
    depth: Literal[1, 2] = 1
    distractors: int = Field(default=3, ge=0)
    repeat_consumes_step: bool = True
    data: str = "data/alchemy2.json"


class EngineConfig(Strict):
    backend: Literal["mock", "hf_local"] = "mock"
    model: str | None = None
    revision: str | None = None
    max_concurrency: int = Field(default=4, ge=1)


class SamplingConfig(Strict):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = Field(default=256, ge=1)
    reasoning: bool = True


class ContextConfig(Strict):
    max_prompt_tokens: int = Field(default=6000, ge=1)
    truncation: Literal["oldest_first"] = "oldest_first"


class RunConfig(Strict):
    run: RunSection = RunSection()
    task: TaskConfig = TaskConfig()
    engine: EngineConfig = EngineConfig()
    sampling: SamplingConfig = SamplingConfig()
    protocol: str | ProtocolConfig = "configs/protocols/isolated.yaml"
    context: ContextConfig = ContextConfig()

    def resolved_protocol(self) -> ProtocolConfig:
        return load_protocol(ROOT / self.protocol) if isinstance(self.protocol, str) else self.protocol

    def max_steps(self) -> int:
        return self.task.max_steps or (6 if self.task.type == "targeted" else 200)


def parse_run_config(text: str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(text) or {})


def load_run_config(path: str | Path) -> RunConfig:
    return parse_run_config(Path(path).read_text(encoding="utf-8"))
