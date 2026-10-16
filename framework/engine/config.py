"""
Run configuration
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SerializeAsAny, field_validator

from framework.comm.compiler import ProtocolConfig, Strict, load_protocol
from framework.tasks.base import TaskConfig
from framework.tasks.registry import parse_task_config

ROOT = Path(__file__).resolve().parents[2]


class RunSection(Strict):
    name: str = "run"
    seed: int = 0
    replicate: int = 0


class EngineConfig(Strict):
    backend: Literal["mock", "hf_local"] = "mock"
    model: str | None = None
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


class RetryConfig(Strict):
    # unusable answers (no action the task can parse) after which the turn is a no-op
    max_parse_attempts: int = Field(default=5, ge=1)
    # model calls per turn in total, re-prompts for an action the task refuses (e.g. a repeat) included
    max_attempts: int = Field(default=6, ge=1)


class RunConfig(Strict):
    run: RunSection = RunSection()
    # validated by the config class of the task named in `task.name` (framework/tasks/registry.py)
    task: SerializeAsAny[TaskConfig]
    engine: EngineConfig = EngineConfig()
    sampling: SamplingConfig = SamplingConfig()
    protocol: str | ProtocolConfig = "configs/protocols/isolated.yaml"
    context: ContextConfig = ContextConfig()
    retries: RetryConfig = RetryConfig()
    # role -> text file (path from the repo root) with that role's persona, goal or behaviour. It is rendered right
    # after the rules for every agent of the role; a role without an entry gets no such block
    personas: dict[str, str] = {}

    @field_validator("task", mode="before")
    @classmethod
    def registered_task(cls, task: Any) -> TaskConfig:
        return parse_task_config(task)

    @field_validator("personas")
    @classmethod
    def persona_files_exist(cls, personas: dict[str, str]) -> dict[str, str]:
        if missing := [path for path in personas.values() if not (ROOT / path).is_file()]:
            raise ValueError(f"persona files not found: {missing}")
        return personas

    def resolved_protocol(self) -> ProtocolConfig:
        return load_protocol(ROOT / self.protocol) if isinstance(self.protocol, str) else self.protocol


def parse_run_config(text: str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(text) or {})


def load_run_config(path: str | Path) -> RunConfig:
    return parse_run_config(Path(path).read_text(encoding="utf-8"))
