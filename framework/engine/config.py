"""
Run configuration
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SerializeAsAny, field_validator, model_validator

from framework.comm.compiler import ProtocolConfig, Strict, load_protocol
from framework.tasks.base import TaskConfig
from framework.tasks.registry import parse_task_config

ROOT = Path(__file__).resolve().parents[2]

# the versions of a run: the condition as written ("base") and the baselines derived from it (framework/baselines.py)
VERSIONS = ("base", "all_independent", "one_for_all", "one_vs_multi_token", "one_vs_multi_completion_token")


class RunSection(Strict):
    name: str = "run"
    seed: int = 0
    replicate: int = 0
    # which versions to run; each one writes to runs/<name>/<version>/seed<seed>_rep<replicate>
    versions: tuple[str, ...] = ("base",)
    # set by framework/baselines.py on a config derived from a condition: which baseline it is
    baseline: str | None = None

    @field_validator("versions")
    @classmethod
    def known_versions(cls, versions: tuple[str, ...]) -> tuple[str, ...]:
        if not versions:
            raise ValueError("run.versions is empty: name at least one of " + ", ".join(VERSIONS))
        if unknown := [v for v in versions if v not in VERSIONS]:
            raise ValueError(f"unknown versions {unknown}; known: {list(VERSIONS)}")
        if len(set(versions)) < len(versions):
            raise ValueError(f"run.versions names a version twice: {list(versions)}")
        return versions


class EngineConfig(Strict):
    backend: Literal["mock", "hf_local"] = "mock"
    model: str | None = None
    max_concurrency: int = Field(default=4, ge=1)
    # baseline B5: one model hosts every agent of the run in one shared conversation (framework/agents/persona_host.py)
    persona_host: bool = False


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


class BudgetMatch(Strict):
    """Caps one unit at what a reference run used, for the same seed and replicate (baseline B2)"""
    # folder of the reference condition, from the repo root (e.g. runs/group_fully_connected)
    reference: str
    unit: Literal["total_tokens", "completion_tokens"]


class BudgetConfig(Strict):
    # None = no cap
    total_tokens: int | None = Field(default=None, ge=1)
    completion_tokens: int | None = Field(default=None, ge=1)
    calls: int | None = Field(default=None, ge=1)
    steps: int | None = Field(default=None, ge=1)
    wall_clock_s: float | None = Field(default=None, gt=0)
    # fixed reserve per counter checked before each step; without one, the largest step cost seen so far
    step_reserve: dict[Literal["total_tokens", "completion_tokens", "calls"], int] = {}
    match: BudgetMatch | None = None

    @model_validator(mode="after")
    def match_sets_its_own_cap(self) -> "BudgetConfig":
        if self.match is not None and getattr(self, self.match.unit) is not None:
            raise ValueError(f"budget.{self.match.unit} is set by budget.match; do not set both")
        return self


class RunConfig(Strict):
    run: RunSection = RunSection()
    # validated by the config class of the task named in `task.name` (framework/tasks/registry.py)
    task: SerializeAsAny[TaskConfig]
    engine: EngineConfig = EngineConfig()
    sampling: SamplingConfig = SamplingConfig()
    protocol: str | ProtocolConfig = "configs/protocols/isolated.yaml"
    context: ContextConfig = ContextConfig()
    retries: RetryConfig = RetryConfig()
    budget: BudgetConfig = BudgetConfig()
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
