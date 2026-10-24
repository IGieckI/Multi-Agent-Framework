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

# the versions of a run (base vs possible baselines): B3, B4, B5, B0, B1, B2 (total and completion tokens)
VERSIONS = ("base", "all_independent", "all_independent_temp_sweep", "one_for_all", "one_agent",
            "one_agent_scaled_steps", "one_vs_multi_token", "one_vs_multi_completion_token")


class RunSection(Strict):
    """All info of a run"""
    name: str = "run"
    seed: int = 0
    # replicate id (part of the response cache key)
    replicate: int = 0
        
    versions: tuple[str, ...] = ("base",)
    
    # set by framework/baselines.py on a config derived from a condition: which baseline it is
    baseline: str | None = None

    @field_validator("versions")
    @classmethod
    def known_versions(cls, versions: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an empty list, unknown version names and duplicates"""
        if not versions:
            raise ValueError("run.versions is empty: name at least one of " + ", ".join(VERSIONS))
        if unknown := [v for v in versions if v not in VERSIONS]:
            raise ValueError(f"unknown versions {unknown}; known: {list(VERSIONS)}")
        if len(set(versions)) < len(versions):
            raise ValueError(f"run.versions names a version twice: {list(versions)}")
        return versions


class EngineConfig(Strict):
    """The LLM backend that serves the agents"""
    # "mock" is the test backend, "hf_local" runs a HuggingFace model on this machine
    backend: Literal["mock", "hf_local"] = "mock"
    # model id, recorded in the run metadata; None for the mock backend
    model: str | None = None
    # commit of the model's weights (configs/models/models.yaml pins one per model); None = the hub's latest
    revision: str | None = None
    # most model calls in flight at once
    max_concurrency: int = Field(default=4, ge=1)
    # baseline B5: one model hosts every agent of the run in one shared conversation (framework/agents/persona_host.py)
    persona_host: bool = False


class SummarizerConfig(Strict):
    """The model that writes the messages of channels with form summary. Required when a channel uses that
    form; a summarizer with the agents' backend and model shares their loaded model. It samples at temperature 1.0
    without reasoning (framework/comm/summary.py)"""
    backend: Literal["mock", "hf_local"]
    model: str | None = None
    revision: str | None = None
    # cap on the length of one summary, in completion tokens
    max_tokens: int = Field(default=128, ge=1)


class SamplingConfig(Strict):
    """Sampling parameters of the agents' calls; frozen across conditions except baseline B4"""
    temperature: float = 1.0
    # nucleus sampling threshold; 1.0 = off
    top_p: float = 1.0
    # sample from the k likeliest tokens only; 0 = off
    top_k: int = 0
    # cap on the completion tokens of one call, reasoning included
    max_tokens: int = Field(default=256, ge=1)
    # whether the model may write reasoning before its action
    reasoning: bool = True
    # baseline B4 only: the temperature of each agent (sorted agent ids, cycling through the list), replacing
    # "temperature". None = every agent samples at `temperature`
    agent_temperatures: tuple[float, ...] | None = None

    @field_validator("agent_temperatures")
    @classmethod
    def temperatures_are_not_negative(cls, temperatures: tuple[float, ...] | None) -> tuple[float, ...] | None:
        """Reject an empty list and a negative temperature"""
        if temperatures is not None and (not temperatures or min(temperatures) < 0):
            raise ValueError(f"sampling.agent_temperatures needs at least one temperature, none negative: {temperatures}")
        return temperatures


class ContextConfig(Strict):
    """Limit on the prompt size, so prompts never grow without bound"""
    # checked before every call, a longer prompt is truncated and the truncation is logged
    max_prompt_tokens: int = Field(default=6000, ge=1)
    # what to drop first when the prompt is too long
    truncation: Literal["oldest_first"] = "oldest_first"


class RetryConfig(Strict):
    """How often one turn may call the model again after an unusable answer"""
    # unusable answers (no action the task can parse) after which the turn is a no-op
    max_parse_attempts: int = Field(default=5, ge=1)
    # model calls per turn in total, re-prompts for an action the task refuses (e.g. a repeat) included
    max_attempts: int = Field(default=6, ge=1)


class BudgetMatch(Strict):
    """Caps one unit at what a reference run used, for the same seed and replicate"""
    # folder of the reference condition, from the repo root (e.g. runs/group_fully_connected)
    reference: str
    # the unit to cap at the reference run's usage
    unit: Literal["total_tokens", "completion_tokens"]


class BudgetConfig(Strict):
    """Caps on the resources of a run; the run ends cleanly at a step boundary when one is reached"""
    # None = no cap
    total_tokens: int | None = Field(default=None, ge=1)
    completion_tokens: int | None = Field(default=None, ge=1)
    # LLM calls, those of channels (like summaries) included
    calls: int | None = Field(default=None, ge=1)
    # environment steps
    steps: int | None = Field(default=None, ge=1)
    # wall clock time, in seconds
    wall_clock_s: float | None = Field(default=None, gt=0)
    # fixed reserve per counter checked before each step; without one, the largest step cost seen so far
    step_reserve: dict[Literal["total_tokens", "completion_tokens", "calls"], int] = {}
    # take the cap of one unit from a reference run instead of setting it above
    match: BudgetMatch | None = None

    @model_validator(mode="after")
    def match_sets_its_own_cap(self) -> "BudgetConfig":
        """Reject a config that sets the matched unit's cap by hand as well as through 'match'"""
        if self.match is not None and getattr(self, self.match.unit) is not None:
            raise ValueError(f"budget.{self.match.unit} is set by budget.match; do not set both")
        return self


class RunConfig(Strict):
    """Top level of a run config file: every setting of one run, validated on load"""
    run: RunSection = RunSection()
    # validated by the config class of the task named in `task.name` (framework/tasks/registry.py)
    task: SerializeAsAny[TaskConfig]
    engine: EngineConfig = EngineConfig()
    # only needed when a channel has form summary
    summarizer: SummarizerConfig | None = None
    sampling: SamplingConfig = SamplingConfig()
    # path (from the repo root) of the protocol YAML, or the protocol written inline
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
        """Build the task's own config class from the raw 'task' section, chosen by 'task.name'"""
        return parse_task_config(task)

    @field_validator("personas")
    @classmethod
    def persona_files_exist(cls, personas: dict[str, str]) -> dict[str, str]:
        """Reject persona paths that do not point to a file under the repo root"""
        if missing := [path for path in personas.values() if not (ROOT / path).is_file()]:
            raise ValueError(f"persona files not found: {missing}")
        return personas

    def resolved_protocol(self) -> ProtocolConfig:
        """The protocol as a config object: loads the YAML file if 'protocol' is a path"""
        return load_protocol(ROOT / self.protocol) if isinstance(self.protocol, str) else self.protocol


def parse_run_config(text: str) -> RunConfig:
    """Parse and validate a run config from YAML text (empty text gives the defaults)"""
    return RunConfig.model_validate(yaml.safe_load(text) or {})


def load_run_config(path: str | Path) -> RunConfig:
    """Read a YAML file and return its validated run config"""
    return parse_run_config(Path(path).read_text(encoding="utf-8"))
