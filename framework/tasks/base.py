"""
The interface between the engine and a task
"""

from typing import Any, Protocol, Sequence
from pydantic import BaseModel, ConfigDict
from framework.agents.backends.base import Sampling
from framework.agents.prompt import Prompt, View
from framework.comm.message import AgentId, MessageDraft

# everything a task exposes to one agent (observation field -> value, before the role filter)
Observation = dict[str, Any]

# a prompt section filled with bus messages (block, label, message types)
SocialBlock = tuple[str, str, tuple[str, ...]]


class TaskConfig(BaseModel):
    """The "task" section of a run config"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str

    def scale_horizon(self, factor: int) -> "TaskConfig":
        """Returns a copy whose episode is factor times longer (needed when comparing multiple agents against a single agent simulating multiples on env steps especially)"""
        raise NotImplementedError(f"task {self.name!r} does not say how to scale its horizon: "
                                  "override TaskConfig.scale_horizon in its config class")


class Task(Protocol):
    # the observation fields a role may list in its "visible_fields"
    fields: tuple[str, ...]
    
    # Messages delivered by the bus
    social_blocks: tuple[SocialBlock, ...]
        
    # the labels an answer is split on ()"combination", "reasoning", "message", etc)
    answer_labels: tuple[str, ...]

    def reset(self, seed: int, agents: Sequence[AgentId]) -> None:
        """Starts the episode: one environment instance per agent"""

    def observe(self, agent: AgentId) -> Observation:
        """Every observation field of the agent, before the role filter"""

    def build_prompt(self, view: View, error: str | None) -> Prompt:
        """The agent's prompt. "error" is the problem found in its previous answer, on a retry"""

    def truncate(self, observation: Observation) -> str | None:
        """Removes the oldest removable history entry from a filtered observation and returns its reference,
        None when nothing is left to remove"""

    def parse(self, agent: AgentId, text: str) -> Any:
        """The action in an answer. Should rais ParseError when there is none"""

    def check(self, agent: AgentId, action: Any) -> str | None:
        """Refuses an action that parses correctly but that the task won't accept, and gives the agent a message to retry with"""

    def step(self, agent: AgentId, action: Any) -> Any:
        """Applies the action to the agent's environment and returns the result"""

    def step_record(self, agent: AgentId, result: Any) -> dict[str, Any]:
        """The fields of the logged env_step event (JSON values)"""

    def no_op(self, agent: AgentId) -> None:
        """A turn without a usable action"""

    def outcome_message(self, agent: AgentId, action: Any, result: Any) -> MessageDraft | None:
        """The outcome message of a step to the acting agent (sender)
           Note: who receives it is up to the bus"""

    def agent_done(self, agent: AgentId) -> bool:
        """Whether the agent's episode is over: it no longer acts"""

    def end_reason(self) -> str | None:
        """None while any agent is still playing, else why the episode ended"""

    def summary(self) -> dict[str, Any]:
        """Final results from the oracle: {"task": ..., "team": ..., "agents": {agent: {...}}}"""

    def evaluate(self, meta: dict[str, Any], events: list[dict]) -> dict[str, float | None]:
        """The task's metrics of a finished run, computed from its metadata and event log only"""

    def series(self, meta: dict[str, Any], events: list[dict]) -> list[dict]:
        """The task's metrics per step, one row {"t": step, ...} per step (optional: a task without it has none)"""

    def state(self, agent: AgentId) -> dict[str, Any]:
        """The agent's current state, for display only"""

    def metadata(self) -> dict[str, Any]:
        """Return metadata: the task instance, and hashes of the data and the prompt templates"""

    def mock_answer(self, prompt: Prompt, sampling: Sampling) -> str:
        """MockBackend's default answer: a random answer in the task's format, from the prompt and seed only"""
