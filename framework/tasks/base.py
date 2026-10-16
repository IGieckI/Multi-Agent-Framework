"""
The interface between the engine and a task (SPEC §4.3)

The engine knows a task only through `Task`: it never reads a task's environment, items or answer format. A task
owns its environment instances (one per agent), its prompt builder, its answer parser and its oracle.

Golden rule: `build_prompt` reads only the View the engine hands it (the observation filtered by the agent's role,
the delivered messages, the message slot, the persona) and the task's static, versioned templates.
"""

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict

from framework.agents.backends.base import Sampling
from framework.agents.prompt import Prompt, View
from framework.comm.message import AgentId, MessageDraft

# observation field -> value, before the role filter
Observation = dict[str, Any]

# (block, label, message types): a prompt section fed by the bus. The engine gives an agent a block only if it can
# receive one of the block's types in the compiled structure, and fills it with those messages
SocialBlock = tuple[str, str, tuple[str, ...]]


class TaskConfig(BaseModel):
    """The `task` section of a run config. Each task subclasses it with `name: Literal["<its name>"]`"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str


class Task(Protocol):
    # the observation fields a role may list in `visible_fields`
    fields: tuple[str, ...]
    social_blocks: tuple[SocialBlock, ...]
    # the labels an answer is split on (lowercase), e.g. ("combination", "reasoning", "message")
    answer_labels: tuple[str, ...]

    def reset(self, seed: int, agents: Sequence[AgentId]) -> None:
        """Starts the episode: one environment instance per agent. `seed` is the task's own random stream"""

    def observe(self, agent: AgentId) -> Observation:
        """Every observation field of the agent, before the role filter, in fresh containers"""

    def build_prompt(self, view: View, error: str | None) -> Prompt:
        """The agent's prompt; `error` is the problem found in its previous answer, on a retry"""

    def truncate(self, observation: Observation) -> str | None:
        """Removes the oldest removable history entry from a filtered observation and returns its reference;
        None when nothing is left to remove. Called by the context guard once the social history is empty"""

    def parse(self, agent: AgentId, text: str) -> Any:
        """The action in an answer. Raises ParseError when there is none (counts towards max_parse_attempts)"""

    def check(self, agent: AgentId, action: Any) -> str | None:
        """An error to re-prompt with for a well-formed action the task refuses (e.g. a repeat), or None"""

    def step(self, agent: AgentId, action: Any) -> Any:
        """Applies the action to the agent's environment and returns the result"""

    def step_record(self, agent: AgentId, result: Any) -> dict[str, Any]:
        """The fields of the logged env_step event (JSON values)"""

    def no_op(self, agent: AgentId) -> None:
        """A turn without a usable action"""

    def outcome_message(self, agent: AgentId, action: Any, result: Any) -> MessageDraft | None:
        """The outcome message of a step (sender = the acting agent); who receives it is up to the bus"""

    def agent_done(self, agent: AgentId) -> bool:
        """Whether the agent's episode is over: it no longer acts"""

    def end_reason(self) -> str | None:
        """None while any agent is still playing, else why the episode ended (e.g. "all_solved")"""

    def summary(self) -> dict[str, Any]:
        """Final results from the oracle: {"task": ..., "team": ..., "agents": {agent: {...}}}"""

    def state(self, agent: AgentId) -> dict[str, Any]:
        """The agent's current state, for display only"""

    def metadata(self) -> dict[str, Any]:
        """Run metadata: the task instance, and hashes of the data and the prompt templates"""

    def mock_answer(self, prompt: Prompt, sampling: Sampling) -> str:
        """MockBackend's default answer: a random answer in the task's format, from the prompt and seed only"""
