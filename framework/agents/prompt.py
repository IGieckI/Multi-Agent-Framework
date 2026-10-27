"""
Prompts are lists of segments, each recording where its text came from
"""

from dataclasses import dataclass, field
from typing import Any

from framework.comm.message import Message


@dataclass
class Segment:
    """Piece of prompt with metadata"""
    text: str

    # template/role/observation/message/memory/own_output
    source: str

    # id of the observation/message/memory item
    ref: str | None = None

    # the chat message the text belongs to (user/assistant/...)
    chat_role: str = "user"


Prompt = tuple[Segment, ...]


def render(prompt: Prompt) -> str:
    return "".join(segment.text for segment in prompt)


@dataclass
class View:
    """What one agent may see on its turn (only prompt input besides static templates)"""
    # the task's observation fields that the agent's role may see
    observation: dict[str, Any]
    # prompt block -> the delivered messages it shows
    social: dict[str, list[Message]] = field(default_factory=dict)
    # whether the prompt asks the agent to write a message
    message_slot: bool = False
    # (file, text) of the agent's role persona, if the run config gives one
    persona: tuple[str, str] | None = None
