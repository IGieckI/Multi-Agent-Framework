"""
Messages and their types
"""

from dataclasses import dataclass, field
from typing import Any

AgentId = str

# message type -> the payload fields it carries
MessageTypes = dict[str, frozenset[str]]

# The types every task has. A task adds its own in the task registry
CORE_MESSAGE_TYPES: MessageTypes = {
    "message": frozenset({"text"}),
    "reasoning": frozenset({"text"}),
    "task": frozenset({"text"}),
    "verdict": frozenset({"text"}),
}


@dataclass
class MessageDraft:
    """What an agent or the task hands to the bus"""
    sender: AgentId
    type: str
    payload: dict[str, Any]
    recipients: tuple[AgentId, ...] | None = None
    parents: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    id: str
    sent_at: int
    sender: AgentId
    type: str
    payload: dict[str, Any]
    parents: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def covers(name: str, msg_type: str) -> bool:
    """Whether a rule written for name applies to type"""
    return msg_type == name or msg_type.startswith(name + ".")


def is_known(name: str, types: MessageTypes) -> bool:
    """Whether "name" is one of the types, or a prefix of anyone"""
    return any(covers(name, t) for t in types)


def text_only(name: str, types: MessageTypes) -> bool:
    """Whether every type "name" covers carries only a text payload (the types a summary can stand in for)"""
    return all(fields == {"text"} for t, fields in types.items() if covers(name, t))
