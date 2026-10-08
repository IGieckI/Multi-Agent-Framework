"""
Messages and their types
"""

from dataclasses import dataclass, field
from typing import Any

AgentId = str

# Every message type and the payload fields it carries
MESSAGE_TYPES: dict[str, set[str]] = {
    "result.success": {"combo", "result"},
    "result.failure": {"combo"},
    "message": {"text"},
    "reasoning": {"text"},
    "task": {"text"},
    "verdict": {"text"},
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
    """Whether a rule or share entry called "name" applies to "msg_type" """
    return msg_type == name or msg_type.startswith(name + ".")


def is_known(name: str) -> bool:
    """A message type, or a prefix of one "result" """
    return any(covers(name, t) for t in MESSAGE_TYPES)
