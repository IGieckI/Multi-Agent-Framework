"""
The communication structure (each agent's role, the links, which types each role may
share, and the channel rule of each message type)
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from framework.comm.message import AgentId, MessageDraft, MessageTypes, covers, is_known, text_only
from framework.log import EventLog


class ChannelRule(BaseModel):
    """How one message type travels"""
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Visibility happens as:
    # - neighbors: along the links
    # - all: everyone (bypassing the links)
    # - a list of agent ids
    # - none
    visible_to: Literal["neighbors", "all", "none"] | tuple[AgentId, ...] = "neighbors"
    
    # For delayed publish (published at tick k, delivered from tick k + 1 + delay)
    delay: int = Field(default=0, ge=0)

    # What the receiver learns of the message:
    # - full: the whole message
    # - summary: a summary written by the run's summarizer model (text types only), in place of the message
    form: Literal["full", "summary"] = "full"

@dataclass
class AddEdge:
    """Adds a link, or adds types to an existing one"""
    src: AgentId
    dst: AgentId
    types: tuple[str, ...]


@dataclass
class RemoveEdge:
    """Removes some types from a link, or the whole link when types is None"""
    src: AgentId
    dst: AgentId
    types: tuple[str, ...] | None = None


@dataclass
class Regroup:
    """Replaces every link with one clique per group, carrying `types`"""
    partition: tuple[tuple[AgentId, ...], ...]
    types: tuple[str, ...]


@dataclass
class SetPolicy:
    """Changes settings of the channel rule in force for a type"""
    type: str
    changes: dict[str, Any]


Edit = AddEdge | RemoveEdge | Regroup | SetPolicy
EDITS = {cls.__name__: cls for cls in (AddEdge, RemoveEdge, Regroup, SetPolicy)}


def edit_record(edit: Edit) -> dict:
    return {"edit": type(edit).__name__, **asdict(edit)}


def edit_from_record(record: dict) -> Edit:
    fields = dict(record)
    return EDITS[fields.pop("edit")](**fields)


@dataclass
class Structure:
    agents: dict[AgentId, str]  # agent id -> role
    links: dict[tuple[AgentId, AgentId], frozenset[str]]  # (src, dst) -> types the link carries
    shares: dict[str, frozenset[str]]  # role -> types its agents may publish
    channels: dict[str, ChannelRule]
    # the run's message types (core + task), fixed for the run: not part of the snapshot
    types: MessageTypes

    def channel(self, msg_type: str) -> ChannelRule:
        """The most specific rule for the type"""
        names = [name for name in self.channels if covers(name, msg_type)]
        return self.channels[max(names, key=len)] if names else ChannelRule()

    def neighbors(self, agent: AgentId, msg_type: str) -> set[AgentId]:
        return {dst for (src, dst), types in self.links.items()
                if src == agent and any(covers(t, msg_type) for t in types)}

    def receivers(self, draft: MessageDraft) -> list[AgentId]:
        """Who receives this draft if it is published now"""
        if not any(covers(t, draft.type) for t in self.shares.get(self.agents[draft.sender], ())):
            return []
        visible_to = self.channel(draft.type).visible_to
        if visible_to == "neighbors":
            found = self.neighbors(draft.sender, draft.type)
        elif visible_to == "all":
            found = set(self.agents)
        elif visible_to == "none":
            found = set()
        else:
            found = set(visible_to) & set(self.agents)
        found.discard(draft.sender)
        if draft.recipients is not None:
            found &= set(draft.recipients)
        return sorted(found)

    def apply(self, edits: list[Edit], log: EventLog, t: int) -> None:
        """Applies the edits in order (an invalid one is logged as edit_rejected and changes nothing)"""
        for edit in edits:
            reason = self.apply_one(edit)
            if reason is None:
                log.write("structure", "edit_applied", t=t, **edit_record(edit))
            else:
                log.write("structure", "edit_rejected", t=t, reason=reason, **edit_record(edit))

    def apply_one(self, edit: Edit) -> str | None:
        """Applies one edit, or returns why it is invalid"""
        if isinstance(edit, SetPolicy):
            if not is_known(edit.type, self.types):
                return f"unknown message type {edit.type!r}"
            try:
                rule = ChannelRule.model_validate({**self.channel(edit.type).model_dump(), **edit.changes})
            except ValidationError as e:
                return f"invalid channel settings: {e.errors()[0]['msg']}"
            if isinstance(rule.visible_to, tuple) and not set(rule.visible_to) <= set(self.agents):
                return "visible_to names unknown agents"
            if rule.form == "summary" and not text_only(edit.type, self.types):
                return f"form summary needs text payloads, and {edit.type!r} covers other types"
            self.channels[edit.type] = rule
            return None

        types = frozenset(edit.types or ())
        if not all(is_known(t, self.types) for t in types):
            return f"unknown message types in {sorted(types)}"
        if isinstance(edit, Regroup):
            if sorted(a for group in edit.partition for a in group) != sorted(self.agents):
                return "the groups must contain every agent exactly once"
            if not types:
                return "Regroup needs at least one type"
            self.links = {(a, b): types for group in edit.partition for a in group for b in group if a != b}
            return None

        if edit.src == edit.dst or not {edit.src, edit.dst} <= set(self.agents):
            return "a link needs two different known agents"
        key = (edit.src, edit.dst)
        if isinstance(edit, AddEdge):
            if not types:
                return "AddEdge needs at least one type"
            self.links[key] = self.links.get(key, frozenset()) | types
        else:
            if key not in self.links:
                return "no such link"
            remaining = frozenset() if edit.types is None else self.links[key] - types
            if remaining:
                self.links[key] = remaining
            else:
                del self.links[key]
        return None

    def snapshot(self) -> dict:
        """The whole structure as plain JSON values, in a stable order"""
        return {
            "agents": dict(sorted(self.agents.items())),
            "links": [[src, dst, sorted(types)] for (src, dst), types in sorted(self.links.items())],
            "shares": {role: sorted(types) for role, types in sorted(self.shares.items())},
            "channels": {name: rule.model_dump(mode="json") for name, rule in sorted(self.channels.items())},
        }

    @classmethod
    def from_snapshot(cls, snap: dict, message_types: MessageTypes) -> "Structure":
        return cls(
            dict(snap["agents"]),
            {(src, dst): frozenset(types) for src, dst, types in snap["links"]},
            {role: frozenset(types) for role, types in snap["shares"].items()},
            {name: ChannelRule.model_validate(rule) for name, rule in snap["channels"].items()},
            message_types,
        )
