"""
A finished run's event log, read into the few shapes the metrics need. Metrics read only this
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActingTurn:
    """The call whose answer the task accepted: what one agent did at one step"""
    t: int
    agent: str
    action: Any
    # ids of the messages delivered to the agent for this turn
    inbox: tuple[str, ...]
    reasoning: str | None
    message: str | None
    # of the whole turn, retries included
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float


@dataclass
class RunView:
    meta: dict
    # the LLM calls of the agents, in log order
    calls: list[dict]
    acting: list[ActingTurn]
    # message id -> {id, sent_at, sender, type, payload, meta} (summaries included)
    messages: dict[str, dict]
    # receiver -> ids of the messages delivered to it, in delivery order
    delivered: dict[str, list[str]]
    env_steps: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def agents(self) -> list[str]:
        return sorted(self.meta["agents"])

    def steps(self) -> list[int]:
        """The steps at which at least one agent acted, in order"""
        return sorted({turn.t for turn in self.acting})

    def acting_at(self, t: int) -> list[ActingTurn]:
        return [turn for turn in self.acting if turn.t == t]

    def turns_of(self, agent: str) -> list[ActingTurn]:
        return [turn for turn in self.acting if turn.agent == agent]


def action_items(action: Any) -> frozenset[str]:
    """The strings an action is made of (a combination's two items); any other action counts as one string"""
    if isinstance(action, (list, tuple)) and all(isinstance(a, str) for a in action):
        return frozenset(action)
    return frozenset({json.dumps(action, sort_keys=True)})


def jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    """1 - |a & b| / |a | b|; 0 for two empty sets"""
    union = a | b
    return 1 - len(a & b) / len(union) if union else 0.0


def build_view(meta: dict, events: list[dict]) -> RunView:
    """Reads the events of a run. A turn's accepted call is its last call without an error"""
    calls = [e["data"] | {"t": e["t"]} for e in events if e["kind"] == "llm_call"]
    first_attempt: dict[tuple[str, int], dict] = {}
    spent: dict[tuple[str, int], list[float]] = {}
    for c in calls:
        key = (c["agent"], c["t"])
        first_attempt.setdefault(key, c)
        totals = spent.setdefault(key, [0, 0, 0.0])
        totals[0] += c["completion"]["prompt_tokens"]
        totals[1] += c["completion"]["generated_tokens"]
        totals[2] += c["completion"]["elapsed_s"]
    acting = []
    for c in calls:
        if c["error"] is not None or c["action"] is None:
            continue
        key = (c["agent"], c["t"])
        completion = c["completion"]
        acting.append(ActingTurn(
            t=c["t"], agent=c["agent"], action=c["action"], inbox=tuple(first_attempt[key]["inbox"]),
            reasoning=completion.get("reasoning") or c["parts"].get("reasoning") or None,
            message=c["parts"].get("message") or None,
            prompt_tokens=int(spent[key][0]), generated_tokens=int(spent[key][1]), elapsed_s=spent[key][2]))
    messages: dict[str, dict] = {}
    delivered: dict[str, list[str]] = {}
    for e in events:
        if e["kind"] in ("msg_published", "msg_summarized"):
            messages[e["data"]["message"]["id"]] = e["data"]["message"]
        elif e["kind"] == "msg_delivered":
            delivered.setdefault(e["data"]["receiver"], []).append(e["data"]["id"])
    env_steps = [e["data"] | {"t": e["t"]} for e in events if e["kind"] == "env_step"]
    return RunView(meta, calls, acting, messages, delivered, env_steps, events)
