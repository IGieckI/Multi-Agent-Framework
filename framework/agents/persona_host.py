"""
One model plays every agent of the run in one shared conversation
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from framework.agents.prompt import Prompt, Segment, render
from framework.comm.message import AgentId

TEMPLATES = Path(__file__).resolve().parents[2] / "prompts" / "v1" / "persona_host"
INTRO = TEMPLATES / "intro.txt"
TURN = TEMPLATES / "turn.txt"


def templates_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(TEMPLATES.glob("*.txt")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


@dataclass
class HostTurn:
    id: str
    agent: AgentId
    # the rendered user message: header and the agent's own part of the prompt
    user: str
    answer: str


class PersonaHost:
    def __init__(self, limit: int | None) -> None:
        # max tokens of the whole conversation; None = no limit
        self.limit = limit
        self.intro = INTRO.read_text(encoding="utf-8")
        self.turn_header = TURN.read_text(encoding="utf-8")
        self.rules: Segment | None = None
        self.turns: list[HostTurn] = []
        # index of the oldest turn still in the conversation
        self.first = 0
        # the user message of the call being made, recorded with its answer
        self.pending: tuple[AgentId, str] | None = None

    def preamble(self) -> list[Segment]:
        return [Segment(self.intro, "template", INTRO.name), self.rules]

    def history(self) -> list[Segment]:
        segments = []
        for turn in self.turns[self.first:]:
            segments.append(Segment(turn.user, "shared_context", f"{turn.id}:user"))
            segments.append(Segment(turn.answer, "shared_context", f"{turn.id}:answer", chat_role="assistant"))
        return segments

    def compose(self, agent: AgentId, prompt: Prompt, count: Callable[[Prompt], int]) -> tuple[Prompt, int, list[str]]:
        """The whole conversation ending with this agent's turn, its token count, and the ids of the turns dropped to
        fit the limit"""
        rules, own = prompt[0], prompt[1:]
        if self.rules is None:
            self.rules = rules
        elif rules.text != self.rules.text:
            raise ValueError("persona_host needs every agent prompt to start with the same rules segment; "
                             f"{agent}'s starts with a different one ({rules.ref!r})")
        current = [Segment(self.turn_header.format(agent=agent), "template", TURN.name), *own]
        removed = []
        while True:
            full = tuple(self.preamble() + self.history() + current)
            tokens = count(full)
            if self.limit is None or tokens <= self.limit or self.first == len(self.turns):
                break
            removed.append(self.turns[self.first].id)
            self.first += 1
        self.pending = (agent, render(tuple(current)))
        return full, tokens, removed

    def record(self, answer: str) -> None:
        """Adds the answer to the call composed last, and that call's turn, to the conversation"""
        agent, user = self.pending
        self.turns.append(HostTurn(f"h{len(self.turns)}", agent, user, answer))
        self.pending = None
