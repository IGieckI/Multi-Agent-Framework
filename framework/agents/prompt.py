"""
Prompts are lists of segments, each recording where its text came from
"""

from dataclasses import dataclass

@dataclass
class Segment:
    text: str
    source: str  # template/role/observation/message/memory/own_output
    ref: str | None = None  # id of the observation/message/memory item


Prompt = tuple[Segment, ...]


def render(prompt: Prompt) -> str:
    return "".join(segment.text for segment in prompt)
