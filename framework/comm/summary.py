"""
Form summary: the prompt of the summarizer model, which sees one message and the versioned template only.
The call itself is made by the engine (async and budget live there).
"""

import hashlib
from pathlib import Path

from framework.agents.backends.base import Sampling
from framework.agents.prompt import Prompt, Segment
from framework.comm.message import Message

TEMPLATE = Path(__file__).resolve().parents[2] / "prompts" / "v1" / "channels" / "summary.txt"
PLACEHOLDER = "{text}"

# fixed for every run: temperature 1.0, no reasoning; max_tokens comes from the summarizer config
TEMPERATURE = 1.0
REASONING = False


class SummaryTemplate:
    def __init__(self, path: Path = TEMPLATE) -> None:
        text = path.read_text(encoding="utf-8")
        if text.count(PLACEHOLDER) != 1:
            raise ValueError(f"{path} needs exactly one {PLACEHOLDER} placeholder")
        self.path = path
        self.before, self.after = text.split(PLACEHOLDER)
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prompt(self, message: Message, text: str | None = None) -> Prompt:
        """The prompt for one message; `text` replaces the message's text when the context guard shortened it"""
        return (
            Segment(self.before, "template", self.path.name),
            Segment(message.payload["text"] if text is None else text, "message", message.id),
            Segment(self.after, "template", self.path.name),
        )


def sampling(max_tokens: int, seed: int | None = None) -> Sampling:
    return Sampling(temperature=TEMPERATURE, max_tokens=max_tokens, reasoning=REASONING, seed=seed)


def mock_summary(prompt: Prompt, sampling: Sampling) -> str:
    """MockBackend's summarizer: the first words of the message, so tests can follow what a summary carries"""
    text = next(segment.text for segment in prompt if segment.source == "message")
    return " ".join(text.split()[:6])


def shortened(text: str, tokens: int, limit: int) -> str:
    """The start of `text`, cut in proportion to how far the prompt is over the limit (always shorter)"""
    keep = min(len(text) - 1, int(len(text) * limit / tokens * 0.9))
    return text[:max(keep, 0)]
