"""
Model backend interface that turns a Prompt into a Completion.
"""

from dataclasses import dataclass, field
from typing import Callable
from framework.agents.prompt import Prompt, render


# Called while a model generates, with the raw text produced so far (reasoning included). Only for display: the
# Completion is still what the run uses and logs. A backend that cannot stream may call it once, at the end
TextListener = Callable[[str], None]


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = off
    max_tokens: int = 256
    reasoning: bool = True
    seed: int | None = None


@dataclass
class Completion:
    # the answer, without the reasoning
    text: str
    # None if reasoning is not available/produced    
    reasoning: str | None
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    # Why the generation stopped (like token or stop)
    finish_reason: str | None
    full_raw_answer: str
    meta: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    """General backend error."""


def to_chat_messages(prompt: Prompt) -> list[dict]:
    """Build the prompt as a list of chat messages, for the HF API"""
    return [{"role": "user", "content": render(prompt)}]


def split_reasoning(raw: str) -> tuple[str | None, str]:
    """Split the reasoning in the output from the answer -> (reasoning, answer)"""
    if "</think>" in raw:
        reasoning, answer = raw.split("</think>", 1)
        reasoning = reasoning.split("<think>", 1)[-1]
    elif "<think>" in raw:
        reasoning, answer = raw.split("<think>", 1)[1], ""
    else:
        return None, raw.strip()
    return (reasoning.strip() or None), answer.strip()
