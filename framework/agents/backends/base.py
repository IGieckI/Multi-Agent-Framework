"""
Model backend interface that turns a Prompt into a Completion.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable
from framework.agents.prompt import Prompt


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
    """Build the prompt as a list of chat messages: consecutive segments with the same chat role form one message, so
    an ordinary prompt is a single user message"""
    messages: list[dict] = []
    for segment in prompt:
        if messages and messages[-1]["role"] == segment.chat_role:
            messages[-1]["content"] += segment.text
        else:
            messages.append({"role": segment.chat_role, "content": segment.text})
    return messages


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


# the line of the answer whose confidence is measured (the task's action); its text after the label
COMBINATION = re.compile(r"combination\W*:(.*)", re.IGNORECASE)


def confidence_summary(raw: str, logprobs: list[float], entropies: list[float], ends: list[int]) -> dict:
    """Confidence of one answer, from the log-probability of each generated token under the distribution it was
    sampled from and that distribution's entropy (nats). `ends[i]`: where token i ends in `raw`. The combination is the
    text after the first 'Combination:' label, up to the end of its line; a token belongs to it if it overlaps it.
    combo_logprob is the log-probability of that whole text (exp of it: how likely the model was to write this pair)"""
    n = len(logprobs)
    out = {"tokens": n,
           "mean_logprob": sum(logprobs) / n if n else None,
           "perplexity": math.exp(-sum(logprobs) / n) if n else None,
           "mean_entropy": sum(entropies) / n if n else None,
           "combo_tokens": 0, "combo_logprob": None, "combo_entropy": None, "combo_first_entropy": None}
    match = COMBINATION.search(raw)
    if match and n:
        start, end = match.start(1), raw.find("\n", match.start(1))
        end = len(raw) if end < 0 else end
        inside = [i for i in range(n) if (ends[i - 1] if i else 0) < end and ends[i] > start
                  and raw[(ends[i - 1] if i else 0):ends[i]].strip()]
        if inside:
            out.update(combo_tokens=len(inside), combo_logprob=sum(logprobs[i] for i in inside),
                       combo_entropy=sum(entropies[i] for i in inside) / len(inside),
                       combo_first_entropy=entropies[inside[0]])
    return out
