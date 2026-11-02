"""
A backend without a model: answers with a function of the prompt and the call's seed (a random agent, or a test
summarizer)
"""

from typing import Callable

from framework.agents.backends.base import Completion, Sampling, TextListener
from framework.agents.prompt import Prompt, render

Answer = Callable[[Prompt, Sampling], str]


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1


class MockBackend:
    def __init__(self, answer: Answer, name: str = "mock") -> None:
        self.answer = answer
        self.name = name

    def engine_info(self) -> dict:
        return {"backend": "mock", "model_id": self.name, "context_window": None}

    async def generate(self, prompt: Prompt, sampling: Sampling, on_text: TextListener | None = None) -> Completion:
        text = self.answer(prompt, sampling)
        if on_text:
            on_text(text)
        return Completion(
            text=text,
            reasoning=None,
            prompt_tokens=approx_tokens(render(prompt)),
            generated_tokens=approx_tokens(text),
            elapsed_s=0.0,
            finish_reason="stop",
            full_raw_answer=text,
        )
