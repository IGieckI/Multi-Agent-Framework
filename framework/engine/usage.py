"""
Usage accounting for tokens, in total and per budget line
"""

import time

COUNTERS = ("total_tokens", "completion_tokens", "calls")
# agents: the agents' own calls; channels: calls made by channel forms (summary)
LINES = ("agents", "channels")
# the logged events of LLM calls, one per line
CALL_EVENTS = {"llm_call": "agents", "channel_call": "channels"}


class Usage:
    def __init__(self) -> None:
        self.used = dict.fromkeys(COUNTERS, 0)
        self.lines = {line: dict.fromkeys(COUNTERS, 0) for line in LINES}
        self.started = time.monotonic()

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def charge(self, prompt_tokens: int, completion_tokens: int, line: str = "agents") -> None:
        """Adds the usage the engine reported for one call"""
        for counters in (self.used, self.lines[line]):
            counters["total_tokens"] += prompt_tokens + completion_tokens
            counters["completion_tokens"] += completion_tokens
            counters["calls"] += 1

    def report(self) -> dict:
        return {**self.used, "wall_clock_s": round(self.elapsed_s(), 3),
                "lines": {line: dict(counters) for line, counters in self.lines.items()}}


def logged_completions(events: list[dict]) -> list[dict]:
    """The completion of every logged LLM call, of every budget line"""
    return [e["data"]["completion"] for e in events if e["kind"] in CALL_EVENTS]
