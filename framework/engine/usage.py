"""
Usage accounting for tokens
"""

import time

COUNTERS = ("total_tokens", "completion_tokens", "calls")


class Usage:
    def __init__(self) -> None:
        self.used = dict.fromkeys(COUNTERS, 0)
        self.started = time.monotonic()

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def charge(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Adds the usage the engine reported for one call"""
        self.used["total_tokens"] += prompt_tokens + completion_tokens
        self.used["completion_tokens"] += completion_tokens
        self.used["calls"] += 1

    def report(self) -> dict:
        return {**self.used, "wall_clock_s": round(self.elapsed_s(), 3)}
