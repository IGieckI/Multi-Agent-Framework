"""
Budget caps for each run (usage is charged from what the engine reports after each call)
"""

import json
from pathlib import Path
from typing import Literal

from framework.engine.usage import Usage, logged_completions
from framework.log import read_events

# endings that interrupt a run instead of finishing it (only a run that ended nicelycounts as a
# reference budget, as a result to evaluate, or as a done replicate
INTERRUPTED = frozenset({"stopped_by_user", "window_closed", "error", "backend_error"})

# the counters a step reserve applies to
METERED = ("total_tokens", "completion_tokens", "calls")
Unit = Literal["total_tokens", "completion_tokens"]


class BudgetEnforcer:
    def __init__(self, caps: dict[str, float | None], usage: Usage, step_reserve: dict[str, int] | None = None) -> None:
        # cap name -> limit; None is no cap
        self.caps = caps
        self.usage = usage
        # counter -> fixed reserve; a counter without one reserves the largest step cost seen so far
        self.step_reserve = dict(step_reserve or {})
        self.largest_step = dict.fromkeys(METERED, 0)
        self.at_step_start = dict.fromkeys(METERED, 0)
        # worst-case cost of the calls currently running
        self.in_flight = dict.fromkeys(METERED, 0)

    def before_step(self, t: int) -> str | None:
        """The cap that ends the run before step t, or None"""
        if self.caps.get("steps") is not None and t >= self.caps["steps"]:
            return "steps"
        if self.caps.get("wall_clock_s") is not None and self.usage.elapsed_s() >= self.caps["wall_clock_s"]:
            return "wall_clock_s"
        for counter in METERED:
            cap = self.caps.get(counter)
            if cap is None:
                continue
            remaining = cap - self.usage.used[counter]
            if remaining <= 0 or remaining < self.step_reserve.get(counter, self.largest_step[counter]):
                return counter
        self.at_step_start = dict(self.usage.used)
        return None

    def end_step(self) -> None:
        """Updates the largest step cost with the step that just ended"""
        for counter in METERED:
            cost = self.usage.used[counter] - self.at_step_start[counter]
            self.largest_step[counter] = max(self.largest_step[counter], cost)

    def remaining(self, counter: str) -> int:
        """What is left of the cap of `counter`"""
        return max(0, int(self.caps[counter] - self.usage.used[counter]))

    @staticmethod
    def worst_case(prompt_tokens: int, max_tokens: int) -> dict[str, int]:
        return {"total_tokens": prompt_tokens + max_tokens, "completion_tokens": max_tokens, "calls": 1}

    def reserve(self, prompt_tokens: int, max_tokens: int) -> str | None:
        """Reserves the worst case of a call; returns the cap it could exceed instead, reserving nothing"""
        if self.caps.get("wall_clock_s") is not None and self.usage.elapsed_s() >= self.caps["wall_clock_s"]:
            return "wall_clock_s"
        cost = self.worst_case(prompt_tokens, max_tokens)
        for counter in METERED:
            cap = self.caps.get(counter)
            if cap is not None and self.usage.used[counter] + self.in_flight[counter] + cost[counter] > cap:
                return counter
        for counter in METERED:
            self.in_flight[counter] += cost[counter]
        return None

    def release(self, prompt_tokens: int, max_tokens: int) -> None:
        """Gives back a reservation (the call failed and reported no usage)"""
        for counter, amount in self.worst_case(prompt_tokens, max_tokens).items():
            self.in_flight[counter] -= amount

    def settle(self, prompt_tokens: int, max_tokens: int, used_prompt: int, used_completion: int,
               line: str = "agents") -> None:
        """Replaces a reservation with the usage the engine reported, charged to the call's budget line"""
        self.release(prompt_tokens, max_tokens)
        self.usage.charge(used_prompt, used_completion, line)


def finished(run_dir: Path) -> bool:
    """Whether the run in `run_dir` ended on its own (its summary.json is written when it ends)"""
    summary = run_dir / "summary.json"
    return summary.exists() and json.loads(summary.read_text(encoding="utf-8"))["ended_by"] not in INTERRUPTED


def reference_usage(run_dir: Path, unit: Unit) -> int:
    """The tokens a finished run used, in "unit", summed from its logged LLM calls (every budget line)"""
    path = run_dir / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"budget.match: no reference run at {run_dir}")
    events = read_events(path)
    ended = next((e["data"]["ended_by"] for e in events if e["kind"] == "run_ended"), None)
    if ended is None:
        raise ValueError(f"budget.match: the reference run {run_dir} has not ended")
    if ended in INTERRUPTED:
        raise ValueError(f"budget.match: the reference run {run_dir} was interrupted ({ended}); only a run that "
                         "ended on its own can set a budget")
    completions = logged_completions(events)
    if unit == "completion_tokens":
        return sum(c["generated_tokens"] for c in completions)
    return sum(c["prompt_tokens"] + c["generated_tokens"] for c in completions)
