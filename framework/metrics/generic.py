"""
Metrics that do not depend on the task: usage, latency, prompt growth, and how much the agents' actions differ from
each other and from their own previous action. Everything is computed from the log (see log_view.py)
"""

from itertools import combinations

import numpy as np

from framework.metrics.log_view import RunView, action_items, jaccard_distance


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def usage_metrics(view: RunView) -> dict[str, float | None]:
    """Latency and prompt length of the agents' calls (token and call totals are in framework/evaluate.py)"""
    first = [c for c in view.calls if c["attempt"] == 0]
    elapsed = [c["completion"]["elapsed_s"] for c in view.calls]
    steps = view.steps()
    # prompt tokens of the first attempt of every turn against the step: the slope is the growth per step
    slope = None
    if len({c["t"] for c in first}) >= 2:
        slope = float(np.polyfit([c["t"] for c in first], [c["counted_prompt_tokens"] for c in first], 1)[0])
    return {
        "latency_total_s": float(sum(elapsed)),
        "latency_mean_s": mean_or_none(elapsed),
        "tokens_per_step": (sum(c["completion"]["prompt_tokens"] + c["completion"]["generated_tokens"]
                                for c in view.calls) / len(steps)) if steps else None,
        "prompt_tokens_first_step": mean_or_none([c["counted_prompt_tokens"] for c in first if c["t"] == steps[0]])
        if steps else None,
        "prompt_tokens_last_step": mean_or_none([c["counted_prompt_tokens"] for c in first if c["t"] == steps[-1]])
        if steps else None,
        "prompt_tokens_slope": slope,
    }


def action_divergence_at(view: RunView, t: int) -> float | None:
    """Mean pairwise distance between the actions the agents took at step t (0 = all equal, 1 = no item shared)"""
    actions = [action_items(turn.action) for turn in view.acting_at(t)]
    pairs = list(combinations(actions, 2))
    return mean_or_none([jaccard_distance(a, b) for a, b in pairs])


def action_change_at(view: RunView, t: int) -> float | None:
    """Mean distance between each agent's action at step t and its previous action (None at its first)"""
    changes = []
    for turn in view.acting_at(t):
        before = [other for other in view.turns_of(turn.agent) if other.t < turn.t]
        if before:
            changes.append(jaccard_distance(action_items(turn.action), action_items(before[-1].action)))
    return mean_or_none(changes)


def series(view: RunView) -> list[dict]:
    """One row per step: the actions' divergence and change, and what the step cost"""
    rows = []
    for t in view.steps():
        calls = [c for c in view.calls if c["t"] == t]
        rows.append({
            "t": t,
            "acting_agents": len(view.acting_at(t)),
            "action_divergence": action_divergence_at(view, t),
            "action_change": action_change_at(view, t),
            "calls": len(calls),
            "tokens": sum(c["completion"]["prompt_tokens"] + c["completion"]["generated_tokens"] for c in calls),
        })
    return rows


def action_metrics(rows: list[dict]) -> dict[str, float | None]:
    """The run's means of the per-step action metrics"""
    return {
        "action_divergence": mean_or_none([r["action_divergence"] for r in rows if r["action_divergence"] is not None]),
        "action_change": mean_or_none([r["action_change"] for r in rows if r["action_change"] is not None]),
    }
