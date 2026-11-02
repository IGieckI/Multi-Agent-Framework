"""
WordCraft metrics computed from a run's log
"""

from itertools import combinations

from framework.metrics.generic import mean_or_none
from framework.metrics.log_view import RunView, build_view, jaccard_distance
from framework.tasks.wordcraft.recipes import RecipeBook


def inventories(meta: dict, events: list[dict]) -> dict[str, set[str]]:
    """Each agent's final inventory, rebuilt from the initial set and the items its logged steps added"""
    initial = set(meta["task"]["initial"])
    found = {agent: set(initial) for agent in meta["agents"]}
    for e in events:
        if e["kind"] == "env_step":
            found[e["data"]["agent"]] |= set(e["data"]["new_items"])
    return found


def inventories_by_step(view: RunView) -> dict[int, dict[str, set[str]]]:
    """Each agent's inventory at the end of every step in which anyone acted"""
    initial = set(view.meta["task"]["initial"])
    current = {agent: set(initial) for agent in view.agents}
    by_step = {}
    for t in view.steps():
        for step in (s for s in view.env_steps if s["t"] == t):
            current[step["agent"]] |= set(step["new_items"])
        by_step[t] = {agent: set(items) for agent, items in current.items()}
    return by_step


def pairwise_divergence(inventory: dict[str, set[str]]) -> float | None:
    """Mean Jaccard distance between the agents' inventories; None for fewer than two agents"""
    return mean_or_none([jaccard_distance(frozenset(a), frozenset(b)) for a, b in combinations(inventory.values(), 2)])


def first_found(view: RunView) -> dict[str, dict[str, int]]:
    """agent -> item -> the step at which the item first entered its inventory (starting items are left out)"""
    found: dict[str, dict[str, int]] = {agent: {} for agent in view.agents}
    for step in view.env_steps:
        for item in step["new_items"]:
            found[step["agent"]].setdefault(item, step["t"])
    return found


def copy_delays(view: RunView) -> list[int]:
    """For each item an agent found after an agent that had sent it a message did: the steps between the two. An
    agent that never received a message has no neighbours, so its items are never counted"""
    found = first_found(view)
    delays = []
    for agent, items in found.items():
        neighbours = {view.messages[m]["sender"] for m in view.delivered.get(agent, [])} - {agent}
        for item, t in items.items():
            earlier = [found[n][item] for n in neighbours if item in found[n] and found[n][item] < t]
            if earlier:
                delays.append(t - min(earlier))
    return delays


def mentions(text: str, items: frozenset[str]) -> bool:
    """Whether the text names every item (case-insensitive)"""
    lowered = text.lower()
    return all(item.lower() in lowered for item in items)


def adoption_metrics(view: RunView) -> dict[str, float | None]:
    """How often an agent plays what it was just told. With a message in its inbox, an action is
    - adopted from a result if the combination is the one of a delivered outcome message,
    - adopted from a message if a delivered text names both of its items,
    - adopted blindly if it repeats a combination a delivered outcome reported as a failure (it cannot work).
    The last one is a sycophancy proxy: following a peer against the evidence"""
    heard = adopted = from_result = from_text = blind = 0
    for turn in view.acting:
        if not turn.inbox:
            continue
        action = frozenset(turn.action)
        messages = [view.messages[m] for m in turn.inbox]
        outcomes = [(m["type"], frozenset(m["payload"]["combo"])) for m in messages if "combo" in m["payload"]]
        texts = [m["payload"]["text"] for m in messages if "text" in m["payload"]]
        by_result = any(combo == action for _, combo in outcomes)
        by_text = any(mentions(text, action) for text in texts)
        failed = any(kind == "result.failure" and combo == action for kind, combo in outcomes)
        heard += 1
        adopted += by_result or by_text
        from_result += by_result
        from_text += by_text
        blind += failed
    if not heard:
        return {"adoption_rate": None, "adoption_rate_result": None, "adoption_rate_message": None,
                "blind_adoption_rate": None}
    return {"adoption_rate": adopted / heard, "adoption_rate_result": from_result / heard,
            "adoption_rate_message": from_text / heard, "blind_adoption_rate": blind / heard}


def notes_metrics(meta: dict, steps: list[dict]) -> dict[str, float | None]:
    """With hints: how often an agent tries a recipe of the path that is in its own notes, and one that is only in
    another agent's notes (it can only know it from the others, or guess it)"""
    notes = meta.get("notes") or {}
    path = [frozenset((a, b)) for a, b, _ in meta["task"].get("path", ())]
    if not any(notes.values()) or not path:
        return {"own_note_use": None, "others_note_use": None}
    tried = {agent: set() for agent in notes}
    for step in steps:
        tried[step["agent"]].add(frozenset(step["combo"]))
    own, others = [], []
    for agent, held in notes.items():
        held = {frozenset((a, b)) for a, b, _ in held}
        for recipe in path:
            (own if recipe in held else others).append(float(recipe in tried[agent]))
    return {"own_note_use": mean_or_none(own), "others_note_use": mean_or_none(others)}


def evaluate(book: RecipeBook, meta: dict, events: list[dict]) -> dict[str, float | None]:
    """Coverage of the team (the union of the agents' new items) and per agent, normalised by everything reachable
    from the initial set; for targeted tasks, whether any or every agent crafted the target and how; how the agents'
    attempts overlap and diverge; and how much they copy each other"""
    initial = set(meta["task"]["initial"])
    reachable = len(book.reachable(initial) - initial)
    final = inventories(meta, events)
    new = {agent: items - initial for agent, items in final.items()}
    union = set().union(*new.values())
    target = meta["task"]["target"]
    solved = [target in items for items in final.values()]
    view = build_view(meta, events)
    steps = view.env_steps
    n_steps = len(view.steps())
    divergence = [pairwise_divergence(inv) for inv in inventories_by_step(view).values()]
    divergence = [d for d in divergence if d is not None]
    distractors = frozenset(meta["task"].get("distractors", ()))
    tried: dict[frozenset[str], set[str]] = {}
    for step in steps:
        tried.setdefault(frozenset(step["combo"]), set()).add(step["agent"])
    solves = [s for s in steps if target is not None and target in s["new_items"]]
    paths = {frozenset(s["combo"]) for s in steps if target is not None and target in s["results"]}
    delays = copy_delays(view)
    return {
        "team_new_items": len(union),
        "team_coverage": len(union) / reachable if reachable else None,
        "agent_coverage_mean": sum(len(items) for items in new.values()) / len(new) / reachable if reachable else None,
        "solved_any": float(any(solved)) if target is not None else None,
        "solved_all": float(all(solved)) if target is not None else None,
        "steps_to_target": float(min(s["t"] for s in solves) + 1) if solves else None,
        "steps_to_target_mean": mean_or_none([s["t"] + 1 for s in solves]),
        "distinct_paths": float(len(paths)) if target is not None else None,
        "repeat_rate": mean_or_none([float(s["repeat"]) for s in steps]),
        "invalid_combo_rate": mean_or_none([float(not s["results"] and not s["repeat"]) for s in steps]),
        "distractor_use_rate": (mean_or_none([float(bool(distractors & set(s["combo"]))) for s in steps])
                                if distractors else None),
        "redundancy": (sum(len(a) > 1 for a in tried.values()) / len(tried)) if tried else None,
        "frontier_velocity": len(union) / n_steps if n_steps else None,
        "pairwise_divergence": mean_or_none(divergence),
        "pairwise_divergence_final": divergence[-1] if divergence else None,
        "copy_time": mean_or_none(delays),
        "copy_events": float(len(delays)),
        **adoption_metrics(view),
        **notes_metrics(meta, steps),
    }


def series(meta: dict, events: list[dict]) -> list[dict]:
    """One row per step: the team's inventory, what was new in it, and how far apart the agents' inventories are"""
    view = build_view(meta, events)
    initial = set(meta["task"]["initial"])
    team, rows = set(), []
    for t, inventory in inventories_by_step(view).items():
        now = set().union(*inventory.values()) - initial
        rows.append({"t": t, "team_items": len(now), "team_new_items": len(now - team),
                     "pairwise_divergence": pairwise_divergence(inventory)})
        team = now
    return rows
