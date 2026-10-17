"""
WordCraft metrics computed from a run's log
"""

from framework.tasks.wordcraft.recipes import RecipeBook


def inventories(meta: dict, events: list[dict]) -> dict[str, set[str]]:
    """Each agent's final inventory, rebuilt from the initial set and the items its logged steps added"""
    initial = set(meta["task"]["initial"])
    found = {agent: set(initial) for agent in meta["agents"]}
    for e in events:
        if e["kind"] == "env_step":
            found[e["data"]["agent"]] |= set(e["data"]["new_items"])
    return found


def evaluate(book: RecipeBook, meta: dict, events: list[dict]) -> dict[str, float | None]:
    """Coverage of the team (the union of the agents' new items) and per agent, normalised by everything reachable
    from the initial set; for targeted tasks, whether any or every agent crafted the target"""
    initial = set(meta["task"]["initial"])
    reachable = len(book.reachable(initial) - initial)
    final = inventories(meta, events)
    new = {agent: items - initial for agent, items in final.items()}
    union = set().union(*new.values())
    target = meta["task"]["target"]
    solved = [target in items for items in final.values()]
    return {
        "team_new_items": len(union),
        "team_coverage": len(union) / reachable if reachable else None,
        "agent_coverage_mean": sum(len(items) for items in new.values()) / len(new) / reachable if reachable else None,
        "solved_any": float(any(solved)) if target is not None else None,
        "solved_all": float(all(solved)) if target is not None else None,
    }
