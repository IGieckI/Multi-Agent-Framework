"""
The two WordCraft task types (SPEC §3.2): targeted and open-ended
"""

import random
from dataclasses import dataclass

from framework.tasks.wordcraft.recipes import RecipeBook


@dataclass(frozen=True)
class TaskSpec:
    initial: tuple[str, ...]
    max_steps: int
    target: str | None = None
    intermediate: tuple[str, ...] = ()
    # starting items that are not ingredients of the target's recipe (logged, for the distractor metrics)
    distractors: tuple[str, ...] = ()


def open_ended(max_steps: int = 200) -> TaskSpec:
    return TaskSpec(initial=("air", "earth", "fire", "water"), max_steps=max_steps)


def sample_targeted(
    book: RecipeBook, rng: random.Random, depth: int = 1, distractors: int = 3, max_steps: int = 6
) -> TaskSpec:
    """Generate a random targeted task(craft one given item, starting from a small inventory): 
    pick a target item, add the deepest starting items, and add distractors
    """
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    craftable = sorted({r for results in book.pairs.values() for r in results})
    while True:
        target = rng.choice(craftable)
        base = list(rng.choice(book.recipes_for(target)))
        intermediate = []
        if depth == 2:
            expandable = [item for item in base if book.recipes_for(item)]
            if not expandable:
                continue
            item = rng.choice(expandable)
            recipe = rng.choice(book.recipes_for(item))
            if target in recipe or item in recipe:
                continue
            base.remove(item)
            base += recipe
            intermediate = [item]
        break

    excluded = {target, *base, *intermediate}
    counts = book.ingredient_counts()
    options = [item for item in book.items if item not in excluded]
    weights = [counts[item] + 1 for item in options]
    chosen: set[str] = set()
    while len(chosen) < distractors:
        chosen.add(rng.choices(options, weights)[0])
    return TaskSpec(
        initial=tuple(sorted(set(base) | chosen)),
        max_steps=max_steps,
        target=target,
        intermediate=tuple(intermediate),
        distractors=tuple(sorted(chosen)),
    )
