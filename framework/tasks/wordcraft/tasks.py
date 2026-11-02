"""
The two WordCraft task types (SPEC §3.2): targeted and open-ended
"""

import random
from dataclasses import dataclass

from framework.tasks.wordcraft.recipes import RecipeBook, combo


@dataclass(frozen=True)
class TaskSpec:
    initial: tuple[str, ...]
    max_steps: int
    target: str | None = None
    intermediate: tuple[str, ...] = ()
    # starting items that are not ingredients of the target's recipe (logged, for the distractor metrics)
    distractors: tuple[str, ...] = ()
    # the recipes that lead to the target, in order: (first item, second item, result)
    path: tuple[tuple[str, str, str], ...] = ()


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
        intermediate, path = [], [(*base, target)]
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
            path = [(*recipe, item), *path]
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
        path=tuple(path),
    )


Note = tuple[str, str, str]


def deal_notes(book: RecipeBook, spec: TaskSpec, agents: int, mode: str, notes: int, rng: random.Random) -> list[list[Note]]:
    """Each agent's private notes, all true recipes (first item, second item, result). split: the path's recipes go to
    different agents, in random order (all of them to a single agent); all: every agent gets the whole path. Each agent's
    notes are filled up to `notes` with recipes of the starting items that do not lead to the target, different for each
    agent as far as there are enough"""
    if mode == "none" or not spec.path:
        return [[] for _ in range(agents)]
    if mode == "all":
        dealt = [list(spec.path) for _ in range(agents)]
    else:
        dealt = [[] for _ in range(agents)]
        order = rng.sample(range(agents), agents)
        for k, note in enumerate(spec.path):
            dealt[order[k % agents]].append(note)
    on_path = {spec.target, *spec.intermediate, *spec.initial}
    path_combos = {combo(a, b) for a, b, _ in spec.path}
    items = spec.initial
    decoys = [(a, b, r) for i, a in enumerate(items) for b in items[i:] if combo(a, b) not in path_combos
              for r in sorted(book.combine(a, b)) if r not in on_path]
    # dealt in turn from a shuffled deck, so agents share a decoy only when there are too few
    rng.shuffle(decoys)
    deck = iter(decoys * agents)
    for own in dealt:
        for _ in range(min(notes - len(own), len(decoys))):
            own.append(next(note for note in deck if note not in own))
        rng.shuffle(own)
    return dealt
