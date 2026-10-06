"""
The Little Alchemy 2 recipe book (which pairs of items make which items)
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

Combo = tuple[str, str]


def combo(a: str, b: str) -> Combo:
    return (a, b) if a <= b else (b, a)


@dataclass
class RecipeBook:
    items: tuple[str, ...]
    
    # every valid combo and all the items it makes
    pairs: dict[Combo, frozenset[str]]
    
    # recipes skipped because their result is also one of their ingredients
    dropped: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "RecipeBook":
        """Read the WordCraft alchemy2.json file"""
        entities = json.loads(Path(path).read_text(encoding="utf-8"))["entities"]
        pairs = defaultdict(set)
        dropped = 0
        for result, item in entities.items():
            for a, b in item["recipes"]:
                if result in (a, b):
                    dropped += 1
                    continue
                pairs[combo(a, b)].add(result)
        return cls(tuple(sorted(entities)), {c: frozenset(r) for c, r in pairs.items()}, dropped)

    def combine(self, a: str, b: str) -> frozenset[str]:
        """The items that a and b make; empty if the combination is invalid"""
        return self.pairs.get(combo(a, b), frozenset())

    def recipes_for(self, item: str) -> list[Combo]:
        return sorted(c for c, results in self.pairs.items() if item in results)

    def reachable(self, initial: set[str]) -> set[str]:
        """Everything that can be crafted from "initial", including "initial" itself"""
        found = set(initial)
        while True:
            new = {r for (a, b), results in self.pairs.items() if a in found and b in found for r in results} - found
            if not new:
                return found
            found |= new

    def ingredient_counts(self) -> Counter:
        """How many valid combos each item takes part in, as an ingredient"""
        counts = Counter()
        for a, b in self.pairs:
            counts[a] += 1
            if b != a:
                counts[b] += 1
        return counts
