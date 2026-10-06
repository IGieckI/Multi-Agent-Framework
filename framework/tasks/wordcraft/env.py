"""
WordCraft env (Agent's inventory and tried combinations)
"""
from dataclasses import dataclass, field

from framework.tasks.wordcraft.recipes import Combo, RecipeBook, combo
from framework.tasks.wordcraft.tasks import TaskSpec


@dataclass
class StepResult:
    """The result of a single step in the WordCraft environment"""
    combo: Combo
    # every item the combo makes, empty if invalid
    results: tuple[str, ...]
    
    # the results that were not in the inventory yet
    new_items: tuple[str, ...]
    
    # the combo had already been tried
    repeat: bool


@dataclass
class WordCraftEnv:
    book: RecipeBook
    task: TaskSpec
    inventory: list[str] = field(default_factory=list)
    
    # valid combos and their results
    valid: dict[Combo, tuple[str, ...]] = field(default_factory=dict)
    invalid: list[Combo] = field(default_factory=list)
    steps: int = 0

    def __post_init__(self) -> None:
        self.inventory = list(self.task.initial)

    @property
    def steps_left(self) -> int:
        return self.task.max_steps - self.steps

    @property
    def solved(self) -> bool:
        return self.task.target is not None and self.task.target in self.inventory

    @property
    def done(self) -> bool:
        return self.solved or self.steps_left <= 0

    def tried(self, a: str, b: str) -> bool:
        c = combo(a, b)
        return c in self.valid or c in self.invalid

    def step(self, a: str, b: str) -> StepResult:
        """Combine two inventory items. Valid combos add their new results + any combo uses one step"""
        if a not in self.inventory or b not in self.inventory:
            raise ValueError(f"not in the inventory: {a!r} and {b!r}")
        c = combo(a, b)
        repeat = self.tried(a, b)
        results = tuple(sorted(self.book.combine(a, b)))
        new_items = tuple(r for r in results if r not in self.inventory)
        self.inventory += new_items
        if results:
            self.valid[c] = results
        elif not repeat:
            self.invalid.append(c)
        self.steps += 1
        return StepResult(c, results, new_items, repeat)

    def no_op(self) -> None:
        """A step where the agent gave no usable combination: it uses a step and changes nothing"""
        self.steps += 1
