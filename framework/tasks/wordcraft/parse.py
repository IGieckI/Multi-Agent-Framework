"""
Read the chosen combination from an answer
"""
import re

from framework.agents.parse import ParseError

# the labelled parts of an answer
ANSWER_LABELS = ("combination", "reasoning", "message")


def clean(item: str) -> str:
    return item.strip().strip("'\"`*.").strip().lower()


def parse_combination(text: str, inventory: list[str]) -> tuple[str, str]:
    match = re.search(r"combination\W*:\W*?(['\"`]?\w.*)", text, re.IGNORECASE)
    if not match:
        raise ParseError("Your answer has no line starting with 'Combination:'. Answer in the required format.")
    # Split at each " and " in turn, because item names can contain it ('mac and cheese')
    pieces = match.group(1).split(" and ")
    splits = [(clean(" and ".join(pieces[:i])), clean(" and ".join(pieces[i:]))) for i in range(1, len(pieces))]
    for a, b in splits:
        if a in inventory and b in inventory:
            return a, b
    if not splits:
        raise ParseError("Write the combination as: Combination: 'first item' and 'second item'")
    missing = sorted({item for item in splits[0] if item not in inventory})
    names = ", ".join(f"'{item}'" for item in missing)
    raise ParseError(f"{names} {'is' if len(missing) == 1 else 'are'} not in your inventory. Use only inventory items.")

