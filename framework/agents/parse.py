"""
Splitting an answer into its labelled parts, and the error a task's parser raises
"""

import re
from typing import Sequence


class ParseError(ValueError):
    """The answer has no usable action. Its text is shown to the model on the retry"""


def output_parts(text: str, labels: Sequence[str]) -> dict[str, str]:
    """The text after each label (e.g. 'Reasoning:'), up to the next label, on one line. Labels are lowercase"""
    label = re.compile(r"^\W*(" + "|".join(re.escape(name) for name in labels) + r")\W*:",
                       re.IGNORECASE | re.MULTILINE)
    marks = list(label.finditer(text))
    parts: dict[str, str] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        parts.setdefault(mark.group(1).lower(), " ".join(text[mark.end():end].split()))
    return parts
