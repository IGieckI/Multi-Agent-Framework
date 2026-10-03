"""
Append JSONL event log, one line per event
"""

import json
from pathlib import Path

class EventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.id = 0

    def write(self, event_type: str, env_step: int | None = None, **data) -> None:
        event = {"id": self.id, "env_step": env_step, "event_type": event_type, "data": data}
        self.file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.file.flush()
        self.id += 1

    def close(self) -> None:
        self.file.close()


def read_events(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]
