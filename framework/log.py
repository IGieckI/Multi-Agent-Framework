"""
Append-only JSONL event log:
{run_id, seq, t, tick, component, kind, data}
"""

import json
from pathlib import Path

class EventLog:
    def __init__(self, path: Path, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.run_id = run_id
        self.seq = 0

    def write(self, component: str, kind: str, *, t: int | None = None, tick: int | None = None, **data) -> None:
        """t is the step and tick the phase clock"""
        event = {"run_id": self.run_id, "seq": self.seq, "t": t, "tick": tick,
                 "component": component, "kind": kind, "data": data}
        self.file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.file.flush()
        self.seq += 1

    def close(self) -> None:
        self.file.close()


def read_events(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]
