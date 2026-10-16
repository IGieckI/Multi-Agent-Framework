"""
The tasks a run config can name in `task.name`, with their config class and builder
"""

from typing import Any, Callable, NamedTuple

from framework.tasks.base import Task, TaskConfig
from framework.tasks.wordcraft.config import WordCraftConfig
from framework.tasks.wordcraft.task import WordCraftTask


class Registered(NamedTuple):
    config: type[TaskConfig]
    build: Callable[[TaskConfig], Task]


TASKS: dict[str, Registered] = {
    "wordcraft": Registered(WordCraftConfig, WordCraftTask.from_config),
}


def registered(name: str) -> Registered:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; known tasks: {sorted(TASKS)}")
    return TASKS[name]


def parse_task_config(data: Any) -> TaskConfig:
    """Validates a `task` section with the config class of the task it names"""
    if isinstance(data, TaskConfig):
        return data
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"the task section needs a `name`; known tasks: {sorted(TASKS)}")
    return registered(data["name"]).config.model_validate(data)


def build_task(cfg: TaskConfig) -> Task:
    return registered(cfg.name).build(cfg)
