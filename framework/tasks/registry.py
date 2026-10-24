"""
The tasks a run config can name in `task.name`, with their config class and builder
"""

from typing import Any, Callable, NamedTuple

from framework.comm.message import CORE_MESSAGE_TYPES, MessageTypes
from framework.tasks.base import Task, TaskConfig
from framework.tasks.wordcraft.config import WordCraftConfig
from framework.tasks.wordcraft.task import WordCraftTask


class Registered(NamedTuple):
    config: type[TaskConfig]
    build: Callable[[TaskConfig], Task]
    # the task's own message types, added to the core ones (known without building the task, e.g. to compile a
    # protocol for a run config)
    message_types: MessageTypes = {}


TASKS: dict[str, Registered] = {
    "wordcraft": Registered(WordCraftConfig, WordCraftTask.from_config, WordCraftTask.message_types),
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


def message_types(name: str) -> MessageTypes:
    """The message types of a run of the task: the core ones and the task's own"""
    own = registered(name).message_types
    if clash := set(own) & set(CORE_MESSAGE_TYPES):
        raise ValueError(f"task {name!r} redefines core message types {sorted(clash)}")
    return {**CORE_MESSAGE_TYPES, **own}
