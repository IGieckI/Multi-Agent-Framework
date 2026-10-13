"""
The message bus (paths from one agent to others)
"""

from dataclasses import asdict
from framework.comm.message import MESSAGE_TYPES, AgentId, Message, MessageDraft
from framework.comm.structure import Structure
from framework.log import EventLog


class MessageBus:
    def __init__(self, structure: Structure, log: EventLog) -> None:
        self.structure = structure
        self.log = log
        
        # receiver -> (due tick, message)
        self.queues: dict[AgentId, list[tuple[int, Message]]] = {}
        self.published = 0

    def publish(self, draft: MessageDraft, tick: int) -> Message | None:
        """Writes the message once and queues it for each receiver, an invalid draft is logged and returns None"""
        if draft.type not in MESSAGE_TYPES:
            error = f"unknown message type {draft.type!r}"
        elif set(draft.payload) != MESSAGE_TYPES[draft.type]:
            error = f"{draft.type} payload needs fields {sorted(MESSAGE_TYPES[draft.type])}"
        elif draft.sender not in self.structure.agents:
            error = f"unknown sender {draft.sender!r}"
        else:
            error = None
        if error is not None:
            self.log.write("bus", "msg_rejected", tick=tick, draft=asdict(draft), reason=error)
            return None

        message = Message(f"m{self.published}", tick, draft.sender, draft.type, dict(draft.payload), draft.parents,
                          dict(draft.meta))
        self.published += 1
        rule = self.structure.channel(draft.type)
        receivers = self.structure.receivers(draft)
        due = tick + 1 + rule.delay
        for receiver in receivers:
            self.queues.setdefault(receiver, []).append((due, message))
        self.log.write("bus", "msg_published", tick=tick, message=asdict(message), receivers=receivers, due=due,
                       form=rule.form)
        return message

    def deliver(self, agent: AgentId, tick: int) -> list[Message]:
        """The messages due for "agent" by "tick", in publication order"""
        queue = self.queues.get(agent, [])
        delivered = [message for due, message in queue if due <= tick]
        self.queues[agent] = [(due, message) for due, message in queue if due > tick]
        for message in delivered:
            self.log.write("bus", "msg_delivered", tick=tick, id=message.id, receiver=agent)
        return delivered
