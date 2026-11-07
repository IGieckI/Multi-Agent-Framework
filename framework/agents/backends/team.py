"""
A mixed team: one backend per model, each agent served by its own (engine.agent_models). The engine asks the team
for each agent's backend; the team itself never generates
"""


class TeamBackend:
    def __init__(self, members: list) -> None:
        # one backend per entry of engine.agent_models, in its order; entries naming the same model share one backend
        self.members = members

    def for_agent(self, index: int):
        """The backend of the index-th agent (sorted agent ids), cycling through the members"""
        return self.members[index % len(self.members)]

    def distinct(self) -> list:
        return list({id(m): m for m in self.members}.values())

    def engine_info(self) -> dict:
        infos = [m.engine_info() for m in self.distinct()]
        windows = [i.get("context_window") for i in infos]
        return {
            "backend": "team",
            "model_id": " + ".join(str(i.get("model_id")) for i in infos),
            "members": infos,
            "context_window": None if None in windows else min(windows),
        }
