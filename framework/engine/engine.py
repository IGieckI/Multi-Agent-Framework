"""
A minimal engine for multi-agent WordCraft play
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from framework.agents.backends.base import Sampling
from framework.agents.prompt import Prompt, render
from framework.comm.bus import MessageBus
from framework.comm.compiler import ProtocolConfig, compile_protocol
from framework.comm.message import AgentId, Message, MessageDraft, covers
from framework.engine.config import ROOT, EngineConfig, RunConfig
from framework.engine.usage import Usage
from framework.log import EventLog
from framework.tasks.wordcraft.env import StepResult, WordCraftEnv
from framework.tasks.wordcraft.parse import ParseError, output_parts, parse_combination
from framework.tasks.wordcraft.prompt import FIELDS, SOCIAL_BLOCKS, View, build_prompt, templates_hash
from framework.tasks.wordcraft.recipes import Combo, RecipeBook
from framework.tasks.wordcraft.tasks import TaskSpec, open_ended, sample_targeted

MAX_PARSE_ATTEMPTS = 5
MAX_ATTEMPTS = 6
AUTHORED = ("message", "task", "verdict")


@dataclass
class Attempt:
    """One LLM call of a turn"""
    attempt: int
    prompt: str
    prompt_tokens: int
    # references of the entries removed cause of context guard
    truncated: list[str]
    answer: str
    reasoning: str | None
    combination: tuple[str, str] | None
    message: str | None
    error: str | None
    completion_tokens: int
    elapsed_s: float


@dataclass
class Turn:
    """What one agent did in one step"""
    t: int
    tick: int
    agent: AgentId
    role: str
    # Messages delivered to the agent at the beginning of the turn
    inbox: list[Message] = field(default_factory=list)
    # List of the attempts
    attempts: list[Attempt] = field(default_factory=list)
    # Status can be:
    # "acted": produced a valid combination
    # "no_valid_answer": all attempts used up without a valid answer
    # "failed": the backend raised an exception
    status: str = "no_valid_answer"
    action: Combo | None = None
    message: str | None = None
    reasoning: str | None = None
    # This is the final result from the env
    result: StepResult | None = None
    # ids of the published messages
    published: list[str] = field(default_factory=list)


def call_seed(seed: int, replicate: int, agent: AgentId, step: int, attempt: int) -> int:
    """A reproducible sampling seed per call"""
    key = f"{seed}/{replicate}/{agent}/{step}/{attempt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") >> 1


def make_task(cfg: RunConfig, book: RecipeBook) -> TaskSpec:
    # If the task is open_ended it finishes with max_steps, otherwise through a random receipe reached
    if cfg.task.type == "open_ended":
        return open_ended(cfg.max_steps())
    stream = int(np.random.SeedSequence([cfg.run.seed, cfg.run.replicate]).generate_state(1)[0])
    return sample_targeted(book, random.Random(stream), cfg.task.depth, cfg.task.distractors, cfg.max_steps())


def make_backend(cfg: EngineConfig):
    # loads the model
    if cfg.backend == "mock":
        from framework.agents.backends.mock import MockBackend
        return MockBackend()
    if not cfg.model:
        raise ValueError(f"engine.backend {cfg.backend!r} needs engine.model")
    from framework.agents.backends.hf_local import HFLocalBackend
    return HFLocalBackend(cfg.model, cfg.revision)


class Engine:
    def __init__(self, cfg: RunConfig, backend, book: RecipeBook, out_dir: Path,
                 listener: Callable[[Turn], None] | None = None, data_sha256: str | None = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.book = book
        self.listener = listener
        protocol = cfg.resolved_protocol()
        # the communication graph
        self.structure, self.phases = compile_protocol(protocol)
        self.agents = sorted(self.structure.agents)
        # for each agent, which observation fields its role can see
        self.fields = {a: self.visible_fields(protocol, a) for a in self.agents}
        self.task = make_task(cfg, book)
        self.envs = {a: WordCraftEnv(book, self.task) for a in self.agents}
        # cumulative social memory
        self.memory: dict[AgentId, list[Message]] = {a: [] for a in self.agents}
        # which social prompt sections the agent gets (fixed for the whole run from the compiled structure)
        self.blocks = {a: {block for block, _, types in SOCIAL_BLOCKS if any(self.can_receive(a, t) for t in types)}
                       for a in self.agents}
        # whether the agent's prompt asks it to write a message: only if something it writes would reach a receiver
        self.slot = {a: any(self.structure.receivers(MessageDraft(a, t, {})) for t in AUTHORED) for a in self.agents}
        self.sampling = Sampling(**cfg.sampling.model_dump())
        self.usage = Usage()
        self.t = 0
        self.tick = 0
        self.ended_by: str | None = None
        self.turns: list[Turn] = []

        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.meta = {
            "run_id": out_dir.name,
            "config": cfg.model_dump(mode="json"),
            "protocol": protocol.model_dump(mode="json", by_alias=True),
            "agents": dict(self.structure.agents),
            "phases": self.phases,
            "visible_fields": {a: sorted(f) for a, f in self.fields.items()},
            "social_blocks": {a: sorted(b) for a, b in self.blocks.items()},
            "message_slot": self.slot,
            "task": asdict(self.task),
            "sampling": asdict(self.sampling),
            "engine": backend.engine_info(),
            "prompt_token_count": "backend" if hasattr(backend, "count_tokens") else "approx_chars_div_4",
            "prompts_hash": templates_hash(),
            "data_sha256": data_sha256,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        
        # Start logging
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        self.log = EventLog(out_dir / "events.jsonl", run_id=out_dir.name)
        self.bus = MessageBus(self.structure, self.log)
        self.log.write("engine", "run_started", t=0, tick=0, **self.meta)
        self.log.write("structure", "snapshot", t=0, tick=0, **self.structure.snapshot())

    def visible_fields(self, protocol: ProtocolConfig, agent: AgentId) -> frozenset[str]:
        # reads the role's visible_fields from the protocol
        fields = protocol.roles[self.structure.agents[agent]].visible_fields
        if fields is None:
            return frozenset(FIELDS)
        if unknown := set(fields) - set(FIELDS):
            raise ValueError(f"unknown visible_fields {sorted(unknown)}; WordCraft has {list(FIELDS)}")
        return frozenset(fields)

    def can_receive(self, agent: AgentId, msg_type: str) -> bool:
        # true if any other agent publishing that type would reach agent
        return any(agent in self.structure.receivers(MessageDraft(sender, msg_type, {}))
                   for sender in self.agents if sender != agent)

    def shares(self, agent: AgentId, msg_type: str) -> bool:
        #  whether the agent's role may publish that type
        return any(covers(t, msg_type) for t in self.structure.shares.get(self.structure.agents[agent], ()))

    def view(self, agent: AgentId) -> View:
        # builds the View object that the prompt builder consumes
        env = self.envs[agent]
        social = {block: [m for m in self.memory[agent] if m.type in types]
                  for block, _, types in SOCIAL_BLOCKS if block in self.blocks[agent]}
        return View(self.fields[agent], list(env.valid.items()), list(env.invalid), social, self.slot[agent])

    def count_tokens(self, prompt: Prompt) -> int:
        # uses the backend's tokenizer if it has one
        if hasattr(self.backend, "count_tokens"):
            return self.backend.count_tokens(prompt)
        return len(render(prompt)) // 4 + 1

    def guarded_prompt(self, agent: AgentId, error: str | None, t: int, tick: int) -> tuple[Prompt, int, list[str]]:
        """Build the prompt, count the tokens, and if it's over max_prompt_tokens, call drop_oldest(view) and rebuild"""
        view = self.view(agent)
        removed: list[str] = []
        limit = self.cfg.context.max_prompt_tokens
        while True:
            prompt = build_prompt(self.envs[agent], error, view)
            tokens = self.count_tokens(prompt)
            if tokens <= limit:
                break
            ref = drop_oldest(view)
            if ref is None:
                self.log.write("agent", "prompt_over_limit", t=t, tick=tick, agent=agent, tokens=tokens, limit=limit)
                break
            removed.append(ref)
        if removed:
            self.log.write("agent", "truncation_event", t=t, tick=tick, agent=agent, removed=removed, tokens=tokens)
        return prompt, tokens, removed

    async def run_agent(self, agent: AgentId, t: int, tick: int, slots: asyncio.Semaphore) -> Turn:
        env = self.envs[agent]
        turn = Turn(t, tick, agent, self.structure.agents[agent])
        
        # append every pending/valid message to agents
        turn.inbox = self.bus.deliver(agent, tick)
        self.memory[agent] += turn.inbox

        error = None
        parse_failures = 0
        for attempt in range(MAX_ATTEMPTS):
            # Build the guarded prompt
            prompt, tokens, removed = self.guarded_prompt(agent, error, t, tick)

            # set the new seed and generate
            sampling = replace(self.sampling, seed=call_seed(self.cfg.run.seed, self.cfg.run.replicate, agent, t, attempt))
            try:
                async with slots:
                    completion = await self.backend.generate(prompt, sampling)
            except Exception as e:
                turn.status = "failed"
                self.log.write("agent", "llm_failed", t=t, tick=tick, agent=agent, attempt=attempt, error=repr(e))
                break
            self.usage.charge(completion.prompt_tokens, completion.generated_tokens)

            # Split the output in different parts
            parts = output_parts(completion.text)
            reasoning = completion.reasoning or parts.get("reasoning") or None
            message = (parts.get("message") or None) if self.slot[agent] else None
            pair, error = None, None
            try:
                pair = parse_combination(completion.text, env.inventory)
                if not self.cfg.task.repeat_consumes_step and env.tried(*pair):
                    error = f"You already tried '{pair[0]}' and '{pair[1]}'. Choose a combination you have not tried."
            except ParseError as e:
                error = str(e)
                parse_failures += 1
                if completion.finish_reason == "length":
                    error += " Your answer was cut off: keep it shorter."

            turn.attempts.append(Attempt(attempt, render(prompt), tokens, removed, completion.text, reasoning,
                                         pair, message, error, completion.generated_tokens, completion.elapsed_s))
            self.log.write(
                "agent", "llm_call", t=t, tick=tick, agent=agent, attempt=attempt,
                prompt=render(prompt),
                segments=[{"source": s.source, "ref": s.ref, "chars": len(s.text)} for s in prompt],
                counted_prompt_tokens=tokens, truncated=removed,
                inbox=[m.id for m in turn.inbox] if attempt == 0 else [],
                sampling=asdict(sampling), completion=asdict(completion),
                parts=parts, combination=pair, error=error,
                usage=self.usage.report(),
            )
            if error is None:
                turn.status, turn.action, turn.message, turn.reasoning = "acted", pair, message, reasoning
                break
            if parse_failures >= MAX_PARSE_ATTEMPTS:
                break
        return turn

    def apply(self, turn: Turn) -> None:
        """Steps the agent's environment and publishes its outcome and shared parts"""
        agent, env = turn.agent, self.envs[turn.agent]
        
        # if the backend failed just exit
        if turn.status == "failed":
            return
        if turn.action is None:
            env.no_op()
            self.log.write("task", "no_op", t=turn.t, tick=turn.tick, agent=agent)
            return
        
        # step on the env + logs
        result = env.step(*turn.action)
        turn.result = result
        self.log.write("task", "env_step", t=turn.t, tick=turn.tick, agent=agent, **asdict(result),
                       inventory_size=len(env.inventory))
        combo = list(result.combo)
        drafts = [MessageDraft(agent, "result.success", {"combo": combo, "result": list(result.results)}, meta={"auto": True})
                  if result.results else MessageDraft(agent, "result.failure", {"combo": combo}, meta={"auto": True})]
        if turn.message:
            drafts += [MessageDraft(agent, t, {"text": turn.message}) for t in AUTHORED if self.shares(agent, t)]
        if turn.reasoning and self.shares(agent, "reasoning"):
            drafts.append(MessageDraft(agent, "reasoning", {"text": turn.reasoning}))
        for draft in drafts:
            if (message := self.bus.publish(draft, turn.tick)) is not None:
                turn.published.append(message.id)

    @property
    def done(self) -> bool:
        return self.ended_by is not None

    async def step(self) -> list[Turn]:
        if self.done:
            return []
        t, turns = self.t, []
        slots = asyncio.Semaphore(self.cfg.engine.max_concurrency)
        try:
            for phase in self.phases:
                tick = self.tick
                active = [a for a in phase if not self.envs[a].done]
                outputs = await asyncio.gather(*(self.run_agent(a, t, tick, slots) for a in active))
                for turn in sorted(outputs, key=lambda turn: turn.agent):
                    self.apply(turn)
                    self.turns.append(turn)
                    turns.append(turn)
                    if self.listener:
                        self.listener(turn)
                self.tick += 1
            self.log.write("structure", "snapshot", t=t, tick=self.tick - 1, **self.structure.snapshot())
            self.t += 1
        except Exception as e:
            self.log.write("engine", "run_failed", t=t, tick=self.tick, error=repr(e))
            self.end("error")
            raise
        statuses = {turn.status for turn in turns}
        if "failed" in statuses:
            self.end("backend_error")
        elif all(env.done for env in self.envs.values()):
            self.end("all_solved" if all(env.solved for env in self.envs.values()) else "max_steps")
        return turns

    async def run(self) -> dict:
        while not self.done:
            await self.step()
        return self.summary()

    def end(self, reason: str) -> None:
        """Ends the run once: writes the summary, the run_ended event and closes the log"""
        if self.done:
            return
        self.ended_by = reason
        summary = self.summary()
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log.write("engine", "run_ended", t=self.t, tick=self.tick, **summary)
        self.log.close()

    def summary(self) -> dict:
        """Final answers: each agent's result and the team's"""
        initial = set(self.task.initial)
        reachable = self.book.reachable(initial) - initial
        agents, union = {}, set()
        for a, env in self.envs.items():
            found = set(env.inventory) - initial
            union |= found
            agents[a] = {
                "role": self.structure.agents[a],
                "solved": env.solved if self.task.target else None,
                "steps": env.steps,
                "new_items": sorted(found),
                "coverage": round(len(found) / len(reachable), 4) if reachable else None,
                "invalid": len(env.invalid),
                "no_valid_answers": sum(turn.agent == a and turn.status == "no_valid_answer" for turn in self.turns),
            }
        solved = [env.solved for env in self.envs.values()]
        return {
            "ended_by": self.ended_by,
            "steps": self.t,
            "task": asdict(self.task),
            "team": {
                "any_solved": any(solved) if self.task.target else None,
                "all_solved": all(solved) if self.task.target else None,
                "union_new_items": len(union),
                "union_coverage": round(len(union) / len(reachable), 4) if reachable else None,
            },
            "agents": agents,
            "usage": self.usage.report(),
        }

    def state(self) -> dict:
        """A plain copy of the current state, safe to hand to another thread"""
        return {
            "t": self.t, "tick": self.tick, "ended_by": self.ended_by, "usage": self.usage.report(),
            "agents": {a: {"role": self.structure.agents[a], "inventory": list(env.inventory),
                           "solved": env.solved, "steps_left": env.steps_left, "done": env.done}
                       for a, env in self.envs.items()},
        }


def drop_oldest(view: View) -> str | None:
    """Removes the oldest history entry removable"""
    social = [(int(m.id[1:]), block, m) for block, messages in view.social.items() for m in messages]
    if social:
        _, block, message = min(social, key=lambda entry: entry[0])
        view.social[block].remove(message)
        return message.id
    if view.own_invalid:
        a, b = view.own_invalid.pop(0)
        return f"own_invalid:{a}+{b}"
    if view.own_valid:
        (a, b), _ = view.own_valid.pop(0)
        return f"own_valid:{a}+{b}"
    return None


def default_out_dir(cfg: RunConfig) -> Path:
    return ROOT / "runs" / cfg.run.name / f"seed{cfg.run.seed}_rep{cfg.run.replicate}"


def create_engine(cfg: RunConfig, out_dir: Path | None = None, listener: Callable[[Turn], None] | None = None,
                  backend=None) -> Engine:
    """Loads the recipe data and the backend named in the config, and builds the engine"""
    data = ROOT / cfg.task.data
    if not data.exists():
        raise FileNotFoundError(f"{data} is missing: run `python scripts/fetch_data.py` first")
    book = RecipeBook.load(data)
    backend = backend or make_backend(cfg.engine)
    return Engine(cfg, backend, book, out_dir or default_out_dir(cfg), listener,
                  hashlib.sha256(data.read_bytes()).hexdigest())
