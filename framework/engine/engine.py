"""
The engine: runs the agents of a compiled protocol on a task, step by step (SPEC §4.2).

It reaches the task only through the Task interface (framework/tasks/base.py), so nothing here depends on a
particular game or environment.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from framework.agents import persona_host
from framework.agents.backends.base import Sampling
from framework.agents.parse import ParseError, output_parts
from framework.agents.persona_host import PersonaHost
from framework.agents.prompt import Prompt, View, render
from framework.comm.bus import MessageBus
from framework.comm.compiler import ProtocolConfig, compile_protocol
from framework.comm.message import AgentId, Message, MessageDraft, covers
from framework.engine.config import ROOT, EngineConfig, RunConfig
from framework.engine.budget import BudgetEnforcer, reference_usage
from framework.engine.usage import Usage
from framework.log import EventLog
from framework.tasks.base import Task
from framework.tasks.registry import build_task

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
    # the action the task read from the answer, None if there is none
    action: Any
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
    # "acted": produced a valid action
    # "no_valid_answer": all attempts used up without a valid answer
    # "failed": the backend raised an exception
    # "budget": a call was refused because it could exceed a budget cap; the run ends after this step
    status: str = "no_valid_answer"
    action: Any = None
    message: str | None = None
    reasoning: str | None = None
    # This is the final result from the task's step
    result: Any = None
    # ids of the published messages
    published: list[str] = field(default_factory=list)


def call_seed(seed: int, replicate: int, agent: AgentId, step: int, attempt: int) -> int:
    """A reproducible sampling seed per call"""
    key = f"{seed}/{replicate}/{agent}/{step}/{attempt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") >> 1


def task_seed(cfg: RunConfig) -> int:
    """The task's own random stream for this seed and replicate"""
    return int(np.random.SeedSequence([cfg.run.seed, cfg.run.replicate]).generate_state(1)[0])


def make_backend(cfg: EngineConfig, task: Task):
    # loads the model; the mock answers at random in the task's format
    if cfg.backend == "mock":
        from framework.agents.backends.mock import MockBackend
        return MockBackend(task.mock_answer)
    if not cfg.model:
        raise ValueError(f"engine.backend {cfg.backend!r} needs engine.model")
    from framework.agents.backends.hf_local import HFLocalBackend
    return HFLocalBackend(cfg.model)


class Engine:
    def __init__(self, cfg: RunConfig, task: Task, backend, out_dir: Path,
                 listener: Callable[[Turn], None] | None = None,
                 stream: Callable[[AgentId, int, int, str | None], None] | None = None) -> None:
        self.cfg = cfg
        self.task = task
        self.backend = backend
        self.listener = listener
        # Display only (never logged, never read back by the run): stream(agent, step, attempt, text) with the raw
        # text generated so far, "" when a call starts and None when it ends
        self.stream = stream
        protocol = cfg.resolved_protocol()
        # the communication graph
        self.structure, self.phases = compile_protocol(protocol)
        self.agents = sorted(self.structure.agents)
        # for each agent, which observation fields its role can see
        self.fields = {a: self.visible_fields(protocol, a) for a in self.agents}
        # one environment instance per agent, inside the task
        self.task.reset(task_seed(cfg), self.agents)
        # cumulative social memory
        self.memory: dict[AgentId, list[Message]] = {a: [] for a in self.agents}
        # which social prompt sections the agent gets (fixed for the whole run from the compiled structure)
        self.blocks = {a: {block for block, _, types in task.social_blocks if any(self.can_receive(a, t) for t in types)}
                       for a in self.agents}
        # whether the agent's prompt asks it to write a message: only if something it writes would reach a receiver
        self.slot = {a: any(self.structure.receivers(MessageDraft(a, t, {})) for t in AUTHORED) for a in self.agents}
        # role -> (file, text) of its persona, rendered after the rules for every agent of the role
        self.personas = self.load_personas(cfg)
        self.sampling = Sampling(**cfg.sampling.model_dump())
        self.usage = Usage()
        caps, budget_info = resolve_budget(cfg)
        self.budget = BudgetEnforcer(caps, self.usage, cfg.budget.step_reserve)
        # the cap that refused a call during the current step
        self.budget_stop: str | None = None
        # baseline B5: every agent's turns accumulate in one shared conversation
        self.host = PersonaHost(self.host_limit(backend)) if cfg.engine.persona_host else None
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
            "personas": {role: {"file": file, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                         for role, (file, text) in sorted(self.personas.items())},
            **task.metadata(),
            "sampling": asdict(self.sampling),
            "engine": backend.engine_info(),
            "prompt_token_count": "backend" if hasattr(backend, "count_tokens") else "approx_chars_div_4",
            "budget": budget_info,
            # B5 is the only run whose agents share a context (CLAUDE.md rule 1)
            "shared_context": self.host is not None,
            "persona_host": None if self.host is None else {"context_limit": self.host.limit,
                                                            "templates_sha256": persona_host.templates_hash()},
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
            return frozenset(self.task.fields)
        if unknown := set(fields) - set(self.task.fields):
            raise ValueError(f"unknown visible_fields {sorted(unknown)}; the task has {list(self.task.fields)}")
        return frozenset(fields)

    def can_receive(self, agent: AgentId, msg_type: str) -> bool:
        # true if any other agent publishing that type would reach agent
        return any(agent in self.structure.receivers(MessageDraft(sender, msg_type, {}))
                   for sender in self.agents if sender != agent)

    def shares(self, agent: AgentId, msg_type: str) -> bool:
        #  whether the agent's role may publish that type
        return any(covers(t, msg_type) for t in self.structure.shares.get(self.structure.agents[agent], ()))

    def view(self, agent: AgentId) -> View:
        # builds the View object that the prompt builder consumes: the observation filtered by the agent's role
        observation = {name: value for name, value in self.task.observe(agent).items() if name in self.fields[agent]}
        social = {block: [m for m in self.memory[agent] if m.type in types]
                  for block, _, types in self.task.social_blocks if block in self.blocks[agent]}
        return View(observation, social, self.slot[agent], self.personas.get(self.structure.agents[agent]))

    def load_personas(self, cfg: RunConfig) -> dict[str, tuple[str, str]]:
        """Reads the persona file of each role named in the run config"""
        roles = set(self.structure.agents.values())
        if unknown := set(cfg.personas) - roles:
            raise ValueError(f"personas name roles the protocol does not have: {sorted(unknown)}; "
                             f"its roles are {sorted(roles)}")
        return {role: (file, (ROOT / file).read_text(encoding="utf-8")) for role, file in cfg.personas.items()}

    def count_tokens(self, prompt: Prompt) -> int:
        # uses the backend's tokenizer if it has one, with the chat template the call will use
        if hasattr(self.backend, "count_tokens"):
            return self.backend.count_tokens(prompt, self.sampling.reasoning)
        return len(render(prompt)) // 4 + 1

    def host_limit(self, backend) -> int:
        """The shared conversation may hold as much as all agents' prompts together, within the model's window"""
        limit = len(self.agents) * self.cfg.context.max_prompt_tokens
        window = backend.engine_info().get("context_window")
        return min(limit, window - self.cfg.sampling.max_tokens) if window else limit

    def hosted_prompt(self, agent: AgentId, prompt: Prompt, t: int, tick: int) -> tuple[Prompt, int, list[str]]:
        """The agent's prompt inside the shared conversation (B5), the oldest turns dropped to fit the host's limit"""
        prompt, tokens, removed = self.host.compose(agent, prompt, self.count_tokens)
        if removed:
            self.log.write("agent", "truncation_event", t=t, tick=tick, agent=agent, removed=removed, tokens=tokens,
                           shared_context=True)
        if tokens > self.host.limit:
            self.log.write("agent", "prompt_over_limit", t=t, tick=tick, agent=agent, tokens=tokens,
                           limit=self.host.limit, shared_context=True)
        return prompt, tokens, removed

    def guarded_prompt(self, agent: AgentId, error: str | None, t: int, tick: int) -> tuple[Prompt, int, list[str]]:
        """Build the prompt, count the tokens, and if it's over max_prompt_tokens, call drop_oldest(view) and rebuild"""
        view = self.view(agent)
        removed: list[str] = []
        limit = self.cfg.context.max_prompt_tokens
        while True:
            prompt = self.task.build_prompt(view, error)
            tokens = self.count_tokens(prompt)
            if tokens <= limit:
                break
            ref = drop_oldest(view, self.task)
            if ref is None:
                self.log.write("agent", "prompt_over_limit", t=t, tick=tick, agent=agent, tokens=tokens, limit=limit)
                break
            removed.append(ref)
        if removed:
            self.log.write("agent", "truncation_event", t=t, tick=tick, agent=agent, removed=removed, tokens=tokens)
        return prompt, tokens, removed

    async def run_agent(self, agent: AgentId, t: int, tick: int, slots: asyncio.Semaphore) -> Turn:
        turn =Turn(t, tick, agent, self.structure.agents[agent])
        
        # append every pending/valid message to agents
        turn.inbox = self.bus.deliver(agent, tick)
        self.memory[agent] += turn.inbox

        error = None
        parse_failures = 0
        for attempt in range(self.cfg.retries.max_attempts):
            # Build the guarded prompt
            prompt, tokens, removed = self.guarded_prompt(agent, error, t, tick)
            if self.host is not None:
                prompt, tokens, dropped = self.hosted_prompt(agent, prompt, t, tick)
                removed = removed + dropped

            # set the new seed and generate
            sampling = replace(self.sampling, seed=call_seed(self.cfg.run.seed, self.cfg.run.replicate, agent, t, attempt))
            if (cap := self.budget.reserve(tokens, sampling.max_tokens)) is not None:
                turn.status = "budget"
                self.budget_stop = cap
                self.log.write("engine", "call_refused", t=t, tick=tick, agent=agent, attempt=attempt, cap=cap,
                               counted_prompt_tokens=tokens, max_tokens=sampling.max_tokens,
                               usage=self.usage.report())
                break
            try:
                async with slots:
                    if self.stream:
                        self.stream(agent, t, attempt, "")
                        completion = await self.backend.generate(
                            prompt, sampling, on_text=lambda text: self.stream(agent, t, attempt, text))
                    else:
                        completion = await self.backend.generate(prompt, sampling)
            except Exception as e:
                self.budget.release(tokens, sampling.max_tokens)
                turn.status = "failed"
                self.log.write("agent", "llm_failed", t=t, tick=tick, agent=agent, attempt=attempt, error=repr(e))
                break
            finally:
                if self.stream:
                    self.stream(agent, t, attempt, None)
            self.budget.settle(tokens, sampling.max_tokens, completion.prompt_tokens, completion.generated_tokens)
            if self.host is not None:
                self.host.record(completion.text)

            # Split the output in different parts
            parts = output_parts(completion.text, self.task.answer_labels)
            reasoning = completion.reasoning or parts.get("reasoning") or None
            message = (parts.get("message") or None) if self.slot[agent] else None
            action, error = None, None
            try:
                action = self.task.parse(agent, completion.text)
                error = self.task.check(agent, action)
            except ParseError as e:
                error = str(e)
                parse_failures += 1
                if completion.finish_reason == "length":
                    error += " Your answer was cut off: keep it shorter."

            turn.attempts.append(Attempt(attempt, render(prompt), tokens, removed, completion.text, reasoning,
                                         action, message, error, completion.generated_tokens, completion.elapsed_s))
            self.log.write(
                "agent", "llm_call", t=t, tick=tick, agent=agent, attempt=attempt,
                prompt=render(prompt),
                segments=[{"source": s.source, "ref": s.ref, "chars": len(s.text), "chat_role": s.chat_role}
                          for s in prompt],
                counted_prompt_tokens=tokens, truncated=removed,
                inbox=[m.id for m in turn.inbox] if attempt == 0 else [],
                sampling=asdict(sampling), completion=asdict(completion),
                parts=parts, action=action, error=error,
                usage=self.usage.report(),
            )
            if error is None:
                turn.status, turn.action, turn.message, turn.reasoning = "acted", action, message, reasoning
                break
            if parse_failures >= self.cfg.retries.max_parse_attempts:
                break
        return turn

    def apply(self, turn: Turn) -> None:
        """Steps the agent's environment and publishes its outcome and shared parts"""
        agent = turn.agent

        # the backend failed or the budget refused the call: nothing happened in the environment
        if turn.status in ("failed", "budget"):
            return
        if turn.action is None:
            self.task.no_op(agent)
            self.log.write("task", "no_op", t=turn.t, tick=turn.tick, agent=agent)
            return

        # step the agent's environment + logs
        result = self.task.step(agent, turn.action)
        turn.result = result
        self.log.write("task", "env_step", t=turn.t, tick=turn.tick, agent=agent,
                       **self.task.step_record(agent, result))
        outcome = self.task.outcome_message(agent, turn.action, result)
        drafts = [outcome] if outcome is not None else []
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
        if (cap := self.budget.before_step(self.t)) is not None:
            self.end(f"budget:{cap}")
            return []
        t, turns = self.t, []
        slots = asyncio.Semaphore(self.cfg.engine.max_concurrency)
        try:
            for phase in self.phases:
                tick = self.tick
                active = [a for a in phase if not self.task.agent_done(a)]
                if self.host is None:
                    outputs = await asyncio.gather(*(self.run_agent(a, t, tick, slots) for a in active))
                else:
                    # one shared conversation: the agents of a phase take their turns one after another
                    outputs = [await self.run_agent(a, t, tick, slots) for a in active]
                for turn in sorted(outputs, key=lambda turn: turn.agent):
                    self.apply(turn)
                    self.turns.append(turn)
                    turns.append(turn)
                    if self.listener:
                        self.listener(turn)
                self.tick += 1
                if self.budget_stop is not None:
                    break
            self.log.write("structure", "snapshot", t=t, tick=self.tick - 1, **self.structure.snapshot())
            self.t += 1
            self.budget.end_step()
        except Exception as e:
            self.log.write("engine", "run_failed", t=t, tick=self.tick, error=repr(e))
            self.end("error")
            raise
        statuses = {turn.status for turn in turns}
        if "failed" in statuses:
            self.end("backend_error")
        elif self.budget_stop is not None:
            self.end(f"budget:{self.budget_stop}")
        elif (reason := self.task.end_reason()) is not None:
            self.end(reason)
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
        """Final answers: the task's results for each agent and the team, with the engine's own counts"""
        results = self.task.summary()
        agents = {a: {"role": self.structure.agents[a], **results["agents"][a],
                      "no_valid_answers": sum(turn.agent == a and turn.status == "no_valid_answer"
                                              for turn in self.turns)}
                  for a in self.agents}
        return {
            "ended_by": self.ended_by,
            "steps": self.t,
            **results,
            "agents": agents,
            "usage": self.usage.report(),
        }

    def state(self) -> dict:
        """A plain copy of the current state, safe to hand to another thread"""
        return {
            "t": self.t, "tick": self.tick, "ended_by": self.ended_by, "usage": self.usage.report(),
            "agents": {a: {"role": self.structure.agents[a], **self.task.state(a)} for a in self.agents},
        }


def drop_oldest(view: View, task: Task) -> str | None:
    """Removes the oldest removable history entry: delivered messages first, then the task's own history"""
    social = [(int(m.id[1:]), block, m) for block, messages in view.social.items() for m in messages]
    if social:
        _, block, message = min(social, key=lambda entry: entry[0])
        view.social[block].remove(message)
        return message.id
    return task.truncate(view.observation)


def resolve_budget(cfg: RunConfig) -> tuple[dict, dict]:
    """The run's caps, and their record for the run metadata. A matched cap is read from the reference run with the
    same seed and replicate (baseline B2)"""
    budget = cfg.budget
    caps = {name: getattr(budget, name)
            for name in ("total_tokens", "completion_tokens", "calls", "steps", "wall_clock_s")}
    info = {"caps": caps, "step_reserve": dict(budget.step_reserve), "match": None}
    if budget.match is not None:
        run_dir = ROOT / budget.match.reference / f"seed{cfg.run.seed}_rep{cfg.run.replicate}"
        caps[budget.match.unit] = reference_usage(run_dir, budget.match.unit)
        info["match"] = {"run_dir": str(run_dir), "unit": budget.match.unit, "reference_usage": caps[budget.match.unit]}
    return caps, info


def default_out_dir(cfg: RunConfig) -> Path:
    """runs/<name>/<version>/seed<seed>_rep<replicate>: every version of a run in one folder"""
    return ROOT / "runs" / cfg.run.name / (cfg.run.baseline or "base") / f"seed{cfg.run.seed}_rep{cfg.run.replicate}"


def create_engine(cfg: RunConfig, out_dir: Path | None = None, listener: Callable[[Turn], None] | None = None,
                  backend=None, stream: Callable[[AgentId, int, int, str | None], None] | None = None) -> Engine:
    """Builds the task and the backend named in the config, and the engine"""
    task = build_task(cfg.task)
    backend = backend or make_backend(cfg.engine, task)
    return Engine(cfg, task, backend, out_dir or default_out_dir(cfg), listener, stream)
