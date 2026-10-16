"""
A small desktop GUI
"""

import asyncio
import copy
import json
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass, field, fields
from math import cos, pi, sin
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from framework.comm.compiler import ProtocolConfig, compile_protocol
from framework.comm.message import Message, MessageDraft, covers
from framework.engine.config import ROOT, RunConfig, parse_run_config
from framework.engine.engine import AUTHORED, Attempt, Engine, Turn, create_engine
from framework.tasks.wordcraft.env import StepResult
from framework.tasks.wordcraft.prompt import SOCIAL_BLOCKS

CONFIGS = ROOT / "configs"
RUN_KEYS = set(RunConfig.model_fields)
MONO = ("Consolas", 10)
ROLE_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#eeca3b", "#9d755d"]
TYPE_COLORS = {"result": "#a0522d", "message": "#1f77b4", "task": "#2ca02c", "verdict": "#d62728",
               "reasoning": "#9467bd", "several": "#222222"}
STATUS_COLORS = {"new item": "#2ca02c", "acted": "#f2a900", "no_valid_answer": "#d62728"}
LIVE_COLOR = "#6a1b9a"
IDLE_EDGE = "#d4d4d4"
EDITOR = "(Config tab editor)"
INLINE = "(inline in the run config)"
BLOCK_LABELS = {block: label for block, label, _ in SOCIAL_BLOCKS}
MESSAGE_FIELDS = [f.name for f in fields(Message)]
NO_USAGE = {"calls": 0, "total_tokens": 0, "completion_tokens": 0, "wall_clock_s": 0.0}

# HELPERS

def is_language_model(repo) -> bool:
    """Whether a cached repo has a causal LM config (so image models and the like are left out)"""
    for revision in repo.revisions:
        try:
            config = json.loads((revision.snapshot_path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return any("CausalLM" in name or "LMHead" in name for name in config.get("architectures") or [])
    return False


def downloaded_models() -> list[str]:
    """The language models in the local HuggingFace cache (empty if there is none)"""
    try:
        from huggingface_hub import scan_cache_dir
        return sorted(repo.repo_id for repo in scan_cache_dir().repos
                      if repo.repo_type == "model" and is_language_model(repo))
    except Exception:
        return []


def message_body(msg_type: str, payload: dict) -> str:
    if msg_type == "result.success":
        return f"{' + '.join(payload['combo'])} -> {', '.join(payload['result'])}"
    if msg_type == "result.failure":
        return f"{' + '.join(payload['combo'])} -> nothing"
    return payload["text"]


def describe_message(m) -> str:
    """A delivered Message object"""
    return f"[{m.id}, tick {m.sent_at}] {m.sender} {m.type}: {message_body(m.type, m.payload)}"


def describe_action(turn: Turn) -> tuple[str, str]:
    if turn.result is None:
        return turn.status, ""
    result = ", ".join(turn.result.results) or "nothing"
    if turn.result.repeat:
        result += " (repeat)"
    return " + ".join(turn.action), result


def scrolled(parent, widget_class, **options):
    """A widget with a vertical scrollbar, packed into its own frame"""
    frame = ttk.Frame(parent)
    widget = widget_class(frame, **options)
    bar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
    widget.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    widget.pack(side="left", fill="both", expand=True)
    return frame, widget


def set_text(widget: tk.Text, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")


def rich_text(parent) -> tuple[ttk.Frame, tk.Text]:
    frame, text = scrolled(parent, tk.Text, wrap="word", font=MONO, state="disabled")
    text.tag_configure("h", font=("Consolas", 10, "bold"), foreground="#1f4e79")
    text.tag_configure("title", font=("Consolas", 12, "bold"))
    text.tag_configure("err", foreground="#b00020")
    text.tag_configure("now", background="#fff3c4")
    text.tag_configure("dim", foreground="#777")
    return frame, text


def write(widget: tk.Text, parts: list[tuple[str, str | None]]) -> None:
    """Replaces the text with (text, tag) parts"""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for text, tag in parts:
        widget.insert("end", text, tag or ())
    widget.configure(state="disabled")


def live_parts(raw: str) -> list[tuple[str, str | None]]:
    """The text a model has produced so far, with its <think> part dimmed"""
    if "</think>" in raw:
        reasoning, answer = raw.split("</think>", 1)
        return [(reasoning.replace("<think>", "").strip() + "\n", "dim"), ("— answer —\n", "h"), (answer.strip(), None)]
    if "<think>" in raw:
        return [(raw.split("<think>", 1)[1].strip(), "dim")]
    return [(raw, None)]


def message_from_log(m: dict) -> Message:
    """A Message from its logged form (the extra routing keys RunHistory adds are dropped)"""
    return Message(**{**{name: m[name] for name in MESSAGE_FIELDS}, "parents": tuple(m["parents"])})


def attempt_from_log(call: dict, slot: bool) -> Attempt:
    """An Attempt from its llm_call event, read the way the engine reads the completion"""
    completion, parts = call["completion"], call["parts"]
    action = call["action"] if "action" in call else call["combination"]  # logs written before the rename
    return Attempt(
        attempt=call["attempt"], prompt=call["prompt"], prompt_tokens=call["counted_prompt_tokens"],
        truncated=call["truncated"], answer=completion["text"],
        reasoning=completion["reasoning"] or parts.get("reasoning") or None,
        action=tuple(action) if isinstance(action, list) else action,
        message=(parts.get("message") or None) if slot else None,
        error=call["error"], completion_tokens=completion["generated_tokens"], elapsed_s=completion["elapsed_s"])


def run_header(meta: dict, folder: Path) -> str:
    """The line above the graph: protocol, backend and model, task, log folder"""
    task, engine = meta["task"], meta["engine"]
    target = f"target '{task['target']}'" if task["target"] else "open-ended"
    protocol = meta["config"]["protocol"] if isinstance(meta["config"]["protocol"], str) else "inline protocol"
    model = f" {engine['model_id']}" if engine.get("model_id") not in (None, "mock") else ""
    where = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
    return (f"{protocol}  |  {engine['backend']}{model}  |  task: {target}; start: {', '.join(task['initial'])}; "
            f"{task['max_steps']} steps  |  log: {where}")


def links_of(snapshot: dict) -> dict[tuple[str, str], set[str]]:
    return {(src, dst): set(types) for src, dst, types in snapshot["links"]}


# Run + Logs

@dataclass
class Activity:
    """What happened in one step, as the graph shows it"""
    nodes: dict[str, str] = field(default_factory=dict)  # agent -> new item | acted | no_valid_answer
    sent: dict[tuple[str, str], set[str]] = field(default_factory=dict)  # published in this step
    delayed: dict[tuple[str, str], set[str]] = field(default_factory=dict)  # published earlier, delivered now


class RunHistory:
    """Reads events.jsonl as it grows: messages and their routes, snapshots, environment events"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.meta: dict = {}
        self.messages: dict[str, dict] = {}  # id -> message + receivers, due, delivered {receiver: tick}
        self.snapshots: dict[int, dict] = {}  # step -> structure at the end of the step
        self.last_tick: dict[int, int] = {}  # step -> its last tick
        self.env: list[dict] = []  # env_step and no_op events, in log order
        self.turns: dict[tuple[int, str], str] = {}  # (step, agent) -> status, from the log
        self.calls: dict[tuple[int, str], list[dict]] = {}  # (step, agent) -> llm_call events of an unfinished turn
        self.finished: list[Turn] = []  # every finished turn, rebuilt from its events, in the order it ended
        self.by_tick: dict[tuple[int, str], Turn] = {}  # (tick, agent) -> turn, to attach what it published
        self.generations: dict[str, list[tuple[int, int, str]]] = {}  # agent -> (step, attempt, raw output) per call
        self.usage: dict | None = None  # usage after the last call
        self.summary: dict | None = None  # the run_ended event, if the run finished
        self.snapshot_count = 0

    def read(self) -> None:
        with open(self.path, "rb") as file:
            file.seek(self.offset)
            data = file.read()
        end = data.rfind(b"\n") + 1  # only complete lines
        self.offset += end
        for line in data[:end].decode("utf-8").splitlines():
            if line:
                self.feed(json.loads(line))

    def feed(self, event: dict) -> None:
        kind, data, t = event["kind"], event["data"], event["t"]
        if kind == "run_started":
            self.meta = data
        elif kind == "snapshot":
            self.snapshots[t] = data
            self.last_tick[t] = event["tick"]
            self.snapshot_count += 1
        elif kind == "msg_published":
            m = data["message"]
            self.messages[m["id"]] = {**m, "receivers": data["receivers"], "due": data["due"], "delivered": {}}
            turn = self.by_tick.get((event["tick"], m["sender"]))
            if turn is not None:
                turn.published.append(m["id"])
        elif kind == "llm_call":
            self.calls.setdefault((t, data["agent"]), []).append(data)
            self.generations.setdefault(data["agent"], []).append(
                (t, data["attempt"], data["completion"]["full_raw_answer"]))
            self.usage = data["usage"]
        elif kind == "llm_failed":
            self.end_turn(event, "failed")
        elif kind == "run_ended":
            self.summary = data
        elif kind == "msg_delivered":
            self.messages[data["id"]]["delivered"][data["receiver"]] = event["tick"]
        elif kind in ("env_step", "no_op"):
            self.env.append(event)
            if kind == "env_step":
                self.turns[t, data["agent"]] = "new item" if data["new_items"] else "acted"
                self.end_turn(event, "acted")
            elif kind == "no_op":
                self.turns[t, data["agent"]] = "no_valid_answer"
                self.end_turn(event, "no_valid_answer")

    def end_turn(self, event: dict, status: str) -> None:
        """Rebuilds a turn, as the engine held it, once the event that ends it is in the log"""
        data, t, tick = event["data"], event["t"], event["tick"]
        agent = data["agent"]
        slot = self.meta["message_slot"][agent]
        attempts = [attempt_from_log(call, slot) for call in self.calls.pop((t, agent), [])]
        inbox = [message_from_log(m) for m in self.messages.values() if m["delivered"].get(agent) == tick]
        turn = Turn(t=t, tick=tick, agent=agent, role=self.meta["agents"][agent], inbox=inbox, attempts=attempts,
                    status=status)
        if status == "acted":
            turn.action = attempts[-1].action  # as the agent wrote it; the environment logs it sorted
            turn.message, turn.reasoning = attempts[-1].message, attempts[-1].reasoning
            turn.result = StepResult(tuple(data["combo"]), tuple(data["results"]), tuple(data["new_items"]),
                                     data["repeat"])
        self.finished.append(turn)
        self.by_tick[tick, agent] = turn

    def state(self) -> dict:
        """The status line's numbers, from the log: steps done, clock, usage and how the run ended"""
        steps = max(self.snapshot_count - 1, 0)  # the first snapshot is taken before step 0
        return {"t": steps, "tick": max(self.last_tick.values()) + 1 if steps else 0,
                "usage": (self.summary or {}).get("usage") or self.usage or NO_USAGE,
                "ended_by": self.summary["ended_by"] if self.summary else None}

    def step_of(self, tick: int) -> int | None:
        for t in sorted(self.last_tick):
            if self.last_tick[t] >= tick:
                return t
        return None

    def snapshot(self, t: int) -> dict:
        known = [s for s in self.snapshots if s <= max(t, 0)]
        return self.snapshots[max(known)] if known else self.snapshots[min(self.snapshots)]

    def activity(self, t: int) -> Activity:
        act = Activity(nodes={agent: status for (step, agent), status in self.turns.items() if step == t})
        for m in self.messages.values():
            sent_now = self.step_of(m["sent_at"]) == t
            for receiver in m["receivers"]:
                pair = (m["sender"], receiver)
                if sent_now:
                    act.sent.setdefault(pair, set()).add(m["type"])
                elif receiver in m["delivered"] and self.step_of(m["delivered"][receiver]) == t:
                    act.delayed.setdefault(pair, set()).add(m["type"])
        return act

    def agent_state(self, agent: str, t: int) -> dict:
        """The agent's environment after step t (t = -1: before the first step), rebuilt from the log"""
        task = self.meta["task"]
        inventory = list(task["initial"])
        valid, invalid, used = [], [], 0
        for event in self.env:
            data = event["data"]
            if event["t"] > t or data["agent"] != agent:
                continue
            used += 1
            if event["kind"] == "env_step":
                inventory += data["new_items"]
                if data["results"] and not data["repeat"]:
                    valid.append((data["combo"], data["results"]))
                elif not data["repeat"]:
                    invalid.append(data["combo"])
        return {"inventory": inventory, "valid": valid, "invalid": invalid, "steps_used": used,
                "max_steps": task["max_steps"], "target": task["target"],
                "solved": task["target"] in inventory if task["target"] else None,
                "new": [item for item in inventory if item not in task["initial"]]}


# Clickable graph

class GraphView:
    """Draws a communication structure on a canvas. With an Activity it highlights what happened in a step"""

    def __init__(self, parent, on_click=None, **canvas_options) -> None:
        self.canvas = tk.Canvas(parent, background="white", highlightthickness=1, highlightbackground="#ccc",
                                **canvas_options)
        self.on_click = on_click
        self.scene: dict | None = None
        self.hits: dict[int, tuple[str, object]] = {}  # canvas item -> ("node", agent) | ("edge", (src, dst))
        self.curves: dict[tuple[str, str], tuple[float, ...]] = {}  # edge -> its three control points
        self.selected: tuple[str, object] | None = None
        self.generating: set[str] = set()  # agents inside a model call right now
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.canvas.bind("<Button-1>", self.click)

    def clear(self) -> None:
        self.scene = None
        self.canvas.delete("all")

    def redraw(self) -> None:
        if self.scene:
            self.draw(**self.scene)

    def draw(self, agents: dict[str, str], links: dict[tuple[str, str], set[str]], phases: list[list[str]],
             activity: Activity | None = None, labels: dict[str, str] | None = None) -> None:
        self.scene = dict(agents=agents, links=links, phases=phases, activity=activity, labels=labels)
        c = self.canvas
        c.delete("all")
        self.hits, self.curves = {}, {}
        width, height = c.winfo_width(), c.winfo_height()
        if width < 50:  # not laid out yet: <Configure> draws it
            return
        names = sorted(agents)
        roles = list(dict.fromkeys(agents[a] for a in names))
        node = 16
        bottom_margin = 62 if activity is not None else 40
        top, bottom = 52, height - bottom_margin
        if len(phases) > 1:  # one row per phase, in turn order
            pos = {}
            for row, phase in enumerate(phases):
                y = top + row * (bottom - top) / (len(phases) - 1)
                for i, a in enumerate(phase):
                    pos[a] = (width * (i + 1) / (len(phase) + 1), y)
        elif len(names) == 1:
            pos = {names[0]: (width / 2, (top + bottom) / 2)}
        else:
            cx, cy, radius = width / 2, (top + bottom) / 2, min(width, bottom - top) / 2 - 20
            pos = {a: (cx + radius * cos(2 * pi * i / len(names) - pi / 2),
                       cy + radius * sin(2 * pi * i / len(names) - pi / 2)) for i, a in enumerate(names)}

        def kind_color(types: set[str]) -> str:
            kinds = {t.split(".")[0] for t in types}
            return TYPE_COLORS[kinds.pop()] if len(kinds) == 1 else TYPE_COLORS["several"]

        pairs = set(links) | (set(activity.sent) | set(activity.delayed) if activity else set())
        # quiet edges first, so that active ones are drawn on top
        order = sorted(pairs, key=lambda p: (activity is not None and (p in activity.sent or p in activity.delayed),
                                             self.selected == ("edge", p)))
        for src, dst in order:
            (x1, y1), (x2, y2) = pos[src], pos[dst]
            dx, dy = x2 - x1, y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1)
            ux, uy = dx / length, dy / length
            mx, my = (x1 + x2) / 2 - uy * 14, (y1 + y2) / 2 + ux * 14  # bend right: the two directions separate
            style = {"fill": kind_color(links.get((src, dst), set())), "width": 1.3}
            if activity is not None:
                style = {"fill": IDLE_EDGE, "width": 1}
                if (src, dst) in activity.sent:
                    style = {"fill": kind_color(activity.sent[src, dst]), "width": 3.2}
                elif (src, dst) in activity.delayed:
                    style = {"fill": kind_color(activity.delayed[src, dst]), "width": 2.4, "dash": (6, 3)}
            if self.selected == ("edge", (src, dst)):
                style = {**style, "fill": "#000000", "width": style["width"] + 1.5}
            points = (x1 + ux * node, y1 + uy * node, mx, my, x2 - ux * node, y2 - uy * node)
            item = c.create_line(*points, smooth=True, arrow="last", arrowshape=(10, 12, 4), **style)
            self.hits[item] = ("edge", (src, dst))
            self.curves[src, dst] = points

        for a in names:
            x, y = pos[a]
            color = ROLE_COLORS[roles.index(agents[a]) % len(ROLE_COLORS)]
            if activity is not None and a in activity.nodes:
                ring = STATUS_COLORS[activity.nodes[a]]
                c.create_oval(x - node - 5, y - node - 5, x + node + 5, y + node + 5, outline=ring, width=4)
            if a in self.generating:
                c.create_oval(x - node - 10, y - node - 10, x + node + 10, y + node + 10, outline=LIVE_COLOR, width=3,
                              dash=(4, 3))
            outline, border = ("#000000", 3) if self.selected == ("node", a) else ("white", 2)
            item = c.create_oval(x - node, y - node, x + node, y + node, fill=color, outline=outline, width=border)
            self.hits[item] = ("node", a)
            text = c.create_text(x, y, text=a.split("_")[-1], fill="white", font=("Segoe UI", 9, "bold"))
            self.hits[text] = ("node", a)
            if labels and a in labels:
                c.create_text(x + node + 6, y + node - 2, text=labels[a], anchor="w", font=("Segoe UI", 8),
                              fill="#333")

        x = 8
        for i, role in enumerate(roles):
            c.create_oval(x, 8, x + 10, 18, fill=ROLE_COLORS[i % len(ROLE_COLORS)], outline="")
            x = c.bbox(c.create_text(x + 14, 13, text=role, anchor="w", font=("Segoe UI", 9)))[2] + 12
        y = height - 12
        if activity is not None:
            x = 8
            for status, color in STATUS_COLORS.items():
                c.create_oval(x, y - 6, x + 12, y + 6, outline=color, width=3)
                x = c.bbox(c.create_text(x + 16, y, text=status, anchor="w", font=("Segoe UI", 8)))[2] + 12
            c.create_oval(x, y - 6, x + 12, y + 6, outline=LIVE_COLOR, width=2, dash=(3, 2))
            x = c.bbox(c.create_text(x + 16, y, text="generating now", anchor="w", font=("Segoe UI", 8)))[2] + 12
            c.create_line(x, y, x + 22, y, fill="#555", width=3)
            x = c.bbox(c.create_text(x + 26, y, text="sent this step", anchor="w", font=("Segoe UI", 8)))[2] + 12
            c.create_line(x, y, x + 22, y, fill="#555", width=2.4, dash=(6, 3))
            c.create_text(x + 26, y, text="delivered this step (sent earlier)", anchor="w", font=("Segoe UI", 8))
            y -= 18
        x = 8
        for kind, color in TYPE_COLORS.items():
            c.create_line(x, y, x + 18, y, fill=color, width=2)
            x = c.bbox(c.create_text(x + 22, y, text=kind, anchor="w", font=("Segoe UI", 8), fill="#444"))[2] + 10

    def distance(self, edge: tuple[str, str], x: float, y: float) -> float:
        """From a point to the edge's curve (a quadratic Bezier, as Tk draws a smoothed 3-point line)"""
        x0, y0, x1, y1, x2, y2 = self.curves[edge]
        best = float("inf")
        for i in range(21):
            u = i / 20
            bx = (1 - u) ** 2 * x0 + 2 * u * (1 - u) * x1 + u * u * x2
            by = (1 - u) ** 2 * y0 + 2 * u * (1 - u) * y1 + u * u * y2
            best = min(best, (bx - x) ** 2 + (by - y) ** 2)
        return best

    def click(self, event) -> None:
        if not self.on_click:
            return
        near = self.canvas.find_overlapping(event.x - 6, event.y - 6, event.x + 6, event.y + 6)
        found = [self.hits[i] for i in near if i in self.hits]
        nodes = [hit for hit in found if hit[0] == "node"]
        edges = [hit for hit in found if hit[0] == "edge"]
        hit = nodes[0] if nodes else min(edges, key=lambda e: self.distance(e[1], event.x, event.y), default=None)
        if hit is not None:
            self.selected = hit
            self.redraw()
            self.on_click(hit)


# The Application

class Worker:
    """An event loop on its own thread: the engine runs here, the window never waits for it"""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Multi-Agent Framework")
        root.geometry("1450x900")
        self.worker = Worker()
        self.events: queue.Queue = queue.Queue()  # worker -> window
        self.engine: Engine | None = None
        self.history: RunHistory | None = None
        self.busy = False
        self.pause = threading.Event()
        self.stop_requested = threading.Event()
        self.turns: list[Turn] = []
        self.state: dict = {}
        self.path: Path | None = None
        self.latest = -1  # last finished step
        self.live: dict[str, tuple[int, int, str]] = {}  # agent -> (step, attempt, text so far), while it generates
        self.run_dir: Path | None = None  # folder of the last started run, even if its engine failed to build
        self.live_history: dict[str, list[tuple[int, int, str]]] = {}  # agent -> its finished generations

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        self.build_config_tab(notebook)
        self.build_run_tab(notebook)
        self.build_results_tab(notebook)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)
        default = CONFIGS / "runs" / "demo_mock.yaml"
        if default.exists():
            self.open_file(default)
            self.setup["base"].set(default.name)
        self.load_setup()

    # Config tab

    def build_config_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Config")
        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        ttk.Label(left, text="configs/").pack(anchor="w", padx=4, pady=(4, 0))
        frame, self.files = scrolled(left, ttk.Treeview, show="tree", selectmode="browse")
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.files.bind("<<TreeviewSelect>>", self.on_file_selected)
        ttk.Button(left, text="Refresh", command=self.refresh_files).pack(fill="x", padx=4, pady=(0, 4))
        panes.add(left, weight=1)

        middle = ttk.Frame(panes)
        bar = ttk.Frame(middle)
        bar.pack(fill="x", padx=4, pady=4)
        for label, command in [("Open…", self.open_dialog),
                               ("Save", self.save), ("Save as…", self.save_as), ("Validate", self.validate),
                               ("Use in Run tab", self.use_in_run)]:
            ttk.Button(bar, text=label, command=command).pack(side="left", padx=2)
        self.path_label = ttk.Label(middle, text="(unsaved)", foreground="#555")
        self.path_label.pack(anchor="w", padx=6)
        frame, self.editor = scrolled(middle, tk.Text, undo=True, wrap="none", font=MONO)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.editor.bind("<Control-s>", lambda _: (self.save(), "break")[1])
        panes.add(middle, weight=3)

        right = ttk.Frame(panes)
        ttk.Label(right, text="Compiled structure").pack(anchor="w", padx=4, pady=(4, 0))
        self.config_graph = GraphView(right, height=360)
        self.config_graph.canvas.pack(fill="x", padx=4, pady=4)
        frame, self.preview = scrolled(right, tk.Text, wrap="word", font=MONO, state="disabled", height=10)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        panes.add(right, weight=2)
        self.refresh_files()

    def refresh_files(self) -> None:
        self.files.delete(*self.files.get_children())
        for folder in sorted(p for p in CONFIGS.iterdir() if p.is_dir()):
            node = self.files.insert("", "end", text=folder.name + "/", open=True)
            for path in sorted(folder.glob("*.yaml")):
                self.files.insert(node, "end", iid=str(path), text=path.name)

    def on_file_selected(self, _event) -> None:
        selected = self.files.selection()
        if selected and selected[0].endswith(".yaml") and Path(selected[0]) != self.path:
            if self.confirm_discard():
                self.open_file(Path(selected[0]))

    def confirm_discard(self) -> bool:
        if not self.editor.edit_modified():
            return True
        return messagebox.askyesno("Unsaved changes", "Discard the changes in the editor?")

    def open_file(self, path: Path) -> None:
        self.path = path
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", path.read_text(encoding="utf-8"))
        self.editor.edit_modified(False)
        self.editor.edit_reset()
        self.path_label.configure(text=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        self.validate()

    def open_dialog(self) -> None:
        name = filedialog.askopenfilename(initialdir=CONFIGS, filetypes=[("YAML", "*.yaml *.yml")])
        if name and self.confirm_discard():
            self.open_file(Path(name))

    def save(self) -> None:
        if self.path is None:
            self.save_as()
            return
        self.path.write_text(self.editor.get("1.0", "end-1c") + "\n", encoding="utf-8")
        self.editor.edit_modified(False)
        self.refresh_files()
        self.refresh_setup_lists()
        self.validate()

    def save_as(self) -> None:
        name = filedialog.asksaveasfilename(initialdir=CONFIGS / "runs", defaultextension=".yaml",
                                            filetypes=[("YAML", "*.yaml")])
        if name:
            self.path = Path(name)
            self.path_label.configure(text=str(self.path))
            self.save()

    def editor_config(self) -> RunConfig | ProtocolConfig:
        """The editor's YAML as a run config, or as a protocol when it has none of the run config's sections"""
        text = self.editor.get("1.0", "end-1c")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("the file must be a YAML mapping")
        if set(data) & RUN_KEYS:
            return parse_run_config(text)
        return ProtocolConfig.model_validate(data)

    def use_in_run(self) -> None:
        """Run the editor's content: a run config as it is, a protocol as the protocol of the selected run config"""
        try:
            cfg = self.editor_config()
        except Exception as e:
            messagebox.showerror("Invalid config", str(e))
            return
        if isinstance(cfg, RunConfig):
            if self.path is not None and self.path.parent == CONFIGS / "runs" and not self.editor.edit_modified():
                self.setup["base"].set(self.path.name)
            else:
                self.setup["base"].set(EDITOR)
            self.load_setup()
        elif self.path is not None and self.path.parent == CONFIGS / "protocols" and not self.editor.edit_modified():
            self.setup["protocol"].set(self.path.name)
        else:
            messagebox.showinfo("Save the protocol first", "Save the protocol in configs/protocols/ to select it, "
                                "or write it inline in a run config.")
            return
        self.notebook.select(1)

    def validate(self) -> None:
        try:
            cfg = self.editor_config()
            protocol = cfg.resolved_protocol() if isinstance(cfg, RunConfig) else cfg
            structure, phases = compile_protocol(protocol)
        except Exception as e:
            set_text(self.preview, f"INVALID\n\n{e}")
            self.config_graph.clear()
            return
        lines = []
        if isinstance(cfg, RunConfig):
            lines += [f"Run config: {cfg.run.name}, seed {cfg.run.seed}, replicate {cfg.run.replicate}",
                      "Task: " + ", ".join(f"{key} {value}" for key, value in cfg.task.model_dump().items()),
                      f"Backend: {cfg.engine.backend}" + (f" ({cfg.engine.model})" if cfg.engine.model else ""),
                      f"Protocol: {cfg.protocol if isinstance(cfg.protocol, str) else 'inline'}"]
            lines += [f"Persona of {role}: {file}" for role, file in cfg.personas.items()]
            if unknown := set(cfg.personas) - set(structure.agents.values()):
                lines.append(f"INVALID: personas name roles the protocol does not have: {sorted(unknown)}")
            lines.append("")
        else:
            lines += [""]
        roles = {}
        for agent, role in structure.agents.items():
            roles.setdefault(role, []).append(agent)
        lines.append(f"{len(structure.agents)} agents")
        for role, agents in roles.items():
            fields = protocol.roles[role].visible_fields
            lines.append(f"  {role}: {', '.join(agents)}" + (f"\n    sees: {', '.join(fields)}" if fields else ""))
            lines.append(f"    shares: {', '.join(sorted(structure.shares.get(role, []))) or 'nothing'}")
        lines.append("Phases per step: " + " | ".join(", ".join(p) for p in phases))
        slot = [a for a in structure.agents if any(structure.receivers(MessageDraft(a, t, {})) for t in AUTHORED)]
        lines.append(f"Message slot: {', '.join(slot) or 'nobody'}")
        lines.append(f"\n{len(structure.links)} links:")
        for (src, dst), types in sorted(structure.links.items()):
            lines.append(f"  {src} -> {dst}: {', '.join(sorted(types))}")
        if structure.channels:
            lines.append("\nChannels:")
            for name, rule in structure.channels.items():
                lines.append(f"  {name}: {rule.model_dump(mode='json')}")
        set_text(self.preview, "\n".join(lines))
        self.config_graph.draw(dict(structure.agents), {k: set(v) for k, v in structure.links.items()}, phases)

    #Run tab setup

    def build_run_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Run")
        self.build_setup(tab)

        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=4)
        self.run_buttons = {}
        for label, command in [("Start", self.start), ("Step", self.step), ("Run", self.run),
                               ("Pause", self.do_pause), ("Stop", self.stop), ("Load run…", self.load_run)]:
            button = ttk.Button(bar, text=label, command=command)
            button.pack(side="left", padx=2)
            self.run_buttons[label] = button
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="left", padx=10)
        self.status = ttk.Label(bar, text="No run yet. Choose the setup above and press Start.")
        self.status.pack(side="left", padx=6)
        self.task_label = ttk.Label(tab, text="", foreground="#333")
        self.task_label.pack(anchor="w", padx=8)

        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.Frame(panes)
        stepper = ttk.Frame(left)
        stepper.pack(fill="x", pady=(0, 4))
        ttk.Label(stepper, text="Showing step").pack(side="left")
        ttk.Button(stepper, text="◀", width=3, command=lambda: self.move_step(-1)).pack(side="left", padx=(6, 0))
        self.step_scale = tk.Scale(stepper, from_=0, to=0, orient="horizontal", showvalue=True, length=380,
                                   command=lambda _: self.show_step(), state="disabled")
        self.step_scale.pack(side="left", padx=2)
        ttk.Button(stepper, text="▶", width=3, command=lambda: self.move_step(1)).pack(side="left")
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(stepper, text="follow the latest step", variable=self.follow,
                        command=self.on_follow).pack(side="left", padx=10)
        self.run_graph = GraphView(left, on_click=self.on_graph_click)
        self.run_graph.canvas.pack(fill="both", expand=True)
        self.step_label = ttk.Label(left, text="Click a node for the agent's state, a link for its messages.",
                                    foreground="#555")
        self.step_label.pack(anchor="w", pady=(4, 0))
        panes.add(left, weight=5)

        right = ttk.Notebook(panes)
        self.right_tabs = right
        frame, self.info = rich_text(right)
        right.add(frame, text="Selection")
        right.add(self.build_turns(right), text="Turns")
        frame, self.live_text = rich_text(right)
        self.live_frame = frame
        right.add(frame, text="Live")
        panes.add(right, weight=4)
        self.update_buttons()

    def build_setup(self, tab) -> None:
        box = ttk.LabelFrame(tab, text="Run setup: a run config, with these settings overridden")
        box.pack(fill="x", padx=6, pady=(6, 0))
        names = ("base", "protocol", "backend", "model")
        self.setup = {name: tk.StringVar() for name in names}

        def label(row, text):
            ttk.Label(row, text=text).pack(side="left", padx=(10, 3))

        row = ttk.Frame(box)
        row.pack(fill="x", padx=4, pady=3)
        label(row, "Run config")
        self.base_box = ttk.Combobox(row, textvariable=self.setup["base"], state="readonly", width=26)
        self.base_box.bind("<<ComboboxSelected>>", lambda _: self.load_setup())
        self.base_box.pack(side="left")
        ttk.Button(row, text="Edit", width=5, command=self.edit_base).pack(side="left", padx=2)
        label(row, "Protocol")
        self.protocol_box = ttk.Combobox(row, textvariable=self.setup["protocol"], state="readonly", width=26)
        self.protocol_box.pack(side="left")
        ttk.Button(row, text="View", width=5, command=self.view_protocol).pack(side="left", padx=2)
        label(row, "Backend")
        ttk.Combobox(row, textvariable=self.setup["backend"], values=["mock", "hf_local"],
                     state="readonly", width=9).pack(side="left")
        label(row, "Model")
        self.model_box = ttk.Combobox(row, textvariable=self.setup["model"], values=downloaded_models(), width=34)
        self.model_box.pack(side="left")
        ttk.Button(row, text="↻", width=3, command=self.refresh_models).pack(side="left", padx=2)
        self.refresh_setup_lists()

    def refresh_models(self) -> None:
        self.model_box.configure(values=downloaded_models())

    def refresh_setup_lists(self) -> None:
        runs = [p.name for p in sorted((CONFIGS / "runs").glob("*.yaml"))]
        self.base_box.configure(values=runs + [EDITOR])
        if not self.setup["base"].get():
            self.setup["base"].set(runs[0] if runs else EDITOR)
        self.protocol_files = [p.name for p in sorted((CONFIGS / "protocols").glob("*.yaml"))]

    def base_text(self) -> str:
        base = self.setup["base"].get()
        if base == EDITOR:
            return self.editor.get("1.0", "end-1c")
        return (CONFIGS / "runs" / base).read_text(encoding="utf-8")

    def load_setup(self) -> None:
        """Fills the setup fields from the chosen run config"""
        try:
            cfg = parse_run_config(self.base_text())
        except Exception as e:
            self.status.configure(text=f"The run config is not valid: {str(e).splitlines()[0]}")
            return
        protocols = list(self.protocol_files)
        if isinstance(cfg.protocol, str):
            path = Path(cfg.protocol)
            chosen = path.name if path.parent == Path("configs/protocols") else cfg.protocol
            if chosen not in protocols:
                protocols.append(chosen)
        else:
            chosen = INLINE
            protocols.insert(0, INLINE)
        self.protocol_box.configure(values=protocols)
        values = {"protocol": chosen, "backend": cfg.engine.backend, "model": cfg.engine.model or ""}
        for name, value in values.items():
            self.setup[name].set(str(value))

    def setup_config(self) -> RunConfig:
        """The chosen run config with the setup fields written over it"""
        data = copy.deepcopy(yaml.safe_load(self.base_text()) or {})
        protocol = self.setup["protocol"].get()
        if protocol != INLINE:
            data["protocol"] = f"configs/protocols/{protocol}" if protocol in self.protocol_files else protocol
        engine = data.setdefault("engine", {})
        engine["backend"] = self.setup["backend"].get()
        if self.setup["model"].get().strip():
            engine["model"] = self.setup["model"].get().strip()
        else:
            engine.pop("model", None)
        return RunConfig.model_validate(data)

    def edit_base(self) -> None:
        base = self.setup["base"].get()
        if base != EDITOR and self.confirm_discard():
            self.open_file(CONFIGS / "runs" / base)
        self.notebook.select(0)

    def view_protocol(self) -> None:
        protocol = self.setup["protocol"].get()
        if protocol in self.protocol_files and self.confirm_discard():
            self.open_file(CONFIGS / "protocols" / protocol)
            self.notebook.select(0)
        elif protocol == INLINE:
            self.edit_base()

    # Run tab: controls

    def update_buttons(self) -> None:
        live = self.engine is not None and not self.engine.done
        states = {"Start": not self.busy, "Step": live and not self.busy, "Run": live and not self.busy,
                  "Pause": self.busy, "Stop": live,
                  "Load run…": not self.busy}
        for label, enabled in states.items():
            self.run_buttons[label].configure(state="normal" if enabled else "disabled")
        if self.busy:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)

    def start(self) -> None:
        try:
            cfg = self.setup_config()
        except Exception as e:
            messagebox.showerror("Invalid setup", str(e))
            return
        if self.engine is not None and not self.engine.done:
            if not messagebox.askyesno("Run in progress", "Stop the current run and start a new one?"):
                return
            self.worker.submit(self.end_quietly(self.engine, "stopped_by_user")).result()
        out = ROOT / "runs" / "gui" / f"{cfg.run.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.clear_run_view()
        self.run_dir = out
        self.status.configure(text=f"Loading the {cfg.engine.backend} backend…")
        self.launch(self.build_engine(cfg, out))

    def clear_run_view(self) -> None:
        """Empties the Run and Results tabs before a new or a loaded run"""
        self.engine, self.history, self.turns, self.state, self.latest = None, None, [], {}, -1
        self.run_graph.selected = None
        self.run_graph.generating = set()
        self.live.clear()
        self.live_history = {}
        self.run_graph.clear()
        self.turn_table.delete(*self.turn_table.get_children())
        for text in [self.info, self.live_text, *self.detail.values()]:
            set_text(text, "")
        self.step_scale.configure(from_=0, to=0, state="disabled")
        self.results_table.delete(*self.results_table.get_children())
        set_text(self.results_text, "")
        self.results_label.configure(text="No finished run yet.")

    def load_run(self) -> None:
        """Shows a finished (or interrupted) run from its folder, rebuilt from its events.jsonl"""
        runs = ROOT / "runs"
        name = filedialog.askdirectory(initialdir=runs / "gui" if (runs / "gui").exists() else runs,
                                       title="Choose the folder of one run (it holds events.jsonl)")
        if name:
            self.show_run(Path(name))

    def show_run(self, folder: Path) -> None:
        if not (folder / "events.jsonl").exists():
            messagebox.showerror("Load run", f"{folder} has no events.jsonl: choose the folder of a single run.")
            return
        with open(folder / "events.jsonl", encoding="utf-8") as file:
            first = json.loads(file.readline() or "{}")
        if first.get("kind") != "run_started" or "phases" not in first.get("data", {}):
            messagebox.showerror("Load run", "This log was not written by the multi-agent engine "
                                             "(old single-agent runs from play_wordcraft.py cannot be shown here).")
            return
        history = RunHistory(folder / "events.jsonl")
        try:
            history.read()
        except (KeyError, TypeError) as e:
            messagebox.showerror("Load run", f"This log was written by an older version of the engine and cannot be "
                                             f"shown (an event has no {e} field).")
            return
        if self.engine is not None and not self.engine.done:
            if not messagebox.askyesno("Run in progress", "End the current run and show the chosen one instead?"):
                return
            self.worker.submit(self.end_quietly(self.engine, "stopped_by_user")).result()
        self.clear_run_view()
        self.run_dir, self.history = folder, history
        self.live_history = {agent: list(calls) for agent, calls in history.generations.items()}
        self.task_label.configure(text=run_header(history.meta, folder))
        self.follow.set(True)
        self.show_state(history.state())
        if history.summary is not None:
            self.show_results(history.summary)
        else:
            self.results_label.configure(text="This run did not finish (no run_ended event): "
                                              "the Run tab shows it up to its last logged step.")
        self.status.configure(text=f"Loaded {folder.name}   " + self.status.cget("text"))
        self.refresh_live(False, jump=True)
        self.update_buttons()

    def launch(self, coroutine) -> None:
        self.busy = True
        self.update_buttons()
        self.worker.submit(coroutine)

    def step(self) -> None:
        self.pause.clear()
        self.right_tabs.select(self.live_frame)
        self.launch(self.run_steps(1))

    def run(self) -> None:
        self.pause.clear()
        self.right_tabs.select(self.live_frame)
        self.launch(self.run_steps(None))

    def do_pause(self) -> None:
        self.pause.set()
        self.status.configure(text="Pausing after this step…")

    def stop(self) -> None:
        if self.engine is None:
            return
        if self.busy:
            self.pause.set()
            self.stop_requested.set()
            self.status.configure(text="Stopping after this step…")
        else:
            self.launch(self.end_engine(self.engine, "stopped_by_user"))

    # coroutines on the worker thread: they talk to the window only through self.events

    async def build_engine(self, cfg: RunConfig, out: Path) -> None:
        try:
            engine = await asyncio.to_thread(
                create_engine, cfg, out, lambda turn: self.events.put(("turn", turn)),
                stream=lambda agent, t, attempt, text: self.events.put(("stream", agent, t, attempt, text)))
            self.events.put(("built", engine, engine.state()))
        except Exception:
            self.events.put(("error", traceback.format_exc()))
        self.events.put(("idle",))

    async def run_steps(self, count: int | None) -> None:
        engine = self.engine
        try:
            done = 0
            while not engine.done and not self.pause.is_set() and (count is None or done < count):
                await engine.step()
                done += 1
                self.events.put(("state", engine.state()))
            if self.stop_requested.is_set():
                self.stop_requested.clear()
                engine.end("stopped_by_user")
            if engine.done:
                self.events.put(("ended", engine.summary(), engine.state()))
        except Exception:
            self.events.put(("error", traceback.format_exc()))
        self.events.put(("idle",))

    async def end_quietly(self, engine: Engine, reason: str) -> None:
        """Ends a run that is being replaced, without posting its results to the window"""
        engine.end(reason)

    async def end_engine(self, engine: Engine, reason: str) -> None:
        engine.end(reason)
        self.events.put(("ended", engine.summary(), engine.state()))
        self.events.put(("idle",))

    # window side

    def poll(self) -> None:
        before, changed = set(self.live), False
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "built":
                    self.on_built(event[1], event[2])
                elif kind == "stream":
                    _, agent, t, attempt, text = event
                    if text is None:
                        if agent in self.live and self.live[agent][2]:
                            self.live_history.setdefault(agent, []).append(self.live[agent])
                        self.live.pop(agent, None)
                    else:
                        self.live[agent] = (t, attempt, text)
                    changed = True
                elif kind == "turn":
                    self.sync_turns()
                elif kind == "state":
                    self.show_state(event[1])
                elif kind == "ended":
                    self.show_state(event[2])
                    self.show_results(event[1])
                elif kind == "error":
                    messagebox.showerror("Error", event[1][-3000:])
                elif kind == "idle":
                    self.busy = False
                    self.live.clear()
                    changed = True
                    self.pause.clear()
                    self.update_buttons()
        except queue.Empty:
            pass
        if changed:
            self.refresh_live(set(self.live) != before)
        self.root.after(100, self.poll)

    def on_built(self, engine: Engine, state: dict) -> None:
        self.engine = engine
        self.history = RunHistory(engine.out_dir / "events.jsonl")
        self.task_label.configure(text=run_header(engine.meta, engine.out_dir))
        self.show_state(state)

    def show_state(self, state: dict) -> None:
        self.state = state
        usage = state["usage"]
        ended = f"   ended: {state['ended_by']}" if state["ended_by"] else ""
        self.status.configure(text=f"step {state['t']}  tick {state['tick']}   calls {usage['calls']}   "
                                   f"tokens {usage['total_tokens']:,} (completion {usage['completion_tokens']:,})"
                                   f"   {usage['wall_clock_s']:.1f}s{ended}")
        if self.history is None:
            return
        self.sync_turns()
        self.latest = state["t"] - 1
        if self.latest >= 0:
            self.step_scale.configure(from_=0, to=self.latest, state="normal")
            if self.follow.get():
                self.step_scale.set(self.latest)
        self.show_step()

    # Run tab: graph and selection

    def shown_step(self) -> int:
        """The step on display; -1 before the first step"""
        return int(self.step_scale.get()) if self.latest >= 0 else -1

    def move_step(self, delta: int) -> None:
        if self.latest >= 0:
            self.follow.set(False)
            self.step_scale.set(min(max(self.shown_step() + delta, 0), self.latest))

    def on_follow(self) -> None:
        if self.follow.get() and self.latest >= 0:
            self.step_scale.set(self.latest)

    def show_step(self) -> None:
        h = self.history
        if h is None or not h.snapshots:
            return
        t = self.shown_step()
        snapshot = h.snapshot(t)
        activity = h.activity(t) if t >= 0 else Activity()
        labels = {}
        for agent in snapshot["agents"]:
            s = h.agent_state(agent, t)
            labels[agent] = f"{len(s['inventory'])} items" + (" ✓" if s["solved"] else "")
        self.run_graph.draw(snapshot["agents"], links_of(snapshot), h.meta["phases"], activity, labels)
        if t < 0:
            self.step_label.configure(text="Before the first step. Click a node for the agent's state, "
                                           "a link for its messages.")
        else:
            sent = sum(len(m["receivers"]) for m in h.messages.values() if h.step_of(m["sent_at"]) == t)
            self.step_label.configure(text=f"Step {t}: {len(activity.nodes)} agents acted, {sent} message "
                                           f"deliveries scheduled, on {len(activity.sent)} links. "
                                           "Click a node or a link for details.")
        self.show_selection()

    def on_graph_click(self, hit) -> None:
        self.right_tabs.select(0)
        self.show_selection()
        self.refresh_turns()
        self.refresh_live(False, jump=True)

    def refresh_live(self, ring_changed: bool, jump: bool = False) -> None:
        """Shows the selected agent's generations (past ones, then the one in progress), or, with no agent selected,
        every agent that is generating now. Keeps the scroll position unless it was at the bottom (or jump)"""
        if ring_changed:
            self.run_graph.generating = set(self.live)
            self.run_graph.redraw()
        selected = self.turn_filter()
        entries: list[tuple[str, int, int, str, bool]] = []  # agent, step, attempt, text, in progress
        notes: list[tuple[str, str | None]] = []
        if selected is None:
            entries = [(a, *self.live[a], True) for a in sorted(self.live)]
            if not entries:
                notes = [("No model call in progress. Click an agent to see its past generations.\n", "dim")]
        else:
            entries = [(selected, *g, False) for g in self.live_history.get(selected, [])]
            if selected in self.live:
                entries.append((selected, *self.live[selected], True))
            elif not entries:
                notes = [(f"{selected} has not called the model yet.\n", "dim")]
            elif self.live:
                notes = [(f"{selected} is not generating now (generating: {', '.join(sorted(self.live))}).\n\n", "dim")]
        parts = notes
        for agent, t, attempt, text, running in entries:
            role = self.history.meta["agents"].get(agent, "") if self.history else ""
            parts.append((f"{agent} ({role}), step {t}, attempt {attempt + 1}" + ("  — generating…" if running else "")
                          + "\n", "title"))
            parts += live_parts(text) if text else [("(reading the prompt…)", "dim")]
            parts.append(("\n\n", None))
        view = self.live_text
        at_bottom = view.bbox("end-1c") is not None  # the last character is on screen (yview fractions are only estimates)
        top = view.index("@0,0")
        write(view, parts)
        if at_bottom or jump:
            view.see("end")
        else:
            view.yview(top)

    def show_selection(self) -> None:
        hit, h = self.run_graph.selected, self.history
        if hit is None or h is None or not h.meta:
            return
        t = self.shown_step()
        if hit[0] == "node":
            write(self.info, self.node_info(hit[1], t))
        else:
            write(self.info, self.edge_info(*hit[1], t))

    def message_line(self, m: dict, t: int, receiver: str | None = None) -> tuple[str, str | None]:
        h = self.history
        step = h.step_of(m["sent_at"])
        line = f"[{m['id']}] step {step}, tick {m['sent_at']}: {m['sender']} {m['type']}: " \
               f"{message_body(m['type'], m['payload'])}"
        if receiver is None:
            line += f"\n      to {', '.join(m['receivers']) or 'nobody'}"
        else:
            delivered = m["delivered"].get(receiver)
            if delivered is not None and h.step_of(delivered) > t:  # after the step on display
                delivered = None
            line += f"\n      due tick {m['due']}, " + (f"delivered at tick {delivered} (step {h.step_of(delivered)})"
                                                         if delivered is not None else f"not delivered by step {t}")
        delivered_now = (receiver is not None and receiver in m["delivered"]
                         and h.step_of(m["delivered"][receiver]) == t)
        return line + "\n", "now" if step == t or delivered_now else None

    def node_info(self, agent: str, t: int) -> list[tuple[str, str | None]]:
        h = self.history
        meta, snapshot = h.meta, h.snapshot(t)
        role = snapshot["agents"][agent]
        state = h.agent_state(agent, t)
        links = links_of(snapshot)
        when = f"after step {t}" if t >= 0 else "before the first step"
        out: list[tuple[str, str | None]] = [(f"{agent}  ·  role {role}\n", "title")]
        blocks = [BLOCK_LABELS[b] for b in meta["social_blocks"][agent]]
        out += [("\nRole and channels\n", "h"),
                (f"Sees: {', '.join(meta['visible_fields'][agent])}\n"
                 f"Persona: {meta.get('personas', {}).get(role, {}).get('file', 'none')}\n"
                 f"May publish: {', '.join(snapshot['shares'].get(role, [])) or 'nothing'}\n"
                 f"Message slot in its prompt: {'yes' if meta['message_slot'][agent] else 'no'}\n"
                 f"Prompt lines from others: {', '.join(blocks) or 'none'}\n", None)]
        outgoing = [f"{dst} ({', '.join(sorted(types))})" for (src, dst), types in sorted(links.items()) if src == agent]
        incoming = [f"{src} ({', '.join(sorted(types))})" for (src, dst), types in sorted(links.items()) if dst == agent]
        out += [(f"Links out: {', '.join(outgoing) or 'none'}\nLinks in: {', '.join(incoming) or 'none'}\n", None)]

        out += [(f"\nState {when}\n", "h"),
                (f"Inventory ({len(state['inventory'])}): {', '.join(state['inventory'])}\n"
                 f"New items: {', '.join(state['new']) or 'none'}\n"
                 + f"Steps used: {state['steps_used']} of {state['max_steps']}"
                 + (f"   Target: {state['target']}   Solved: {'yes' if state['solved'] else 'no'}" if state["target"] else "")
                 + "\n", None),
                (f"Valid combinations ({len(state['valid'])}):\n", "h")]
        out += [(f"  {' + '.join(combo)} -> {', '.join(results)}\n", None) for combo, results in state["valid"]]
        out += [(f"Invalid combinations ({len(state['invalid'])}):\n", "h")]
        out += [(f"  {' + '.join(combo)}\n", None) for combo in state["invalid"]]

        turn = next((turn for turn in self.turns if turn.agent == agent and turn.t == t), None)
        if t >= 0:
            out.append((f"\nIn step {t}\n", "h"))
            if turn is None:
                out.append(("Did not act (its task was over).\n", "dim"))
            else:
                action, result = describe_action(turn)
                calls = len(turn.attempts)
                tokens = sum(a.prompt_tokens + a.completion_tokens for a in turn.attempts)
                out.append((f"{turn.status}: {action}" + (f" -> {result}" if result else "") + "\n"
                            + (f'Message: "{turn.message}"\n' if turn.message else "")
                            + f"{calls} LLM call{'s' if calls != 1 else ''}, {tokens} tokens"
                            + "  (prompt and output: Turns tab)\n", "now"))

        received = [(m, r) for m in h.messages.values() for r, tick in m["delivered"].items()
                    if r == agent and (h.step_of(tick) or 0) <= max(t, -1)]
        out.append((f"\nMessages received up to this step ({len(received)})\n", "h"))
        out += [self.message_line(m, t, agent) for m, _ in received]
        sent = [m for m in h.messages.values() if m["sender"] == agent and (h.step_of(m["sent_at"]) or 0) <= max(t, -1)]
        out.append((f"\nMessages sent up to this step ({len(sent)})\n", "h"))
        out += [self.message_line(m, t) for m in sent]
        return out

    def edge_info(self, src: str, dst: str, t: int) -> list[tuple[str, str | None]]:
        h = self.history
        snapshot = h.snapshot(t)
        types = links_of(snapshot).get((src, dst))
        out: list[tuple[str, str | None]] = [(f"{src}  →  {dst}\n", "title")]
        if types:
            out.append((f"\nLink in force at step {max(t, 0)}: carries {', '.join(sorted(types))}\n", None))
        else:
            out.append(("\nNo link: these messages reached the receiver through a channel whose visible_to "
                        "bypasses the links (all, or a list of agents).\n", None))
        rules = [f"  {name}: {rule}" for name, rule in snapshot["channels"].items()
                 if not types or any(covers(t_, name) or covers(name, t_) for t_ in types)]
        out.append(("Channel rules: " + ("\n" + "\n".join(rules) if rules
                                          else "default (neighbors, no delay, full form)") + "\n", None))
        along = [m for m in h.messages.values() if m["sender"] == src and dst in m["receivers"]
                 and (h.step_of(m["sent_at"]) or 0) <= max(t, -1)]
        out.append((f"\nMessages along this link up to step {t} ({len(along)}); highlighted: this step\n", "h"))
        out += [self.message_line(m, t, dst) for m in along]
        if not along:
            out.append(("none\n", "dim"))
        return out

    # Run tab: turns

    def build_turns(self, parent) -> ttk.PanedWindow:
        panes = ttk.PanedWindow(parent, orient="vertical")
        top = ttk.Frame(panes)
        head = ttk.Frame(top)
        head.pack(fill="x")
        self.turns_label = ttk.Label(head, text="All turns (click a node in the graph to filter)")
        self.turns_label.pack(side="left")
        ttk.Button(head, text="Show all", command=self.show_all_turns).pack(side="right")
        columns = ("t", "tick", "agent", "action", "result", "message")
        frame, self.turn_table = scrolled(top, ttk.Treeview, columns=columns, show="headings", selectmode="browse")
        for column, width in [("t", 35), ("tick", 40), ("agent", 75), ("action", 150), ("result", 120),
                              ("message", 200)]:
            self.turn_table.heading(column, text=column)
            self.turn_table.column(column, width=width, anchor="w")
        self.turn_table.bind("<<TreeviewSelect>>", self.on_turn_selected)
        frame.pack(fill="both", expand=True)
        panes.add(top, weight=2)

        detail = ttk.Notebook(panes)
        self.detail = {}
        for name in ("Output", "Prompt", "Inbox"):
            frame, text = rich_text(detail)
            detail.add(frame, text=name)
            self.detail[name] = text
        panes.add(detail, weight=3)
        return panes

    def turn_filter(self) -> str | None:
        hit = self.run_graph.selected
        return hit[1] if hit and hit[0] == "node" else None

    def sync_turns(self) -> None:
        """Reads new log lines and adds the turns they finished (the Turns tab shows the log, not engine memory)"""
        if self.history is None:
            return
        self.history.read()
        for turn in self.history.finished[len(self.turns):]:
            self.add_turn(turn)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if self.turn_filter() in (None, turn.agent):
            self.insert_turn(len(self.turns) - 1, turn)

    def insert_turn(self, index: int, turn: Turn) -> None:
        action, result = describe_action(turn)
        self.turn_table.insert("", "end", iid=str(index),
                               values=(turn.t, turn.tick, turn.agent, action, result, turn.message or ""))
        self.turn_table.see(str(index))

    def refresh_turns(self) -> None:
        agent = self.turn_filter()
        self.turns_label.configure(text=f"Turns of {agent}" if agent else "All turns (click a node to filter)")
        self.turn_table.delete(*self.turn_table.get_children())
        for index, turn in enumerate(self.turns):
            if agent in (None, turn.agent):
                self.insert_turn(index, turn)

    def show_all_turns(self) -> None:
        self.run_graph.selected = None
        self.run_graph.redraw()
        self.refresh_turns()

    def on_turn_selected(self, _event) -> None:
        selected = self.turn_table.selection()
        if not selected:
            return
        turn = self.turns[int(selected[0])]
        if self.latest >= turn.t and self.shown_step() != turn.t:
            self.follow.set(False)
            self.step_scale.set(turn.t)
        parts: list[tuple[str, str | None]] = [
            (f"{turn.agent} ({turn.role}), step {turn.t}, tick {turn.tick}: {turn.status}\n", "h")]
        for a in turn.attempts:
            parts.append((f"\n— Attempt {a.attempt + 1}: {a.prompt_tokens} prompt tokens, "
                          f"{a.completion_tokens} completion tokens, {a.elapsed_s:.2f}s —\n", "h"))
            if a.truncated:
                parts.append((f"Context guard removed {len(a.truncated)} history entries\n", "err"))
            if a.reasoning:
                parts += [("Reasoning:\n", "h"), (a.reasoning.strip() + "\n", None)]
            parts += [("Answer:\n", "h"), (a.answer.strip() + "\n", None), ("Parsed: ", "h"),
                      (f"action {a.action}" +(f", message {a.message!r}" if a.message else "") + "\n", None)]
            if a.error:
                parts.append((f"Error (sent back to the model): {a.error}\n", "err"))
        if turn.result is not None:
            r = turn.result
            parts += [("\nEnvironment: ", "h"),
                      (f"{' + '.join(r.combo)} -> {', '.join(r.results) or 'nothing'}; new items: "
                       f"{', '.join(r.new_items) or 'none'}" + ("; repeat" if r.repeat else "") + "\n", None)]
        if turn.published:
            parts.append((f"Published: {', '.join(turn.published)}\n", None))
        write(self.detail["Output"], parts)
        prompt = turn.attempts[-1].prompt if turn.attempts else "(no call)"
        header = f"Prompt of attempt {len(turn.attempts)} of {len(turn.attempts)}\n{'=' * 60}\n" if turn.attempts else ""
        set_text(self.detail["Prompt"], header + prompt)
        inbox = "\n".join(describe_message(m) for m in turn.inbox) or "(nothing delivered at this turn)"
        set_text(self.detail["Inbox"], f"Delivered to {turn.agent} at tick {turn.tick}:\n\n{inbox}")

    # Results tab

    def build_results_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Results")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Button(bar, text="Open run folder", command=self.open_run_folder).pack(side="left")
        self.results_label = ttk.Label(bar, text="No finished run yet.")
        self.results_label.pack(side="left", padx=10)
        columns = ("role", "solved", "steps", "new", "coverage", "invalid", "no_valid_answers", "items", "last_message")
        frame, self.results_table = scrolled(tab, ttk.Treeview, columns=columns, height=10)
        self.results_table.heading("#0", text="agent")
        self.results_table.column("#0", width=90)
        for column, width in [("role", 80), ("solved", 55), ("steps", 45), ("new", 45), ("coverage", 70),
                              ("invalid", 55), ("no_valid_answers", 110), ("items", 330), ("last_message", 330)]:
            self.results_table.heading(column, text=column.replace("_", " "))
            self.results_table.column(column, width=width, anchor="w")
        frame.pack(fill="both", expand=True, padx=6, pady=6)
        frame, self.results_text = scrolled(tab, tk.Text, wrap="word", font=MONO, state="disabled", height=12)
        frame.pack(fill="both", expand=False, padx=6, pady=6)

    def show_results(self, summary: dict) -> None:
        self.results_table.delete(*self.results_table.get_children())
        for agent, info in summary["agents"].items():
            last = next((t.message for t in reversed(self.turns) if t.agent == agent and t.message), "")
            solved = "" if info["solved"] is None else ("yes" if info["solved"] else "no")
            coverage = "" if info["coverage"] is None else f"{info['coverage']:.1%}"
            self.results_table.insert("", "end", text=agent, values=(
                info["role"], solved, info["steps"], len(info["new_items"]), coverage, info["invalid"],
                info["no_valid_answers"], ", ".join(info["new_items"]), last))
        team, task = summary["team"], summary["task"]
        lines = [f"Ended by: {summary['ended_by']} after {summary['steps']} steps",
                 f"Task: {'target ' + repr(task['target']) if task['target'] else 'open-ended'}, "
                 f"start {', '.join(task['initial'])}"]
        if task["target"]:
            lines.append(f"Team: any agent solved = {team['any_solved']}, all solved = {team['all_solved']}")
        lines.append(f"Team discovered {team['union_new_items']} distinct new items"
                     + (f" ({team['union_coverage']:.1%} of what is reachable)" if team["union_coverage"] is not None else ""))
        usage = summary["usage"]
        lines.append(f"Usage: {usage['calls']} calls, {usage['total_tokens']:,} tokens "
                     f"({usage['completion_tokens']:,} completion), {usage['wall_clock_s']:.1f}s")
        if self.run_dir is not None:
            lines.append(f"Log: {self.run_dir}")
        set_text(self.results_text, "\n".join(lines))
        self.results_label.configure(text=f"Run ended: {summary['ended_by']}")

    def open_run_folder(self) -> None:
        folder = self.run_dir if self.run_dir is not None else (self.engine.out_dir if self.engine else None)
        if folder is None:
            messagebox.showinfo("Open run folder", "There is no run yet: press Start first.")
        elif not folder.exists():
            messagebox.showinfo("Open run folder", f"{folder} does not exist (the run was not created).")
        elif not hasattr(os, "startfile"):
            messagebox.showinfo("Open run folder", f"This system cannot open folders from here. The run is in:\n{folder}")
        else:
            try:
                os.startfile(folder)
            except OSError as e:
                messagebox.showerror("Open run folder", f"Could not open {folder}:\n{e}")

    # Closing

    def close(self) -> None:
        if self.engine is not None and not self.engine.done:
            if self.busy:
                self.pause.set()
                self.stop_requested.set()
            else:
                self.worker.submit(self.end_engine(self.engine, "window_closed")).result(timeout=10)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
