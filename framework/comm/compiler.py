"""
Structure compiler
"""

import inspect
import operator
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal
import networkx as nx
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from framework.comm.generators import GENERATORS
from framework.comm.message import MESSAGE_TYPES, AgentId, covers, is_known
from framework.comm.structure import ChannelRule, Structure


class CompileError(ValueError):
    """General error for this process"""


class Strict(BaseModel):
    """Base class, !!!"""
    model_config = ConfigDict(extra="forbid", frozen=True)


class Role(Strict):
    # how many agents get this role
    count: int | None = Field(default=None, ge=1)
    # choose nodes that satisfy these conditions (where: leaf, {hub: true}, ...)
    where: str | dict[str, Any] | None = None
    # observation fields the role sees (None = all of them)
    visible_fields: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def count_or_where(self) -> "Role":
        """Checks that only count or where has been used, not both of them"""
        if (self.count is None) == (self.where is None):
            raise ValueError("a role needs exactly one of `count` and `where`")
        return self


class Rule(Strict):
    model_config = ConfigDict(populate_by_name=True)
    # The role of the sending agent
    from_: str | None = Field(default=None, alias="from")
    # The role of the receiving agent
    to: str | None = None
    # Type of relation between from and to
    along: str | None = None
    # Kind of messages transmitted
    share: tuple[str, ...]


class ByAttribute(Strict):
    """Dict form of schedule (ex. {by: depth, order: top_down})"""
    by: str
    order: Literal["top_down", "bottom_up"] = "top_down"


class EachHas(Strict):
    """Count Constraint (each A has exactly n agents of role of B)"""
    each: str
    has_exactly: int
    of: str


class ProtocolConfig(Strict):
    """top-level config"""
    roles: dict[str, Role]
    structure: dict[str, Any] = {"generator": "complete"}
    rules: tuple[Rule, ...] = ()
    schedule: Literal["all_at_once"] | tuple[str, ...] | ByAttribute = "all_at_once"
    constraints: tuple[Literal["acyclic", "connected"] | EachHas, ...] = ()
    channels: dict[str, ChannelRule] = {}


def load_protocol(path: str | Path) -> ProtocolConfig:
    """Read and validate a protocol YAML file"""
    return ProtocolConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def agent_ids(n: int) -> list[AgentId]:
    """Generate sorted ids (agent_02 < agent_10)"""
    width = max(2, len(str(n - 1)))
    return [f"agent_{i:0{width}d}" for i in range(n)]


# GRAPH and ROLES

def build_graph(cfg: ProtocolConfig) -> nx.DiGraph:
    """Run the protocol's generator (without an explicit "n", the size is the sum of the role counts)"""
    params = dict(cfg.structure)
    name = params.pop("generator", "complete")
    if name not in GENERATORS:
        raise CompileError(f"unknown generator {name!r}; known: {sorted(GENERATORS)}")
    generator = GENERATORS[name]
    
    # Some generators take a size "n" (complete, path ...), others derive it (tree, subgroups)
    if "n" in inspect.signature(generator).parameters and "n" not in params:
        counts = [role.count for role in cfg.roles.values()]
        if None in counts:
            raise CompileError("roles assigned with \"where\" need an explicit structure size \"n\"")
        params["n"] = sum(counts)
    try:
        return generator(**params)
    except (TypeError, ValueError) as e:
        raise CompileError(f"structure {cfg.structure}: {e}") from e


OPERATORS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt,
             "ge": operator.ge, "in": lambda value, options: value in options}


def selects(attrs: dict[str, Any], where: str | dict[str, Any]) -> bool:
    """Whether a node matches a selector (structured dicts only, never eval)"""
    if isinstance(where, str):
        where = {where: True}
    for name, test in where.items():
        if name not in attrs:
            raise CompileError(f"selector uses {name!r}, but the nodes have only {sorted(attrs)}")
        if not isinstance(test, dict):
            test = {"eq": test}
        for op, arg in test.items():
            if op not in OPERATORS:
                raise CompileError(f"unknown selector operator {op!r}; use one of {sorted(OPERATORS)}")
            if not OPERATORS[op](attrs[name], arg):
                return False
    return True


def assign_roles(g: nx.DiGraph, roles: dict[str, Role]) -> dict[int, str]:
    """Roles with `where` take the nodes they select; roles with `count` then take the remaining nodes in order"""
    assigned: dict[int, str] = {}
    for name, role in roles.items():
        if role.where is None:
            continue
        for node in g:
            if selects(g.nodes[node], role.where):
                # raise error if a node can be assigned to two roles
                if node in assigned:
                    raise CompileError(f"node {node} matches both {assigned[node]!r} and {name!r}")
                assigned[node] = name

    free = [node for node in g if node not in assigned]
    for name, role in roles.items():
        if role.count is None:
            continue
        if role.count > len(free):
            raise CompileError(f"role {name!r} needs {role.count} agents but only {len(free)} nodes are left")
        for node in free[:role.count]:
            assigned[node] = name
        free = free[role.count:]

    if free:
        raise CompileError(f"{len(free)} nodes have no role")
    if missing := set(roles) - set(assigned.values()):
        raise CompileError(f"roles {sorted(missing)} match no node")
    return dict(sorted(assigned.items()))


# Rules


def check_rule(rule: Rule, role_names: set[str], relations: set[str]) -> None:
    """Raise CompileError if the rule names an unknown role, relation or message type"""
    if {rule.from_, rule.to} - {None} - role_names:
        raise CompileError(f"rule {rule} names an unknown role")
    if rule.along is not None and rule.along not in relations:
        raise CompileError(f"rule {rule}: the structure has no relation {rule.along!r} (it has {sorted(relations)})")
    if unknown := [t for t in rule.share if not is_known(t)]:
        raise CompileError(f"rule {rule} shares unknown message types {unknown}")


def edge_matches(rule: Rule, src_role: str, dst_role: str, relation: str | None) -> bool:
    """Whether a rule keeps an edge; a key missing from the rule matches anything"""
    return (rule.from_ in (None, src_role)
            and rule.to in (None, dst_role)
            and rule.along in (None, relation))


def compile_rules(
    cfg: ProtocolConfig, g: nx.DiGraph, roles: dict[int, str], ids: list[AgentId]
) -> tuple[dict[tuple[AgentId, AgentId], frozenset[str]], dict[str, frozenset[str]]]:
    """The links (src, dst) -> types they carry, and the types each role may publish"""
    relations = {relation for _, _, relation in g.edges(data="relation")}
    links: dict[tuple[AgentId, AgentId], frozenset[str]] = defaultdict(frozenset)
    shares: dict[str, frozenset[str]] = defaultdict(frozenset)
    for rule in cfg.rules:
        check_rule(rule, set(cfg.roles), relations)
        share = frozenset(rule.share)
        kept = [(u, v) for u, v, relation in g.edges(data="relation")
                if edge_matches(rule, roles[u], roles[v], relation)]
        if share:  # a rule that shares nothing adds no link
            for u, v in kept:
                links[ids[u], ids[v]] |= share
        publishers = {rule.from_} if rule.from_ else {roles[u] for u, _ in kept}
        for role in publishers:
            shares[role] |= share
    return dict(links), dict(shares)


def check_channels(cfg: ProtocolConfig, ids: list[AgentId]) -> None:
    """Raise CompileError if a channel rule names an unknown message type or agent"""
    for name, channel in cfg.channels.items():
        if not is_known(name):
            raise CompileError(f"channel rule for unknown message type {name!r}")
        if isinstance(channel.visible_to, tuple) and not set(channel.visible_to) <= set(ids):
            raise CompileError(f"channel {name!r} is visible to unknown agents")


# Schedule

def build_phases(schedule, g: nx.DiGraph, roles: dict[int, str], ids: list[AgentId]) -> list[list[AgentId]]:
    """Returns a list of phases (sheduling): each phase is a list of agent ids that act together"""
    if schedule == "all_at_once":
        return [ids]

    if isinstance(schedule, tuple):
        if set(schedule) != set(roles.values()):
            raise CompileError(f"schedule {list(schedule)} must name every role, and only roles")
        return [[ids[node] for node, r in roles.items() if r == role] for role in schedule]

    attr = "index" if schedule.by == "position" else schedule.by
    if any(attr not in g.nodes[node] for node in g):
        raise CompileError(f"schedule by {schedule.by!r}: the nodes have no attribute {attr!r}")
    values = sorted({g.nodes[node][attr] for node in g}, reverse=schedule.order == "bottom_up")
    return [[ids[node] for node in g if g.nodes[node][attr] == value] for value in values]


# Constraints

def link_graph(s: Structure, msg_type: str | None = None) -> nx.DiGraph:
    """Rebuilds the graph from s.links, or given a type, it keeps only the links that carry that typere"""
    g = nx.DiGraph()
    g.add_nodes_from(s.agents)
    g.add_edges_from(pair for pair, types in s.links.items()
                     if msg_type is None or any(covers(t, msg_type) for t in types))
    return g


def check_acyclic(s: Structure) -> None:
    for msg_type in MESSAGE_TYPES:
        if not nx.is_directed_acyclic_graph(link_graph(s, msg_type)):
            raise CompileError(f"constraint acyclic: the {msg_type!r} links have a cycle")


def check_connected(s: Structure) -> None:
    if not nx.is_weakly_connected(link_graph(s)):
        raise CompileError("constraint connected: the links do not connect every agent")


def check_each_has(c: EachHas, s: Structure, role_names: set[str]) -> None:
    if {c.each, c.of} - role_names:
        raise CompileError(f"constraint {c.model_dump()} names an unknown role")
    g = link_graph(s)
    for agent, role in s.agents.items():
        if role != c.each:
            continue
        count = sum(s.agents[other] == c.of for other in set(nx.all_neighbors(g, agent)))
        if count != c.has_exactly:
            raise CompileError(f"constraint {c.model_dump()}: {agent} has {count}")


def check_constraints(cfg: ProtocolConfig, s: Structure) -> None:
    """Raise CompileError if the compiled structure breaks one of the protocol's constraints"""
    for c in cfg.constraints:
        if c == "acyclic":
            check_acyclic(s)
        elif c == "connected":
            check_connected(s)
        else:
            check_each_has(c, s, set(cfg.roles))

def compile_protocol(cfg: ProtocolConfig) -> tuple[Structure, list[list[AgentId]]]:
    """Start compiling"""
    g = build_graph(cfg)
    roles = assign_roles(g, cfg.roles)
    ids = agent_ids(len(g))
    links, shares = compile_rules(cfg, g, roles, ids)
    check_channels(cfg, ids)
    structure = Structure({ids[node]: role for node, role in roles.items()}, links, shares, dict(cfg.channels))
    check_constraints(cfg, structure)
    return structure, build_phases(cfg.schedule, g, roles, ids)
