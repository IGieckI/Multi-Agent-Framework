"""
Builds the starting graphs for the communication structure
"""
import networkx as nx

def _nodes(n: int) -> nx.DiGraph:
    """generate n nodes and no edges"""
    if n < 1:
        raise ValueError(f"a structure needs at least 1 agent, got {n}")
    g = nx.DiGraph()
    g.add_nodes_from((i, {"index": i}) for i in range(n))
    return g


def complete(n: int) -> nx.DiGraph:
    """Generate a graph where every node is linked to every other both ways"""
    g = _nodes(n)
    g.add_edges_from(((u, v) for u in g for v in g if u != v), relation="peer")
    return g


def empty(n: int) -> nx.DiGraph:
    """No edges"""
    return _nodes(n)


def path(n: int, directed: bool = False) -> nx.DiGraph:
    """Generate a chain graph"""
    g = _nodes(n)
    g.add_edges_from(((i, i + 1) for i in range(n - 1)), relation="next")
    if not directed:
        g.add_edges_from(((i + 1, i) for i in range(n - 1)), relation="prev")
    return g


def star(n: int) -> nx.DiGraph:
    """Generate an hub (node 0) linked to every other node graph"""
    g = _nodes(n)
    for i in g:
        g.nodes[i].update(hub=i == 0, depth=min(i, 1))
    g.add_edges_from(((i, 0) for i in range(1, n)), relation="to_hub")
    g.add_edges_from(((0, i) for i in range(1, n)), relation="from_hub")
    return g


def tree(branching: int, depth: int) -> nx.DiGraph:
    """Generate a tree graph (parent with branching children), numbered through BFS"""
    g = _nodes(sum(branching**k for k in range(depth + 1)))
    g.nodes[0]["depth"] = 0
    for i in range(1, len(g)):
        parent = (i - 1) // branching
        g.nodes[i]["depth"] = g.nodes[parent]["depth"] + 1
        g.add_edge(parent, i, relation="down")
        g.add_edge(i, parent, relation="up")
    for i in g:
        g.nodes[i]["leaf"] = g.nodes[i]["depth"] == depth
    return g


def subgroups(sizes: list[int]) -> nx.DiGraph:
    """Generate a separate groups of given sizes graph"""
    g = _nodes(sum(sizes))
    groups = [group for group, size in enumerate(sizes) for _ in range(size)]
    for i in g:
        g.nodes[i]["group"] = groups[i]
    g.add_edges_from(((u, v) for u in g for v in g if u != v and groups[u] == groups[v]), relation="peer")
    return g


GENERATORS = {"complete": complete, "empty": empty, "path": path, "star": star, "tree": tree, "subgroups": subgroups}
