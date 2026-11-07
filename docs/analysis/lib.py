import csv, random
from collections import defaultdict

def load(path):
    return list(csv.DictReader(open(path)))

def f(x):
    return float(x) if x not in ('', None) else None

def index(rows, keys=('model', 'protocol', 'depth', 'distractors', 'hints', 'replicate', 'version')):
    return {tuple(r.get(k, '') for k in keys): r for r in rows}

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def boot(diffs, n=4000, seed=0, cluster=None):
    """mean and 95% bootstrap interval of paired differences; cluster: list of cluster ids to resample by"""
    rng = random.Random(seed)
    if cluster is None:
        outs = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    else:
        groups = defaultdict(list)
        for d, c in zip(diffs, cluster): groups[c].append(d)
        ks = list(groups)
        outs = []
        for _ in range(n):
            s = [d for k in rng.choices(ks, k=len(ks)) for d in groups[k]]
            outs.append(sum(s) / len(s))
        outs.sort()
    return sum(diffs) / len(diffs), outs[int(0.025 * n)], outs[int(0.975 * n)]

def fmt(m, lo, hi, scale=100, d=1):
    return f"{m*scale:+.{d}f} [{lo*scale:+.{d}f}, {hi*scale:+.{d}f}]"
