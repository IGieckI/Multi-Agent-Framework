# mixed (mix-qgl) vs main, mixed_hints vs hints2 (same targets and notes), F46-: mixed - each single-model team,
# paired by target; for silenced teams also the success expected from the single-model runs' agents (agents are
# independent without links, so a mixed silenced team is a draw of 2 (FC) or 1 (star) agents per model)
import sys, json, glob
from math import comb
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import *
M3 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
SH = json.load(open('docs/analysis/short200.json'))
def solvers(run):
    meta = json.load(open(run + '/meta.json')); t = meta['task']['target']; s = set()
    for l in open(run + '/events.jsonl'):
        e = json.loads(l)
        if e['kind'] == 'env_step' and t in e['data']['new_items']: s.add(e['data']['agent'])
    return len(s), len(meta['agents']) if isinstance(meta['agents'], list) else len(meta['agents'])
def expected(cell_of, R, per):
    """per target: 1 - prod over models of P(no solver among `per` agents drawn from that model's silenced run)"""
    out = {}
    for r in R:
        q = 1.0
        for m in M3:
            k, n = solvers(cell_of(m, r)); q *= comb(n - k, per) / comb(n, per)
        out[r] = 1 - q
    return out
def section(title, mx, sg, cells, R, cellpath):
    print(f"\n==== {title}")
    for c in cells:
        for v in ['all_independent', 'base']:
            a = {r: f(mx[c + (str(r), v)]['solved_any']) for r in R}
            line = f"  {'/'.join(c):40s} {v:16s} mixed {100*mean(list(a.values())):5.1f}"
            best = None
            for m in M3:
                b = {r: f(sg[(m,) + c + (str(r), v)]['solved_any']) for r in R}
                line += f" | {m.split('-')[0][:5]} {100*mean(list(b.values())):5.1f} {fmt(*boot([a[r]-b[r] for r in R], cluster=list(R)), d=0)}"
            print(line)
            if v == 'all_independent':
                per = 2 if c[0] == 'fully_connected' else 1
                e = expected(lambda m, r: cellpath(m, c, v, r), R, per)
                print(f"  {'':40s} {'':16s} expected from single-model agents {100*mean(list(e.values())):5.1f}; mixed - expected {fmt(*boot([a[r]-e[r] for r in R], cluster=list(R)))}")
        dd = [f(mx[c + (str(r), 'base')]['solved_any']) - f(mx[c + (str(r), 'all_independent')]['solved_any']) for r in R]
        print(f"  {'/'.join(c):40s} mixed talk - silenced {fmt(*boot(dd, cluster=list(R)))}")
KM = ('protocol', 'depth', 'distractors', 'replicate', 'version')
MX = index(load('run_mixed/mixed/runs.csv'), KM); MN = index(load('run_main/main/runs.csv'), ('model',) + KM)
cells = [(p, d, x) for p in ['fully_connected', 'star'] for d, x in [('1', '3'), ('2', '0')]]
section('mixed vs main (20 targets per setting)', MX, MN, cells, range(20),
        lambda m, c, v, r: f"run_main/main/{m}__{c[0]}__depth{c[1]}__distractors{c[2]}/{v}/seed0_rep{r}")
# pooled over the two settings
for p in ['fully_connected', 'star']:
    for v in ['all_independent', 'base']:
        out = f"  POOLED {p:16s} {v:16s}"
        for m in M3:
            dd, cc = [], []
            for d, x in [('1', '3'), ('2', '0')]:
                for r in range(20):
                    dd.append(f(MX[(p, d, x, str(r), v)]['solved_any']) - f(MN[(m, p, d, x, str(r), v)]['solved_any'])); cc.append(d + str(r))
            out += f" | -{m.split('-')[0][:5]} {fmt(*boot(dd, cluster=cc))}"
        print(out)
KH = ('protocol', 'hints', 'decoys', 'replicate', 'version')
MH = index(load('run_mixed_hints/mixed_hints/runs.csv'), KH); H2 = index(load('run_hints2/hints2/runs.csv'), ('model',) + KH)
cells = [(p, h, 'True') for p in ['fully_connected', 'star'] for h in ['split', 'all']]
section('mixed_hints vs hints2 (40 targets)', MH, H2, cells, range(40),
        lambda m, c, v, r: f"run_hints2/hints2/{m}__{c[0]}__hints{c[1]}__decoys{c[2]}/{v}/seed0_rep{r}")
