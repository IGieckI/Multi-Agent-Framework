# hints2_made vs hints2 / hints2_8b (same targets, notes, seeds; only the "Items you made" line differs), F46-:
# success per cell, made - not made paired by target, and the share of actions that remake an intermediate already held
import sys, json, glob
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import *
K = ('model', 'protocol', 'hints', 'decoys', 'replicate', 'version')
MD = index(load('run_hints2_made/hints2_made/runs.csv'), K)
H2 = index(load('run_hints2/hints2/runs.csv'), K); H8 = index(load('run_hints2_8b/hints2_8b/runs.csv'), K)
M4 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b', 'qwen3-8b']
def get(src, m, p, h, v, r, col='solved_any'):
    row = src.get((m, p, h, 'True', str(r), v)); return f(row[col]) if row else None
def remake(folder):
    xs = []
    for run in sorted(glob.glob(folder + '/seed0_rep*')):
        if int(run.rsplit('rep', 1)[1]) >= 40: continue
        meta = json.load(open(run + '/meta.json')); a, b, inter = meta['task']['path'][0]; have = defaultdict(set)
        for l in open(run + '/events.jsonl'):
            e = json.loads(l)
            if e['kind'] != 'env_step': continue
            ag, c = e['data']['agent'], sorted(e['data']['combo'])
            if inter in have[ag]: xs.append(float(c == sorted([a, b])))
            have[ag] |= set(e['data']['new_items'])
    return mean(xs), len(xs)
CELLS = [('fully_connected', 'all_independent'), ('fully_connected', 'base'), ('star', 'all_independent'), ('star', 'base'), ('fully_connected', 'one_agent')]
for h in ['all', 'split']:
    print(f"\n==== notes {h} (decoys), 40 targets: not made -> made, made - not made [95%], remake share not made -> made")
    pool = defaultdict(lambda: ([], []))
    for m in M4:
        base = H8 if m == 'qwen3-8b' else H2; root = 'run_hints2_8b/hints2_8b' if m == 'qwen3-8b' else 'run_hints2/hints2'
        for p, v in CELLS:
            a = [get(MD, m, p, h, v, r) for r in range(40)]; b = [get(base, m, p, h, v, r) for r in range(40)]
            dd = [x - y for x, y in zip(a, b)]
            for x, y, r in zip(a, b, range(40)):
                if m != 'qwen3-8b': pool[(p, v)][0].append(x - y); pool[(p, v)][1].append(r)
            r0, n0 = remake(f'{root}/{m}__{p}__hints{h}__decoysTrue/{v}'); r1, n1 = remake(f'run_hints2_made/hints2_made/{m}__{p}__hints{h}__decoysTrue/{v}')
            rr = f"{100*r0:4.0f}% -> {100*r1:4.0f}% (n {n0}/{n1})" if r0 is not None and r1 is not None else ''
            print(f"  {m:12s} {p:16s} {v:16s} {100*mean(b):5.1f} -> {100*mean(a):5.1f}  {fmt(*boot(dd, cluster=list(range(40))))}   {rr}")
    for (p, v), (dd, cc) in pool.items():
        print(f"  three 4B pooled {p:16s} {v:16s} {fmt(*boot(dd, cluster=cc))} n={len(dd)}")
print("\n==== talk - silenced with the made line (notes all, decoys)")
for m in M4:
    for p in ['fully_connected', 'star']:
        dd = [get(MD, m, p, 'all', 'base', r) - get(MD, m, p, 'all', 'all_independent', r) for r in range(40)]
        print(f"  {m:12s} {p:16s} {fmt(*boot(dd, cluster=list(range(40))))}")
