# hints2_use, hints2_8b, hints2_8b_share: rates and paired contrasts (F42-). Pairs with hints2 and hints2_share (k1.py)
import sys; sys.path.insert(0, sys.argv[1])
from lib import *
K = ('model', 'protocol', 'hints', 'decoys', 'replicate', 'version')
src = {
    'h2': index(load('run_hints2/hints2/runs.csv'), K),
    'share': index(load('run_hints2_share/hints2_share/runs.csv'), K),
    'use': index(load('run_hints2_use/hints2_use/runs.csv'), K),
    '8b': index(load('run_hints2_8b/hints2_8b/runs.csv'), K),
}
B = load('run_hints2_8b_share/hints2_8b_share/runs.csv')
src['8b_share'] = index([r for r in B if r['use_notes'] == 'False'], K)
src['8b_use'] = index([r for r in B if r['use_notes'] == 'True'], K)
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
FC = 'fully_connected'
def G(s, p, h, d, v, col='solved_any'):
    """getter (model, replicate) -> value; s: which grid; for 8b grids the model is forced to qwen3-8b"""
    def g(m, r):
        key = ('qwen3-8b' if s.startswith('8b') else m, p, h, d, str(r), v)
        row = src[s].get(key); return f(row[col]) if row else None
    return g
def rate(fn, models):
    xs = [fn(m, r) for m in models for r in range(40)]; xs = [x for x in xs if x is not None]
    return f"{100 * mean(xs):5.1f} (n={len(xs)})"
def contrast(a, b, models):
    dd, cc = [], []
    for m in models:
        for r in range(40):
            x, y = a(m, r), b(m, r)
            if x is None or y is None: continue
            dd.append(x - y); cc.append(r)
    return fmt(*boot(dd, cluster=cc)) + f" (n={len(dd)})"
def pmodel(a, b):
    return '  '.join(f"{m.split('-')[0][:5]} {100*mean([a(m, r) - b(m, r) for r in range(40)]):+.0f}" for m in LLM)
def pm_rate(fn):
    return '  '.join(f"{m.split('-')[0][:5]} {100*mean([fn(m, r) for r in range(40)]):.0f}" for m in LLM)

for d in ['True', 'False']:
    print(f"\n==== 4B models, fully connected, notes split, decoys={d}")
    C = {'one agent holding both': G('h2', FC, 'split', d, 'one_agent'),
         'silenced': G('h2', FC, 'split', d, 'all_independent'),
         'talk': G('h2', FC, 'split', d, 'base'),
         'talk + share': G('share', FC, 'split', d, 'base'),
         'talk + share + use': G('use', FC, 'split', d, 'base'),
         'everyone holds both, silenced': G('h2', FC, 'all', 'True', 'all_independent')}
    for n, fn in C.items(): print(f"  {n:32s} {rate(fn, LLM)}   {pm_rate(fn)}")
    u = C['talk + share + use']
    for n in ['talk + share', 'talk', 'silenced', 'one agent holding both', 'everyone holds both, silenced']:
        print(f"  use - {n:32s} {contrast(u, C[n], LLM)}   {pmodel(u, C[n])}")

print("\n==== qwen3-8b (hints2_8b, hints2_8b_share)")
Q = ['qwen3-8b']
for h, d in [('split', 'True'), ('split', 'False'), ('all', 'True')]:
    for p in [FC, 'star']:
        for v in ['one_agent', 'all_independent', 'base']:
            if v == 'one_agent' and p == 'star': continue
            print(f"  {h:5s} decoys={d:5s} {p:16s} {v:16s} {rate(G('8b', p, h, d, v), Q)}")
    if h == 'split':
        print(f"  {h:5s} decoys={d:5s} {FC:16s} {'base + share':16s} {rate(G('8b_share', FC, h, d, 'base'), Q)}")
        print(f"  {h:5s} decoys={d:5s} {FC:16s} {'base + share+use':16s} {rate(G('8b_use', FC, h, d, 'base'), Q)}")
print("  -- contrasts (40 targets)")
for h, d in [('split', 'True'), ('split', 'False'), ('all', 'True')]:
    for p in [FC, 'star']:
        print(f"  talk - silenced {h} decoys={d} {p:16s} {contrast(G('8b', p, h, d, 'base'), G('8b', p, h, d, 'all_independent'), Q)}")
for d in ['True', 'False']:
    sh, us = G('8b_share', FC, 'split', d, 'base'), G('8b_use', FC, 'split', d, 'base')
    print(f"  decoys={d}: share - talk {contrast(sh, G('8b', FC, 'split', d, 'base'), Q)}")
    print(f"  decoys={d}: use - share {contrast(us, sh, Q)}")
    print(f"  decoys={d}: use - talk {contrast(us, G('8b', FC, 'split', d, 'base'), Q)}")
    print(f"  decoys={d}: use - one agent both {contrast(us, G('8b', FC, 'split', d, 'one_agent'), Q)}")
    print(f"  decoys={d}: use - everyone both silenced {contrast(us, G('8b', FC, 'all', 'True', 'all_independent'), Q)}")
    print(f"  decoys={d}: everyone both silenced - split silenced {contrast(G('8b', FC, 'all', 'True', 'all_independent'), G('8b', FC, 'split', d, 'all_independent'), Q)}")
    print(f"  decoys={d}: one agent both - talk {contrast(G('8b', FC, 'split', d, 'one_agent'), G('8b', FC, 'split', d, 'base'), Q)}")
for p, v in [(FC, 'all_independent'), ('star', 'all_independent'), (FC, 'base'), ('star', 'base')]:
    print(f"  decoys: without - with {p} {v} {contrast(G('8b', p, 'split', 'False', v), G('8b', p, 'split', 'True', v), Q)}")
print(f"  6 silenced all - one agent all {contrast(G('8b', FC, 'all', 'True', 'all_independent'), G('8b', FC, 'all', 'True', 'one_agent'), Q)}")

print("\n==== qwen3-8b - qwen3-4b, same targets")
pairs = [('h2', '8b', FC, 'split', 'True', 'one_agent'), ('h2', '8b', FC, 'split', 'True', 'all_independent'),
         ('h2', '8b', FC, 'split', 'True', 'base'), ('h2', '8b', 'star', 'split', 'True', 'base'),
         ('h2', '8b', FC, 'split', 'False', 'base'), ('h2', '8b', FC, 'all', 'True', 'all_independent'),
         ('h2', '8b', FC, 'all', 'True', 'base'),
         ('share', '8b_share', FC, 'split', 'True', 'base'), ('share', '8b_share', FC, 'split', 'False', 'base'),
         ('use', '8b_use', FC, 'split', 'True', 'base'), ('use', '8b_use', FC, 'split', 'False', 'base')]
for s4, s8, p, h, d, v in pairs:
    a = G(s8, p, h, d, v); b = G(s4, p, h, d, v)
    print(f"  {s8:9s} {p:16s} {h:5s} decoys={d:5s} {v:16s} 8b {rate(a, Q)}  4b {rate(b, ['qwen3-4b'])}  diff {contrast(a, b, ['qwen3-4b'])}")
