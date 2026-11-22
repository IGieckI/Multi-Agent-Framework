# hints2_12b, hints2_12b_share, hints2_8b on 80 targets (F46-): rates and paired contrasts, with hints2 / share / use
import sys; sys.path.insert(0, sys.argv[1])
from lib import *
K = ('model', 'protocol', 'hints', 'decoys', 'replicate', 'version')
def split_use(path):
    B = load(path); return (index([r for r in B if r['use_notes'] == 'False'], K), index([r for r in B if r['use_notes'] == 'True'], K))
src = {'h2': index(load('run_hints2/hints2/runs.csv'), K), 'share': index(load('run_hints2_share/hints2_share/runs.csv'), K),
       'use': index(load('run_hints2_use/hints2_use/runs.csv'), K),
       '12b': index(load('run_hints2_12b/hints2_12b/runs.csv'), K), '8b': index(load('run_hints2_8b/hints2_8b/runs.csv'), K)}
src['12b_share'], src['12b_use'] = split_use('run_hints2_12b_share/hints2_12b_share/runs.csv')
src['8b_share'], src['8b_use'] = split_use('run_hints2_8b_share/hints2_8b_share/runs.csv')
FC, ST = 'fully_connected', 'star'
def G(s, m, p, h, d, v, col='solved_any'):
    def g(r):
        row = src[s].get((m, p, h, d, str(r), v)); return f(row[col]) if row else None
    return g
def rate(fn, R):
    xs = [fn(r) for r in R]; xs = [x for x in xs if x is not None]; return f"{100*mean(xs):5.1f}" if xs else "  -  "
def con(a, b, R):
    dd, cc = [], []
    for r in R:
        x, y = a(r), b(r)
        if x is None or y is None: continue
        dd.append(x - y); cc.append(r)
    return fmt(*boot(dd, cluster=cc)) + f" n={len(dd)}"
R40, R80, RNEW = range(40), range(80), range(40, 80)
CELLS = [('split', 'True'), ('split', 'False'), ('all', 'True')]
def table(name, s, m, s_sh, s_us, R):
    print(f"\n==== {name}")
    for h, d in CELLS:
        row = [rate(G(s, m, FC, h, d, 'one_agent'), R)]
        for p in (FC, ST): row += [rate(G(s, m, p, h, d, 'all_independent'), R), rate(G(s, m, p, h, d, 'base'), R)]
        extra = '' if h == 'all' or not s_sh else f"  share {rate(G(s_sh, m, FC, h, d, 'base'), R)}  use {rate(G(s_us, m, FC, h, d, 'base'), R)}"
        print(f"  {h:5s} dec={d:5s} one {row[0]}  FC mute {row[1]} talk {row[2]}  ST mute {row[3]} talk {row[4]}{extra}")
    for h, d in CELLS:
        for p in (FC, ST):
            print(f"  talk-mute {h:5s} dec={d:5s} {p:16s} {con(G(s, m, p, h, d, 'base'), G(s, m, p, h, d, 'all_independent'), R)}")
    if s_sh:
        for d in ('True', 'False'):
            t = G(s, m, FC, 'split', d, 'base'); sh = G(s_sh, m, FC, 'split', d, 'base'); us = G(s_us, m, FC, 'split', d, 'base')
            print(f"  dec={d:5s} share-talk {con(sh, t, R)}   use-share {con(us, sh, R)}   use-one both {con(us, G(s, m, FC, 'split', d, 'one_agent'), R)}")
table('gemma3-12b (40 targets)', '12b', 'gemma3-12b', '12b_share', '12b_use', R40)
table('gemma3-4b (hints2, same 40 targets)', 'h2', 'gemma3-4b', 'share', 'use', R40)
table('qwen3-8b, targets 0-39', '8b', 'qwen3-8b', '8b_share', '8b_use', R40)
table('qwen3-8b, targets 40-79 (new)', '8b', 'qwen3-8b', '8b_share', '8b_use', RNEW)
table('qwen3-8b, 80 targets', '8b', 'qwen3-8b', '8b_share', '8b_use', R80)
print("\n==== gemma3-12b - gemma3-4b and - qwen3-8b, same 40 targets")
for s4, s12, s8, p, h, d, v in [('h2', '12b', '8b', FC, 'split', 'True', 'one_agent'), ('h2', '12b', '8b', FC, 'split', 'True', 'all_independent'),
        ('h2', '12b', '8b', FC, 'split', 'True', 'base'), ('h2', '12b', '8b', ST, 'split', 'True', 'base'),
        ('h2', '12b', '8b', FC, 'split', 'False', 'base'), ('h2', '12b', '8b', FC, 'all', 'True', 'base'),
        ('share', '12b_share', '8b_share', FC, 'split', 'True', 'base'), ('use', '12b_use', '8b_use', FC, 'split', 'True', 'base')]:
    a = G(s12, 'gemma3-12b', p, h, d, v)
    print(f"  {s12:9s} {p:16s} {h:5s} dec={d:5s} {v:16s} -4b {con(a, G(s4, 'gemma3-4b', p, h, d, v), R40)}   -8b {con(a, G(s8, 'qwen3-8b', p, h, d, v), R40)}")
