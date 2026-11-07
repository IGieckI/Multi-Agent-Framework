import sys; sys.path.insert(0, sys.argv[1])
from lib import *
H = index(load('run_hints2/hints2/runs.csv'), keys=('model', 'protocol', 'hints', 'decoys', 'replicate', 'version'))
SH = index(load('run_hints2_share/hints2_share/runs.csv'), keys=('model', 'protocol', 'hints', 'decoys', 'replicate', 'version'))
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
SET = [('split', 'True'), ('split', 'False'), ('all', 'True')]
def g(m, p, h, d, r, v, col='solved_any', share=False):
    row = (SH if share else H).get((m, p, h, d, str(r), v)); return f(row[col]) if row else None
def rate(fn, models=LLM): 
    xs = [fn(m, r) for m in models for r in range(40)]; return round(100 * mean(xs), 1)
print('== success, LLMs pooled (120 runs) | per model')
for h, d in SET:
    for p, v in [('fully_connected', 'one_agent'), ('fully_connected', 'all_independent'), ('star', 'all_independent'), ('fully_connected', 'base'), ('star', 'base')]:
        pm = ' '.join(f"{m.split('-')[0][:5]} {rate(lambda mm, r: g(mm, p, h, d, r, v), [m]):5.1f}" for m in LLM)
        print(f"  {h:5s} decoys={d:5s} {p:16s} {v:16s} {rate(lambda m, r: g(m, p, h, d, r, v)):5.1f}   {pm}   random {rate(lambda m, r: g(m, p, h, d, r, v), ['mock_norepeat'])}")
    if h == 'split':
        pm = ' '.join(f"{m.split('-')[0][:5]} {rate(lambda mm, r: g(mm, 'fully_connected', h, d, r, 'base', share=True), [m]):5.1f}" for m in LLM)
        print(f"  {h:5s} decoys={d:5s} {'fully_connected':16s} {'base + SHARE':16s} {rate(lambda m, r: g(m, 'fully_connected', h, d, r, 'base', share=True)):5.1f}   {pm}")
def contrast(a, b, models=LLM):
    dd, cc = [], []
    for m in models:
        for r in range(40):
            x, y = a(m, r), b(m, r)
            if x is None or y is None: continue
            dd.append(x - y); cc.append(r)
    return fmt(*boot(dd, cluster=cc)) + f"  (n={len(dd)})"
def pmodel(a, b):
    return ' '.join(f"{m.split('-')[0][:5]} {100*mean([a(m, r) - b(m, r) for r in range(40)]):+.0f}" for m in LLM)
print('== talk vs silenced')
for h, d in SET:
    for p in ['fully_connected', 'star']:
        a = lambda m, r, p=p, h=h, d=d: g(m, p, h, d, r, 'base'); b = lambda m, r, p=p, h=h, d=d: g(m, p, h, d, r, 'all_independent')
        print(f"  {h} decoys={d} {p:16s} base - ind: {contrast(a, b)}   {pmodel(a, b)}")
print('== share instruction (fully_connected)')
for d in ['True', 'False']:
    sh = lambda m, r, d=d: g(m, 'fully_connected', 'split', d, r, 'base', share=True)
    for name, b in [('share - talk (no instruction)', lambda m, r, d=d: g(m, 'fully_connected', 'split', d, r, 'base')),
                    ('share - silenced', lambda m, r, d=d: g(m, 'fully_connected', 'split', d, r, 'all_independent')),
                    ('share - everyone holds both (silenced)', lambda m, r: g(m, 'fully_connected', 'all', 'True', r, 'all_independent')),
                    ('share - one agent holding both', lambda m, r, d=d: g(m, 'fully_connected', 'split', d, r, 'one_agent'))]:
        print(f"  decoys={d:5s} {name:40s} {contrast(sh, b)}   {pmodel(sh, b)}")
print('== decoys (without - with)')
for p, v in [('fully_connected', 'all_independent'), ('star', 'all_independent'), ('fully_connected', 'base'), ('star', 'base'), ('fully_connected', 'one_agent')]:
    a = lambda m, r, p=p, v=v: g(m, p, 'split', 'False', r, v); b = lambda m, r, p=p, v=v: g(m, p, 'split', 'True', r, v)
    print(f"  {p:16s} {v:16s} {contrast(a, b)}   {pmodel(a, b)}")
a = lambda m, r: g(m, 'fully_connected', 'split', 'False', r, 'base', share=True); b = lambda m, r: g(m, 'fully_connected', 'split', 'True', r, 'base', share=True)
print(f"  share            base            {contrast(a, b)}")
print('== information')
for p in ['fully_connected', 'star']:
    for d in ['True', 'False']:
        a = lambda m, r, p=p: g(m, p, 'all', 'True', r, 'all_independent'); b = lambda m, r, p=p, d=d: g(m, p, 'split', d, r, 'all_independent')
        print(f"  {p:16s} ind all - ind split(decoys={d}) {contrast(a, b)}")
    a = lambda m, r: g(m, 'fully_connected', 'split', 'True', r, 'one_agent'); b = lambda m, r, p=p: g(m, p, 'split', 'True', r, 'base')
    print(f"  one agent holding both - {p} talk (split, decoys) {contrast(a, b)}")
a = lambda m, r: g(m, 'fully_connected', 'all', 'True', r, 'all_independent'); b = lambda m, r: g(m, 'fully_connected', 'all', 'True', r, 'one_agent')
print(f"  6 silenced all - one agent all {contrast(a, b)}")
a = lambda m, r: g(m, 'fully_connected', 'all', 'True', r, 'one_agent'); b = lambda m, r: g(m, 'fully_connected', 'split', 'True', r, 'one_agent')
print(f"  check one_agent all - one_agent split (same information) {contrast(a, b)}")
a = lambda m, r: g(m, 'fully_connected', 'split', 'False', r, 'one_agent'); b = lambda m, r: g(m, 'fully_connected', 'split', 'True', r, 'one_agent')
print(f"  check one_agent split nodecoys - decoys (same information) {contrast(a, b)}")
