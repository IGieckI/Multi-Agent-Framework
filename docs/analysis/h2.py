import sys; sys.path.insert(0, sys.argv[1])
from lib import *
rows = load('run_hints/hints/runs.csv')
I = index(rows)
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
def get(m, p, h, rep, v, col='solved_any'):
    r = I.get((m, p, '2', '6', h, str(rep), v)); return f(r[col]) if r else None
IND = {'fully_connected': 'fully_connected', 'fully_connected_results': 'fully_connected', 'fully_connected_messages': 'fully_connected', 'fully_connected_failures': 'fully_connected', 'star': 'star'}
def contrast(a, b, models=LLM, col='solved_any'):
    """a, b: functions (m, rep) -> value; paired over models x targets, resampled by target"""
    d, c = [], []
    for m in models:
        for rep in range(20):
            x, y = a(m, rep), b(m, rep)
            if x is None or y is None: continue
            d.append(x - y); c.append(rep)
    return fmt(*boot(d, cluster=c)), len(d)
def per_model(a, b):
    return ' '.join(f"{m.split('-')[0][:5]} {100*mean([a(m,r)-b(m,r) for r in range(20)]):+.0f}" for m in LLM)
print('A. talk vs no links, by hints setting (base - all_independent of the same structure)')
for h in ['none', 'split', 'all']:
    for p in IND:
        a = lambda m, r, p=p, h=h: get(m, p, h, r, 'base'); b = lambda m, r, p=p, h=h: get(m, IND[p], h, r, 'all_independent')
        print(f"  {h:5s} {p:26s} {contrast(a, b)[0]}   {per_model(a, b)}")
print('B. information: all_independent with notes all/split vs none')
for p in ['fully_connected', 'star']:
    for h1, h2 in [('split', 'none'), ('all', 'split'), ('all', 'none')]:
        a = lambda m, r, p=p, h1=h1: get(m, p, h1, r, 'all_independent'); b = lambda m, r, p=p, h2=h2: get(m, p, h2, r, 'all_independent')
        print(f"  {p:16s} ind {h1} - ind {h2}: {contrast(a, b)[0]}   {per_model(a, b)}")
print('C. talk with split notes vs the team where all know (no links): head-room left')
for p in IND:
    a = lambda m, r, p=p: get(m, p, 'split', r, 'base'); b = lambda m, r, p=p: get(m, IND[p], 'all', r, 'all_independent')
    print(f"  {p:26s} base(split) - ind(all): {contrast(a, b)[0]}")
print('D. one agent with both recipes vs teams with split notes')
for p in IND:
    a = lambda m, r: get(m, 'fully_connected', 'split', r, 'one_agent'); b = lambda m, r, p=p: get(m, p, 'split', r, 'base')
    print(f"  one_agent(split) - {p:26s} base(split): {contrast(a, b)[0]}   {per_model(a, b)}")
a = lambda m, r: get(m, 'fully_connected', 'split', r, 'one_agent'); b = lambda m, r: get(m, 'fully_connected', 'split', r, 'all_independent')
print(f"  one_agent(split) - fc ind(split): {contrast(a, b)[0]}")
a = lambda m, r: get(m, 'fully_connected', 'all', r, 'all_independent'); b = lambda m, r: get(m, 'fully_connected', 'all', r, 'one_agent')
print(f"  fc ind(all) - one_agent(all): {contrast(a, b)[0]}")
a = lambda m, r: get(m, 'fully_connected', 'all', r, 'one_agent'); b = lambda m, r: get(m, 'fully_connected', 'split', r, 'one_agent')
print(f"  replicate check one_agent(all) - one_agent(split) (same information): {contrast(a, b)[0]}")
print('E. pooled over the 4 fully connected kinds + star, split, base - ind:')
d, c = [], []
for p in IND:
    for m in LLM:
        for r in range(20):
            d.append(get(m, p, 'split', r, 'base') - get(m, IND[p], 'split', r, 'all_independent')); c.append(r)
print('  ', fmt(*boot(d, cluster=c)), len(d))
