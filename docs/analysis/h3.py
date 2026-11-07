import sys, json; sys.path.insert(0, sys.argv[1])
from lib import *
sc = json.load(open(sys.argv[1] + '/shortcuts.json'))
rows = load('run_hints/hints/runs.csv'); I = index(rows)
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
short = {int(r) for r, v in sc['hints'].items() if v}
def get(m, p, h, rep, v, col='solved_any'):
    r = I.get((m, p, '2', '6', h, str(rep), v)); return f(r[col]) if r else None
print('success on shortcut targets / the other 16, LLMs pooled')
for h in ['none', 'split', 'all']:
    for p, v in [('fully_connected', 'one_agent'), ('fully_connected', 'all_independent'), ('fully_connected', 'base'), ('star', 'base'), ('fully_connected_results', 'base')]:
        a = mean([get(m, p, h, r, v) for m in LLM for r in short]); b = mean([get(m, p, h, r, v) for m in LLM for r in range(20) if r not in short])
        print(f"  {h:5s} {p:24s} {v:16s} shortcut {100*a:5.1f}  others {100*b:5.1f}")
IND = {'fully_connected': 'fully_connected', 'fully_connected_results': 'fully_connected', 'fully_connected_messages': 'fully_connected', 'fully_connected_failures': 'fully_connected', 'star': 'star'}
keep = [r for r in range(20) if r not in short]
def contrast(a, b):
    d, c = [], []
    for m in LLM:
        for r in keep:
            d.append(a(m, r) - b(m, r)); c.append(r)
    return fmt(*boot(d, cluster=c))
print('contrasts without shortcut targets (48 pairs)')
for p in IND:
    print(f"  split base - ind  {p:26s}", contrast(lambda m, r, p=p: get(m, p, 'split', r, 'base'), lambda m, r, p=p: get(m, IND[p], 'split', r, 'all_independent')))
for p in IND:
    print(f"  none  base - ind  {p:26s}", contrast(lambda m, r, p=p: get(m, p, 'none', r, 'base'), lambda m, r, p=p: get(m, IND[p], 'none', r, 'all_independent')))
print("  ind all - ind split fc", contrast(lambda m, r: get(m, 'fully_connected', 'all', r, 'all_independent'), lambda m, r: get(m, 'fully_connected', 'split', r, 'all_independent')))
print("  ind split - ind none fc", contrast(lambda m, r: get(m, 'fully_connected', 'split', r, 'all_independent'), lambda m, r: get(m, 'fully_connected', 'none', r, 'all_independent')))
print("  one_agent split - fc base split", contrast(lambda m, r: get(m, 'fully_connected', 'split', r, 'one_agent'), lambda m, r: get(m, 'fully_connected', 'split', r, 'base')))
print("  one_agent split - star base split", contrast(lambda m, r: get(m, 'fully_connected', 'split', r, 'one_agent'), lambda m, r: get(m, 'star', 'split', r, 'base')))
for h in ['none', 'split', 'all']:
    print(h, 'rates w/o shortcuts: one', round(100*mean([get(m, 'fully_connected', h, r, 'one_agent') for m in LLM for r in keep]),1),
          'fc ind', round(100*mean([get(m, 'fully_connected', h, r, 'all_independent') for m in LLM for r in keep]),1),
          'star ind', round(100*mean([get(m, 'star', h, r, 'all_independent') for m in LLM for r in keep]),1),
          *[(p, round(100*mean([get(m, p, h, r, 'base') for m in LLM for r in keep]),1)) for p in IND],
          'random fc ind', round(100*mean([get('mock_norepeat', 'fully_connected', h, r, 'all_independent') for r in keep]),1))
# main depth 2
mrows = load('run_main/main/runs.csv'); M = index(mrows)
shm = {int(r) for r, v in sc['main d2'].items() if v}
print('main depth 2 success by shortcut / not')
for p, v in [('fully_connected', 'one_agent'), ('fully_connected', 'all_independent'), ('fully_connected', 'base'), ('star', 'all_independent'), ('star', 'base')]:
    ms = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
    g = lambda m, r: f(M[(m, p, '2', '0', '', str(r), v)]['solved_any'])
    print(f"  {p:16s} {v:16s} shortcut {100*mean([g(m, r) for m in ms for r in shm]):5.1f} others {100*mean([g(m, r) for m in ms for r in range(20) if r not in shm]):5.1f}")
