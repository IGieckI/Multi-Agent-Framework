import sys; sys.path.insert(0, sys.argv[1])
from lib import *
rows = load('run_hints/hints/runs.csv')
I = index(rows)
models = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b', 'mock_norepeat']
prots = ['fully_connected', 'star', 'fully_connected_results', 'fully_connected_messages', 'fully_connected_failures']
def get(m, p, h, rep, v, col='solved_any'):
    r = I.get((m, p, '2', '6', h, str(rep), v)); return f(r[col]) if r else None
print('success (solved_any %), 20 targets each')
for h in ['none', 'split', 'all']:
    print('== hints', h)
    for m in models:
        line = [f"{m:14s}"]
        line.append(f"one_agent {100*mean([get(m,'fully_connected',h,r,'one_agent') for r in range(20)]):5.1f}")
        for p in ['fully_connected', 'star']:
            line.append(f"{p[:4]}_ind {100*mean([get(m,p,h,r,'all_independent') for r in range(20)]):5.1f}")
        for p in prots:
            line.append(f"{p.replace('fully_connected','fc')}:{100*mean([get(m,p,h,r,'base') for r in range(20)]):5.1f}")
        print('  ' + ' | '.join(line))
