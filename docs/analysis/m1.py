import sys, json; sys.path.insert(0, sys.argv[1])
from lib import *
sh = json.load(open(sys.argv[1] + '/short200.json'))
M = index(load('run_main/main/runs.csv'))
ms = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
for p in ['fully_connected', 'star']:
    for excl in [False, True]:
        d, c = [], []
        for dep, D in [('1', '3'), ('2', '0')]:
            for m in ms:
                for r in range(20):
                    if excl and sh[f"{dep},{D},{r}"]: continue
                    d.append(f(M[(m, p, dep, D, '', str(r), 'base')]['solved_any']) - f(M[(m, p, dep, D, '', str(r), 'all_independent')]['solved_any'])); c.append((dep, r))
        print(p, 'without shortcut targets' if excl else 'all targets', len(d), fmt(*boot(d, cluster=c)))
