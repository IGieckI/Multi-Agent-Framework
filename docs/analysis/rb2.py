import sys, random, json
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, sys.argv[1])
from lib import *
from framework.tasks.wordcraft.recipes import RecipeBook
from framework.tasks.wordcraft.tasks import sample_targeted
book = RecipeBook.load('data/alchemy2.json')
def tseed(rep): return int(np.random.SeedSequence([0, rep]).generate_state(1)[0])
short = {}
for d, D in [(1, 3), (2, 0), (2, 6)]:
    for rep in range(200):
        t = sample_targeted(book, random.Random(tseed(rep)), d, D, 4)
        direct = [c for c in book.recipes_for(t.target) if set(c) <= set(t.initial)]
        short[(d, D, rep)] = len(direct) > (1 if d == 1 else 0)
for d, D in [(1, 3), (2, 0), (2, 6)]:
    print(f"depth {d} distractors {D}: shortcut/extra direct recipe in {sum(short[(d, D, r)] for r in range(200))}/200 targets, reps 0-19: {sum(short[(d, D, r)] for r in range(20))}")
# check against the earlier 20-target count (hints 4/20, main d2 3/20)
rb = load('run_random_baseline/random_baseline/runs.csv')
print('random agents, depth 2 distractors 0, on non-shortcut / shortcut targets (200)')
for m in ['mock', 'mock_norepeat']:
    for p, v in [('fully_connected', 'all_independent'), ('star', 'all_independent'), ('star', 'one_agent'), ('fully_connected', 'base')]:
        sel = [r for r in rb if r['depth'] == '2' and r['model'] == m and r['protocol'] == p and r['version'] == v]
        a = mean([f(r['solved_any']) for r in sel if not short[(2, 0, int(r['replicate']))]]); b = mean([f(r['solved_any']) for r in sel if short[(2, 0, int(r['replicate']))]])
        print(f"  {m:14s} {p:16s} {v:16s} non-shortcut {100*a:.1f}  shortcut {100*b:.1f}")
# paired LLM - random on reps 0-19 (main), by model, cluster by target
main = index(load('run_main/main/runs.csv'))
R = index(rb)
print('LLM - random (mock_norepeat), main targets 0-19, pairs over 4 models x 20 targets')
for d, D in [('1', '3'), ('2', '0')]:
    for p, v, rv in [('fully_connected', 'all_independent', 'all_independent'), ('fully_connected', 'base', 'all_independent'), ('star', 'all_independent', 'all_independent'), ('star', 'base', 'all_independent'), ('fully_connected', 'one_agent', 'one_agent')]:
        dd, cc = [], []
        for m in ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']:
            for rep in range(20):
                x = f(main[(m, p, d, D, '', str(rep), v)]['solved_any'])
                y = f(R[('mock_norepeat', p, d, D, '', str(rep), rv)]['solved_any'])
                dd.append(x - y); cc.append(rep)
        print(f"  depth {d} LLM {p} {v} - random {p} {rv}: {fmt(*boot(dd, cluster=cc))}")
json.dump({f"{k[0]},{k[1]},{k[2]}": v for k, v in short.items()}, open(sys.argv[1] + '/short200.json', 'w'))
