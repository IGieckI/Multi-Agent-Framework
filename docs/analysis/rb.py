import sys; sys.path.insert(0, sys.argv[1])
from lib import *
rb = load('run_random_baseline/random_baseline/runs.csv')
print(len(rb), 'rows')
def rate(rows):
    xs = [f(r['solved_any']) for r in rows]; return f"{100*mean(xs):.1f}% (n={len(xs)})"
for d in ('1', '2'):
    print('== depth', d)
    for m in ('mock', 'mock_norepeat'):
        for p, v in [('fully_connected', 'all_independent'), ('fully_connected', 'base'), ('fully_connected_results', 'base'), ('fully_connected3', 'all_independent'), ('star', 'all_independent'), ('star6', 'all_independent'),  ('star', 'base'), ('star', 'one_agent'), ('fully_connected', 'one_agent_scaled_steps'), ('star', 'one_agent_scaled_steps')]:
            sel = [r for r in rb if r['depth'] == d and r['model'] == m and r['protocol'] == p and r['version'] == v]
            e = [r for r in sel if int(r['replicate']) < 20]
            print(f"  {m:14s} {p:24s} {v:24s} all200 {rate(sel)}  reps0-19 {rate(e)}  repeat {mean([f(r['repeat_rate']) for r in sel]):.3f}")
