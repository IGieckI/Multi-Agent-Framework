import sys, json; sys.path.insert(0, sys.argv[1])
from lib import *
out = {}
M = index(load('run_main/main/runs.csv')); ms = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
S = [('1', '3'), ('2', '0')]
g = lambda m, p, d, D, r, v: f(M[(m, p, d, D, '', str(r), v)]['solved_any'])
# A. main versions pooled
A = {}
for p in ['fully_connected', 'star']:
    for v in ['one_agent', 'one_vs_multi_token', 'all_independent_temp_sweep', 'all_independent', 'base']:
        A[f"{p}|{v}"] = round(100 * mean([g(m, p, d, D, r, v) for m in ms for d, D in S for r in range(20)]), 1)
out['main_rates'] = A
# B. forest main per model
B = []
for p in ['fully_connected', 'star']:
    for m in ms + ['all']:
        dd, cc = [], []
        for mm in (ms if m == 'all' else [m]):
            for d, D in S:
                for r in range(20):
                    dd.append(g(mm, p, d, D, r, 'base') - g(mm, p, d, D, r, 'all_independent')); cc.append((d, r))
        mu, lo, hi = boot(dd, cluster=cc)
        B.append({'protocol': p, 'model': m, 'd': round(mu*100, 1), 'lo': round(lo*100, 1), 'hi': round(hi*100, 1)})
out['main_forest'] = B
# C. followup protocols
F = index(load('run_followup/followup/runs.csv')); fm = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
ind = {'fully_connected': 'fully_connected', 'fully_connected3': 'fully_connected3', 'star': 'star', 'star6': 'star6',
       'fully_connected_results': 'fully_connected', 'fully_connected_messages': 'fully_connected'}
C = []
for p, ip in ind.items():
    dd, cc = [], []
    for m in fm:
        for d, D in S:
            for r in range(20):
                a = F.get((m, p, d, D, '', str(r), 'base')); b = F.get((m, ip, d, D, '', str(r), 'all_independent'))
                dd.append(f(a['solved_any']) - f(b['solved_any'])); cc.append((d, r))
    mu, lo, hi = boot(dd, cluster=cc)
    C.append({'protocol': p, 'd': round(mu*100, 1), 'lo': round(lo*100, 1), 'hi': round(hi*100, 1),
              'talk': round(100*mean([f(F[(m, p, d, D, '', str(r), 'base')]['solved_any']) for m in fm for d, D in S for r in range(20)]), 1),
              'ind': round(100*mean([f(F[(m, ip, d, D, '', str(r), 'all_independent')]['solved_any']) for m in fm for d, D in S for r in range(20)]), 1)})
out['followup_forest'] = C
# D. random vs LLM
R = load('run_random_baseline/random_baseline/runs.csv')
def rr(d, p, v, model='mock_norepeat'):
    return round(100*mean([f(r['solved_any']) for r in R if r['depth'] == d and r['protocol'] == p and r['version'] == v and r['model'] == model]), 1)
D = []
for d, DD in S:
    D.append({'depth': d,
              'llm': {'one': round(100*mean([g(m, 'fully_connected', d, DD, r, 'one_agent') for m in ms for r in range(20)]), 1),
                      'three': round(100*mean([g(m, 'star', d, DD, r, 'all_independent') for m in ms for r in range(20)]), 1),
                      'six': round(100*mean([g(m, 'fully_connected', d, DD, r, 'all_independent') for m in ms for r in range(20)]), 1),
                      'six_talk': round(100*mean([g(m, 'fully_connected', d, DD, r, 'base') for m in ms for r in range(20)]), 1)},
              'random': {'one': rr(d, 'star', 'one_agent'), 'three': rr(d, 'star', 'all_independent'), 'six': rr(d, 'fully_connected', 'all_independent'),
                         'six_share': rr(d, 'fully_connected', 'base')}})
out['random'] = D
json.dump(out, open(sys.argv[1] + '/pagedata.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:4000])
