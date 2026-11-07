import sys, json, glob
from collections import defaultdict, Counter
sys.path.insert(0, sys.argv[1])
from lib import mean
sc = json.load(open(sys.argv[1] + '/shortcuts.json')); short = {int(r) for r, v in sc['hints'].items() if v}
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
same = lambda c, n: sorted(c) == sorted(n[:2])
print('one agent holding both recipes (no shortcut targets): first two actions')
for h in ['split', 'all']:
    for m in LLM:
        c = Counter(); fails = []
        for d in glob.glob(f'run_hints/hints/{m}__fully_connected__depth2__distractors6__hints{h}/one_agent/seed0_rep*'):
            if int(d.rsplit('rep', 1)[1]) in short: continue
            meta = json.load(open(d + '/meta.json')); path = meta['task']['path']
            ev = [json.loads(l) for l in open(d + '/events.jsonl')]
            steps = [e['data']['combo'] for e in ev if e['kind'] == 'env_step']
            s0 = 'r0' if steps and same(steps[0], path[0]) else 'other'
            used1 = any(same(s, path[1]) for s in steps)
            used0 = any(same(s, path[0]) for s in steps)
            c[(s0, 'used r0' if used0 else 'never r0', 'used r1' if used1 else 'never r1')] += 1
        print(f"  {h:5s} {m:12s}", dict(c))
print('decoy notes: share of agents that try one of their decoys, and of all steps spent on decoys')
for h in ['split', 'all']:
  for p, v in [('fully_connected', 'all_independent'), ('fully_connected', 'base'), ('star', 'base')]:
    agents_tried, steps_decoy = [], []
    for m in LLM:
        for d in glob.glob(f'run_hints/hints/{m}__{p}__depth2__distractors6__hints{h}/{v}/seed0_rep*'):
            if int(d.rsplit('rep', 1)[1]) in short: continue
            meta = json.load(open(d + '/meta.json')); path = [sorted(x[:2]) for x in meta['task']['path']]
            ev = [json.loads(l) for l in open(d + '/events.jsonl')]
            steps = [e['data'] for e in ev if e['kind'] == 'env_step']
            for ag, ns in meta['notes'].items():
                decoys = [sorted(n[:2]) for n in ns if sorted(n[:2]) not in path]
                if not decoys: continue
                mine = [sorted(s['combo']) for s in steps if s['agent'] == ag]
                agents_tried.append(float(any(x in decoys for x in mine)))
                steps_decoy += [float(x in decoys) for x in mine]
    print(f"  {h:5s} {p:16s} {v:16s} agents {mean(agents_tried):.2f}  steps {mean(steps_decoy):.2f}" if agents_tried else f"  {h} {p} {v} no decoys")
