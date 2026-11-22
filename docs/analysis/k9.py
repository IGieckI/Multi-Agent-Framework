# mixed / mixed_hints follow-ups (F46-): talk - silenced, mixed vs each single model (difference in differences);
# herding measures; mixed_hints success by which family holds the path recipes; remake share
import sys, json, glob
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import *
M3 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
KM = ('protocol', 'depth', 'distractors', 'replicate', 'version'); KH = ('protocol', 'hints', 'decoys', 'replicate', 'version')
MX = index(load('run_mixed/mixed/runs.csv'), KM); MN = index(load('run_main/main/runs.csv'), ('model',) + KM)
MH = index(load('run_mixed_hints/mixed_hints/runs.csv'), KH); H2 = index(load('run_hints2/hints2/runs.csv'), ('model',) + KH)
def tm(src, pre, mid, R, col='solved_any'):
    return {r: f(src[pre + mid + (str(r), 'base')][col]) - f(src[pre + mid + (str(r), 'all_independent')][col]) for r in R}
print("==== talk - silenced: mixed, each model, mixed - model (diff in diff)")
for name, mx, sg, mids, R in [('main', MX, MN, [('1', '3'), ('2', '0')], range(20)), ('hints2', MH, H2, [('split', 'True'), ('all', 'True')], range(40))]:
    for p in ['fully_connected', 'star']:
        for mid in mids + ['pooled']:
            ms = mids if mid == 'pooled' else [mid]
            a, keys = [], []
            for x in ms:
                t = tm(mx, (), (p,) + x, R); a += [t[r] for r in R]; keys += [''.join(x) + str(r) for r in R]
            line = f"  {name:6s} {p:16s} {'/'.join(mid) if mid != 'pooled' else 'pooled':12s} mixed {fmt(*boot(a, cluster=keys))}"
            for m in M3:
                b = []
                for x in ms:
                    t = tm(sg, (m,), (p,) + x, R); b += [t[r] for r in R]
                line += f" | {m[:5]} {100*mean(b):+5.1f} dd {fmt(*boot([u - v for u, v in zip(a, b)], cluster=keys), d=0)}"
            print(line)
print("\n==== herding measures, fully connected with talk (main settings pooled; hints2 split)")
for col in ['blind_adoption_rate', 'adoption_rate', 'distinct_paths', 'repeat_rate', 'copy_events']:
    line = f"  {col:20s}"
    for lab, src, pre in [('mixed', MX, ())] + [(m[:5], MN, (m,)) for m in M3]:
        xs = [f(src[pre + ('fully_connected', d, x, str(r), v)][col]) for d, x in [('1', '3'), ('2', '0')] for r in range(20) for v in ['base']]
        line += f" {lab} {mean(xs):.2f}"
    xs0 = [f(MX[('fully_connected', d, x, str(r), 'all_independent')]['distinct_paths']) for d, x in [('1', '3'), ('2', '0')] for r in range(20)]
    print(line + (f"   (mixed silenced {mean(xs0):.2f})" if col == 'distinct_paths' else ''))
print("\n==== mixed_hints split: success by the family holding each path recipe (talk / silenced)")
fam = lambda s: s.split('/')[0].split('-')[0].lower()
for p in ['fully_connected', 'star']:
    for v in ['base', 'all_independent']:
        by = defaultdict(list)
        for run in glob.glob(f'run_mixed_hints/mixed_hints/mix-qgl__{p}__hintssplit__decoysTrue/{v}/seed0_rep*'):
            meta = json.load(open(run + '/meta.json')); am = meta['agent_models']; notes = meta['notes']; path = meta['task']['path']
            hold = []
            for a, b, r in path:
                hs = [ag for ag, ns in notes.items() if any(sorted(n[:2]) == sorted([a, b]) for n in ns)]
                hold.append('+'.join(sorted({fam(am[h]) for h in hs})) + ('(hub)' if p == 'star' and 'agent_00' in hs else ''))
            solved = f(MH[(p, 'split', 'True', run.rsplit('rep', 1)[1], v)]['solved_any'])
            by['inter ' + hold[0]].append(solved); by['target ' + hold[1]].append(solved)
        print(f"  {p:16s} {v:16s} " + '  '.join(f"{k} {100*mean(x):.0f}% (n={len(x)})" for k, x in sorted(by.items())))
def remake(folder):
    xs = []
    for run in sorted(glob.glob(folder + '/seed0_rep*')):
        meta = json.load(open(run + '/meta.json')); a, b, inter = meta['task']['path'][0]; have = defaultdict(set)
        for l in open(run + '/events.jsonl'):
            e = json.loads(l)
            if e['kind'] != 'env_step': continue
            ag, c = e['data']['agent'], sorted(e['data']['combo'])
            if inter in have[ag]: xs.append(float(c == sorted([a, b])))
            have[ag] |= set(e['data']['new_items'])
    return f"{100*mean(xs):.0f}% (n={len(xs)})"
print("\n==== mixed_hints all: remake share")
for p in ['fully_connected', 'star']:
    print(f"  {p:16s} silenced {remake(f'run_mixed_hints/mixed_hints/mix-qgl__{p}__hintsall__decoysTrue/all_independent')}  talk {remake(f'run_mixed_hints/mixed_hints/mix-qgl__{p}__hintsall__decoysTrue/base')}")
