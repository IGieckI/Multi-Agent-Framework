# Confidence and team size (the tutor's hypothesis, F55). Three parts:
# 1. from the logs (no confidence needed): the first answers of silenced 6-agent teams, which all get the same prompt,
#    sample the model's answer distribution: share right, modal share, distinct pairs; per-target agent solve rate and
#    the team success it predicts for 1/3/6/12 independent agents
# 2. with run_confidence/ (scripts/rescore.py) or runs that logged it: probability of the first combination, right vs
#    wrong, and whether targets where the model is unsure are the ones where a bigger team gains
# 3. with run_size12/: 12 silenced agents against 6 and 3 (followup), paired by target
import sys, json, glob, math
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from lib import load, boot, fmt, sign_test, mean
SRC = {'qwen3-4b': 'run_followup/followup', 'gemma3-4b': 'run_followup/followup', 'llama3.2-3b': 'run_followup/followup',
       'qwen3-8b': 'run_main/main'}
SETTINGS = ['depth1__distractors3', 'depth2__distractors0']


def first_moves(run):
    """agent -> (pair, right) of its first action, and the agents that made the target"""
    meta = json.load(open(run + '/meta.json'))
    target, inter = meta['task']['target'], (meta['task'].get('intermediate') or [None])[0]
    first, solved = {}, set()
    for line in open(run + '/events.jsonl'):
        e = json.loads(line)
        if e['kind'] == 'env_step':
            a, res = e['data']['agent'], e['data']['results']
            if e['t'] == 0:
                first[a] = (tuple(sorted(e['data']['combo'])), target in res or (inter is not None and inter in res))
            if target in e['data']['new_items']:
                solved.add(a)
    return first, solved


def confidence_lines(grid_root, grid, cell, version, rep):
    """The rescored lines of a run, or its own logged confidence, or None"""
    f = Path(f'run_confidence/{grid}/{cell}/{version}/seed0_rep{rep}.jsonl')
    if f.exists():
        return [json.loads(l) for l in open(f)]
    ev = Path(f'{grid_root}/{cell}/{version}/seed0_rep{rep}/events.jsonl')
    if ev.exists():
        out = []
        for l in open(ev):
            e = json.loads(l)
            if e['kind'] == 'llm_call' and (c := (e['data']['completion'].get('meta') or {}).get('confidence')):
                out.append({'t': e['t'], 'agent': e['data']['agent'], 'attempt': e['data']['attempt'],
                            'action': e['data']['action'], 'error': e['data']['error'], **c})
        return out or None
    return None


print('# 1. first answers of silenced 6-agent teams (fully_connected all_independent)')
per_target = {}  # (model, setting, rep) -> agent solve rate
for setting in SETTINGS:
    print('==', setting)
    for m, root in SRC.items():
        rows = []
        for run in sorted(glob.glob(f'{root}/{m}__fully_connected__{setting}/all_independent/seed0_rep*')):
            rep = int(run.rsplit('rep', 1)[1])
            first, solved = first_moves(run)
            q = len(solved) / 6
            per_target[(m, setting, rep)] = q
            if len(first) < 6:
                continue
            c = Counter(x[0] for x in first.values())
            rows.append(dict(p=sum(x[1] for x in first.values()) / 6, modal=c.most_common(1)[0][1] / 6,
                             distinct=len(c), q=q, team=float(bool(solved))))
        n = len(rows)
        mixed = Counter('right' if r['p'] == 1 else 'wrong' if r['p'] == 0 else 'mixed' for r in rows)
        pred = '/'.join(f"{sum(1 - (1 - r['q']) ** k for r in rows) / n:.2f}" for k in (1, 3, 6, 12))
        print(f"  {m:12s} n={n}: first pair right {mean([r['p'] for r in rows]):.2f}, modal share "
              f"{mean([r['modal'] for r in rows]):.2f}, distinct pairs {mean([r['distinct'] for r in rows]):.1f}/6, "
              f"targets all right/all wrong/mixed {mixed['right']}/{mixed['wrong']}/{mixed['mixed']}; agent solves "
              f"{mean([r['q'] for r in rows]):.2f}, team {mean([r['team'] for r in rows]):.2f}, "
              f"independent draws n=1/3/6/12 {pred}")

print('\n# 2. confidence of the first combination (rescored or logged)')
for m, root in SRC.items():
    grid = Path(root).name
    right, wrong, by_target = [], [], {}
    for setting in SETTINGS:
        for version in ['one_agent', 'all_independent']:
            cell = f'{m}__fully_connected__{setting}'
            for run in sorted(glob.glob(f'{root}/{cell}/{version}/seed0_rep*')):
                rep = int(run.rsplit('rep', 1)[1])
                lines = confidence_lines(root, grid, cell, version, rep)
                if not lines:
                    continue
                first, _ = first_moves(run)
                ps = []
                for x in lines:
                    if x['t'] == 0 and x['error'] is None and x.get('combo_logprob') is not None and x['agent'] in first:
                        p = math.exp(x['combo_logprob']); ps.append(p)
                        (right if first[x['agent']][1] else wrong).append(p)
                if version == 'all_independent' and ps:
                    by_target[(setting, rep)] = mean(ps)
    if not right and not wrong:
        print(f'  {m}: no confidence yet (run scripts/rescore.py, download runs_confidence/ to run_confidence/)')
        continue
    print(f"  {m:12s} P(first combination): right {mean(right):.2f} (n={len(right)}), wrong {mean(wrong):.2f} "
          f"(n={len(wrong)})")
    # do unsure targets gain more from 3 -> 6 agents? (followup, silenced)
    if root.endswith('followup'):
        rows = load(f'{root}/runs.csv')
        sv = {(r['protocol'], r['depth'], r['distractors'], r['replicate']): float(r['solved_any']) for r in rows
              if r['model'] == m and r['version'] == 'all_independent'}
        for half, cond in [('unsure (P < median)', lambda p, med: p < med), ('sure (P >= median)', lambda p, med: p >= med)]:
            med = sorted(by_target.values())[len(by_target) // 2]
            d = [sv[('fully_connected', s[5], s[-1], str(r))] - sv[('fully_connected3', s[5], s[-1], str(r))]
                 for (s, r), p in by_target.items() if cond(p, med)]
            print(f"    6 - 3 silenced agents, {half}: {100 * mean(d):+.1f} points (n={len(d)})")

print('\n# 3. 12 silenced agents (size12) against 6 and 3 (followup)')
if Path('run_size12/size12/runs.csv').exists():
    s12 = {(r['model'], r['depth'], r['distractors'], r['replicate']): float(r['solved_any'])
           for r in load('run_size12/size12/runs.csv')}
    f = load('run_followup/followup/runs.csv')
    for m in ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']:
        for p, lab in [('fully_connected', '6'), ('fully_connected3', '3')]:
            o = {(r['model'], r['depth'], r['distractors'], r['replicate']): float(r['solved_any']) for r in f
                 if r['model'] == m and r['protocol'] == p and r['version'] == 'all_independent'}
            ks = sorted(set(s12) & set(o))
            a, b = [s12[k] for k in ks], [o[k] for k in ks]
            print(f"  {m:12s} 12 - {lab}: {fmt(*boot([x - y for x, y in zip(a, b)], cluster=[k[1:] for k in ks]))} "
                  f"(12: {100 * mean(a):.1f}%, {lab}: {100 * mean(b):.1f}%; sign test {sign_test(a, b)})")
else:
    print('  run_size12/ not downloaded yet')
