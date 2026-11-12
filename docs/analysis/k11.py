# Sequential and chain turns (chain, chain_hints) against the parallel fully connected team and the silenced one,
# paired by target: success, herding measures, and a cascade measure from the logs (does each next agent copy the
# first agent's first pair, and do later agents solve more?)
import sys, json, glob
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from lib import load, boot, fmt, sign_test, mean
TWIN = {'qwen3-4b': 'run_followup/followup', 'gemma3-4b': 'run_followup/followup', 'llama3.2-3b': 'run_followup/followup',
        'qwen3-8b': 'run_main/main'}
HTWIN = {'qwen3-4b': 'run_hints2/hints2', 'gemma3-4b': 'run_hints2/hints2', 'llama3.2-3b': 'run_hints2/hints2',
         'qwen3-8b': 'run_hints2_8b/hints2_8b'}
sv = lambda r: float(r['solved_any'])
METRICS = ['distinct_paths', 'repeat_rate', 'blind_adoption_rate', 'adoption_rate', 'copy_events', 'total_tokens']


def table(rows, keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def cascade(run):
    """first pairs in turn order: share of agents 1-5 that play agent_00's first pair, and who made the target"""
    first, solved = {}, set()
    target = json.load(open(run + '/meta.json'))['task']['target']
    for line in open(run + '/events.jsonl'):
        e = json.loads(line)
        if e['kind'] == 'env_step':
            a = e['data']['agent']
            if e['t'] == 0:
                first[a] = tuple(sorted(e['data']['combo']))
            if target in e['data']['new_items']:
                solved.add(a)
    agents = sorted(first)
    same = [float(first[a] == first[agents[0]]) for a in agents[1:]] if agents else []
    return mean(same), [float(f'agent_{i:02d}' in solved) for i in range(6)]


def compare(label, new_rows, twin_rows, key, twin_filter, new_dir, twin_dir, cells='*'):
    for proto in sorted({r['protocol'] for r in new_rows}):
        print(f"\n== {label}: {proto}")
        pooled = defaultdict(list)
        for m in sorted({r['model'] for r in new_rows}):
            new = {key(r): r for r in new_rows if r['model'] == m and r['protocol'] == proto}
            for ver, name in [('base', 'parallel fully connected'), ('all_independent', 'silenced')]:
                tw = {key(r): r for r in twin_rows[m] if r['model'] == m and twin_filter(r) and r['version'] == ver}
                ks = sorted(set(new) & set(tw))
                a, b = [sv(new[k]) for k in ks], [sv(tw[k]) for k in ks]
                pooled[name] += [(k, x - y, x, y) for k, x, y in zip(ks, a, b)]
                print(f"  {m:12s} {proto} - {name}: {fmt(*boot([x - y for x, y in zip(a, b)], cluster=ks))} "
                      f"({100 * mean(a):.1f}% vs {100 * mean(b):.1f}%, n={len(ks)}, discordant {sign_test(a, b)})")
            tw = {key(r): r for r in twin_rows[m] if r['model'] == m and twin_filter(r) and r['version'] == 'base'}
            print('    herding (new / parallel): ' + ', '.join(
                f"{x} {mean([float(new[k][x]) for k in new if new[k][x] not in ('', None)]) or 0:.2f} / "
                f"{mean([float(tw[k][x]) for k in tw if tw[k][x] not in ('', None)]) or 0:.2f}" for x in METRICS))
            same, by_pos = [], defaultdict(list)
            for run in glob.glob(f"{new_dir}/{m}__{proto}__{cells}/base/seed0_rep*"):
                s, sol = cascade(run)
                if s is not None: same.append(s)
                for i, x in enumerate(sol): by_pos[i].append(x)
            par = [cascade(run)[0] for run in glob.glob(f"{twin_dir(m)}/{m}__fully_connected__{cells}/base/seed0_rep*")]
            par = [x for x in par if x is not None]
            print(f"    agents 1-5 playing agent_00's first pair: {mean(same):.2f} (parallel {mean(par):.2f}); "
                  f"solve rate by turn position " + ' '.join(f"{mean(by_pos[i]):.2f}" for i in range(6)))
        for name, xs in pooled.items():
            print(f"  pooled {proto} - {name}: {fmt(*boot([d for _, d, _, _ in xs], cluster=[k for k, *_ in xs]))} "
                  f"(discordant {sign_test([x for *_, x, _ in xs], [y for *_, y in xs])})")


if Path('run_chain/chain/runs.csv').exists():
    new = load('run_chain/chain/runs.csv')
    twins = {m: load(f'{d}/runs.csv') for m, d in TWIN.items()}
    compare('main settings', new, twins, lambda r: (r['depth'], r['distractors'], r['replicate']),
            lambda r: r['protocol'] == 'fully_connected', 'run_chain/chain', lambda m: TWIN[m])
else:
    print('run_chain/ not downloaded yet')
if Path('run_chain_hints/chain_hints/runs.csv').exists():
    new = load('run_chain_hints/chain_hints/runs.csv')
    twins = {m: [r for r in load(f'{d}/runs.csv') if int(r['replicate']) < 40] for m, d in HTWIN.items()}
    for hints in ['split', 'all']:
        compare(f'hints {hints}', [r for r in new if r['hints'] == hints], twins, lambda r: (r['replicate'],),
                lambda r, h=hints: r['protocol'] == 'fully_connected' and r['hints'] == h and r['decoys'] == 'True',
                'run_chain_hints/chain_hints', lambda m: HTWIN[m], f'hints{hints}__decoysTrue')
else:
    print('run_chain_hints/ not downloaded yet')
