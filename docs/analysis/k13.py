# Adoption and copy measures against a silenced baseline (F64). The logged adoption_rate_result,
# blind_adoption_rate and copy_events count only what an agent received, so a silenced team scores 0 or nothing by
# construction. Here the silenced side gets the inbox fully connected would have delivered (every teammate's action
# of the previous round) and copies count items any teammate found first. The same proxy on talking runs reproduces
# the logged values exactly (checked on main and followup), so talk - silenced is what talk adds.
# Parts: (1) main and followup fully connected, talk - silenced paired by target; (2) mixed teams (F49);
# (3) chain6 against the parallel team with copies counted against every teammate (F63).
import sys, json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from lib import load, boot, fmt, mean

MEASURES = ['result_adoption', 'blind_adoption', 'copy_events']
RESULTS = {}   # grid -> protocol -> model -> measure -> {sil, talk, d, lo, hi, n} (percent, copies per run)


def local(run_dir):
    """runs.csv holds giano paths; the download keeps everything after runs/ under run_<grid>/"""
    rest = run_dir.split('/runs/')[1]
    return Path('run_' + rest.split('/')[0]) / rest


def measures(run):
    steps = [dict(json.loads(l)['data'], t=json.loads(l)['t']) for l in open(run / 'events.jsonl') if '"env_step"' in l]
    by_t = defaultdict(dict)
    for s in steps:
        by_t[s['t']][s['agent']] = s
    heard = res = blind = 0
    for t in sorted(by_t):
        for a, s in by_t[t].items():
            mates = [p for b, p in by_t.get(t - 1, {}).items() if b != a]
            if t == 0 or not mates:
                continue
            act = frozenset(s['combo'])
            heard += 1
            res += any(frozenset(p['combo']) == act for p in mates)
            blind += any(frozenset(p['combo']) == act and not p['results'] for p in mates)
    found = defaultdict(dict)
    for s in sorted(steps, key=lambda s: s['t']):
        for item in s['new_items']:
            found[s['agent']].setdefault(item, s['t'])
    copies = sum(1 for a, items in found.items() for item, t in items.items()
                 if any(item in found[b] and found[b][item] < t for b in found if b != a))
    return {'result_adoption': res / heard if heard else None, 'blind_adoption': blind / heard if heard else None,
            'copy_events': float(copies)}


def paired(grid, protocol, models=None):
    runs = defaultdict(dict)
    for r in load(f'{grid}/runs.csv'):
        if r['protocol'] == protocol and r['version'] in ('base', 'all_independent') and (not models or r['model'] in models):
            runs[(r['model'], r['depth'], r['distractors'], r['replicate'])][r['version']] = measures(local(r['run_dir']))
    print(f'\n{grid}, {protocol}: silenced (would-be inbox) / talk / talk - silenced, bootstrap by target')
    res = RESULTS.setdefault(grid.split('/')[-1], {}).setdefault(protocol, {})
    for m in sorted({k[0] for k in runs}) + ['pooled']:
        for meas in MEASURES:
            rows = [(k, v['all_independent'][meas], v['base'][meas]) for k, v in runs.items()
                    if (m == 'pooled' or k[0] == m) and len(v) == 2
                    and v['all_independent'][meas] is not None and v['base'][meas] is not None]
            sil, talk = mean([a for _, a, _ in rows]), mean([b for _, _, b in rows])
            scale = 1 if meas == 'copy_events' else 100
            d = boot([b - a for _, a, b in rows], cluster=[k[1:] for k, _, _ in rows])
            res.setdefault(m, {})[meas] = {'sil': round(sil * scale, 2), 'talk': round(talk * scale, 2),
                                           **dict(zip(('d', 'lo', 'hi'), (round(x * scale, 2) for x in d))), 'n': len(rows)}
            print(f'  {m:12s} {meas:16s} {sil * scale:6.2f} {talk * scale:6.2f}  {fmt(*d, scale=scale, d=2)}  n={len(rows)}')


paired('run_main/main', 'fully_connected')
paired('run_followup/followup', 'fully_connected')
paired('run_followup/followup', 'fully_connected_results')
paired('run_followup/followup', 'fully_connected_messages')
paired('run_mixed/mixed', 'fully_connected')

print('\nchain6 vs parallel fully connected: copies counted against every teammate (logged: only against senders)')
TWIN = {'qwen3-8b': 'run_main/main'}
chain = defaultdict(list)
for r in load('run_chain/chain/runs.csv'):
    if r['protocol'] == 'chain6':
        chain[r['model']].append((measures(local(r['run_dir']))['copy_events'], float(r['copy_events'] or 0)))
for m in sorted(chain):
    rows = load(f"{TWIN.get(m, 'run_followup/followup')}/runs.csv")
    par = {v: [(measures(local(r['run_dir']))['copy_events'], float(r['copy_events'] or 0)) for r in rows
               if r['model'] == m and r['protocol'] == 'fully_connected' and r['version'] == v]
           for v in ('base', 'all_independent')}
    RESULTS.setdefault('chain_copies', {})[m] = {
        'sil': round(mean([a for a, _ in par['all_independent']]), 2), 'par': round(mean([a for a, _ in par['base']]), 2),
        'par_logged': round(mean([b for _, b in par['base']]), 2), 'chain': round(mean([a for a, _ in chain[m]]), 2),
        'chain_logged': round(mean([b for _, b in chain[m]]), 2)}
    print(f"  {m:12s} silenced {mean([a for a, _ in par['all_independent']]):.2f}  parallel talk "
          f"{mean([a for a, _ in par['base']]):.2f} (logged {mean([b for _, b in par['base']]):.2f})  chain6 "
          f"{mean([a for a, _ in chain[m]]):.2f} (logged {mean([b for _, b in chain[m]]):.2f})")
