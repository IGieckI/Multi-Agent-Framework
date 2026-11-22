# Every number of the report, from the downloaded runs (runs.csv and the runs' logs). Run from the repo root:
#   python3 docs/report/make_data.py
# Writes docs/report/data/data.json (read by make_figures.py and report.typ) and one CSV per table.
import csv, glob, io, json, os, runpy, sys, contextlib
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from lib import load, f, mean, boot

OUT = os.path.join(HERE, 'data')
os.makedirs(OUT, exist_ok=True)
D = {}


def rows_where(rows, **kw):
    return [r for r in rows if all(r.get(k) == str(v) for k, v in kw.items())]


def rate(rows, **kw):
    """success rate (solved_any) over distinct runs, with a 95% bootstrap interval"""
    seen, xs = set(), []
    for r in rows_where(rows, **kw):
        k = r.get('run_dir') or id(r)
        if k in seen: continue
        seen.add(k); xs.append(f(r['solved_any']))
    if not xs: return None
    m, lo, hi = boot(xs)
    return {'p': round(100 * m, 1), 'lo': round(100 * lo, 1), 'hi': round(100 * hi, 1), 'n': len(xs)}


def colmean(rows, col, **kw):
    seen, xs = set(), []
    for r in rows_where(rows, **kw):
        k = r.get('run_dir') or id(r)
        if k in seen: continue
        seen.add(k); xs.append(f(r[col]))
    v = mean(xs)
    return round(v, 3) if v is not None else None


def contrast(rows, a, b, keys, col='solved_any', keep=None):
    """paired difference a - b (in points), 95% bootstrap resampled by target; a, b: dicts of column values"""
    idx = {tuple(r.get(k, '') for k in keys): r for r in rows}
    dd, cc = [], []
    for k, r in idx.items():
        row = dict(zip(keys, k))
        if not all(row.get(x) == str(y) for x, y in a.items()): continue
        if keep and not keep(row): continue
        kb = tuple(str(b.get(x, row[x])) for x in keys)
        if kb not in idx: continue
        dd.append(f(r[col]) - f(idx[kb][col]))
        cc.append((row.get('depth'), row.get('distractors'), row['replicate']))
    if not dd: return None
    m, lo, hi = boot(dd, cluster=cc)
    return {'d': round(100 * m, 1), 'lo': round(100 * lo, 1), 'hi': round(100 * hi, 1), 'n': len(dd)}


def write_csv(name, header, table):
    with open(os.path.join(OUT, name), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(table)


M4 = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
M3 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']

# ---------------------------------------------------------------- pilot (prompts v1)
P = load('run_62/runs.csv')
pilot = {}
for m in sorted({r['model'] for r in P}):
    pilot[m] = {
        'one': rate(P, model=m, protocol='fully_connected', version='one_agent'),
        'fc_mute': rate(P, model=m, protocol='fully_connected', version='all_independent'),
        'fc_talk': rate(P, model=m, protocol='fully_connected', version='base'),
        'st_mute': rate(P, model=m, protocol='star', version='all_independent'),
        'st_talk': rate(P, model=m, protocol='star', version='base'),
        'refused': round(100 * colmean(P, 'parse_failure_rate', model=m), 1),
        'redundancy': colmean(P, 'redundancy', model=m, protocol='fully_connected', version='base'),
    }
D['pilot'] = pilot
write_csv('pilot.csv', ['model', 'one_agent', 'fc_silenced', 'fc_talk', 'star_silenced', 'star_talk', 'refused_pct'],
          [[m, v['one']['p'], v['fc_mute']['p'], v['fc_talk']['p'], v['st_mute']['p'], v['st_talk']['p'], v['refused']]
           for m, v in pilot.items()])

# ---------------------------------------------------------------- prompt v2 check (paired with the pilot)
V2 = load('run_62/prompt_v2_check/runs.csv')
key = ('model', 'protocol', 'depth', 'distractors', 'replicate', 'version')
p_idx = {tuple(r[k] for k in key): r for r in P}
v2 = {}
for m in sorted({r['model'] for r in V2}):
    seen, pairs = set(), []
    for r in V2:
        if r['model'] != m or r['run_dir'] in seen: continue
        seen.add(r['run_dir'])
        t = p_idx.get(tuple(r[k] for k in key))
        if t: pairs.append((t, r))
    v2[m] = {'n': len(pairs),
             'refused_v1': round(100 * mean([f(a['parse_failure_rate']) for a, b in pairs]), 1),
             'refused_v2': round(100 * mean([f(b['parse_failure_rate']) for a, b in pairs]), 1),
             'solved_v1': round(100 * mean([f(a['solved_any']) for a, b in pairs]), 1),
             'solved_v2': round(100 * mean([f(b['solved_any']) for a, b in pairs]), 1)}
D['v2'] = v2
write_csv('prompt_v2.csv', ['model', 'pairs', 'refused_v1', 'refused_v2', 'solved_v1', 'solved_v2'],
          [[m, v['n'], v['refused_v1'], v['refused_v2'], v['solved_v1'], v['solved_v2']] for m, v in v2.items()])

# ---------------------------------------------------------------- main
MN = load('run_main/main/runs.csv')
K = ('model', 'protocol', 'depth', 'distractors', 'replicate', 'version')
main = {}
for m in M4 + ['all']:
    sel = {} if m == 'all' else {'model': m}
    e = {}
    for p in ['fully_connected', 'star']:
        e[p] = {v: rate(MN, protocol=p, version=v, **sel) for v in
                ['base', 'all_independent', 'all_independent_temp_sweep', 'one_vs_multi_token']}
        e[p]['talk_minus_mute'] = contrast(MN, dict(protocol=p, version='base', **sel), dict(version='all_independent'), K)
        e[p]['tok_minus_talk'] = contrast(MN, dict(protocol=p, version='one_vs_multi_token', **sel), dict(version='base'), K)
        e[p]['tok_minus_mute'] = contrast(MN, dict(protocol=p, version='one_vs_multi_token', **sel), dict(version='all_independent'), K)
        e[p]['temp_minus_mute'] = contrast(MN, dict(protocol=p, version='all_independent_temp_sweep', **sel), dict(version='all_independent'), K)
    e['one'] = rate(MN, protocol='fully_connected', version='one_agent', **sel)
    e['herding'] = {v: {'blind': colmean(MN, 'blind_adoption_rate', protocol='fully_connected', version=v, **sel),
                        'adoption': colmean(MN, 'adoption_rate', protocol='fully_connected', version=v, **sel),
                        'paths': colmean(MN, 'distinct_paths', protocol='fully_connected', version=v, **sel),
                        'tokens': colmean(MN, 'total_tokens', protocol='fully_connected', version=v, **sel)}
                    for v in ['base', 'all_independent']}
    e['by_depth'] = {}
    for d, x in [(1, 3), (2, 0)]:
        e['by_depth'][d] = {
            'fc_mute': rate(MN, protocol='fully_connected', depth=d, version='all_independent', **sel),
            'fc_talk': rate(MN, protocol='fully_connected', depth=d, version='base', **sel),
            'st_mute': rate(MN, protocol='star', depth=d, version='all_independent', **sel),
            'st_talk': rate(MN, protocol='star', depth=d, version='base', **sel),
            'one': rate(MN, protocol='fully_connected', depth=d, version='one_agent', **sel),
            'repeat_one': colmean(MN, 'repeat_rate', protocol='fully_connected', depth=d, version='one_agent', **sel)}
    main[m] = e
D['main'] = main
write_csv('main.csv', ['model', 'fc_silenced', 'fc_talk', 'fc_talk_minus_silenced', 'ci_lo', 'ci_hi',
                       'star_silenced', 'star_talk', 'star_talk_minus_silenced', 'ci_lo', 'ci_hi', 'one_agent',
                       'token_matched', 'blind_adoption_fc', 'distinct_paths_silenced', 'distinct_paths_talk'],
          [[m, e['fully_connected']['all_independent']['p'], e['fully_connected']['base']['p'],
            e['fully_connected']['talk_minus_mute']['d'], e['fully_connected']['talk_minus_mute']['lo'], e['fully_connected']['talk_minus_mute']['hi'],
            e['star']['all_independent']['p'], e['star']['base']['p'],
            e['star']['talk_minus_mute']['d'], e['star']['talk_minus_mute']['lo'], e['star']['talk_minus_mute']['hi'],
            e['one']['p'], e['fully_connected']['one_vs_multi_token']['p'], e['herding']['base']['blind'],
            e['herding']['all_independent']['paths'], e['herding']['base']['paths']] for m, e in main.items()])

# ---------------------------------------------------------------- follow-up
FU = load('run_followup/followup/runs.csv')
PROT = ['fully_connected', 'fully_connected3', 'star', 'star6', 'fully_connected_results', 'fully_connected_messages']
fu = {}
for m in M3 + ['all']:
    sel = {} if m == 'all' else {'model': m}
    e = {p: {'mute': rate(FU, protocol=p, version='all_independent', **sel),
             'talk': rate(FU, protocol=p, version='base', **sel),
             'talk_minus_mute': contrast(FU, dict(protocol=p, version='base', **sel), dict(version='all_independent'), K)}
         for p in PROT}
    for big, small in [('fully_connected', 'fully_connected3'), ('star6', 'star')]:
        for v in ['all_independent', 'base']:
            e[f'{big}-{small}:{v}'] = contrast(FU, dict(protocol=big, version=v, **sel), dict(protocol=small), K)
    e['temp_minus_mute'] = contrast(FU, dict(protocol='fully_connected', version='all_independent_temp_sweep', **sel), dict(version='all_independent'), K)
    fu[m] = e
D['followup'] = fu
write_csv('followup.csv', ['model', 'protocol', 'silenced', 'talk', 'talk_minus_silenced', 'ci_lo', 'ci_hi'],
          [[m, p, e[p]['mute']['p'], e[p]['talk']['p'], e[p]['talk_minus_mute']['d'], e[p]['talk_minus_mute']['lo'],
            e[p]['talk_minus_mute']['hi']] for m, e in fu.items() for p in PROT])
write_csv('team_size.csv', ['model', 'contrast', 'version', 'diff', 'ci_lo', 'ci_hi'],
          [[m, c.split(':')[0], c.split(':')[1], v['d'], v['lo'], v['hi']] for m, e in fu.items() for c, v in e.items()
           if ':' in c])

# ---------------------------------------------------------------- random agents
RB = load('run_random_baseline/random_baseline/runs.csv')
rnd = {}
for m in ['mock', 'mock_norepeat']:
    rnd[m] = {d: {'fc_mute': rate(RB, model=m, protocol='fully_connected', depth=d, version='all_independent'),
                  'fc_talk': rate(RB, model=m, protocol='fully_connected', depth=d, version='base'),
                  'st_mute': rate(RB, model=m, protocol='star', depth=d, version='all_independent'),
                  'one': rate(RB, model=m, protocol='fully_connected', depth=d, version='one_agent')} for d in [1, 2]}
D['random'] = rnd

# ---------------------------------------------------------------- hints (20 targets, 16 without shortcuts)
SC = json.load(open(os.path.join(HERE, '..', 'analysis', 'shortcuts.json')))['hints']
HI = [r for r in load('run_hints/hints/runs.csv') if not SC[r['replicate']]]
hints = {}
for m in M3 + ['mock_norepeat']:
    hints[m] = {h: {'one': rate(HI, model=m, protocol='fully_connected', hints=h, version='one_agent'),
                    'fc_mute': rate(HI, model=m, protocol='fully_connected', hints=h, version='all_independent'),
                    'fc_talk': rate(HI, model=m, protocol='fully_connected', hints=h, version='base'),
                    'st_talk': rate(HI, model=m, protocol='star', hints=h, version='base'),
                    'fail_talk': rate(HI, model=m, protocol='fully_connected_failures', hints=h, version='base')}
                for h in ['none', 'split', 'all']}
D['hints'] = hints
write_csv('hints.csv', ['model', 'notes', 'one_agent', 'fc_silenced', 'fc_talk', 'star_talk', 'fc_failures_only'],
          [[m, h, v['one']['p'], v['fc_mute']['p'], v['fc_talk']['p'], v['st_talk']['p'], v['fail_talk']['p']]
           for m, e in hints.items() for h, v in e.items()])

# ---------------------------------------------------------------- hints2 family (40 targets without shortcuts)
HK = ('model', 'protocol', 'hints', 'decoys', 'replicate', 'version')
H2 = load('run_hints2/hints2/runs.csv') + load('run_hints2_8b/hints2_8b/runs.csv')
SH = load('run_hints2_share/hints2_share/runs.csv') + [r for r in load('run_hints2_8b_share/hints2_8b_share/runs.csv') if 'use_notesFalse' in r['cell']]
US = load('run_hints2_use/hints2_use/runs.csv') + [r for r in load('run_hints2_8b_share/hints2_8b_share/runs.csv') if 'use_notesTrue' in r['cell']]
H2, SH, US = ([r for r in X if int(r['replicate']) < 40] for X in (H2, SH, US))  # the report uses the first 40 targets (hints2_8b* now hold 80)
h2 = {}
for m in M4 + ['mock_norepeat']:
    e = {}
    for d in ['True', 'False']:
        c = {'one': rate(H2, model=m, protocol='fully_connected', hints='split', decoys=d, version='one_agent'),
             'fc_mute': rate(H2, model=m, protocol='fully_connected', hints='split', decoys=d, version='all_independent'),
             'fc_talk': rate(H2, model=m, protocol='fully_connected', hints='split', decoys=d, version='base'),
             'st_mute': rate(H2, model=m, protocol='star', hints='split', decoys=d, version='all_independent'),
             'st_talk': rate(H2, model=m, protocol='star', hints='split', decoys=d, version='base'),
             'share': rate(SH, model=m, decoys=d), 'use': rate(US, model=m, decoys=d)}
        for p in ['fully_connected', 'star']:
            c[f'{p}:talk_minus_mute'] = contrast(H2, dict(model=m, protocol=p, hints='split', decoys=d, version='base'), dict(version='all_independent'), HK)
        e[d] = c
    e['all'] = {'one': rate(H2, model=m, protocol='fully_connected', hints='all', version='one_agent'),
                'fc_mute': rate(H2, model=m, protocol='fully_connected', hints='all', version='all_independent'),
                'fc_talk': rate(H2, model=m, protocol='fully_connected', hints='all', version='base'),
                'st_mute': rate(H2, model=m, protocol='star', hints='all', version='all_independent'),
                'st_talk': rate(H2, model=m, protocol='star', hints='all', version='base')}
    for p in ['fully_connected', 'star']:
        e['all'][f'{p}:talk_minus_mute'] = contrast(H2, dict(model=m, protocol=p, hints='all', version='base'), dict(version='all_independent'), HK)
    h2[m] = e
D['hints2'] = h2
write_csv('hints2.csv', ['model', 'notes', 'one_agent_both', 'fc_silenced', 'fc_talk', 'fc_share', 'fc_share_use',
                         'star_silenced', 'star_talk'],
          [[m, {'True': 'split+decoys', 'False': 'split'}[d], e[d]['one']['p'], e[d]['fc_mute']['p'], e[d]['fc_talk']['p'],
            e[d]['share']['p'] if e[d]['share'] else '', e[d]['use']['p'] if e[d]['use'] else '',
            e[d]['st_mute']['p'], e[d]['st_talk']['p']] for m, e in h2.items() for d in ['True', 'False']] +
          [[m, 'all', e['all']['one']['p'], e['all']['fc_mute']['p'], e['all']['fc_talk']['p'], '', '',
            e['all']['st_mute']['p'], e['all']['st_talk']['p']] for m, e in h2.items()])

# ---------------------------------------------------------------- who tells, who acts (the logs; docs/analysis/k4.py)
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ['k4.py', os.path.join(HERE, '..', 'analysis')]
    k4 = runpy.run_path(os.path.join(HERE, '..', 'analysis', 'k4.py'))
logs = {}
for m in M4:
    logs[m] = {}
    for d in ['True', 'False']:
        for n in ['talk', 'share', 'share+use']:
            st = k4['stats'][(m, n, d)]
            g = lambda name: round(100 * mean(st[name]), 1) if st.get(name) else None
            logs[m][f'{n}|{d}'] = {
                'tell_inter_first': g('inter recipe: holder names it in 1st message'),
                'tell_target_first': g('target recipe: holder names it in 1st message'),
                'recv_inter_next': g('told, can play: next action plays the intermediate recipe'),
                'recv_inter_n': len(st.get('told, can play: next action plays the intermediate recipe', [])),
                'recv_target_next': g('told, can play: next action plays the target recipe'),
                'decoy_msgs': g('messages naming a decoy note')}
D['logs'] = logs
write_csv('tell_receive.csv', ['model', 'condition', 'decoys', 'holder_names_inter_recipe_1st_msg', 'holder_names_target_recipe_1st_msg',
                               'receiver_plays_inter_recipe_next', 'n_receiver_cases', 'messages_naming_decoy'],
          [[m, k.split('|')[0], k.split('|')[1], v['tell_inter_first'], v['tell_target_first'], v['recv_inter_next'],
            v['recv_inter_n'], v['decoy_msgs']] for m, e in logs.items() for k, v in e.items()])

# ---------------------------------------------------------------- remaking the intermediate (F44), everyone holds both
def remake(folder):
    """share of an agent's actions, once it holds the intermediate, that remake it"""
    xs = []
    for run in sorted(glob.glob(folder + '/seed0_rep*')):
        if int(run.rsplit('rep', 1)[1]) >= 40: continue
        meta = json.load(open(run + '/meta.json'))
        a, b, inter = meta['task']['path'][0]
        have = defaultdict(set)
        for l in open(run + '/events.jsonl'):
            e = json.loads(l)
            if e['kind'] != 'env_step': continue
            ag, c = e['data']['agent'], sorted(e['data']['combo'])
            if inter in have[ag]: xs.append(float(c == sorted([a, b])))
            have[ag] |= set(e['data']['new_items'])
    return round(100 * mean(xs), 1) if xs else None, len(xs)
rm = {}
for m in M4:
    root = 'run_hints2_8b/hints2_8b' if m == 'qwen3-8b' else 'run_hints2/hints2'
    rm[m] = {f'{p}|{v}': remake(f'{root}/{m}__{p}__hintsall__decoysTrue/{v}')
             for p in ['fully_connected', 'star'] for v in ['all_independent', 'base']}
D['remake'] = rm
write_csv('remake.csv', ['model', 'protocol', 'version', 'remake_pct', 'actions'],
          [[m, k.split('|')[0], k.split('|')[1], v[0], v[1]] for m, e in rm.items() for k, v in e.items()])

json.dump(D, open(os.path.join(OUT, 'data.json'), 'w'), indent=1)
print('wrote', OUT)
