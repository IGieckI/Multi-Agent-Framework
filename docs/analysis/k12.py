# The review's grids (2026-11-12, F50-F53), paired by target with the grids they rerun (targets 0-39, fully connected):
# 1. hints2_share_pub: share instruction + publish_on_no_op - share alone (F50)
# 2. hints2_channels(_share): results only / messages only against silenced and both channels (F51)
# 3. hints2_ex2: prompts v3 (the example plays step 2) - v2, everyone holds both; remake share (F53)
# 4. renamed(_share): made-up names, so no recipe can be known (F52); who solves a silenced split team
# Logs: who tells (k6.py's `mentions`), what a told receiver plays, who makes the target first, remakes (k7.py)
import sys, json, glob
from collections import defaultdict, Counter
sys.path.insert(0, sys.argv[1])
from lib import load, boot, fmt, sign_test, mean
_rows = {}


def S(root, cell, version, reps=range(40)):
    """replicate -> solved_any"""
    if root not in _rows:
        _rows[root] = load(f'{root}/runs.csv')
    return {int(r['replicate']): float(r['solved_any']) for r in _rows[root]
            if r['cell'] == cell and r['version'] == version and int(r['replicate']) in reps}


def contrast(label, a, b, width=46):
    ks = sorted(set(a) & set(b)); x, y = [a[k] for k in ks], [b[k] for k in ks]
    if not ks:
        print(f'  {label:{width}s} (no pairs)'); return
    w, l, p = sign_test(x, y)
    print(f"  {label:{width}s} {100 * mean(x):5.1f}% vs {100 * mean(y):5.1f}%  {fmt(*boot([u - v for u, v in zip(x, y)], cluster=ks))}"
          f"  sign {w}/{l} p={p:.3g}  n={len(ks)}")


def mentions(text, items):
    t = text.lower(); return all(i.lower() in t for i in items)


same = lambda c, n: sorted(c) == sorted(n[:2])


def logs(run):
    """per run, from events.jsonl: who holds / tells / plays the path recipes, who made the target first, remakes"""
    meta = json.load(open(run + '/meta.json')); task, notes = meta['task'], meta['notes']
    path, target = task['path'], task['target']
    inter = path[0][2]
    holders = {k: [ag for ag, ns in notes.items() if any(same(n[:2], p) for n in ns)] for k, p in enumerate(path)}
    have = {ag: set(task['initial']) for ag in notes}
    msgs, heard, tried = {}, defaultdict(set), defaultdict(set)
    named = {0: 0, 1: 0}; plays = {0: [], 1: []}; remakes = []; solver = None; texts = 0
    for l in open(run + '/events.jsonl'):
        e = json.loads(l); k_ = e['kind']
        if k_ == 'msg_published':
            x = e['data']['message']; msgs[x['id']] = x
            if x['type'] == 'message':
                texts += 1
                for k, p in enumerate(path):
                    if x['sender'] in holders[k] and mentions(x['payload']['text'], p[:2]):
                        named[k] += 1
        elif k_ == 'msg_delivered':
            x = msgs.get(e['data']['id'])
            if x and x['type'] == 'message':
                for k, p in enumerate(path):
                    if x['sender'] in holders[k] and mentions(x['payload']['text'], p[:2]):
                        heard[e['data']['receiver']].add(k)
        elif k_ == 'env_step':
            ag, c = e['data']['agent'], e['data']['combo']
            for k, (a, b, r) in enumerate(path):
                if ag in holders[k] or k not in heard[ag] or r in have[ag] or k in tried[ag]:
                    continue
                if a in have[ag] and b in have[ag]:
                    plays[k].append(float(same(c, (a, b))))
            for k, p in enumerate(path):
                if same(c, p):
                    tried[ag].add(k)
            if inter in have[ag]:
                remakes.append(float(same(c, path[0])))
            if target in e['data']['new_items'] and solver is None:
                solver = ('target holder' if ag in holders[1] else 'inter holder' if ag in holders[0] else 'other')
            have[ag] |= set(e['data']['new_items'])
    receivers = {k: sum(1 for ag in notes if ag not in holders[k] and k in heard[ag]) for k in (0, 1)}
    return dict(named=named, told=receivers, plays=plays, remakes=remakes, solver=solver, messages=texts)


def summary(folder, reps=range(40)):
    runs = [r for r in sorted(glob.glob(folder + '/seed0_rep*')) if int(r.rsplit('rep', 1)[1]) in reps]
    L = [logs(r) for r in runs]
    return dict(n=len(L), messages=mean([x['messages'] for x in L]),
                tgt_named=mean([float(x['named'][1] > 0) for x in L]), tgt_told=mean([x['told'][1] for x in L]),
                int_named=mean([float(x['named'][0] > 0) for x in L]),
                tgt_play=mean(sum((x['plays'][1] for x in L), [])), n_play=len(sum((x['plays'][1] for x in L), [])),
                remake=mean(sum((x['remakes'] for x in L), [])), n_remake=len(sum((x['remakes'] for x in L), [])),
                solvers=Counter(x['solver'] for x in L if x['solver']))


def show(label, s, keys):
    parts = []
    for k in keys:
        v = s[k]
        parts.append(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {dict(v) if isinstance(v, Counter) else v}")
    print(f'    {label:28s} ' + ', '.join(parts))


H2, H8, H12 = 'run_hints2/hints2', 'run_hints2_8b/hints2_8b', 'run_hints2_12b/hints2_12b'
SRC = {'qwen3-4b': H2, 'gemma3-4b': H2, 'llama3.2-3b': H2, 'qwen3-8b': H8, 'gemma3-12b': H12}
split = lambda m, d='True', p='fully_connected': f'{m}__{p}__hintssplit__decoys{d}'
both = lambda m, p='fully_connected': f'{m}__{p}__hintsall__decoysTrue'
TELL = ['messages', 'tgt_named', 'tgt_told', 'tgt_play', 'n_play']

print('# 1. share + publish_on_no_op (hints2_share_pub) - share alone (F50)')
PUB = 'run_hints2_share_pub/hints2_share_pub'
for m in ['qwen3-4b', 'gemma3-4b', 'qwen3-8b']:
    for d in ['True', 'False']:
        sh_root, sh_cell = (('run_hints2_8b_share/hints2_8b_share', split(m, d) + '__use_notesFalse') if m == 'qwen3-8b'
                            else ('run_hints2_share/hints2_share', split(m, d)))
        pub, sh = S(PUB, split(m, d), 'base'), S(sh_root, sh_cell, 'base')
        print(f'== {m} decoys {d}')
        contrast('share+pub - share', pub, sh)
        contrast('share+pub - silenced', pub, S(SRC[m], split(m, d), 'all_independent'))
        contrast('share+pub - everyone holds both, silenced', pub, S(SRC[m], both(m), 'all_independent')) if d == 'True' else None
        show('share (logs)', summary(f'{sh_root}/{sh_cell}/base'), TELL)
        show('share+pub (logs)', summary(f'{PUB}/{split(m, d)}/base'), TELL)

print('\n# 2. channels (split, decoys): results only / messages only (hints2_channels) vs silenced and both (F51)')
CH, CHS = 'run_hints2_channels/hints2_channels', 'run_hints2_channels_share/hints2_channels_share'
for m, shr in [('qwen3-8b', 'run_hints2_8b_share/hints2_8b_share'), ('gemma3-12b', 'run_hints2_12b_share/hints2_12b_share')]:
    print(f'== {m}')
    sil, bo = S(SRC[m], split(m), 'all_independent'), S(SRC[m], split(m), 'base')
    res, mes = S(CH, split(m, p='fully_connected_results'), 'base'), S(CH, split(m, p='fully_connected_messages'), 'base')
    msh, bsh = S(CHS, split(m, p='fully_connected_messages'), 'base'), S(shr, split(m) + '__use_notesFalse', 'base')
    print(f'  rates: silenced {100 * mean(list(sil.values())):.1f}, results only {100 * mean(list(res.values())):.1f}, '
          f'messages only {100 * mean(list(mes.values())):.1f}, both {100 * mean(list(bo.values())):.1f}; with the share '
          f'instruction: messages only {100 * mean(list(msh.values())):.1f}, both {100 * mean(list(bsh.values())):.1f}')
    contrast('results only - silenced', res, sil)
    contrast('messages only - silenced', mes, sil)
    contrast('both - results only', bo, res)
    contrast('both - messages only', bo, mes)
    contrast('messages only + share - messages only', msh, mes)
    contrast('messages only + share - both + share', msh, bsh)
    for lab, fo in [('results only', f'{CH}/{split(m, p="fully_connected_results")}/base'),
                    ('messages only', f'{CH}/{split(m, p="fully_connected_messages")}/base'),
                    ('both', f'{SRC[m]}/{split(m)}/base'),
                    ('messages only + share', f'{CHS}/{split(m, p="fully_connected_messages")}/base'),
                    ('both + share', f'{shr}/{split(m)}__use_notesFalse/base')]:
        show(lab, summary(fo), TELL + ['solvers'])

print('\n# 3. prompts v3 (hints2_ex2) - v2 (hints2 / hints2_8b / hints2_12b), everyone holds both (F53)')
EX = 'run_hints2_ex2/hints2_ex2'
for m in ['llama3.2-3b', 'gemma3-4b', 'qwen3-8b', 'gemma3-12b']:
    print(f'== {m}')
    for v in ['all_independent', 'base', 'one_agent']:
        contrast(f'{v}: v3 - v2', S(EX, both(m), v), S(SRC[m], both(m), v))
    for p_, root in [('v2', SRC[m]), ('v3', EX)]:
        contrast(f'{p_} talk - silenced', S(root, both(m), 'base'), S(root, both(m), 'all_independent'))
    for v in ['all_independent', 'base', 'one_agent']:
        a, b = summary(f'{SRC[m]}/{both(m)}/{v}'), summary(f'{EX}/{both(m)}/{v}')
        print(f"    remake share {v:16s} v2 {100 * a['remake']:4.0f}% (n {a['n_remake']}) -> v3 {100 * b['remake']:4.0f}% (n {b['n_remake']})")

print('\n# 4. renamed items (renamed, renamed_share), prompts v2r + publish_on_no_op (F52)')
RN, RNS = 'run_renamed/renamed', 'run_renamed_share/renamed_share'
for m in ['qwen3-4b', 'gemma3-4b', 'qwen3-8b', 'gemma3-12b']:
    print(f'== {m}')
    cells = {('split', v): S(RN, split(m), v) for v in ['all_independent', 'base', 'one_agent']}
    cells |= {('all', v): S(RN, both(m), v) for v in ['all_independent', 'base', 'one_agent']}
    cells[('split', 'share')] = S(RNS, split(m), 'base')
    print('  rates: ' + ', '.join(f'{h} {v} {100 * mean(list(x.values())):.1f}' for (h, v), x in cells.items()))
    contrast('split: talk - silenced', cells[('split', 'base')], cells[('split', 'all_independent')])
    contrast('split: share - talk', cells[('split', 'share')], cells[('split', 'base')])
    contrast('split: share - silenced', cells[('split', 'share')], cells[('split', 'all_independent')])
    contrast('split: share - one agent holding both', cells[('split', 'share')], cells[('split', 'one_agent')])
    contrast('split: share - everyone holds both, silenced', cells[('split', 'share')], cells[('all', 'all_independent')])
    contrast('all: talk - silenced', cells[('all', 'base')], cells[('all', 'all_independent')])
    for h, v in [('split', 'all_independent'), ('split', 'base'), ('split', 'one_agent'), ('all', 'all_independent'),
                 ('all', 'base'), ('all', 'one_agent')]:
        cell = split(m) if h == 'split' else both(m)
        contrast(f'renamed - original names: {h} {v}', S(RN, cell, v), S(SRC[m], cell, v))
    contrast('renamed - original names: split share', cells[('split', 'share')],
             S('run_hints2_8b_share/hints2_8b_share', split(m) + '__use_notesFalse', 'base') if m == 'qwen3-8b' else
             S('run_hints2_12b_share/hints2_12b_share', split(m) + '__use_notesFalse', 'base') if m == 'gemma3-12b' else
             S('run_hints2_share/hints2_share', split(m), 'base'))
    for lab, fo in [('split silenced', f'{RN}/{split(m)}/all_independent'), ('split talk', f'{RN}/{split(m)}/base'),
                    ('split share', f'{RNS}/{split(m)}/base'), ('all talk', f'{RN}/{both(m)}/base')]:
        show(lab, summary(fo), TELL + ['remake', 'solvers'])
