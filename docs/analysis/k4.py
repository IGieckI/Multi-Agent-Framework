# hints2_use / hints2_8b / hints2_8b_share from the logs: who tells, and what a receiver does once a holder's recipe
# reached it (F42-). Fully connected, notes split. Same measures for hints2 and hints2_share, for comparison
import sys, json, glob
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import mean
def mentions(text, items): t = text.lower(); return all(i.lower() in t for i in items)
same = lambda c, n: sorted(c) == sorted(n[:2])
LLM4 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
conds = []  # (group, name, folder pattern per decoys)
for m in LLM4 + ['qwen3-8b']:
    big = m == 'qwen3-8b'
    h2 = 'run_hints2_8b/hints2_8b' if big else 'run_hints2/hints2'
    conds += [(m, 'silenced', lambda d, h2=h2, m=m: f'{h2}/{m}__fully_connected__hintssplit__decoys{d}/all_independent'),
              (m, 'talk', lambda d, h2=h2, m=m: f'{h2}/{m}__fully_connected__hintssplit__decoys{d}/base')]
    if big:
        conds += [(m, 'share', lambda d, m=m: f'run_hints2_8b_share/hints2_8b_share/{m}__fully_connected__hintssplit__decoys{d}__use_notesFalse/base'),
                  (m, 'share+use', lambda d, m=m: f'run_hints2_8b_share/hints2_8b_share/{m}__fully_connected__hintssplit__decoys{d}__use_notesTrue/base')]
    else:
        conds += [(m, 'share', lambda d, m=m: f'run_hints2_share/hints2_share/{m}__fully_connected__hintssplit__decoys{d}/base'),
                  (m, 'share+use', lambda d, m=m: f'run_hints2_use/hints2_use/{m}__fully_connected__hintssplit__decoys{d}/base')]
stats = defaultdict(lambda: defaultdict(list))
for m, name, pat in conds:
    for d in ['True', 'False']:
        runs = glob.glob(pat(d) + '/seed0_rep*')
        assert len(runs) == 40, (pat(d), len(runs))
        grp = '4B' if m != 'qwen3-8b' else '8B'
        for run in runs:
            meta = json.load(open(run + '/meta.json')); path = meta['task']['path']; notes = meta['notes']
            inter, target = meta['task']['intermediate'][0], meta['task']['target']
            ev = [json.loads(l) for l in open(run + '/events.jsonl')]
            msgs = {e['data']['message']['id']: e['data']['message'] for e in ev if e['kind'] == 'msg_published'}
            texts = [x for x in msgs.values() if x['type'] == 'message']
            heard = defaultdict(set)   # agent -> ids of the path recipes a holder told it so far
            have = {ag: set(meta['task']['initial']) for ag in notes}
            holders = {k: [ag for ag, ns in notes.items() if any(same((n[0], n[1]), (a, b)) for n in ns)] for k, (a, b, r) in enumerate(path)}
            for k_ in (grp, m):
                st = stats[(k_, name, d)]
                st_list = st
            acts = defaultdict(list)  # (k) -> list of 0/1: a receiver's action when recipe k reached it and it can play it
            first = defaultdict(list)
            waiting = {}  # (agent, k) -> True when heard, not yet acted on
            tried = defaultdict(set)
            for e in ev:
                if e['kind'] == 'msg_delivered':
                    x = msgs.get(e['data']['id'])
                    if x and x['type'] == 'message':
                        for k, (a, b, r) in enumerate(path):
                            if x['sender'] in holders[k] and mentions(x['payload']['text'], [a, b]) and k not in heard[e['data']['receiver']]:
                                heard[e['data']['receiver']].add(k); waiting[(e['data']['receiver'], k)] = True
                elif e['kind'] == 'env_step':
                    ag = e['data']['agent']; c = e['data']['combo']
                    for k, (a, b, r) in enumerate(path):
                        if ag in holders[k] or k not in heard[ag] or r in have[ag] or k in tried[ag]: continue
                        if a in have[ag] and b in have[ag]:
                            acts[k].append(float(same(c, (a, b))))
                            if waiting.pop((ag, k), False): first[k].append(float(same(c, (a, b))))
                    for k, (a, b, r) in enumerate(path):
                        if same(c, (a, b)): tried[ag].add(k)
                    have[ag] |= set(e['data']['new_items'])
            made = lambda item: {ag for ag in have if item in have[ag]}
            for key in [(grp, name, d), (m, name, d)]:
                st = stats[key]
                for k, (a, b, r) in enumerate(path):
                    lab = 'inter' if k == 0 else 'target'
                    hm = [x for x in texts if x['sender'] in holders[k]]
                    if name != 'silenced':
                        st[f'{lab} recipe: holder names it in 1st message'].append(float(bool(hm) and mentions(hm[0]['payload']['text'], [a, b])))
                        st[f'{lab} recipe: holder names it at least once'].append(float(any(mentions(x['payload']['text'], [a, b]) for x in hm)))
                        st[f'{lab} recipe: receivers told (of 5)'].append(sum(1 for ag in notes if ag not in holders[k] and k in heard[ag]))
                    st[f'{lab} recipe: share of non-holders that try it'].append(mean([float(k in tried[o]) for o in notes if o not in holders[k]]))
                    st[f'{lab} recipe: holder tries it'].append(mean([float(k in tried[o]) for o in holders[k]]))
                st['told, can play: next action plays the intermediate recipe'] += first[0]
                st['told, can play: next action plays the target recipe'] += first[1]
                st['told, can play: all actions playing the intermediate recipe'] += acts[0]
                st['told, can play: all actions playing the target recipe'] += acts[1]
                st['agents that made the intermediate (of 6)'].append(len(made(inter)))
                st['agents that made the target (of 6)'].append(len(made(target)))
                st['solved'].append(float(bool(made(target))))
                if name != 'silenced':
                    st['messages per run'].append(len(texts))
                    st['messages naming a decoy note'].append(mean([float(any(mentions(x['payload']['text'], n[:2]) for n in notes[x['sender']] if not any(same((n[0], n[1]), p) for p in path))) for x in texts]) if texts else 0.0)
order = ['silenced', 'talk', 'share', 'share+use']
for grp in ['4B', 'qwen3-4b', 'gemma3-4b', 'llama3.2-3b', '8B']:
    for d in ['True', 'False']:
        keys = [(grp, n, d) for n in order]
        print(f"\n== {grp}, decoys={d}" + ''.join(f"{n:>11s}" for n in order))
        names = []
        for k in keys:
            for n in stats[k]:
                if n not in names: names.append(n)
        for n in names:
            row = []
            for k in keys:
                xs = stats[k].get(n, [])
                row.append(f"{mean(xs):11.2f}" if xs else f"{'-':>11s}")
            cnt = '  n=' + '/'.join(str(len(stats[k].get(n, []))) for k in keys) if n.startswith('told') else ''
            print(f"  {n:56s}" + ''.join(row) + cnt)
