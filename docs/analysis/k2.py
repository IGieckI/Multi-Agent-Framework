import sys, json, glob
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import mean
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
def mentions(text, items): t = text.lower(); return all(i.lower() in t for i in items)
same = lambda c, n: sorted(c) == sorted(n[:2])
stats = defaultdict(lambda: defaultdict(list))
for share, root in [(False, 'run_hints2/hints2'), (True, 'run_hints2_share/hints2_share')]:
    for d in ['True', 'False']:
        for v in (['base'] if share else ['base', 'all_independent']):
            for m in LLM:
                for run in glob.glob(f'{root}/{m}__fully_connected__hintssplit__decoys{d}/{v}/seed0_rep*'):
                    meta = json.load(open(run + '/meta.json')); path = meta['task']['path']; notes = meta['notes']
                    inter, target = meta['task']['intermediate'][0], meta['task']['target']
                    ev = [json.loads(l) for l in open(run + '/events.jsonl')]
                    steps = [e for e in ev if e['kind'] == 'env_step']
                    msgs = [(e['data']['message'], e.get('t')) for e in ev if e['kind'] == 'msg_published' and e['data']['message']['type'] == 'message']
                    errs = [e for e in ev if e['kind'] == 'llm_call' and e['data'].get('error')]
                    key = ('share' if share else v.replace('base', 'talk').replace('all_independent', 'silenced'), 'decoys' if d == 'True' else 'no decoys')
                    st = stats[key]
                    holders = {}
                    for k, (a, b, r) in enumerate(path):
                        holders[k] = [ag for ag, ns in notes.items() if any(same((n[0], n[1]), (a, b)) for n in ns)]
                        others = [ag for ag in notes if ag not in holders[k]]
                        hm = [x for x, t in msgs if x['sender'] in holders[k]]
                        if v == 'base':
                            st[f'r{k} holder names it in its first message'].append(float(any(mentions(x['payload']['text'], [a, b]) for x in hm[:1])) if hm else 0.0)
                            st[f'r{k} holder names it at least once'].append(float(any(mentions(x['payload']['text'], [a, b]) for x in hm)))
                        tried = lambda ag: any(s['data']['agent'] == ag and same(s['data']['combo'], (a, b)) for s in steps)
                        st[f'r{k} a teammate tries it'].append(mean([float(tried(o)) for o in others]))
                    made = lambda item: {s['data']['agent'] for s in steps if item in s['data']['new_items']}
                    mi, mt = made(inter), made(target)
                    st['agents that made the intermediate (of 6)'].append(len(mi))
                    st['agents that made the target (of 6)'].append(len(mt))
                    st['target made by an agent holding neither recipe'].append(float(bool(mt - set(holders[0]) - set(holders[1]))))
                    st['solved'].append(float(bool(mt)))
                    st['refusals: target recipe before the intermediate'].append(sum(1 for e in errs if f"'{inter}'" in e['data']['error'] and 'not in your inventory' in e['data']['error']))
                    if v == 'base':
                        st['messages naming a decoy note'].append(mean([float(any(mentions(x['payload']['text'], n[:2]) for n in notes[x['sender']] if not any(same((n[0], n[1]), p) for p in path))) for x, t in msgs]) if msgs else 0.0)
order = [('talk', 'decoys'), ('share', 'decoys'), ('silenced', 'decoys'), ('talk', 'no decoys'), ('share', 'no decoys'), ('silenced', 'no decoys')]
names = []
for k in order:
    for n in stats[k]:
        if n not in names: names.append(n)
print(f"{'':52s}" + ''.join(f"{a[:5]+'/'+b[:4]:>13s}" for a, b in order))
for n in names:
    print(f"{n:52s}" + ''.join(f"{(mean(stats[k][n]) if stats[k].get(n) else float('nan')):13.2f}" for k in order))
print('n runs', [len(stats[k]['solved']) for k in order])
