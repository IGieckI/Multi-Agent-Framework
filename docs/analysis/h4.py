import sys, json, glob
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from lib import mean
sc = json.load(open(sys.argv[1] + '/shortcuts.json')); short = {int(r) for r, v in sc['hints'].items() if v}
LLM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
prots = ['fully_connected', 'fully_connected_results', 'fully_connected_messages', 'fully_connected_failures', 'star']
def mentions(text, items):
    t = text.lower(); return all(i.lower() in t for i in items)
stats = defaultdict(lambda: defaultdict(list))
for m in LLM:
    for p in prots:
        for v in ['base', 'all_independent']:
            if v == 'all_independent' and p not in ('fully_connected', 'star'): continue
            for d in glob.glob(f'run_hints/hints/{m}__{p}__depth2__distractors6__hintssplit/{v}/seed0_rep*'):
                rep = int(d.rsplit('rep', 1)[1])
                if rep in short: continue
                meta = json.load(open(d + '/meta.json')); path = meta['task']['path']; notes = meta['notes']
                inter, target = meta['task']['intermediate'][0], meta['task']['target']
                ev = [json.loads(l) for l in open(d + '/events.jsonl')]
                steps = [e for e in ev if e['kind'] == 'env_step']
                msgs = [e['data']['message'] for e in ev if e['kind'] == 'msg_published' and e['data']['message']['type'] == 'message']
                key = (p, v)
                for k, (a, b, r) in enumerate(path):
                    holders = [ag for ag, ns in notes.items() if any(sorted(n[:2]) == sorted([a, b]) for n in ns)]
                    others = [ag for ag in notes if ag not in holders]
                    tried = lambda ag: [s['t'] for s in steps if s['data']['agent'] == ag and sorted(s['data']['combo']) == sorted([a, b])]
                    stats[key][f'r{k} holder used it'].append(mean([float(bool(tried(h))) for h in holders]))
                    stats[key][f'r{k} holder used it at step 0'].append(mean([float(0 in tried(h)) for h in holders]))
                    stats[key][f'r{k} others tried it'].append(mean([float(bool(tried(o))) for o in others]))
                    if msgs or v == 'base':
                        hm = [x for x in msgs if x['sender'] in holders]
                        stats[key][f'r{k} holder messages naming it'].append(mean([float(mentions(x['payload']['text'], [a, b])) for x in hm]) if hm else None)
                        stats[key][f'r{k} holder ever names it'].append(float(any(mentions(x['payload']['text'], [a, b]) for x in hm)) if hm else None)
                # did anyone make the intermediate; did the holder of r1 (target recipe) ever have the intermediate
                made_inter = {s['data']['agent'] for s in steps if inter in s['data']['new_items']}
                h1 = [ag for ag, ns in notes.items() if any(sorted(n[:2]) == sorted(path[1][:2]) for n in ns)]
                stats[key]['someone made the intermediate'].append(float(bool(made_inter)))
                stats[key]['target-recipe holder made the intermediate'].append(float(any(h in made_inter for h in h1)))
                solved_by = {s['data']['agent'] for s in steps if target in s['data']['new_items']}
                stats[key]['solved'].append(float(bool(solved_by)))
                stats[key]['solved by a target-recipe holder'].append(float(any(h in solved_by for h in h1)))
for key, d in stats.items():
    print(key, f"n={len(d['solved'])}")
    for name, xs in d.items():
        mm = mean(xs); print(f"   {name:45s} {mm:.2f}" if mm is not None else f"   {name:45s} -")
