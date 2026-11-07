import json, random, glob, sys, csv
from collections import defaultdict
sys.path.insert(0, '.')
from framework.tasks.wordcraft.recipes import RecipeBook, combo
book = RecipeBook.load('data/alchemy2.json')
tasks = {}
for m in glob.glob('run_main/main/qwen3-4b__star__*/base/*/meta.json'):
    meta = json.load(open(m))
    t = json.load(open(m.replace('meta.json', 'summary.json')))['task']
    d = meta['config']['task']['depth']
    tasks[(d, meta['config']['run']['replicate'])] = t
def play(team, steps, t, rng, shared_fail=False):
    invs = [list(t['initial']) for _ in range(team)]
    tried = [set() for _ in range(team)]; failed = set()
    for s in range(steps):
        for i in range(team):
            inv = invs[i]
            pairs = [combo(a, b) for k, a in enumerate(inv) for b in inv[k:]]
            opts = pairs if shared_fail == 2 else ([p for p in pairs if p not in tried[i] and not (shared_fail == 1 and p in failed)] or pairs)
            p = rng.choice(opts); tried[i].add(p)
            res = book.combine(*p)
            if not res: failed.add(p)
            for r in res:
                if r not in inv: inv.append(r)
            if t['target'] in inv: return True
    return False
rng = random.Random(0); N = 2000
out = defaultdict(list)
for (d, rep), t in sorted(tasks.items()):
    for name, team, steps, sf in [('rand1x4', 1, 4, 0), ('rand1x17', 1, 17, 0), ('rand3x4', 3, 4, 0), ('rand6x4', 6, 4, 0), ('rand6x4_sharedfail', 6, 4, 1), ('repl1x4', 1, 4, 2), ('repl6x4', 6, 4, 2)]:
        out[(d, name)].append(sum(play(team, steps, t, rng, sf) for _ in range(N)) / N)
    out[(d, 'n_items')].append(len(t['initial']))
for k, v in sorted(out.items()):
    print(k, round(sum(v) / len(v), 3))
# LLM success from runs.csv (count shared run_dir once)
rows = list(csv.DictReader(open('run_main/main/runs.csv')))
seen = set(); acc = defaultdict(list)
for r in rows:
    key = (r['run_dir'])
    if key in seen: continue
    seen.add(key)
    prot = r['protocol'] if r['version'] not in ('one_agent',) else '-'
    acc[(r['depth'], prot, r['version'])].append(float(r['solved_any']))
for k, v in sorted(acc.items()):
    print(k, len(v), round(sum(v) / len(v), 3))
