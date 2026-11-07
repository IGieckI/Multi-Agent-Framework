import json, glob, sys
sys.path.insert(0, '.')
from framework.tasks.wordcraft.recipes import RecipeBook
book = RecipeBook.load('data/alchemy2.json')
def info(t):
    init = set(t['initial']); target = t['target']
    direct = [c for c in book.recipes_for(target) if set(c) <= init]
    # pairs in the initial inventory that make the target or any other route of length 2
    return direct
out = {}
for name, pat in [('hints', 'run_hints/hints/qwen3-4b__fully_connected__depth2__distractors6__hintsnone/base/*/meta.json'),
                  ('main d1', 'run_main/main/qwen3-4b__star__depth1__distractors3/base/*/meta.json'),
                  ('main d2', 'run_main/main/qwen3-4b__star__depth2__distractors0/base/*/meta.json')]:
    n = sh = 0; ex = []
    for m in sorted(glob.glob(pat)):
        meta = json.load(open(m)); t = meta['task']; rep = meta['config']['run']['replicate']
        d = info(t); n += 1
        if name == 'main d1':
            sh += len(d) > 1
            if len(d) > 1: ex.append((rep, t['target'], d))
        else:
            sh += bool(d)
            if d: ex.append((rep, t['target'], d))
        out.setdefault(name, {})[rep] = d
    print(name, f'{sh}/{n}', 'with a shortcut' if name != 'main d1' else 'with more than one direct recipe', ex[:8])
json.dump({k: {str(r): bool(v) for r, v in d.items()} for k, d in out.items()}, open(sys.argv[1] + '/shortcuts.json', 'w'))
