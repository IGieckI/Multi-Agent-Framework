# The numbers of the report's v2 sections (the grids after the first report: F46-F64). Run from the repo root:
#   python3 docs/report/make_data_v2.py
# Writes docs/report/data/data_v2.json (read by make_figures_v2.py and report_v2.typ); data.json (v1) is untouched.
# Every contrast is paired by target: d, 95% bootstrap by target (lo, hi), exact sign test (w, l discordant pairs, p).
import contextlib, glob, io, json, math, os, runpy, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, '..', 'analysis')
sys.path.insert(0, AN)
from lib import load, mean, boot, sign_test

OUT = os.path.join(HERE, 'data')
D = {}
_rows = {}
R40 = range(40)


def S(root, cell, version, reps=R40):
    """replicate -> solved_any"""
    if root not in _rows:
        _rows[root] = load(f'{root}/runs.csv')
    return {int(r['replicate']): float(r['solved_any']) for r in _rows[root]
            if r['cell'] == cell and r['version'] == version and int(r['replicate']) in reps}


def rate(x):
    if not x: return None
    m, lo, hi = boot(list(x.values()))
    return {'p': round(100 * m, 1), 'lo': round(100 * lo, 1), 'hi': round(100 * hi, 1), 'n': len(x)}


def pair(a, b, cl=lambda k: k):
    ks = sorted(set(a) & set(b))
    if not ks: return None
    x, y = [a[k] for k in ks], [b[k] for k in ks]
    m, lo, hi = boot([u - v for u, v in zip(x, y)], cluster=[cl(k) for k in ks])
    w, l, p = sign_test(x, y)
    return {'d': round(100 * m, 1), 'lo': round(100 * lo, 1), 'hi': round(100 * hi, 1), 'w': w, 'l': l,
            'p': round(p, 4), 'n': len(ks)}


def quiet(script):
    with contextlib.redirect_stdout(io.StringIO()):
        sys.argv = [script, AN]
        return runpy.run_path(os.path.join(AN, script))


k12 = quiet('k12.py')   # logs() / summary(): who tells, who is told, who plays, who solves, remakes
summary = k12['summary']
H2, H8, H12 = 'run_hints2/hints2', 'run_hints2_8b/hints2_8b', 'run_hints2_12b/hints2_12b'
SRC = {'qwen3-4b': H2, 'gemma3-4b': H2, 'llama3.2-3b': H2, 'qwen3-8b': H8, 'gemma3-12b': H12}
SHARE = {'qwen3-4b': ('run_hints2_share/hints2_share', ''), 'gemma3-4b': ('run_hints2_share/hints2_share', ''),
         'llama3.2-3b': ('run_hints2_share/hints2_share', ''),
         'qwen3-8b': ('run_hints2_8b_share/hints2_8b_share', '__use_notesFalse'),
         'gemma3-12b': ('run_hints2_12b_share/hints2_12b_share', '__use_notesFalse')}
split = lambda m, d='True', p='fully_connected': f'{m}__{p}__hintssplit__decoys{d}'
both = lambda m, p='fully_connected': f'{m}__{p}__hintsall__decoysTrue'
share = lambda m, d='True', reps=R40: S(SHARE[m][0], split(m, d) + SHARE[m][1], 'base', reps)

# ---------------------------------------------------------------- bigger models: gemma3-12b (40 targets), qwen3-8b (80): F46, F47
big = {}
for m, reps in [('gemma3-12b', R40), ('qwen3-8b', range(80)), ('qwen3-4b', R40), ('gemma3-4b', R40)]:
    e = {}
    for d in ['True', 'False']:
        mute, talk = S(SRC[m], split(m, d), 'all_independent', reps), S(SRC[m], split(m, d), 'base', reps)
        sh, one = share(m, d, reps), S(SRC[m], split(m, d), 'one_agent', reps)
        e[d] = {'mute': rate(mute), 'talk': rate(talk), 'share': rate(sh), 'one': rate(one),
                'talk_minus_mute': pair(talk, mute), 'share_minus_talk': pair(sh, talk), 'share_minus_one': pair(sh, one)}
    am, at = S(SRC[m], both(m), 'all_independent', reps), S(SRC[m], both(m), 'base', reps)
    sm, st = S(SRC[m], both(m, 'star'), 'all_independent', reps), S(SRC[m], both(m, 'star'), 'base', reps)
    e['all'] = {'mute': rate(am), 'talk': rate(at), 'talk_minus_mute': pair(at, am), 'star_talk_minus_mute': pair(st, sm)}
    e['targets'] = len(reps)
    big[m] = e
D['big'] = big

# ---------------------------------------------------------------- the "Items you made" line (hints2_made): F48
MD = 'run_hints2_made/hints2_made'
made = {}
for m in ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']:
    e = {}
    for v in ['all_independent', 'base', 'one_agent']:
        a, b = S(MD, both(m), v), S(SRC[m], both(m), v)
        ra, rb = summary(f'{MD}/{both(m)}/{v}'), summary(f'{SRC[m]}/{both(m)}/{v}')
        e[v] = {'without': rate(b), 'with': rate(a), 'diff': pair(a, b),
                'remake_without': round(100 * rb['remake'], 1), 'remake_with': round(100 * ra['remake'], 1)}
    made[m] = e
D['made'] = made

# ---------------------------------------------------------------- share + publish_on_no_op: F58
PUB = 'run_hints2_share_pub/hints2_share_pub'
pub = {}
for m in ['qwen3-4b', 'gemma3-4b', 'qwen3-8b']:
    e = {}
    for d in ['True', 'False']:
        a, b = S(PUB, split(m, d), 'base'), share(m, d)
        sa, sb = summary(f'{PUB}/{split(m, d)}/base'), summary(f'{SHARE[m][0]}/{split(m, d) + SHARE[m][1]}/base')
        e[d] = {'share': rate(b), 'pub': rate(a), 'diff': pair(a, b), 'told_share': round(sb['tgt_told'], 2),
                'told_pub': round(sa['tgt_told'], 2), 'play_share': round(100 * sb['tgt_play'], 1),
                'play_pub': round(100 * sa['tgt_play'], 1)}
    pub[m] = e
D['pub'] = pub

# ---------------------------------------------------------------- channels (results only / messages only): F59
CH, CHS = 'run_hints2_channels/hints2_channels', 'run_hints2_channels_share/hints2_channels_share'
ch = {}
for m in ['qwen3-8b', 'gemma3-12b']:
    c = {'mute': S(SRC[m], split(m), 'all_independent'), 'results': S(CH, split(m, p='fully_connected_results'), 'base'),
         'messages': S(CH, split(m, p='fully_connected_messages'), 'base'), 'both': S(SRC[m], split(m), 'base'),
         'messages_share': S(CHS, split(m, p='fully_connected_messages'), 'base'), 'both_share': share(m)}
    e = {k: rate(v) for k, v in c.items()}
    e |= {'results_minus_mute': pair(c['results'], c['mute']), 'messages_minus_mute': pair(c['messages'], c['mute']),
          'both_minus_results': pair(c['both'], c['results']), 'messages_share_minus_messages': pair(c['messages_share'], c['messages']),
          'messages_share_minus_both_share': pair(c['messages_share'], c['both_share'])}
    s = summary(f'{CH}/{split(m, p="fully_connected_messages")}/base')
    e['messages_logs'] = {'named': round(100 * s['tgt_named'], 1), 'told': round(s['tgt_told'], 2), 'play': round(100 * s['tgt_play'], 1)}
    r = summary(f'{CH}/{split(m, p="fully_connected_results")}/base')
    e['results_solvers'] = dict(r['solvers'])
    ch[m] = e
D['channels'] = ch

# ---------------------------------------------------------------- renamed items: F61, F62
RN, RNS = 'run_renamed/renamed', 'run_renamed_share/renamed_share'
rn = {}
for m in ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'gemma3-12b']:
    new = {'mute': S(RN, split(m), 'all_independent'), 'talk': S(RN, split(m), 'base'), 'share': S(RNS, split(m), 'base'),
           'one': S(RN, split(m), 'one_agent'), 'all_mute': S(RN, both(m), 'all_independent'), 'all_talk': S(RN, both(m), 'base')}
    old = {'mute': S(SRC[m], split(m), 'all_independent'), 'talk': S(SRC[m], split(m), 'base'), 'share': share(m),
           'one': S(SRC[m], split(m), 'one_agent'), 'all_mute': S(SRC[m], both(m), 'all_independent'),
           'all_talk': S(SRC[m], both(m), 'base')}
    e = {'renamed': {k: rate(v) for k, v in new.items()}, 'real': {k: rate(v) for k, v in old.items()},
         'renamed_minus_real': {k: pair(new[k], old[k]) for k in new},
         'talk_minus_mute': pair(new['talk'], new['mute']), 'share_minus_talk': pair(new['share'], new['talk']),
         'share_minus_one': pair(new['share'], new['one']), 'share_minus_all_mute': pair(new['share'], new['all_mute']),
         'all_talk_minus_mute': pair(new['all_talk'], new['all_mute']),
         'real_talk_minus_mute': pair(old['talk'], old['mute'])}
    st, ss = summary(f'{RN}/{split(m)}/base'), summary(f'{RNS}/{split(m)}/base')
    so, sr = summary(f'{RN}/{split(m)}/one_agent'), summary(f'{SRC[m]}/{split(m)}/one_agent')
    e['logs'] = {'talk_named': round(100 * st['tgt_named'], 1), 'share_named': round(100 * ss['tgt_named'], 1),
                 'share_play': round(100 * ss['tgt_play'], 1), 'remake_one_renamed': round(100 * so['remake'], 1),
                 'remake_one_real': round(100 * sr['remake'], 1)}
    rn[m] = e
D['renamed'] = rn

# ---------------------------------------------------------------- prompts v3 (the example plays step 2): F60
EX = 'run_hints2_ex2/hints2_ex2'
ex = {}
for m in ['qwen3-8b', 'gemma3-12b', 'gemma3-4b', 'llama3.2-3b']:
    e = {}
    for v in ['all_independent', 'base', 'one_agent']:
        a, b = S(EX, both(m), v), S(SRC[m], both(m), v)
        e[v] = {'v2': rate(b), 'v3': rate(a), 'diff': pair(a, b),
                'remake_v2': round(100 * summary(f'{SRC[m]}/{both(m)}/{v}')['remake'], 1),
                'remake_v3': round(100 * summary(f'{EX}/{both(m)}/{v}')['remake'], 1)}
    ex[m] = e
D['ex2'] = ex

# ---------------------------------------------------------------- team size 1 / 3 / 6 / 12 and token confidence: F55, F57
FU = 'run_followup/followup'
cells = lambda m, p: [f'{m}__{p}__depth1__distractors3', f'{m}__{p}__depth2__distractors0']
def Sm(root, m, p, v):
    out = {}
    for i, c in enumerate(cells(m, p)):
        out |= {(i, k): x for k, x in S(root, c, v, range(20)).items()}
    return out
size = {}
for m in ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']:
    one, three = Sm(FU, m, 'fully_connected', 'one_agent'), Sm(FU, m, 'fully_connected3', 'all_independent')
    six, twelve = Sm(FU, m, 'fully_connected', 'all_independent'), Sm('run_size12/size12', m, 'fully_connected12', 'all_independent')
    size[m] = {'1': rate(one), '3': rate(three), '6': rate(six), '12': rate(twelve),
               '12-6': pair(twelve, six), '12-3': pair(twelve, three), '12-1': pair(twelve, one)}
k10 = quiet('k10.py')
first_moves, confidence_lines, auc, spearman = k10['first_moves'], k10['confidence_lines'], k10['auc'], k10['spearman']
from collections import Counter
for m in ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']:
    right, wrong, tg = [], [], []
    for run in sorted(glob.glob(f'run_size12/size12/{m}__fully_connected12__*/all_independent/seed0_rep*')):
        first, solved = first_moves(run)
        lines = confidence_lines('run_size12/size12', 'size12', run.split('/')[2], 'all_independent', run.rsplit('rep', 1)[1])
        ps = {}
        for c in lines or []:
            if c['t'] == 0 and c['error'] is None and c.get('combo_logprob') is not None and c['agent'] in first:
                ps.setdefault(c['agent'], math.exp(c['combo_logprob']))
        for a, p in ps.items():
            (right if first[a][1] else wrong).append(p)
        if len(first) == 12 and ps:
            tg.append((mean(list(ps.values())), Counter(v[0] for v in first.values()).most_common(1)[0][1] / 12))
    size[m]['conf'] = {'p_right': round(mean(right), 2), 'p_wrong': round(mean(wrong), 2), 'auroc': round(auc(right, wrong), 2),
                       'rho_agreement': round(spearman([t[0] for t in tg], [t[1] for t in tg]), 2),
                       'targets': [[round(p, 3), round(a, 3)] for p, a in tg]}
D['size'] = size

# ---------------------------------------------------------------- sequential turns and the relay chain: F63
k11 = quiet('k11.py')
TW = {'qwen3-4b': ('run_followup/followup', 20), 'gemma3-4b': ('run_followup/followup', 20),
      'llama3.2-3b': ('run_followup/followup', 20), 'qwen3-8b': ('run_main/main', 20)}
chn = {}
for m, (root, n) in TW.items():
    par, mute = Sm(root, m, 'fully_connected', 'base'), Sm(root, m, 'fully_connected', 'all_independent')
    seq, c6 = Sm('run_chain/chain', m, 'sequential', 'base'), Sm('run_chain/chain', m, 'chain6', 'base')
    hs = {}
    for h in ['split', 'all']:
        cell = f'{m}__sequential__hints{h}__decoysTrue'; tw = f'{m}__fully_connected__hints{h}__decoysTrue'
        a, b, c = S('run_chain_hints/chain_hints', cell, 'base'), S(SRC[m], tw, 'base'), S(SRC[m], tw, 'all_independent')
        hs[h] = {'seq': rate(a), 'par': rate(b), 'mute': rate(c), 'seq_minus_par': pair(a, b), 'seq_minus_mute': pair(a, c)}
    chn[m] = {'mute': rate(mute), 'par': rate(par), 'seq': rate(seq), 'chain': rate(c6),
              'chain_minus_par': pair(c6, par), 'chain_minus_mute': pair(c6, mute),
              'seq_minus_par': pair(seq, par), 'seq_minus_mute': pair(seq, mute), 'hints': hs}
pool = lambda a_, b_: pair({(m, k): v for m in TW for k, v in a_(m).items()}, {(m, k): v for m in TW for k, v in b_(m).items()},
                           cl=lambda k: k[1])  # resampled by target, as k11.py
chn['pooled'] = {
    'chain_minus_par': pool(lambda m: Sm('run_chain/chain', m, 'chain6', 'base'), lambda m: Sm(TW[m][0], m, 'fully_connected', 'base')),
    'chain_minus_mute': pool(lambda m: Sm('run_chain/chain', m, 'chain6', 'base'), lambda m: Sm(TW[m][0], m, 'fully_connected', 'all_independent')),
    'seq_minus_par': pool(lambda m: Sm('run_chain/chain', m, 'sequential', 'base'), lambda m: Sm(TW[m][0], m, 'fully_connected', 'base')),
    'seq_minus_mute': pool(lambda m: Sm('run_chain/chain', m, 'sequential', 'base'), lambda m: Sm(TW[m][0], m, 'fully_connected', 'all_independent'))}
D['chain'] = chn

# ---------------------------------------------------------------- adoption and copies against a silenced baseline: F64
D['adopt'] = quiet('k13.py')['RESULTS']   # main / followup / mixed -> protocol -> model -> measure; chain_copies

json.dump(D, open(os.path.join(OUT, 'data_v2.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'data_v2.json'))
