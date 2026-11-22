# The report's figures, from data/data.json (make_data.py). Run from the repo root:
#   python docs/report/make_figures.py      (needs matplotlib)
# Writes docs/report/figures/*.svg. Colour = model, hollow marker = silenced team, filled = talking team.
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'data', 'data.json')))
OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
COL = {'qwen3-4b': '#2a78d6', 'gemma3-4b': '#eb6834', 'llama3.2-3b': '#1baf7a', 'qwen3-8b': '#eda100'}
MK = {'qwen3-4b': 'o', 'gemma3-4b': 's', 'llama3.2-3b': '^', 'qwen3-8b': 'D'}
NAME = {'qwen3-4b': 'Qwen3-4B', 'gemma3-4b': 'Gemma 3-4B', 'llama3.2-3b': 'Llama 3.2-3B', 'qwen3-8b': 'Qwen3-8B'}
M4 = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
M3 = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']

plt.rcParams.update({
    'font.family': 'Noto Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'svg.fonttype': 'path', 'legend.frameon': False, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def num(x, sign=True):
    s = f'{x:+.1f}' if sign else f'{x:.1f}'
    return s.replace('.0', '').replace('.', ',').replace('-', '−')


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)


def dumbbell(ax, rows, title, xlabel='successo (%)', xlim=(0, 100)):
    """rows: (label, model, silenced, talk); arrow from the silenced team (hollow) to the talking one (filled)"""
    for i, (lab, m, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=COL[m], lw=2, solid_capstyle='round', zorder=2)
        ax.scatter([a], [y], s=46, marker=MK[m], facecolor='white', edgecolor=COL[m], lw=1.6, zorder=3)
        ax.scatter([b], [y], s=46, marker=MK[m], color=COL[m], edgecolor='white', lw=0.8, zorder=4)
        ax.annotate(num(b - a), (max(a, b), y), xytext=(8, 0), textcoords='offset points', va='center',
                    color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(axis='y', visible=False)


def marker_legend(fig, extra=()):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, mew=1.4, ms=6, label='squadra muta'),
         Line2D([], [], ls='', marker='o', color=INK2, ms=6, label='squadra con comunicazione')] + list(extra)
    fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(0.5, -0.06), fontsize=8)


# 1. main: talk vs silenced, per model
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, p, t in [(axs[0], 'fully_connected', 'Completamente connesso, 6 agenti'), (axs[1], 'star', 'Stella, 3 agenti')]:
    dumbbell(ax, [(NAME[m], m, D['main'][m][p]['all_independent']['p'], D['main'][m][p]['base']['p']) for m in M4], t)
marker_legend(fig); fig.tight_layout(); save(fig, 'main.svg')


# 2. follow-up: talk - silenced per protocol, per model, with 95% intervals
PROT = [('fully_connected', 'Compl. connesso, 6'), ('fully_connected3', 'Compl. connesso, 3'), ('star', 'Stella, 3'),
        ('star6', 'Stella, 6'), ('fully_connected_results', 'Solo risultati, 6'), ('fully_connected_messages', 'Solo messaggi, 6')]
def forest(ax, cells, title, xlabel):
    """cells: list of (row label, {model: {'d','lo','hi'}}); one point per model with its interval"""
    for i, (lab, vals) in enumerate(cells):
        y0 = len(cells) - 1 - i
        for j, m in enumerate([m for m in M4 if m in vals]):
            v = vals[m]; y = y0 + (1 - j) * 0.22
            ax.plot([v['lo'], v['hi']], [y, y], color=COL[m], lw=1.6, solid_capstyle='round')
            ax.scatter([v['d']], [y], marker=MK[m], s=34, color=COL[m], edgecolor='white', lw=0.8, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(cells)), [c[0] for c in cells][::-1]); ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, len(cells) - 0.4); ax.set_title(title); ax.set_xlabel(xlabel)
def model_legend(fig, models, y=-0.05):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls='', marker=MK[m], color=COL[m], ms=6, label=NAME[m]) for m in models],
               loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, y), fontsize=8)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
forest(ax, [(lab, {m: D['followup'][m][p]['talk_minus_mute'] for m in M3}) for p, lab in PROT],
       'Effetto della comunicazione per protocollo', 'comunicazione − muti (punti percentuali)')
model_legend(fig, M3); fig.tight_layout(); save(fig, 'followup.svg')


# 3. team size: 6 - 3 agents
fig, ax = plt.subplots(figsize=(6.2, 2.6))
SZ = [('fully_connected-fully_connected3:all_independent', 'Compl. connesso, muti'),
      ('fully_connected-fully_connected3:base', 'Compl. connesso, con comunicazione'),
      ('star6-star:all_independent', 'Stella, muti'), ('star6-star:base', 'Stella, con comunicazione')]
forest(ax, [(lab, {m: D['followup'][m][k] for m in M3}) for k, lab in SZ],
       'Vantaggio di 6 agenti su 3', '6 agenti − 3 agenti (punti percentuali)')
model_legend(fig, M3, -0.07); fig.tight_layout(); save(fig, 'team_size.svg')


# 4. LLM teams against random agents, by depth
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
from matplotlib.lines import Line2D
for ax, d in zip(axs, ['1', '2']):
    bd = lambda m: D['main'][m]['by_depth'][d]
    dumbbell(ax, [(NAME[m], m, bd(m)['fc_mute']['p'], bd(m)['fc_talk']['p']) for m in M4],
             f'Profondità {d}' + (' (3 distrattori)' if d == '1' else ' (0 distrattori)'))
    for i, m in enumerate(M4):
        ax.scatter([bd(m)['one']['p']], [len(M4) - 1 - i], marker='|', s=90, color=COL[m], lw=1.8, zorder=3)
    r6, r1 = D['random']['mock'][d]['fc_mute']['p'], D['random']['mock'][d]['one']['p']
    for x, lab in [(r6, '6 casuali'), (r1, '1 casuale')]:
        ax.axvline(x, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.text(x + 1, -0.55, lab, ha='left', va='bottom', fontsize=7, color=INK2)
marker_legend(fig, [Line2D([], [], ls='', marker='|', color=INK2, ms=9, mew=1.8, label='agente singolo')])
fig.tight_layout(); save(fig, 'random.svg')


# 5. notes split between agents: success per condition and model
CND = [('fc_mute', 'muti'), ('fc_talk', 'comuni-\ncano'), ('share', '+ istr.\ncondividi'),
       ('use', '+ istr.\nusa'), ('one', '1 agente,\nentrambe'), ('all_mute', 'tutti,\nentrambe')]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, d, t in [(axs[0], 'True', 'Note divise, con esche'), (axs[1], 'False', 'Note divise, senza esche')]:
    for j, m in enumerate(M4):
        for i, (k, lab) in enumerate(CND):
            v = D['hints2'][m]['all']['fc_mute'] if k == 'all_mute' else D['hints2'][m][d][k]
            x = i + (j - 1.5) * 0.17
            ax.plot([x, x], [v['lo'], v['hi']], color=COL[m], lw=1.3, alpha=0.55, solid_capstyle='round')
            ax.scatter([x], [v['p']], marker=MK[m], s=30, color=COL[m], edgecolor='white', lw=0.7, zorder=3)
    ax.axvspan(3.5, 5.5, color='#f0efec', zorder=0)
    ax.set_xticks(range(len(CND)), [c[1] for c in CND], fontsize=7.5); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 100); ax.set_title(t); ax.set_xlim(-0.5, 5.5)
    ax.text(4.5, 2, 'riferimenti', ha='center', va='bottom', fontsize=7.5, color=INK2)
axs[0].set_ylabel('successo (%)')
model_legend(fig, M4, -0.04); fig.tight_layout(); save(fig, 'hints2.svg')


# 6. who tells and who uses: holder names its recipe vs receiver plays it
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, d, t in [(axs[0], 'True', 'Con esche'), (axs[1], 'False', 'Senza esche')]:
    for m in M4:
        a, b = D['logs'][m][f'talk|{d}'], D['logs'][m][f'share|{d}']
        ax.annotate('', (b['tell_inter_first'], b['recv_inter_next']), (a['tell_inter_first'], a['recv_inter_next']),
                    arrowprops=dict(arrowstyle='-|>', color=COL[m], lw=1.4, shrinkA=5, shrinkB=5))
        ax.scatter([a['tell_inter_first']], [a['recv_inter_next']], marker=MK[m], s=44, facecolor='white', edgecolor=COL[m], lw=1.5, zorder=3)
        ax.scatter([b['tell_inter_first']], [b['recv_inter_next']], marker=MK[m], s=44, color=COL[m], edgecolor='white', lw=0.8, zorder=4)
        ax.annotate(NAME[m], (b['tell_inter_first'], b['recv_inter_next']), xytext=(-6, 7), textcoords='offset points',
                    fontsize=7.5, color=INK, ha='right')
    ax.set_xlim(-3, 105); ax.set_ylim(-3, 60); ax.set_title(t)
    ax.set_xlabel('il detentore nomina la ricetta\nnel primo messaggio (%)')
axs[0].set_ylabel('chi la riceve la gioca\nall\'azione successiva (%)')
fig.legend(handles=[Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, mew=1.4, ms=6, label='senza istruzione'),
                    Line2D([], [], ls='', marker='o', color=INK2, ms=6, label='con l\'istruzione a condividere')],
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=8)
fig.tight_layout(); save(fig, 'tell_receive.svg')


# 7. remaking the intermediate, everyone holds both recipes
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, p, t in [(axs[0], 'fully_connected', 'Completamente connesso, 6 agenti'), (axs[1], 'star', 'Stella, 3 agenti')]:
    dumbbell(ax, [(NAME[m], m, D['remake'][m][f'{p}|all_independent'][0], D['remake'][m][f'{p}|base'][0]) for m in M4],
             t, xlabel='azioni che rifanno l\'intermedio già creato (%)')
marker_legend(fig); fig.tight_layout(); save(fig, 'remake.svg')

print('wrote', OUT)
