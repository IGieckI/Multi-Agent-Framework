# The figures of the report's v2 sections, from data/data_v2.json (make_data_v2.py). Run from the repo root:
#   python docs/report/make_figures_v2.py      (needs matplotlib)
# Writes docs/report/figures/v2_*.svg, in make_figures.py's style (colour = model).
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'data', 'data_v2.json')))
OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
COL = {'qwen3-4b': '#2a78d6', 'gemma3-4b': '#eb6834', 'llama3.2-3b': '#1baf7a', 'qwen3-8b': '#eda100', 'gemma3-12b': '#b8327a'}
MK = {'qwen3-4b': 'o', 'gemma3-4b': 's', 'llama3.2-3b': '^', 'qwen3-8b': 'D', 'gemma3-12b': 'P'}
NAME = {'qwen3-4b': 'Qwen3-4B', 'gemma3-4b': 'Gemma 3-4B', 'llama3.2-3b': 'Llama 3.2-3B', 'qwen3-8b': 'Qwen3-8B',
        'gemma3-12b': 'Gemma 3-12B'}
plt.rcParams.update({
    'font.family': 'Noto Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'svg.fonttype': 'path', 'legend.frameon': False, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight'); plt.close(fig)


def model_legend(fig, models, y=-0.05):
    fig.legend(handles=[Line2D([], [], ls='', marker=MK[m], color=COL[m], ms=6, label=NAME[m]) for m in models],
               loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, y), fontsize=8)


def dots(ax, models, cats, get, title, shade=None):
    """success (%) per category (x) and model (colour), with 95% intervals"""
    for j, m in enumerate(models):
        for i, (k, _) in enumerate(cats):
            v = get(m, k)
            if v is None: continue
            x = i + (j - (len(models) - 1) / 2) * 0.17
            ax.plot([x, x], [v['lo'], v['hi']], color=COL[m], lw=1.3, alpha=0.55, solid_capstyle='round')
            ax.scatter([x], [v['p']], marker=MK[m], s=30, color=COL[m], edgecolor='white', lw=0.7, zorder=3)
    if shade:
        ax.axvspan(shade[0], shade[1], color='#f0efec', zorder=0)
        ax.text((shade[0] + shade[1]) / 2, 2, 'riferimenti', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(cats)), [c[1] for c in cats], fontsize=7.5); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 102); ax.set_xlim(-0.5, len(cats) - 0.5); ax.set_title(title)


def forest(ax, cells, models, title, xlabel):
    for i, (lab, vals) in enumerate(cells):
        y0 = len(cells) - 1 - i
        for j, m in enumerate(models):
            v = vals.get(m)
            if v is None: continue
            y = y0 + ((len(models) - 1) / 2 - j) * 0.18
            ax.plot([v['lo'], v['hi']], [y, y], color=COL.get(m, INK2), lw=1.6, solid_capstyle='round')
            ax.scatter([v['d']], [y], marker=MK.get(m, 'o'), s=34, color=COL.get(m, INK2), edgecolor='white', lw=0.8, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(cells)), [c[0] for c in cells][::-1]); ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, len(cells) - 0.4); ax.set_title(title); ax.set_xlabel(xlabel)


# 1. renamed items: the same cells with real and with made-up names
RM = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'gemma3-12b']
CND = [('mute', 'muti'), ('talk', 'comuni-\ncano'), ('share', '+ istr.\ncondividi'), ('one', '1 agente,\nentrambe'),
       ('all_mute', 'tutti,\nentrambe')]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, k, t in [(axs[0], 'real', 'Nomi veri (note divise, con esche)'), (axs[1], 'renamed', 'Nomi inventati (stesse note)')]:
    dots(ax, RM, CND, lambda m, c: D['renamed'][m][k][c], t, shade=(2.5, 4.5))
axs[0].set_ylabel('successo (%)')
model_legend(fig, RM, -0.04); fig.tight_layout(); save(fig, 'v2_renamed.svg')

# 2. channels: results only / messages only
CM = ['qwen3-8b', 'gemma3-12b']
CC = [('mute', 'muti'), ('results', 'solo\nrisultati'), ('messages', 'solo\nmessaggi'), ('both', 'entrambi'),
      ('messages_share', 'solo messaggi\n+ condividi'), ('both_share', 'entrambi\n+ condividi')]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
dots(ax, CM, CC, lambda m, c: D['channels'][m][c], 'Note divise con esche: quale canale porta l\'informazione')
ax.axvline(3.5, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.set_ylabel('successo (%)')
model_legend(fig, CM, -0.06); fig.tight_layout(); save(fig, 'v2_channels.svg')

# 3. team size and token confidence
SM = ['qwen3-4b', 'gemma3-4b', 'llama3.2-3b']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax = axs[0]
for m in SM:
    xs = [1, 3, 6, 12]; ys = [D['size'][m][str(n)]['p'] for n in xs]
    ax.plot(xs, ys, color=COL[m], lw=1.8, marker=MK[m], ms=5.5, mec='white', mew=0.7)
    ax.annotate(f"{ys[-1]:.1f}".replace('.0', '').replace('.', ','), (12, ys[-1]), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=7.5, color=INK)
ax.set_xscale('log', base=2); ax.set_xticks([1, 3, 6, 12], ['1', '3', '6', '12']); ax.minorticks_off()
ax.set_ylim(0, 100); ax.set_xlabel('agenti muti'); ax.set_ylabel('successo (%)'); ax.set_title('Successo per numero di agenti')
ax = axs[1]
for m in SM:
    pts = D['size'][m]['conf']['targets']
    ax.scatter([p[0] for p in pts], [100 * p[1] for p in pts], marker=MK[m], s=20, color=COL[m], alpha=0.75,
               edgecolor='white', lw=0.5)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 104)
ax.set_xlabel('probabilità media della prima combinazione'); ax.set_ylabel('agenti sulla coppia più comune (%)')
ax.set_title('Confidenza e accordo, per obiettivo (12 agenti)')
model_legend(fig, SM, -0.06); fig.tight_layout(); save(fig, 'v2_size.svg')

# 4. sequential turns and the relay chain
TM = ['qwen3-4b', 'qwen3-8b', 'gemma3-4b', 'llama3.2-3b']
CH = [('chain_minus_par', 'catena − parallelo'), ('chain_minus_mute', 'catena − muti'),
      ('seq_minus_par', 'in sequenza − parallelo'), ('seq_minus_mute', 'in sequenza − muti')]
fig, ax = plt.subplots(figsize=(6.2, 2.9))
forest(ax, [(lab, {m: D['chain'][m][k] for m in TM}) for k, lab in CH], TM,
       'Turni in sequenza e catena, impostazioni principali', 'differenza (punti percentuali)')
model_legend(fig, TM, -0.07); fig.tight_layout(); save(fig, 'v2_chain.svg')

print('wrote', OUT)
