# Analysis scripts

Run from the repo root with the system python (needs numpy; no networkx, no venv), passing this folder:

    python3 docs/analysis/<script>.py docs/analysis

| script | what it computes | findings |
|---|---|---|
| `lib.py` | helpers: `load`, `index` (key a runs.csv by columns), `mean`, `boot` (paired bootstrap, `cluster=` resamples by target), `fmt`, `sign_test` (exact, on discordant pairs) | all |
| `sim_random.py` | simulated random agents on the main targets (run as `python3 docs/analysis/sim_random.py`) | F30 |
| `rb.py`, `rb2.py` | random-baseline grid rates; shortcut targets over 200 reps (writes `short200.json`); LLM − random paired | F31, F32 |
| `short.py`, `m1.py` | shortcut targets in main/hints (writes `shortcuts.json`); F16/F18 without them | F32 |
| `h1.py` – `h5.py` | hints grid: rates, paired contrasts, shortcut split, note use from the logs, decoys, refusals | F33–F36 |
| `k1.py`, `k2.py` | hints2 + hints2_share: rates and contrasts; who tells and who acts, from the logs | F37–F40 |
| `k3.py`, `k4.py` | hints2_use, hints2_8b, hints2_8b_share: rates and contrasts (with hints2 / hints2_share); who tells, what a receiver does once told | F42–F44 |
| `k5.py`, `k6.py` | hints2_12b(_share), hints2_8b(_share) on 80 targets: rates and contrasts; who tells and who uses (`k4.py` keeps the first 40 targets) | F46, F47 |
| `k7.py` | hints2_made vs hints2 / hints2_8b: success and remake share with the made line | F48 |
| `k8.py`, `k9.py` | mixed / mixed_hints vs main / hints2: mixed − each model, expected success of a silenced mix, herding, holders' family | F49 |
| `k10.py` | confidence and team size: agreement of silenced agents' first answers, token confidence (rescored or logged), 12 vs 6 vs 3 agents | F55 |
| `k11.py` | sequential and chain turns (chain, chain_hints) vs parallel fully connected and silenced; cascade measure | chain |
| `pagedata.py` | chart data for the results page (writes `pagedata.json`) | page |

`short200.json` keys are `"depth,distractors,replicate"` → whether the target has a shortcut (F32); use it to report
depth-2 results of main / followup / random_baseline / hints without shortcut targets.
