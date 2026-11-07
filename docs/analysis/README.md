# Analysis scripts (git-ignored, local only)

Run from the repo root with the system python (needs numpy; no networkx, no venv), passing this folder:

    python3 docs/analysis/<script>.py docs/analysis

| script | what it computes | findings |
|---|---|---|
| `lib.py` | helpers: `load`, `index` (key a runs.csv by columns), `mean`, `boot` (paired bootstrap, `cluster=` resamples by target), `fmt` | all |
| `sim_random.py` | simulated random agents on the main targets (run as `python3 docs/analysis/sim_random.py`) | F30 |
| `rb.py`, `rb2.py` | random-baseline grid rates; shortcut targets over 200 reps (writes `short200.json`); LLM − random paired | F31, F32 |
| `short.py`, `m1.py` | shortcut targets in main/hints (writes `shortcuts.json`); F16/F18 without them | F32 |
| `h1.py` – `h5.py` | hints grid: rates, paired contrasts, shortcut split, note use from the logs, decoys, refusals | F33–F36 |
| `k1.py`, `k2.py` | hints2 + hints2_share: rates and contrasts; who tells and who acts, from the logs | F37–F40 |
| `k3.py`, `k4.py` | hints2_use, hints2_8b, hints2_8b_share: rates and contrasts (with hints2 / hints2_share); who tells, what a receiver does once told | F42–F44 |
| `pagedata.py` | chart data for the results page (writes `pagedata.json`) | page |

`short200.json` keys are `"depth,distractors,replicate"` → whether the target has a shortcut (F32); use it to report
depth-2 results of main / followup / random_baseline / hints without shortcut targets.
