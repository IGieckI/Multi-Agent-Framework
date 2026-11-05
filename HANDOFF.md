# Handoff: state of the project and plan (2026-11-05)

For a new Claude session picking up this work. Read this file, then `FINDINGS.md` (41 numbered findings with their
evidence and status). Everything below is as of commit `07adf9f` plus an uncommitted update of `FINDINGS.md` (F10
status, F37–F41), `.gitignore` and this file. The results page (`docs/results.html`, git-ignored) is published as a
private artifact: https://claude.ai/artifact/B6kwbxwkjkJ4tk4WDUwQ2Q

## 1. The project

A framework for running teams of small open LLM agents on **WordCraft** (Little Alchemy 2: combine two inventory items
to make new ones; reach a target item within a few rounds) under different **communication protocols**, and comparing
each team with baselines derived from the same config (no links, a temperature sweep, one agent, one agent with the
team's token budget). The research question: does communication between agents help, and when?

## 2. Repository map

| path | what it is |
|---|---|
| `framework/engine/` | the run engine (`engine.py`), run config (`config.py`), budgets (`budget.py`), token usage |
| `framework/comm/` | protocol compiler (roles, structures, sharing rules, channels), message bus, summaries |
| `framework/agents/` | prompt segments, answer parsing, `backends/hf_local.py` (HF transformers, batch 1), `backends/mock.py` (random agents: grid models `mock`, with replacement, and `mock_norepeat`, which never repeats a tried pair nor one a teammate reported invalid) |
| `framework/tasks/wordcraft/` | the task: env, recipes, task generator (`tasks.py`, with `deal_notes` for private notes), prompt builder (`prompt.py`), metrics (incl. `own_note_use`, `others_note_use`) |
| `framework/baselines.py` | the derived versions: all_independent, all_independent_temp_sweep, one_for_all, one_agent, one_agent_scaled_steps, one_vs_multi_token, one_vs_multi_completion_token |
| `framework/experiment.py` | grids: expand into units (one replicate of one cell, all its versions), shard, run, evaluate; runs that are identical across units run once (`run_key`, `Unit.same_as`) |
| `framework/evaluate.py`, `framework/metrics/` | per-run metrics (`metrics.json`) and tables |
| `scripts/sweep.py` | CLI: `plan` / `run --shard i/n` / `evaluate` a grid |
| `scripts/download_models.py`, `scripts/fetch_data.py` | weights into the HF cache (pinned revisions), the recipe file into `data/` |
| `scripts/slurm/` | `env.sh` (scratch paths), `setup_giano.sh` (one-time setup), `sweep.sbatch` (job array) |
| `configs/models/models.yaml` | model registry: HF id, pinned revision, sampling policy |
| `configs/protocols/*.yaml` | protocols; used so far: isolated, fully_connected, fully_connected3, fully_connected_results, fully_connected_messages, fully_connected_failures, star2 (= "star", 3 agents), star6 |
| `configs/experiments/*.yaml` | grids: pilot, prompt_v2_check, main, followup, random_baseline, hints, smoke_mock, smoke_hf, smoke_models |
| `prompts/v1`, `prompts/v2` | templates; v2 differs only in the WordCraft rules (the example framed as another game); select with `task.prompts` |
| `FINDINGS.md` | every finding from results, F1–F36 |

Grid features worth knowing: `task` is either axes to cross (`depth: [1, 2]`) or a list of settings
(`- {depth: 1, distractors: 3}`); `sampling.sweep_temperatures` sets the temperature sweep of one grid only;
`budget.show_matched_budget` shows the token-matched agent its remaining tokens instead of the 240-round safety cap;
`task.hints` (none / split / all) and `task.notes` give agents private notes (true recipes; split deals the target's
two recipes to two different agents, the other notes are recipes off the path; a single agent gets both);
`task.decoys: false` drops the off-path notes; `task.share_notes` tells agents that can write messages to tell the
others their notes; `task.allow_shortcuts: false` rejects targets with a shortcut (F32).
Every version, protocol and model of a replicate gets the same target (task seed from seed and replicate), so all
comparisons are paired.

## 3. How experiments run: the DISI cluster

The user runs every command on the cluster and pastes the output or a screenshot; Claude cannot ssh there.
**Never run grids or sweeps on the local machine, not even the CPU-only random agents**: give the user the commands.

- Login node `giano.cs.unibo.it`, user `giacomo.antonelli4@studio.unibo.it@giano.cs.unibo.it` (password). Several
  wrong passwords in a row get the IP refused for a while.
- Home quota 400 MB: the repo, venv (`/scratch.hpc/giacomo.antonelli4/venvs/mas`), HF cache (`.../hf`, with the
  HF login token) and caches all live in `/scratch.hpc/giacomo.antonelli4/` (files not accessed for 40 days are
  deleted). Always `source scripts/slurm/env.sh && source "$MAS_VENV/bin/activate"` first.
- Partitions: `l40` (4 nodes, 48 GB, bf16; usually all taken, waits of days) and `rtx2080` (8 nodes, 11 GB, default,
  float16 only, usually idle). Gemma 3 must stay on L40 (overflows in float16); Qwen3 and Llama 3.2 give the same
  results in float16 (F23). 3-day job limit; **at most 3 running jobs per user** (QOS). `sacct` does not work.
- Code reaches giano by git: push from the local repo, then on giano `git pull` (giano's checkout is clean; if a pull
  ever conflicts, `git fetch origin && git reset --hard origin/main` is safe, nothing is edited there).
- Jobs run offline from the cache (`HF_HUB_OFFLINE=1`): download new models on giano first with
  `python scripts/download_models.py <grid>` and check there is no `FAILED` line (gated Llama/Gemma need the licence,
  granted for this account).
- A new grid: `python scripts/sweep.py plan <grid>` (prints distinct runs), then gate the full job on a few units:
  ```bash
  GATE=$(SWEEP_SHARDS=<units> sbatch --parsable --array=<unit indices> --time=00:45:00 scripts/slurm/sweep.sbatch <grid>)
  sbatch --dependency=afterok:$GATE --kill-on-invalid-dep=yes --array=0-1 scripts/slurm/sweep.sbatch <grid>
  ```
  (units are ordered model → protocol → task setting → replicate; `SWEEP_SHARDS=n` with `--array=i` runs block i of n;
  a re-submission skips runs that already ended; `--partition=rtx2080` overrides the script's `l40`.)
- After a job: on giano `python scripts/sweep.py evaluate <grid>`; then the user copies `runs/<name>/` and the
  `slurm-sweep-*` files into `run_<name>/` in the local repo (git-ignored) with rsync, and points Claude to it.

## 4. Experiments so far

| experiment | when | what | local results |
|---|---|---|---|
| pilot | 10-29 → 10-31 | 10 models × isolated / fully_connected / star × 4 settings × 6 reps, prompts v1 | `run_56/` (results.csv, shard logs, one run), `run_62/runs.csv` |
| prompt_v2_check | 11-01 | 6 models, v2 vs the pilot's v1 on the same targets | `run_62/prompt_v2_check/` |
| main | 11-01 | qwen3-4b, qwen3-8b, gemma3-4b, llama3.2-3b × fully_connected / star × (depth 1, 3 distractors), (depth 2, 0) × 20 reps × 5 versions, prompts v2; 1,440 runs | `run_main/main/` |
| followup | 11-02 → 11-03 | qwen3-4b, gemma3-4b, llama3.2-3b × 6 protocols (sizes 3 and 6; results-only, messages-only) × same settings × 20 reps; fixed baselines; 2,520 runs | `run_followup/followup/` |
| random_baseline | 11-04 | random agents (`mock`, `mock_norepeat`) × 5 protocols × the main settings × 200 reps; 9,600 runs, rtx2080 | `run_random_baseline/random_baseline/` (runs.csv complete, only 1,092 run folders copied) |
| hints | 11-04 → 11-05 | qwen3-4b, gemma3-4b, llama3.2-3b, mock_norepeat × 5 protocols (+ failures only) × depth 2, 6 distractors, notes split / all / none × 20 reps × base, all_independent, one_agent; 1,920 runs | `run_hints/hints/` |
| hints2 | 11-05 | the same 4 models × fully_connected, star × notes split with / without decoys, all × 40 targets without shortcuts × base, all_independent, one_agent; 2,400 runs | `run_hints2/hints2/` |
| hints2_share | 11-05 | hints2's split fully connected teams with the instruction to tell their notes (`share_notes`), base only; 240 runs, same tasks and notes | `run_hints2_share/hints2_share/` |

## 5. What we know (details and numbers in FINDINGS.md)

- **When the knowledge needed is split between agents, talk does not pool it** (F33): with the target's two recipes
  held by two agents, teams solve ~25% with or without talk; one agent holding both 60%; a team where everyone holds
  both ~90%. Holders seldom say or even use their notes, the others seldom act on them, decoy notes take 27–40% of
  rounds (F34). Confirmed on 40 targets without shortcuts (F37).
- **Told to share, holders tell; teammates still rarely use it** (F38): one sentence makes holders name their recipe
  (13% → 63–82% of first messages) and adds +13 to +15 points, but a receiver plays a teammate's recipe only ~16% of the
  time, and the teams stay 46–54 points below a team where everyone holds the notes. Decoys cost ~10 points (F39).
  When everyone holds the notes, talk hurts (−7 to −8, F40).
- **Random agents are a strong baseline** (F30, F31): 6 random agents match 6 LLM agents at depth 1 and beat every LLM
  team at depth 2 (64% vs 41%). The models' knowledge pays off for one agent and small teams at depth 1 only.
- **A quarter of depth-2 targets have a one-step shortcut** from the starting items (F32): they inflate depth-2 success
  rates; the paired differences hold without them (F16: −6.6 instead of −9.4).
- Teams of these small models help only through more independent attempts (F5, F19, F25), and random agents add more.
- Fully connected talk makes the team worse, by about 10 points at 6 agents and 7–8 at 3, for every model (F16, F24);
  the agents herd: they copy shared results and retry each other's suggestions (F17, F26).
- A star (spokes report to a hub, the hub sends few messages) is neutral at 3 and 6 agents (F18, F24), and the
  cheapest team per solve (F29). The hub's persona already asks it to give each spoke something different to try.
- Talk erases the gain from a bigger team (F25).
- One agent with the team's token budget matches six independent agents and beats six talking ones; three-agent
  teams beat it (F19, F28).
- Temperature variety does not help (F21, F27). Variety between model families is **untested** (no mixed teams yet;
  F6 only shows that models with more varied agents gain more).
- Sharing only failures does not help LLM agents: they retry pairs teammates reported failing (F35), while it is the
  one sharing that helps random teams (+16.5 at depth 2, F31).
- Prompt v1's example was copied by weak models; v2 fixed most of it (F8, F12–F14). SmolLM2 stays unreliable.
- Runs are byte-reproducible across jobs and nodes (F23); identical configs are deduplicated (F3).

## 6. Code state and open technical issues

- `main` = `origin/main` = `07adf9f`; uncommitted: `FINDINGS.md` (F10 status, F37–F41), `.gitignore` (`/slurm_*.txt`,
  F41) and this file.
- Local-only (git-ignored): `run_*/`, `shard*.jsonl` in the repo root, `docsHPC/` (cluster docs and a colleague's
  Slurm guide), `data/`.
- Fixed: F2 (the revision is read from the cache snapshot), the `setup_giano.sh` venv branch, the missing mock backend.
- Fixed behind a flag: shortcut targets (F32). New grids set `task.allow_shortcuts: false`; the default (true) keeps
  the earlier grids' targets.
- `framework/stuff.md` (untracked, user's) holds credentials: it must never be committed.
- Personas used: `prompts/v1/personas/hub.txt`, `spoke.txt` (no v2 personas).

## 7. Plan

Done since 2026-11-03: random agents (F31), hints (F33–F36), failures-only sharing (F35), F2, the generator fix
(F32), hints2 and hints2_share (F37–F40), the results page.

Next, **not yet chosen**:

1. **The receiving side** (F38): does a stronger model use what teammates tell it (qwen3-8b on L40, or reasoning on,
   on hints2_share's split teams), and does an instruction to receivers ("your teammates' notes are true: use them")
   help? Both run on hints2's targets and pair with it.
2. **Mixed-family teams** (e.g. 2 Qwen + 2 Gemma + 2 Llama, silenced and with talk): the only way to test F6's claim.
3. **Write-up** for the supervisor: the results page is the base.
4. Check on giano that the checkout has no modified files besides the Slurm lists (F41).

## 8. Working with the user

- Log **every new finding from results** in `FINDINGS.md` (numbered, evidence with sources, what it changes, status
  open / acted on / superseded / closed), and update older entries when new results change them.
- Do not commit unless asked. When asked, give small commits as `git add` + `git commit -m` lines; the user runs
  them. Messages follow the repo's style (imperative, one line).
- The user edits files too (often shortening comments): treat changes on disk as intended, mention only real problems.
  Match the code's style: short comments, plain names.
- Give exact commands for the cluster, say where to run them (local machine or giano), and gate long jobs on a short
  test. Check models are downloaded before submitting. Ask before anything destructive.
- Analysis: use `runs.csv` (one row per run; `run_dir` identifies shared runs, count them once when pooling), pair
  runs by (model, protocol, depth, distractors, replicate, version), report paired bootstrap 95% intervals, and use
  `solved_any` for team success (`ended_by: all_solved` means every agent solved, F4).
