# Handoff: state of the project and plan (2026-11-22)

For a new Claude session picking up this work. Read this file, then `FINDINGS.md` (64 numbered findings with their
evidence and status), then `docs/LITERATURE_2026-11-22.md` (the related work checked so far). Code as of commit `076fb11` (local = `origin/main` = giano). **Every experiment has finished,
been downloaded and been analysed (F1–F63)**: the 2026-11-07 grids (F46–F49), the review's grids and the tutor's
grids submitted on 2026-11-12 (F56–F63, analysed 2026-11-20). Nothing is queued on giano. The report's v2
(`docs/report/report_v2.pdf`) covers everything through F64 (2026-11-22: the adoption/copy correction and a
related-work section). Section 7 lists what is left.

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
| `framework/agents/` | prompt segments, answer parsing, `backends/hf_local.py` (HF transformers, batch 1), `backends/mock.py` (random agents: grid models `mock`, with replacement, and `mock_norepeat`, which never repeats a tried pair nor one a teammate reported invalid), `backends/team.py` (mixed teams: one backend per agent, `engine.agent_models`) |
| `framework/tasks/wordcraft/` | the task: env, recipes, task generator (`tasks.py`, with `deal_notes` for private notes), prompt builder (`prompt.py`), metrics (incl. `own_note_use`, `others_note_use`) |
| `framework/baselines.py` | the derived versions: all_independent, all_independent_temp_sweep, one_for_all, one_agent, one_agent_scaled_steps, one_vs_multi_token, one_vs_multi_completion_token |
| `framework/experiment.py` | grids: expand into units (one replicate of one cell, all its versions), shard, run, evaluate; runs that are identical across units run once (`run_key`, `Unit.same_as`) |
| `framework/evaluate.py`, `framework/metrics/` | per-run metrics (`metrics.json`) and tables |
| `scripts/sweep.py` | CLI: `plan` / `run --shard i/n` / `evaluate` a grid |
| `scripts/download_models.py`, `scripts/fetch_data.py` | weights into the HF cache (pinned revisions), the recipe file into `data/` |
| `scripts/slurm/` | `env.sh` (scratch paths), `setup_giano.sh` (one-time setup), `sweep.sbatch` (job array) |
| `configs/models/models.yaml` | model registry: HF id, pinned revision, sampling policy; `teams:` mixed teams a grid can name as a model (`mix-qgl`: qwen3-4b, gemma3-4b, llama3.2-3b in turn) |
| `configs/protocols/*.yaml` | protocols; used so far: isolated, fully_connected, fully_connected3, fully_connected_results, fully_connected_messages, fully_connected_failures, star2 (= "star", 3 agents), star6 |
| `configs/experiments/*.yaml` | grids: pilot, prompt_v2_check, main, followup, random_baseline, hints, hints2, hints2_share, hints2_use, hints2_8b, hints2_8b_share (both extended to 80 replicates), hints2_12b, hints2_12b_share, hints2_made, mixed, mixed_hints, hints2_share_pub, hints2_channels, hints2_channels_share, hints2_ex2, renamed, renamed_share, chain, chain_hints, size12, confidence_check, smoke_mock, smoke_hf, smoke_models |
| `prompts/v1`, `v2`, `v2r`, `v3` | templates; v2 differs from v1 only in the WordCraft rules (the example framed as another game); v2r: made-up example, no "real-world intuition" (for `task.rename`); v3: v2 plus the example one round later (F60); select with `task.prompts` |
| `FINDINGS.md` | every finding from results, F1–F63 |
| `docs/results.html` | the results web page (git-ignored, see section 6) |
| `docs/analysis/` | the analysis scripts behind F30–F63 and the page (`README.md` lists them) |
| `docs/report/` | the Italian report for the supervisor: v1 `report.typ` (user-edited, through F45) and v2 `report_v2.typ` (through F63); see section 6 |
| `docs/REVIEW_2026-11-11.md` | critical review of F1–F49 and the experiments it proposed (C1–C10, D1–D3) |
| `docs/Communication in LLM-Based Multi-Agent Systems_ ...md` | a deep-research review of the field written without knowledge of this project (metrics M1–M12, ideas A1–C6); its §9 updated with what was checked |
| `docs/LITERATURE_2026-11-22.md`, `docs/downloaded_papers/` | notes on the 17 papers that review left unverified (PDFs alongside): what each says, what it changes here |

Grid features worth knowing: `task` is either axes to cross (`depth: [1, 2]`) or a list of settings
(`- {depth: 1, distractors: 3}`); `sampling.sweep_temperatures` sets the temperature sweep of one grid only;
`budget.show_matched_budget` shows the token-matched agent its remaining tokens instead of the 240-round safety cap;
`task.hints` (none / split / all) and `task.notes` give agents private notes (true recipes; split deals the target's
two recipes to two different agents, the other notes are recipes off the path; a single agent gets both);
`task.decoys: false` drops the off-path notes; `task.share_notes` tells agents that can write messages to tell the
others their notes; `task.use_notes` tells agents that receive messages that the recipes others pass on from their
notes are true and to use them; `task.allow_shortcuts: false` rejects targets with a shortcut (F32); `task.mark_made` lists under the inventory the items
the agent made ("Items you made (already in your inventory): ...", F44). A grid's model may be a team of the registry:
its agents take the members in turn (sorted agent ids), all loaded at once, and `meta.json` records `agent_models`.
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
  ever conflicts, `git fetch origin && git reset --hard origin/main` is safe, nothing is edited there). Pull only when
  `squeue -u $USER` is empty: every run calls `git status` (provenance), and a killed one can leave
  `.git/index.lock` behind (2026-11-07: a lock from 6 Nov 10:01 blocked the pull). If a pull reports the lock and no
  job and no git process runs (`ps -u $USER -o pid,lstart,cmd | grep -w git`), `rm .git/index.lock` and pull again.
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
- After a job, on giano: `squeue -u $USER` (empty), `python scripts/sweep.py evaluate <grid>` (must say `0 unfinished`),
  then list the grid's Slurm logs:
  ```bash
  for f in $(grep -l "runs/<grid>/" slurm-sweep-*.out); do echo "$f"; echo "${f%.out}.err"; done > slurm_<grid>.txt
  ```
  Then on the local machine, from the repo root (one password for all copies; re-running an rsync resumes it):
  ```bash
  H=giacomo.antonelli4@studio.unibo.it@giano.cs.unibo.it; R=/scratch.hpc/giacomo.antonelli4/Multi-Agent-Framework
  S=~/.ssh/giano.sock; ssh -fNM -S $S $H; mkdir -p run_<grid>
  rsync -av -e "ssh -S $S" $H:$R/runs/<grid>/ run_<grid>/<grid>/
  rsync -av -e "ssh -S $S" --files-from=:$R/slurm_<grid>.txt $H:$R/ run_<grid>/
  ssh -S $S -O exit $H
  ```
  Check the download before analysing: number of `events.jsonl` = distinct `run_dir` in `runs.csv`, `unfinished.json`
  is `[]`, every `.out` ends with `0 failed runs`, and `meta.json`'s `environment.git` (commit, dirty).
- Job IDs so far: main 74066/74067; followup 74110, 74114, 74115; random_baseline 74155; hints 74158, 74161–74163;
  hints2 74189, 74192–74194; hints2_share 74195, 74196; hints2_use 74232, 74234–74236; hints2_8b 74237, 74238;
  hints2_8b_share 74239, 74240; the 2026-11-07 grids: gates 74395–74397, full jobs 74398–74405.
- 2026-11-12: the review grids' L40 blocks (74629–74637) were still pending when `076fb11` was pulled, so they run
  on it while their rtx2080 blocks ran on the commit before (the new code only adds `engine.log_confidence`, off
  there: same outputs). New jobs: gates 74697–74700 (all passed), confidence_check 74701, chain 74702 / 74704,
  chain_hints 74706 / 74707 (74705 was submitted without the grid argument and exited at once), size12 74708 /
  74711, rescore 74712–74715 (test 74700). All finished and were downloaded on 2026-11-20 (`run_<grid>/`,
  rescore output in `run_confidence/`); checked: 0 unfinished, 0 failed runs, clean `076fb11` / `e7ff8e3`.
  Rescore 74714 (gemma3-4b) ran a second time on 2026-11-20 16:34 (slurmng-n1) and scored 0 runs (all done); its
  logs, downloaded again on 2026-11-22, replaced the first run's (1,440 runs in 1,345 s on slurmng-n2). The 1,440
  result files are the first run's (18–19 Nov), unchanged.
- **Analysis jobs while grids run** (2026-11-22): a second checkout, `/scratch.hpc/giacomo.antonelli4/MAF-tools`
  (a clone of the same GitHub repo), is where scoring jobs are pulled and submitted. Pulling or writing untracked
  output in the main checkout while a grid runs changes its runs' provenance (F41). The venv's editable install
  imports `framework` from the main checkout, so only `scripts/` comes from the clone; `counterfactual.py` writes to
  the clone's own `runs_counterfactual/`.
- `sweep.py` takes the grid's file, not its name: `python scripts/sweep.py evaluate configs/experiments/<grid>.yaml`.
  Run the download's rsync lines from the repo root (on 2026-11-06 they ran one folder up and had to be moved).

## 4. Experiments so far

| experiment | when | what | local results |
|---|---|---|---|
| pilot | 10-29 → 10-31 | 10 models × isolated / fully_connected / star × 4 settings × 6 reps, prompts v1 | `run_56/` (results.csv, shard logs, one run), `run_62/runs.csv` |
| prompt_v2_check | 11-01 | 6 models, v2 vs the pilot's v1 on the same targets | `run_62/prompt_v2_check/` |
| main | 11-01 | qwen3-4b, qwen3-8b, gemma3-4b, llama3.2-3b × fully_connected / star × (depth 1, 3 distractors), (depth 2, 0) × 20 reps × 5 versions, prompts v2; 1,440 runs | `run_main/main/` |
| followup | 11-02 → 11-03 | qwen3-4b, gemma3-4b, llama3.2-3b × 6 protocols (sizes 3 and 6; results-only, messages-only) × same settings × 20 reps; fixed baselines; 2,520 runs | `run_followup/followup/` |
| random_baseline | 11-04 | random agents (`mock`, `mock_norepeat`) × 5 protocols × the main settings × 200 reps; 9,600 runs, rtx2080 | `run_random_baseline/random_baseline/` |
| hints | 11-04 → 11-05 | qwen3-4b, gemma3-4b, llama3.2-3b, mock_norepeat × 5 protocols (+ failures only) × depth 2, 6 distractors, notes split / all / none × 20 reps × base, all_independent, one_agent; 1,920 runs | `run_hints/hints/` |
| hints2 | 11-05 | the same 4 models × fully_connected, star × notes split with / without decoys, all × 40 targets without shortcuts × base, all_independent, one_agent; 2,400 runs | `run_hints2/hints2/` |
| hints2_share | 11-05 | hints2's split fully connected teams with the instruction to tell their notes (`share_notes`), base only; 240 runs, same tasks and notes | `run_hints2_share/hints2_share/` |
| hints2_use | 11-06 | hints2_share's teams with receivers also told to use teammates' notes (`use_notes`); qwen3-4b, gemma3-4b, llama3.2-3b; 240 runs | `run_hints2_use/hints2_use/` |
| hints2_8b | 11-06 | hints2's cells with qwen3-8b (L40); 600 distinct runs | `run_hints2_8b/hints2_8b/` |
| hints2_8b_share | 11-06, 11-07 → 11-10 | qwen3-8b, split fully connected teams with share_notes, use_notes false / true; 320 runs (80 targets) | `run_hints2_8b_share/hints2_8b_share/` |
| hints2_8b (ext.) | 11-07 → 11-10 | the same cells on targets 40–79: 1,200 distinct runs in all | `run_hints2_8b/hints2_8b/` |
| hints2_12b | 11-07 → 11-10 | hints2's cells with gemma3-12b (L40); 600 distinct runs | `run_hints2_12b/hints2_12b/` |
| hints2_12b_share | 11-07 → 11-10 | gemma3-12b split fully connected teams with share_notes, use_notes false / true; 160 runs | `run_hints2_12b_share/hints2_12b_share/` |
| hints2_made | 11-07 → 11-10 | `mark_made` on: qwen3-4b, gemma3-4b, llama3.2-3b, qwen3-8b × FC / star × split (decoys), all × 3 versions; 1,600 distinct runs, same tasks and notes as hints2 | `run_hints2_made/hints2_made/` |
| mixed | 11-07 → 11-10 | mix-qgl (qwen3-4b, gemma3-4b, llama3.2-3b in turn) × FC 6 / star 3 × main's settings × 20 reps × base, all_independent; 160 runs | `run_mixed/mixed/` |
| mixed_hints | 11-07 → 11-10 | mix-qgl on hints2's split (decoys) and all cells, 40 targets; 320 runs | `run_mixed_hints/mixed_hints/` |
| hints2_share_pub | 11-12 → | share + `publish_on_no_op`; qwen3-4b, gemma3-4b, qwen3-8b; split ± decoys; 240 runs (F58) | `run_hints2_share_pub/` |
| hints2_channels(_share) | 11-12 → | qwen3-8b, gemma3-12b; results only / messages only (+ share); split, decoys; 160 + 80 (F59) | `run_hints2_channels*/` |
| hints2_ex2 | 11-12 → | prompts v3; 4 models; everyone holds both; 3 versions; 480 (F60) | `run_hints2_ex2/` |
| renamed(_share) | 11-12 → | made-up names, v2r, publish_on_no_op; 4 models; split, all; + share; 960 + 160 (F61, F62) | `run_renamed*/` |
| confidence_check | 11-12 | log_confidence on main's first gemma3-4b units: byte-identical (F56) | `run_confidence_check/` |
| chain, chain_hints | 11-12 → | sequential / chain6 × main's settings, sequential × hints2 split, all; 4 models; 320 + 320 (F63) | `run_chain*/` |
| size12 | 11-12 → | 12 silenced agents, main's settings, 3 models, confidence logged; 120 (F57) | `run_size12/` |
| rescore | 11-12 → | token confidence of followup, main, hints2, hints2_8b (5,880 runs, 4 models) (F56, F57) | `run_confidence/` |
| renamed_made | 11-22 → | renamed + `mark_made` (+ `show_matched_budget`); 4 models; split, all; base, all_independent, one_agent, one_vs_multi_token; 1,280 runs | (not run yet) |
| renamed_made_share | 11-22 → | its split teams + share instruction; base, one_vs_multi_token; 320 runs | (not run yet) |

## 5. What we know (details and numbers in FINDINGS.md)

New on 2026-11-22 (F64, literature check, no runs):
- **Adoption and copy measures had no silenced baseline** (F64): given the inbox fully connected would have
  delivered, silenced agents already "blindly adopt" as often (pooled 26% vs 27% with talk; Qwen agents agree
  without talk, F55). Talk adds result adoption (+16 points) and copies (+0.5 per run). F45's "Qwen follows
  blindly", F49's "mixed teams follow blindly less" and F63's "the chain herds less" fall; F16 and distinct paths stand.
- The literature closest to the project (`docs/LITERATURE_2026-11-22.md`): ICR (2610.01042, Qwen3-4B/8B, correction
  vs preservation with a no-message control), "Who Convinced Whom?" (2610.35930, counterfactual message scoring and
  track record), "The Cost of Consensus" (2605.00914, 7–8B homogeneous debate, noise control), "Not All Flips"
  (2606.00820), "At Equal Inference Cost…" (2609.04217, iso-call, agentic), the information-bottleneck view
  (2607.16133).

New on 2026-11-20 (F56–F63):
- **With made-up item names the split-notes leak is gone and talk pools** (F61): silenced split teams 0% for every
  model; talk +35 (qwen3-4b), +45 (qwen3-8b), +17.5 (gemma3-12b), +7.5 (gemma3-4b); with the share instruction
  qwen3-8b 92.5% (above one agent holding both, 37.5%), gemma3-12b 87.5%, gemma3-4b 7.5%. With real names Qwen's
  talk gain was hidden by silenced teams guessing; Gemma3-12B's talk solves were guesses. Size matters for using told
  recipes (share: 8B − 4B +35, 12B − 4B +80), much less for unprompted talk (+10).
- **Qwen3-8B's unprompted gain is the results channel** (F59): results only +22.5 vs silenced, messages only +2.5;
  messages carry it only with the share instruction (+37.5).
- **The remake loop is state tracking, not the example** (F60, F62): prompts v3 barely moves remakes (80% → 75%;
  qwen3-8b talk +17.5); `mark_made` halves them (F48); made-up names make Qwen's loop much worse (one agent 56% → 85%).
- **Publishing refused turns' messages adds little** (F58): reach rises (gemma3-4b 2.1 → 3.2 of 5), success 0 to +7.5.
- **Team size and confidence** (F57): 12 − 6 silenced agents 0 / +2.5 / +5 (qwen3-4b / gemma3-4b / llama); token
  probability tracks agreement between agents (ρ 0.49–0.85), not correctness (AUROC 0.44–0.64); unsure targets gain
  more from more agents. Rescore = sampled confidence (r 0.996, F56).
- **Relay chain beats parallel fully connected (+11.2, 20/2) only by talking less**: it ties silenced; sequential
  turns tie parallel and lose to silenced (−7.5, 4/16); with everyone holding both they amplify the remake lock (F63).

Earlier:

- **Pooling split notes: telling is Qwen's, using grows with size** (F46): Gemma3-12B never names its notes unprompted
  (0% of first messages; talk +2.5), but with the share instruction its team jumps +40 points (52.5% → 92.5%), as
  much as one agent holding both. Its receivers play a told target recipe next 73–92% (gemma3-4b 40–45%).
- **Qwen3-8B on 80 targets** (F47, confirms F43): talk lifts split teams +25 to +32.5 (fully connected), +20 (star);
  with everyone holding both, fully connected talk costs −38.8, the star gains +13.8 (96%).
- **The "Items you made" line breaks the remake loop for Qwen only** (F48): qwen3-8b fully connected talk 62.5% →
  92.5% (remakes 80% → 41%), one agent +15; qwen3-4b +10; Gemma and Llama keep remaking, no change.
- **Mixing families adds no variety beyond the best family** (F49): silenced mixed teams solve what their members'
  agents predict and no more than Llama alone; talking mixed teams herd less (fully connected talk −2.5 vs −5 to
  −12.5; blind adoption 0.20 vs Qwen 0.42), and a mixed star pools split notes (+17.5), but intervals are wide.
- **A bigger model pools split notes through talk** (F43): with Qwen3-8B, talk lifts split teams by +15 to +28 points
  (the 4B models: 0 to +3), and with the share instruction the team matches one agent holding both recipes (65–73% vs
  68%). Alone or silenced Qwen3-8B is no better than Qwen3-4B; its gain is all in communication.
- **Telling receivers to use teammates' notes changes nothing** (F42): the 4B models still play a teammate's recipe
  about one time in ten (−5 points vs the share instruction alone; qwen3-8b −5 too).
- **Agents remake an intermediate they already hold** (F44), and fully connected talk that restates the plan ("first
  make X") locks the team into it: Qwen3-8B with everyone holding both recipes loses 25 points to talk; in the lost
  runs every action remakes the intermediate. The star avoids it (95%, the best cell). Explains much of F40 and F34.

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
- Temperature variety does not help (F21, F27), nor does variety between model families (F49); F6 only shows that
  models with more varied agents gain more.
- Sharing only failures does not help LLM agents: they retry pairs teammates reported failing (F35), while it is the
  one sharing that helps random teams (+16.5 at depth 2, F31).
- Prompt v1's example was copied by weak models; v2 fixed most of it (F8, F12–F14). SmolLM2 stays unreliable.
- Runs are byte-reproducible across jobs and nodes (F23); identical configs are deduplicated (F3).

## 6. Code state and open technical issues

- `main` = `origin/main` = `076fb11` on the local machine and on giano. Uncommitted (2026-11-20): this file,
  FINDINGS.md, `docs/analysis/` (k5–k12, README; `k10.py` parts 4–5 added), `docs/report/` (the whole folder is
  untracked: v1 and v2), `docs/analysis/k13.py` and `docs/LITERATURE_2026-11-22.md` and `docs/downloaded_papers/`
  (2026-11-22), the deep-research review's §9, the downloaded `run_*/` folders of the 2026-11-07 and 2026-11-12 grids, `run_confidence/`,
  re-downloaded `metrics.json` files of hints2_8b (new `evaluated_at` only), the user's `framework/mail.txt` and
  `framework/stuff.md` (see below), and the git-ignored pages.
- **Report v2** (`docs/report/report_v2.typ` → `report_v2.pdf`, 18 pages, Italian): v1's sections with forward
  pointers, then F46–F64 (sections 10–22: bigger models, the loop, mixed teams, the review, lost messages, channels,
  renamed items, confidence and size, sequential/chain, the adoption baseline (F64, `sec-baseline`), related work
  (`sec-lett`), conclusions, next steps). The main table's last column is now "Ripete un fallimento (muti → com.)". Its numbers come from
  `make_data_v2.py` → `data/data_v2.json` (pairs by target, bootstrap + exact sign test; reuses `k10.py`, `k11.py`, `k13.py`,
  `k12.py`), figures from `make_figures_v2.py` → `figures/v2_*.svg`; v1's `data.json` and figures are untouched.
  Build: `python docs/report/build.py --v2` (`--pdf` for the PDF only). typst and matplotlib are not in the local
  system python: use a scratchpad venv (`uv venv`, `uv pip install typst matplotlib numpy`) and call build.py with it.
- New on 2026-11-07 (`6d0202f`–`bd60792`, four commits):
  - **mixed teams**: `engine.agent_models` (list of `{model, revision}`; agents take them in turn by sorted id) and
    `framework/agents/backends/team.py` (`TeamBackend`: one backend per distinct model; the engine keeps
    `agent_backends` and calls each agent's own backend, token counts with its tokenizer). A grid names a team of
    `configs/models/models.yaml` `teams:` as a model (`experiment.load_teams`, `members`); `runs.csv` gets family
    `mixed`. `meta.json` gets `agent_models` (agent → model id) and `engine.backend: team` with `members`. Not
    allowed with `persona_host`. Checked locally with a mock team (one run in the scratchpad): routing and metadata
    right. Its first real test is the `mixed` gate (74397).
  - **`task.mark_made`**: an "Items you made (already in your inventory): ..." line under the inventory
    (`WordCraftTask.build_prompt` passes the inventory minus the starting items).
  - **gemma3-12b** in the registry (revision `96b6f1ec…` from the HF API, 12,187,325,040 parameters; downloaded on
    giano).
- Since 2026-11-07 (`9b33f0b`–`9d0a8ee`, "Add current results") the user commits the downloaded `run_*/` folders and
  `docs/analysis/`: `.gitignore` no longer excludes them. A re-download rewrites the old runs' `metrics.json` with a
  new `evaluated_at` and evaluator commit only (checked 2026-11-11: 760 files, no value changed).
- Fixed: F2 (the revision is read from the cache snapshot), the `setup_giano.sh` venv branch, the missing mock backend.
- Fixed behind a flag: shortcut targets (F32). New grids set `task.allow_shortcuts: false`; the default (true) keeps
  the earlier grids' targets.
- `framework/stuff.md` (untracked, user's) holds what look like an HF token and a password: it must never be
  committed (never use `git add -A` or `git add framework/` in the commit lines), and its content never repeated.
- Local Python: the system python (3.14) has numpy, pydantic and yaml, enough for the analysis scripts and for calling
  task functions by hand (`framework.tasks.wordcraft.*`). It lacks networkx, so anything that imports
  `framework.engine` or `framework.experiment` needs a throwaway venv (`uv venv -p 3.11` in the session's scratchpad,
  then `uv pip install pydantic pyyaml numpy networkx`). Use it only for static checks (expanding a grid with
  `experiment.expand`), never to run grids.
- **Results page**: `docs/results.html` is published as a private artifact, https://claude.ai/artifact/B6kwbxwkjkJ4tk4WDUwQ2Q
  (slug `51c92f70-f2b6-46fa-ab4a-8277cac3a8ed`, last version `1793913770-7eda`). To update it: edit the file, then
  republish with the artifacts server's `publish_files` (files `[{as: index.html, path: <absolute path of the file>}]`,
  that slug, `base_version` = the last version). Its chart data are written into the page's script at the bottom
  (from `docs/analysis/pagedata.py` and FINDINGS.md); it covers every experiment through hints2 and F1–F41. Read the
  server's `pages` and `design` guides before changing it. **Not yet republished**: the file covers F42–F44 (section
  3.8 and summary keys 05 and 07, added 2026-11-07), but the artifact tools failed in that session. Republish it
  from the file (call `get_artifact` first for the live version).
- **Two-week report**: `docs/changes_2026-11-06.html` (git-ignored), private artifact
  https://claude.ai/artifact/T5GV5UVfWCorzNySfMMhuX (slug `d3261f65-e21a-48c9-82a0-e7d336856ae6`, version
  `1793978928-cde5`): code changes 24 Oct – 5 Nov and a summary per experiment through hints2_share (F1–F41). **Not
  yet republished**: the file also covers F42–F44 and the queued grids (2026-11-07); republish it the same way.
- Personas used: `prompts/v1/personas/hub.txt`, `spoke.txt` (no v2 personas).

## 7. Plan

Done since 2026-11-05: the receiving side (hints2_use, F42), Qwen3-8B (hints2_8b, hints2_8b_share, F43, and on 80
targets F47), the intermediate loop (F44) and the made line (F48), Gemma3-12B (F46), mixed teams (F49), F41 closed.
The seven grids queued on 2026-11-07 (jobs 74395–74405) all finished and were downloaded and checked on 2026-11-11
(0 unfinished, 0 failed runs, `bd60792` clean, pairing with hints2 / main checked run by run).

Analysis scripts added (`docs/analysis/`): `k5.py` (gemma3-12b, qwen3-8b on 80 targets), `k6.py` (who
tells / who uses for both), `k7.py` (made line), `k8.py`, `k9.py` (mixed teams). `k4.py` now reads only the first 40
targets (hints2_8b has 80): F43–F45 and the report's `make_data.py`, which runs it, stay as they were.

### Done: the review's grids (prepared 2026-11-11, run 2026-11-12, analysed 2026-11-20: F56–F63)

`docs/REVIEW_2026-11-11.md` is a critical review of F1–F49 (new entries F50–F54 in FINDINGS.md). Its first four
experiments are ready; code checked locally only by expanding the grids, rebuilding the tasks (560 of 560 have
hints2's task and notes) and one stub-backend run per feature in the scratchpad. Their first real test is the gates.

- **`engine.publish_on_no_op`** (F50): the message of a turn whose answers were all refused is still sent
  (`Engine.run_agent` keeps the last message written, `apply` publishes it after the no-op). Off by default.
  `written_messages` in `runs.csv` still counts acting turns only: count `msg_published` of type `message` in the logs.
- **`task.rename`** (F52): every item shown under a made-up name (`tasks.pseudo_names`, seeded by the task seed); the
  task and notes are drawn as without it, `RecipeBook.renamed` is what the agents play. `meta.json` gets
  `task_original` (the task under the book's names, equal to hints2's) and `names_seed`; `task` and `notes` hold the
  made-up names. Use with `prompts: v2r` (no "real-world intuition", a made-up example without a plan).
- **`prompts/v3`** (F53): v2 plus the example game one round later, playing the second step.
- Grids (all on hints2's targets 0–39, fully connected):

| grid | what | runs | pairs with |
|---|---|---|---|
| `hints2_share_pub` | share instruction + `publish_on_no_op`; qwen3-4b, gemma3-4b, qwen3-8b; split with / without decoys | 240 | hints2_share, hints2_8b_share |
| `hints2_channels` | qwen3-8b, gemma3-12b; results only and messages only; split, decoys | 160 | hints2_8b, hints2_12b |
| `hints2_channels_share` | the same two, messages only + share instruction | 80 | hints2_8b_share, hints2_12b_share |
| `hints2_ex2` | prompts v3; llama3.2-3b, gemma3-4b, qwen3-8b, gemma3-12b; everyone holds both; 3 versions | 480 | hints2, hints2_8b, hints2_12b |
| `renamed` | `rename` + v2r + `publish_on_no_op`; qwen3-4b, gemma3-4b, qwen3-8b, gemma3-12b; split (decoys) and all; 3 versions | 960 | itself, hints2 (by target) |
| `renamed_share` | renamed's split teams + share instruction | 160 | renamed |

  Each grid's header says which `SWEEP_SHARDS` block is which model (qwen3-4b and llama on rtx2080, the rest on L40).
- Analysis plan: share_pub − share per model (does the gain grow, F50); results only / messages only − silenced and −
  both for the two big models (F51); remake share and talk − silenced with v3 against v2 (F53; `k7.py`'s measure);
  renamed: silenced split should be near 0, then share − talk and share − everyone holding both by model (F52). Report
  exact sign tests beside the bootstrap (F54).

### Prepared on 2026-11-12: confidence, sequential turns, 12 agents (the tutor's suggestions)

- **`engine.log_confidence`** (hf_local; off by default): a `TokenStats` logits processor records, without changing
  the scores, each generated token's log-probability and its distribution's entropy. `completion.meta` of every
  `llm_call` gets `confidence` (`base.confidence_summary`: perplexity, mean entropy, `combo_logprob` = log-probability
  of the text after "Combination:", `combo_entropy`, `combo_first_entropy`) and the per-token lists. `runs.csv` gains
  `perplexity_mean`, `token_entropy_mean`, `combo_prob_mean`, `combo_prob_first` (round 1), `combo_entropy_mean`,
  `combo_entropy_first` (empty for runs without it). Part of `engine_key`, so a shard never reuses a backend across
  the flag. `confidence_check` is its gate: the two gemma3-4b units must be byte-identical to main's.
- **`scripts/rescore.py`** + `scripts/slurm/rescore.sbatch`: the same measures for finished runs, by teacher
  forcing the logged answers (retokenized; the end-of-turn token is not scored). Writes `runs_confidence/<grid>/...`
  on giano (git-ignored); download it to `run_confidence/`.
- **Protocols**: `sequential` (fully_connected, `schedule: {by: position}`: agent k sees agents 0..k−1 of the same
  round), `chain6` (directed relay, results and message to the next agent only; the last one has no message slot),
  `fully_connected12`. Stub runs checked who hears whom in round 1.
- **Grids**: `chain` (sequential, chain6 × main's settings × 4 models, base; 320 runs; pairs with followup / main),
  `chain_hints` (sequential × hints2 split and all × 4 models, base; 320; pairs with hints2 / hints2_8b),
  `size12` (12 silenced agents × main's settings × 3 models; 120), `confidence_check` (4). All log confidence.
- Analysis: `k10.py` (F55, token confidence, 12 vs 6 vs 3), `k11.py` (sequential / chain − parallel and − silenced,
  herding, "agents 1–5 playing agent_00's first pair", solve rate by turn position); `lib.sign_test`.

### Prepared on 2026-11-22: renamed with the made line and a token-matched single agent

- `renamed_made` and `renamed_made_share` (YAML only). Checked locally by expanding (320 + 160 units, 1,280 + 320
  distinct runs) and rebuilding every task: the team runs have renamed's / renamed_share's task, notes and made-up
  names, run by run; the token-matched agent's task differs only in its step cap (4 → 240) and it holds both path
  notes, as one_agent does. SWEEP_SHARDS=4: block 0 (qwen3-4b) on rtx2080, 1–3 on L40.
- Questions: renamed + made − renamed for one agent, everyone-holds-both and talk (does the loop explain F62?);
  share − one agent holding both, and share − token-matched agent (is the 92.5% vs 37.5% of F61 more than compute?).
- Estimated GPU time from renamed's latencies: about 7–11 h per model for renamed_made, 2–4 h for the share grid.
- Submitted 2026-11-22 on `e8693bc` (pushed with F64, the literature notes and the report): gates 76396 (renamed_made
  unit 0) and 76397 (renamed_made_share unit 0), rtx2080; full jobs renamed_made 76398 (L40, blocks 1–3) and 76400
  (rtx2080, block 0), renamed_made_share 76402 (L40) and 76403 (rtx2080), all `afterok` on their gate. 76399 was
  submitted without the grid argument and cancelled.

### Prepared on 2026-11-22: counterfactual scoring of logged turns (`scripts/counterfactual.py`)

- For every turn after round 0 (first attempt) of finished runs: log P("Combination: 'a' and 'b'", both orders) for
  the pair answered, the path's recipes (from the recipe book for main, which did not log the path), pairs named in
  received messages, and teammates' previous-round pairs; under the logged prompt and edited ones: `no_msgs`,
  `swap_msgs` (messages the same agent got in the same round of another replicate), `no_results`, `drop:<ref>` (one
  message). Prompts are cut by the logged `segments` (each message and result has its ref and length): checked on
  1,147 turns that removing everything gives exactly the round-0 layout and that rejoining gives the logged prompt.
  `--dry` builds everything without a model (checked locally on main and renamed). GPU path untested until the gate.
- `scripts/slurm/counterfactual.sbatch <model> <grid folders> [options]`; output `runs_counterfactual/` (git-ignored).
- Submitted 2026-11-22 from `MAF-tools` (a sparse, shallow clone: `scripts`, `configs` and the top-level files):
  test 76408 (qwen3-4b, main, `--limit 2`, rtx2080), then `afterok`: 76410 qwen3-4b (renamed, renamed_share, main;
  rtx2080, `--batch 4`), 76411 llama3.2-3b (main; rtx2080), 76412 gemma3-4b, 76413 qwen3-8b (renamed, renamed_share,
  main), 76414 gemma3-12b (renamed, renamed_share), on L40. All with `--cells fully_connected`.
- 2026-11-23: 76408 and 76411 (llama, 40 main runs) finished, 76410 (qwen3-4b) ran out of memory on the rtx2080 at
  batch 4. Llama's scores were wrong: P(the pair it played) ≈ 0.001 against ≈ 0.25 rescored (Qwen3-4B's test runs
  were right, 0.75). Cause: the right-padded batch (Llama has no pad token; padded with token 0); never used by
  rescore.py. Fixed: each prompt runs once and each answer continues a copy of its cache (one sequence, no padding,
  no `--batch`); every run checks one answer against a plain full forward (`check` in each output line, largest
  difference printed at the end). 76412–76414 cancelled (old code) and everything resubmitted; the outputs scored
  before the fix were deleted on giano. A llama job on followup's fully connected depth-1 runs validates against
  the rescore (`run_confidence/followup`).
- Questions: does a told recipe raise the receiver's probability of playing it (Gemma3-4B vs the rest, F45, F61)?
  In fully connected main, do the messages' content or only their presence pull agents (swap vs orig vs no_msgs)?
  Validation: `orig` of the played pair against the rescored `combo_logprob` (F56) on main.

### Next

1. **Write-ups.** The report's v2 is written (section 6); the user may edit it as they did v1. `framework/mail.txt`
   (the user's mail) is not updated. The report covers F64 and related work (2026-11-22).
2. Republish the two pages (section 6), adding F42–F63.
3. Commit the new analysis, report and results (the user decides; never `git add -A` or `framework/`).
4. Possible next grids (the user decides):
   - **renamed with `mark_made`** (F61, F62): separates pooling from the remake loop, which with made-up names costs
     Qwen most (one agent 67.5% → 37.5%, everyone-holds-both talk 15%);
   - `mark_made` as the default for new grids (F48, F60), and made-up names as the default task for pooling questions
     (F61: no leak);
   - gemma3-4b under renamed + share tells in 30% of runs only: why (prompt, refusals)?
   - a second family at 8B (e.g. Llama 3.1-8B) to separate size from family once more, now on renamed;
   - mixed teams on more targets (F49) and with the star's hub rotated through the families;
   - the review's C5 (stamped / deduplicated message and result lists) is still untested;
   - from the literature (no GPU unless stated): counterfactual scoring of logged receiver turns (prompt with / without
     / with a swapped message, scored like `rescore.py`; method of 2610.35930); a placebo-message arm on main's fully
     connected cells (GPU; noise control of 2605.00914); a track-record line for receivers instead of "notes are
     true" (GPU; 2601.21742); a summary relay for the small models (GPU; 2607.16133);
   - `metrics.py`: log the would-be-inbox baseline for silenced runs (F64) so it is standard.

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
  runs by (model, protocol, task setting, replicate, version), report paired bootstrap 95% intervals resampled by
  target (`lib.boot(..., cluster=replicate)` in `docs/analysis/`), and use `solved_any` for team success
  (`ended_by: all_solved` means every agent solved, F4). Report depth-2 results of the grids before hints2 also
  without shortcut targets (`docs/analysis/short200.json`, F32). Compare LLM teams with the random agents.
- Look at the logs (`events.jsonl`: `llm_call`, `env_step`, `msg_published`) to explain a result, not only the
  success rates: most findings came from there (F17, F34, F38).
- Claude's memory for this project holds: log findings, the cluster notes, this handoff, and "no local runs".
