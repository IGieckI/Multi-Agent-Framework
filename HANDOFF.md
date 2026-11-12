# Handoff: state of the project and plan (2026-11-11)

For a new Claude session picking up this work. Read this file, then `FINDINGS.md` (44 numbered findings with their
evidence and status). Code as of commit `bd60792` (local = `origin/main` = giano). **Every experiment has finished,
been downloaded and been analysed (F1–F49)**, including the seven grids queued on 2026-11-07 (F46–F49, analysed
2026-11-11). Nothing is queued on giano. A review on 2026-11-11
(`docs/REVIEW_2026-11-11.md`) added F50–F54 and prepared six new grids, not yet committed or submitted (section 7). Section 7 lists what is left: the write-ups and possible next grids.

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
| `configs/experiments/*.yaml` | grids: pilot, prompt_v2_check, main, followup, random_baseline, hints, hints2, hints2_share, hints2_use, hints2_8b, hints2_8b_share (both extended to 80 replicates), hints2_12b, hints2_12b_share, hints2_made, mixed, mixed_hints (queued 2026-11-07), smoke_mock, smoke_hf, smoke_models |
| `prompts/v1`, `prompts/v2` | templates; v2 differs only in the WordCraft rules (the example framed as another game); select with `task.prompts` |
| `FINDINGS.md` | every finding from results, F1–F41 |
| `docs/results.html` | the results web page (git-ignored, see section 6) |
| `docs/analysis/` | the analysis scripts behind F30–F40 and the page (git-ignored; `README.md` lists them) |

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

## 5. What we know (details and numbers in FINDINGS.md)

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

- `main` = `origin/main` = `bd60792` on the local machine and on giano; nothing uncommitted except the user's
  `framework/stuff.md` (see below) and the git-ignored pages.
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

### Prepared on 2026-11-11, not yet committed or submitted: the review's grids

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

### Next

1. **Write-ups.** `docs/report/report.typ` (the Italian PDF for the supervisor, user-edited) still has the 2026-11-07
   grids under "In corso": replace that section with F46–F49 (a short section each, or one "Nuovi risultati"), and
   update the Qwen3-8B paragraph with the 80-target numbers (F47). Its `make_data.py` filters Qwen3-8B to 40 targets
   in `remake`; check every other read of `run_hints2_8b` before rebuilding the data, or keep the report at 40 targets
   and quote F47 in the text. `framework/mail.txt` (the user's mail) likewise.
2. Republish the two pages (section 6), adding F42–F49.
3. Possible next grids (the user decides):
   - `mark_made` with gemma3-12b (it remakes most of all, 61–66%, F46) and as the default for new grids (F48);
   - Gemma3-12B with everyone holding both in a star (−22.5, F46): is it the hub restating the plan?
   - mixed teams on more targets (F49's talk effects are suggestive only) and with the star's hub rotated through
     the families (the hub is always qwen3-4b); a mix with qwen3-8b / gemma3-12b members;
   - a second family at 8B (e.g. Llama 3.1-8B) to separate size from family once more.

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
