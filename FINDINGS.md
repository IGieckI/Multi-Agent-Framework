# Findings

What the results have shown, one entry per finding, newest last. Each entry says what was seen, the evidence (numbers
and where they come from), what it changes, and its status: **open**, **acted on** (with the change), or **superseded**.

Success counts below are "x/24" or percentages of runs. With 24 runs a 95% interval is about ±20 points, with 240
about ±6: differences smaller than that are not findings.

---

## Pilot: `runs/pilot` (2026-10-29 → 2026-10-31)

Grid `configs/experiments/pilot.yaml`: 10 models × 3 protocols (isolated 6 agents, fully_connected 6, star 3) × 4 task
settings (depth 1/2 × distractors 0/3) × 6 replicates × 3 versions (base, all_independent, one_agent) = 2,160 runs,
prompts v1, 4 steps, temperature 1.0, on the DISI L40s. All 2,160 runs ended on their own (0 failed).
Sources: `run_56/shard0of2.jsonl`, `run_56/shard1of2.jsonl` (sweep logs), `run_56/results.csv` (`sweep.py evaluate`),
`run_56/seed0_rep3/` (one run: llama3.2-3b, star, depth 2, distractors 3, base).

### F1. Gemma 3-4B did not load: our backend read a field its config does not have
- `AttributeError: 'Gemma3Config' object has no attribute 'max_position_embeddings'` on every gemma3-4b run of the
  first submission (216 runs). Gemma 3-4B is multimodal: the field is in its nested text config.
- **Acted on**: `framework/agents/backends/hf_local.py` reads `config.get_text_config()`; the resubmission ran all 216.

### F2. The loaded weights' revision is not recorded when the hub is offline
- Every run's `engine.revision` is `null` (transformers sets no `_commit_hash` with `HF_HUB_OFFLINE=1`); only
  `requested_revision` (the pinned commit) is recorded. Offline loading can only serve that snapshot, so results stay
  traceable, but the record is not the check it was meant to be.
- **Open**: read the commit from the snapshot folder the weights were loaded from.

### F3. 44% of the pilot's runs were exact duplicates of other runs
- `one_agent` drops the protocol, so it is the same run in all 3 protocols of a model, task setting and replicate:
  480 of its 720 runs are repeats. `isolated` has no links, so its base, its all_independent and fully_connected's
  all_independent are the same run: 480 more. Checked on the logs: identical ended_by, steps, calls and tokens in 240/240
  cases for each. Only 1,200 of 2,160 runs are distinct.
- It also shows that runs are reproducible: same config and seed → byte-identical results.
- Consequence: pooled sample sizes are smaller than they look (one_agent: 24 distinct runs per model, not 72).
- **Acted on**: `framework/experiment.py` runs a run once when its config differs from another only in the cell name
  (`run_key`, `Unit.same_as`); `runs.csv` gets a `run_dir` column to count shared runs once. This removes the one_agent
  repeats (pilot: 2,160 → 1,680 runs). The `isolated` duplicates differ in config (protocol structure), so they are
  removed by design instead: **open**, drop `isolated` from the next grid (it is fully_connected's all_independent).

### F4. `ended_by: all_solved` means every agent solved, not the team
- `framework/tasks/wordcraft/task.py:106`: a multi-agent run ends `all_solved` only if *each* agent's own environment
  reached the target. Comparing it with one_agent (one agent must solve) penalises bigger teams: star (3 agents) looked
  better than fully_connected (6 agents) on it (39% vs 29%) only because fewer agents had to succeed.
- Use `solved_any` (any agent reached the target) from `results.csv` for team success; `all_solved` in the sweep log
  is a progress signal only.

### F5. Teams beat a single agent through more attempts, not through communication
- `solved_any`, all models pooled (240 runs per cell): one agent 39% [33–45]; fully_connected without messages
  (all_independent) 52% [46–58], with messages (base) 47% [41–53]; star without messages 54% [47–60], with 52% [46–58].
- Messages never help and sometimes hurt (not significant pooled). Clearest per model: gemma3-4b fully_connected 11/24
  with messages vs 17/24 without, star 15 vs 17; smollm2-360m fully_connected 7 vs 11.
- Same direction for `team_coverage` (FC 0.269 vs 0.289) and `steps_to_target`.
- **Open**: the main question of the study; to be tested with more replicates and in the settings of F10.

### F6. The gain from a team follows how different its agents are
- Qwen3 agents behave almost like copies: `redundancy` 0.76–0.95 and final `pairwise_divergence` 0.00–0.03 in base,
  and a Qwen3-8B team barely beats one agent (58% vs 54%). Llama 3.2 and SmolLM2-1.7B teams are the most varied
  (redundancy 0.50–0.66, divergence 0.07–0.13; smollm2-360m 0.75 and 0.08) and gain the most: llama3.2-3b 17% alone →
  46–62% as a team, smollm2-1.7b 29% → 54–62%.
- Not a seeding bug: every call gets its own seed from (seed, replicate, agent, step, attempt) (`call_seed`,
  `framework/engine/engine.py:82`), checked in `seed0_rep3` (15 calls, 15 different seeds).
- Part of the Llama/SmolLM2 variety is noise from failed answers (F8).
- Consequence: temperature-driven variety (`all_independent_temp_sweep`) is the baseline to beat.

### F7. Model ranking (single agent, one_agent, 24 distinct runs each, `solved_any`)
- gemma3-4b 62%, qwen3-8b 54%, qwen3-4b 50%, qwen3-1.7b 42%, gemma3-1b 38%, qwen3-0.6b 33%, smollm2-360m 33%,
  llama3.2-1b 29%, smollm2-1.7b 29%, llama3.2-3b 17%. Only the ends of the list are clearly apart (±20 points).
- Qwen3 improves with size in every version and protocol.
- Llama and SmolLM2 are ranked low partly because of the prompt (F8): **superseded for Llama** by F14 (with prompts
  v2 llama3.2-3b solves as often as qwen3-4b on the same targets); SmolLM2 stays low.

### F8. Llama and SmolLM2 mistake the rules' example for their own game (prompt v1)
- `parse_failure_rate` (answers refused and asked again): llama3.2-1b 76%, llama3.2-3b 74%, smollm2-1.7b 66%,
  smollm2-360m 59%, gemma3-1b 21%; qwen3 ≤2%, gemma3-4b 1%. Calls per counted answer 2.0–2.4 vs ~1.0.
- In `seed0_rep3` (llama3.2-3b; target 'roe', inventory 'ant', 'day', 'net', 'sea', 'wind') the first answer was
  `'metal' and 'human'`, "a cyborg often has metallic parts": the items and target of the example in
  `prompts/v1/wordcraft/rules_targeted.txt`. Other refused answers used items that are in no inventory ('fish',
  'water'). 4 of 15 calls were refused.
- Cause: in v1 the example is laid out exactly like the real task, under the same `<Current task>` header.
- **Acted on**: `prompts/v2` (task setting `prompts: v2`), which frames the example as another game (`<Example task>`
  … `<End of example>`) and says to use only items of the own inventory; nothing else changes.
  `configs/experiments/prompt_v2_check.yaml` measures it against the pilot, run by run: see F14 (v2 works).
  How often the example is copied in the whole pilot: see F13, its items are the most invented ones.

### F9. Some models barely use the communication channel
- Messages written per run (base, FC and star): qwen3 8.4–10.2, gemma3-4b 10.3, llama3.2 4.8–5.0, gemma3-1b 4.8,
  smollm2-1.7b 2.3, smollm2-360m 0.6. For SmolLM2 "with messages" is close to "without".
- Messages are followed: `adoption_rate` 0.81–0.96 for qwen3 and 0.81 for smollm2-360m, 0.45–0.71 for the others;
  blind adoption (following without own reason) 0.07–0.32, highest for smollm2-360m.

### F10. Two of the four task settings are at the ceiling or the floor
- `solved_any` (FC with messages / FC without / star with / one agent), all models:
  depth 1, 0 distractors 98/98/100/92%; depth 1, 3 distractors 57/63/62/38%; depth 2, 0 distractors 20/33/35/17%;
  depth 2, 3 distractors 13/13/12/8%.
- Only depth 1 with distractors and depth 2 without separate the versions. Depth 2 with distractors in 4 steps is
  nearly impossible. **Open**: drop the two extreme settings, or give depth 2 more steps.

### F11. Cost: about 20 GPU-hours for the pilot, nearly independent of model size
- Mean seconds per run (base / all_independent / one_agent): qwen3-0.6b 29/24/5, qwen3-8b 35/30/6, gemma3-4b 29/21/4,
  llama3.2-3b 74/72/16, smollm2-1.7b 96/80/17. Generation (HF transformers, batch 1) is ~26 tokens/s and bound by
  per-token overhead, so an 8B model costs about what a 0.6B one does; the slow ones are slow because of retries (F8)
  and long answers.

### F12. Copying the example explains only 4–18% of the refused answers; most are invented items
**Superseded by F13**: the 4–18% counted only answers whose *two* items both come from the example. Most refused
answers pair one example item with one real item, so this undercounts the copying badly.
- Over every LLM call of the pilot's distinct runs (duplicates of F3 skipped; counted on giano over
  `runs/pilot/*/*/seed0_rep*/events.jsonl`, completions only), per model: refused answers, share of them refused for an
  item not in the inventory, share whose combination uses only the example's items ('fire', 'human', 'life',
  'metal'), and answers mentioning "cyborg":
  llama3.2-1b 2,528/3,433 refused (74%): 81% not in inventory, 18% example items, 569 mention cyborg;
  llama3.2-3b 2,874/3,609 (80%): 83%, 15%, 563; smollm2-360m 2,412/3,239 (74%): 58%, 12%, 438;
  smollm2-1.7b 2,428/3,289 (74%): 51%, 4%, 187; gemma3-1b 824/2,016 (41%): 54%, 4%, 22;
  gemma3-4b 38/1,293 (3%) and qwen3 0–44 per model (≤3%), none with example items.
- So the example of F8 is copied, but most refusals use items that are not from it ('fish', 'water' in `seed0_rep3`).
  The rules' inline examples ('fire' and 'water' → 'steam', 'wall' and 'wall' → 'house') may leak too: not yet counted.
  For SmolLM2 and gemma3-1b about half the refusals are something else than a missing item (likely format).
- **Open**: which items are invented and what the other refusals are (tally on giano); then decide whether v2 needs a
  further change (reword the inline examples, or repeat the inventory in the retry message).

### F13. What the refused answers are: the example's items, the game's classic elements, and broken format
- Tally on giano of the refused answers of llama3.2 ×2, smollm2 ×2 and gemma3-1b (pilot, distinct runs, prompts v1):
  items named in "not in your inventory" refusals, and the other error messages. Counts are of item mentions.
- The most invented items come from the prompt itself:
  - the v1 example ('metal', 'life', 'human', 'fire', 'robot'): llama3.2-1b 439/398/306/204/32, llama3.2-3b
    400/319/193/347, smollm2-360m 279/328/105/151/32, smollm2-1.7b 79/79/38/56;
  - the rules' inline examples ('fire' and 'water' → 'steam', 'wall' and 'wall' → 'house'): 'water' 143 (llama 1B),
    374 (llama 3B), 98 and 85 (smollm2); 'wall' 47 and 24; 'house' 33;
  - the answer format itself: smollm2-360m answered `'first item' and 'second item'` 162 times.
- Items of the real Little Alchemy 2, which starts from air, earth, fire and water: 'stone' 737, 'earth' 280, 'wood' 188,
  'rock' 163, 'air' 71 (llama3.2-3b); the model plays the game it knows, not the inventory it is given.
- The target used as an ingredient: 'roe' 94 (llama3.2-1b).
- gemma3-1b invents other plausible items ('mermaid' 61, 'petroleum' 61, 'explosion' 60, 'mineral' 56, 'steel' 56),
  more like items it hopes to make than copies.
- The refusals that are not about items are format: no line starting with `Combination:` (llama 388 and 463, smollm2
  985 and 539, gemma3-1b 353), a malformed combination line (smollm2-360m 352, llama3.2-1b 70), answers cut off at
  256 tokens (≤56 per model).
- (The ', ' "item" in the tally is an artifact of splitting the error message, not an answer.)
- Consequence: v2 targets the biggest single source (the example) but leaves the inline examples, the format
  placeholders and the inventory reminder as they are. **Open**: wait for `prompt_v2_check`; if the example items
  drop but failures stay high, a v3 could (a) reword the inline examples with items that are not Little Alchemy
  basics, (b) write the format as `Combination: '<an item of your inventory>' and '<another one>'`, (c) repeat the
  inventory in the retry message. These are failures of instruction following, so how far to remove them is a choice
  of the study, not a bug fix. See F14 for what v2 left.

---

## Prompt v2 check: `runs/prompt_v2_check` (2026-11-01, job 74062)

Grid `configs/experiments/prompt_v2_check.yaml`: llama3.2 ×2, smollm2 ×2, gemma3-1b and qwen3-4b (control) ×
fully_connected and star × depth 1 and 2 with 3 distractors × replicates 0–3 × base and one_agent, prompts v2. 192 runs,
144 distinct (one_agent shared across protocols, F3), 0 failed. Each run has the target of its pilot twin (same seed and
replicate), so v1 and v2 are compared pair by pair: 144 pairs, 24 per model. Intervals are 95% paired bootstraps.
Sources: `run_62/prompt_v2_check/` (runs, `runs.csv`), `run_62/runs.csv` (the pilot's `runs.csv`).

### F14. Prompt v2 cuts the weak models' refused answers by a third and lifts Llama's success; Qwen is unaffected
- `parse_failure_rate` v1 → v2: llama3.2-3b 0.59 → 0.22 [−0.52, −0.22], llama3.2-1b 0.63 → 0.45 [−0.27, −0.10],
  smollm2-1.7b 0.64 → 0.38 [−0.35, −0.18], smollm2-360m 0.69 → 0.63 [−0.12, 0.00], gemma3-1b 0.26 → 0.19
  [−0.16, +0.01]; all five pooled 0.56 → 0.37 [−0.24, −0.14]. Control qwen3-4b 0.04 → 0.03 (no change).
- Copying the example almost disappears: answers mentioning "cyborg" went from 16–17% of all Llama calls in the pilot
  (F13) to 6 of 483 (llama3.2-1b) and 0 of 323 (llama3.2-3b); smollm2-360m 14% → 30 of 648 (5%).
- Success (`solved_any`, pairs where v1 / v2 solved: both, v1 only, v2 only, neither): llama3.2-3b 9/0/4/11,
  0.38 → 0.54 [+0.04, +0.33]; llama3.2-1b 2/1/5/16, 0.12 → 0.29 [0.00, +0.38]; smollm2-1.7b 5/3/2/14,
  smollm2-360m 1/1/1/21 and gemma3-1b 6/3/1/14 (no change); qwen3-4b 15/0/0/9, the same outcome on every target.
  `team_coverage` and `invalid_combo_rate` do not change.
- Cheaper: calls per run llama3.2-3b 24.0 → 13.5, smollm2-1.7b 27.0 → 21.1, llama3.2-1b 24.5 → 20.1.
- What v2 leaves (refusals for an item not in the inventory, share by cause; 'fire' and 'water' are both in the
  rules' inline example and Little Alchemy's starting elements):
  llama3.2-3b: 'fire'/'water' 63%; llama3.2-1b: the target or its intermediate used as ingredient 25%, 'fire'/'water'
  18%; smollm2-360m: example items 22%, 'fire'/'water' 20%, the format's placeholders 14% (`'first item' and
  'second item'` 38 times), plus 203 answers with no or a malformed `Combination:` line; smollm2-1.7b: mostly format
  (100 answers with no or a malformed line); gemma3-1b: 'human' 63 and 'mermaid' 62 (46% count as example items, but it
  mentions "cyborg" once, so this looks like planning a recipe, not copying).
- Consequences: **v2 replaces v1** for the next experiments (it helps the weak models and does not change the strong
  one). With v2, llama3.2-3b (54%) is close to qwen3-4b (62%) on the same targets, so Llama 3.2-3B is a candidate third
  family again, with the most varied agents (F6). SmolLM2 stays unreliable (F7). **Open**: whether a v3 (inventory in
  the retry message, placeholders that cannot be copied, inline examples without 'fire'/'water') is worth another
  check; only the weak models would gain.
