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
- **Closed**: `hf_local.py` reads the commit from the cache snapshot folder when transformers records none
  (`c59c6b9`); every hints run records it (e.g. Qwen3-4B `1cfa9a7…`, the pinned commit).

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
- **Acted on**: tested in the main experiment, see F16 (fully_connected messages hurt) and F18 (star: neutral).

### F6. The gain from a team follows how different its agents are
- Qwen3 agents behave almost like copies: `redundancy` 0.76–0.95 and final `pairwise_divergence` 0.00–0.03 in base,
  and a Qwen3-8B team barely beats one agent (58% vs 54%). Llama 3.2 and SmolLM2-1.7B teams are the most varied
  (redundancy 0.50–0.66, divergence 0.07–0.13; smollm2-360m 0.75 and 0.08) and gain the most: llama3.2-3b 17% alone →
  46–62% as a team, smollm2-1.7b 29% → 54–62%.
- Not a seeding bug: every call gets its own seed from (seed, replicate, agent, step, attempt) (`call_seed`,
  `framework/engine/engine.py:82`), checked in `seed0_rep3` (15 calls, 15 different seeds).
- Part of the Llama/SmolLM2 variety is noise from failed answers (F8).
- Consequence: temperature-driven variety (`all_independent_temp_sweep`) is the baseline to beat; but the sweep as
  built lowers the variety instead (F21), and a sweep centred on 1.0 changes nothing (F27): **superseded**, variety
  from temperature does not help. Whether variety *between families* helps is **untested**: no mixed-model team has
  been run; this entry only shows that models whose agents differ more gain more from a team. Random agents add more
  variety than any LLM team (F30, F31).

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
  nearly impossible. **Acted on**: the main grid and the later ones use only the two middle settings (`hints`
  and `hints2` use depth 2 with 6 distractors instead).

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

---

## Main experiment: `runs/main` (started 2026-11-01)

Grid `configs/experiments/main.yaml`: qwen3-4b, qwen3-8b, gemma3-4b, llama3.2-3b × fully_connected and star × (depth 1,
3 distractors) and (depth 2, 0 distractors) × 20 replicates × base, all_independent, all_independent_temp_sweep,
one_agent, one_vs_multi_token; prompts v2. 1,600 runs, 1,440 distinct. Gated by its first unit (`shard0of320.jsonl`).

### F15. The token-matched single agent is told the safety cap as its remaining rounds, not its real budget
- In the first unit (qwen3-4b, fully_connected, depth 1, rep 0) `one_vs_multi_token` and `one_agent` gave the same
  answer (57 completion tokens, solved in 1 step) but the first prompt of `one_vs_multi_token` was 2 tokens longer
  (493 vs 491): its "Remaining rounds:" line shows the horizon cap of `framework/baselines.py` (SAFETY_FACTOR 10 ×
  6 agents × 4 steps = 240), while the team's agents see 4. Its real limit is the team's token budget, which it does not
  see.
- So when it does not solve at once, it plays as if it had 240 rounds; the comparison "same tokens, one agent" holds,
  but the agent's urgency differs from the team's. All 5 versions of the unit ran without errors.
- Confirmed relevant by the full run (F19): 21 of its 95 fully_connected solves came after step 4.
- **Acted on**: `budget.show_matched_budget` (off by default) replaces the agent's "Remaining rounds" with
  "Remaining budget: N tokens"; on in `configs/experiments/followup.yaml`. It changes nothing (F28): **closed**, the
  main grid's `one_vs_multi_token` stands.

Full run (jobs 74066/74067, 2026-11-01 14:05 → 20:38; shard 0 6.5 h, shard 1 4.6 h): all 1,440 distinct runs ended on
their own, 0 failed. Every version of a replicate has the same target (task seed from seed and replicate), so all
comparisons below are paired: 160 pairs per protocol (4 models × 2 settings × 20 replicates), 40 per model and
protocol. Intervals are 95% paired bootstraps. Success is `solved_any`. Source: `run_main/main/runs.csv`.

### F16. In fully_connected, letting the agents talk makes the team worse
- Same 6 agents with and without messages: base − all_independent = −9.4 points [−14.4, −5.0] (53% vs 62%).
- Negative for every model (qwen3-4b −12 [−23, −3], qwen3-8b −7 [−17, 0], gemma3-4b −12 [−25, 0], llama3.2-3b −5
  [−12, 0]) and in both settings (depth 1 −8.7 [−15.0, −3.7], depth 2 −10.0 [−17.5, −2.5]).
- Team coverage falls too for qwen3-4b (−0.050 [−0.086, −0.019]) and qwen3-8b (−0.034 [−0.057, −0.014]).
- This is the clearest result of the study so far, and it confirms the pilot's direction (F5) with 4× the data.

### F17. Why: fully connected agents herd onto the same combinations
- base vs all_independent in fully_connected (means over runs): messages written 16.2 per run; agents adopt what
  they receive 87% of the time, 26% of the time with no reason of their own (blind adoption); 1.8 copy events per run.
- The team explores less: distinct paths 0.544 vs 0.644, final pairwise divergence 0.040 vs 0.060, repeat rate 0.360
  vs 0.316, new items found by the team 1.42 vs 1.61. Qwen3 agents were already near-copies (F6) and become more so
  (redundancy 0.97 vs 0.89 for qwen3-4b).
- So messages do not pool different knowledge; they pull six agents onto one plan. Refined by F26: both shared results
  and free messages do it.

### F18. In star, messages neither help nor hurt
- base − all_independent = +1.2 points [−4.4, +6.9] (corrected 2026-11-03: it read −1.3 [−6.9, +4.4], the depth-1 number); depth 1 −1.3 [−8.7, +6.2], depth 2 +3.7 [−5.0, +12.5].
- The hub writes few messages (2.5 per run) and they are followed less, and rarely blindly (adoption 0.60, blind 0.06).
- Only cell with a clear gain: qwen3-4b +10 [+3, +20] (coverage +0.043 [+0.021, +0.068]). One positive cell in eight
  tests is what chance can give: not a finding on its own.
- Caveat: star has 3 agents and fully_connected 6 (F22), so F16 vs F18 compares two structures *and* two team sizes.

### F19. A single agent with the team's token budget matches the teams without messages and beats fully_connected
- Paired, vs the team with messages (base): one agent with the same number of steps (one_agent) −5.0 [−11.9, +1.9]
  vs fully_connected, −15.6 [−22.5, −8.7] vs star. One agent with the team's token budget (one_vs_multi_token):
  +6.9 [+0.6, +13.1] vs fully_connected, −9.4 [−16.3, −2.5] vs star.
- vs the team without messages: token-matched − all_independent = −2.5 [−6.9, +1.9] (fully_connected), −8.1
  [−13.8, −2.5] (star).
- So at equal tokens one agent is as good as six independent agents and better than six talking ones; three
  independent or star agents still beat one agent at equal tokens.
- It wins partly by playing longer: 17.3 steps per run on fully_connected's budget (up to 49), and 21 of its 95 solves
  came after step 4, the team's horizon. It is also told 240 remaining rounds (F15).

### F20. Messages in fully_connected cost 54% more tokens for less success
- Mean tokens per run / success / tokens per solved run: fully_connected with messages 14,461 / 53% / 27,545; without
  9,399 / 62% / 15,191 (1.8× cheaper per solve); star with 6,006 / 63% / 9,514; without 5,468 / 62% / 8,837; one
  agent 1,608 / 47% / 3,385.
- The difference, 5,062 tokens per run over about 20 agent turns (6 agents × 3.3 steps), is about 250 extra tokens
  per agent turn in fully_connected, where everyone reads everyone's messages.

### F21. The temperature sweep lowers variety instead of raising it
- `TEMP_SWEEP = (0.4, 0.7, 1.0, 1.3)` in `framework/baselines.py` averages 0.85, while every other version samples at
  1.0. Against all_independent (all at 1.0): distinct paths 0.606 vs 0.644, final divergence 0.049 vs 0.060, team new
  items 1.48 vs 1.61 (fully_connected); success −3.7 points [−7.5, −0.6] (fully_connected), −1.9 [−5.0, +1.3] (star).
- So it does not test "more variety helps"; it tests "some colder agents". **Acted on**: `sampling.sweep_temperatures`
  sets the sweep per grid without touching the other versions; the follow-up uses (0.7, 1.0, 1.3), mean 1.0: see F27.

### F22. Protocol and team size are confounded in this grid
- fully_connected has 6 agents and star 3, so "star is better than fully_connected" (63% vs 53% with messages) may be
  structure or size: the no-message teams are equal (62% vs 62%) although one has twice the agents, and the herding of
  F17 grows with the number of senders each agent reads.
- **Acted on**: the follow-up adds `fully_connected3` and `star6` (a hub and 5 spokes), and fully_connected sharing
  only results or only messages, to test where the herding of F17 comes from. **Closed** by F24 and F25: it is the
  structure, not the size.

---

## Follow-up: `runs/followup` (2026-11-02 → 2026-11-03)

Grid `configs/experiments/followup.yaml`: qwen3-4b, gemma3-4b, llama3.2-3b × 6 protocols (fully_connected and star
again; fully_connected3 with 3 agents; star6 with a hub and 5 spokes; fully_connected sharing only results or only
messages) × the main grid's 2 settings × 20 replicates × the main grid's 5 versions, with the temperature sweep centred on
1.0 (0.7, 1.0, 1.3) and the token-matched agent shown its budget. 3,600 runs, 2,520 distinct, all ended on their own,
code `78e7719` clean. The L40s were taken for days, so Qwen and Llama ran on RTX 2080 Ti in float16 (jobs 74110, 74114)
and Gemma on an L40 in bfloat16 (74115). Every version, protocol and model of a replicate has the same target, so all
comparisons are paired: 120 pairs per protocol (3 models × 2 settings × 20). Intervals are 95% paired bootstraps.
Source: `run_followup/followup/runs.csv`, compared with `run_main/main/runs.csv`.

### F23. Runs are reproducible across jobs and nodes, and the results hold in float16
- Gemma (bfloat16 in both): every repeated fully_connected and star run of base, all_independent and one_agent is
  byte-identical to the main experiment's (same steps, calls and tokens), on another node and another day.
- Qwen and Llama in float16: the exact tokens change (3–26 of 80 runs identical), the outcomes do not: success
  float16 − bfloat16 = 0.00 [−0.03, +0.03] (qwen3-4b) and −0.02 [−0.06, +0.02] (llama3.2-3b), 200 pairs each;
  parse failures 0.010 → 0.012 and 0.088 → 0.082. So the follow-up and the main experiment can be read together.

### F24. Fully connected talk hurts at both sizes; star is neutral at both: it is the structure, not the size
- base − no links: fully_connected 6 agents −10.8 [−17.5, −5.0] (the main experiment again: −9.4), 3 agents −7.5
  [−15.0, 0.0]; star 3 agents +4.2 [−3.3, +11.7], 6 agents +0.8 [−6.7, +8.3].
- Same direction for each model in fully_connected (−10 to −12 at 6 agents, −3 to −12 at 3); star +5/+5/+3 at 3 agents.
- Resolves F22: the gap between fully_connected and star is the structure (everyone reads everyone vs a hub).

### F25. Talking erases the gain from a bigger team
- Without links, 6 agents beat 3: fully_connected 6 − 3 = +5.8 [+2.5, +10.0], star 6 − 3 = +4.2 [+0.8, +8.3] (more
  independent attempts).
- With links, size makes no difference: fully_connected 6 − 3 = +2.5 [−3.3, +8.3], star 6 − 3 = +0.8 [−5.8, +7.5].
- In fully_connected the extra agents' attempts are lost to herding (F26); in star the hub reads 5 spokes instead of 2
  but writes as few messages (2.5 per run), and copy events double (0.6 → 1.35).

### F26. Herding comes through both channels, through messages more
- fully_connected (6 agents) base − no links: sharing only results −4.2 [−10.0, +1.7], only messages −6.7 [−13.3,
  −0.8], both −10.8 [−17.5, −5.0]: roughly additive.
- Different mechanisms: shared results are copied (2.6 copy events per run, the most of any protocol) but agents repeat
  less (repeat rate 0.27); messages make agents retry the same suggestions (repeat rate 0.37, the highest; adoption
  0.47); with both, copying 2.3, repeats 0.33, blind adoption 0.22.
- Sharing only results hurts only Llama (−12; Qwen and Gemma 0); messages hurt all three (−5 to −10).
- Star avoids both: spokes send results to the hub only, and the hub's few messages are rarely followed blindly (blind
  adoption 0.04–0.05 vs 0.16–0.22 in fully_connected).

### F27. A temperature sweep centred on the base temperature changes nothing
- (0.7, 1.0, 1.3) − all at 1.0, no links: fully_connected 6 −0.8 [−4.2, +2.5], 3 −2.5 [−5.8, +0.8], star 3 0.0
  [−4.2, +5.0], star 6 0.0 [−3.3, +3.3].
- With F21 (a sweep averaging 0.85 hurt by 3.7): varying the temperature adds no useful variety; lowering it costs a
  little. Supersedes F6's "baseline to beat".

### F28. Showing the token-matched agent its budget changes nothing; F19 holds
- Paired with the main experiment (same targets), the agent told its budget instead of 240 rounds: Gemma (bfloat16 in
  both) −2.5 [−15.0, +7.5] (fully_connected budget) and −2.5 [−12.5, +7.5] (star); Qwen and Llama 0.0 and −1.3; steps
  per run about the same (Gemma 13.8 → 14.6). Closes F15.
- F19 again: token-matched − team with talk +6.7 [−0.8, +14.2] (fully_connected 6), −13.3 [−20.8, −5.8] (star 3);
  − team without talk −4.2 [−9.2, +0.8] and −9.2 [−15.8, −2.5].

### F29. Cost per solved run: the 3-agent star is the cheapest team
- Tokens per solved run (team with talk / without): star 3 9.2k / 9.2k, fully_connected 3 12.0k / 8.2k, star 6
  17.8k / 16.4k, fully_connected 6 26.9k / 14.8k; fully_connected 6 sharing only results 17.8k, only messages 23.5k.
  One agent: about 3.3k (F20).
- Talk is nearly free in star (+7% tokens per run) and doubles the cost of a solve in fully_connected 6.

---

## Calibration against random agents (2026-11-03, no GPU)

Simulated random agents on the main grid's 40 targets (2 settings × 20 replicates; the follow-up uses the same ones),
2,000 games per target, with the real recipe book and environment rules: each round an agent picks a pair of its
inventory uniformly, either never repeating its own pairs ("no-repeat") or with replacement. Compared with the LLM
runs of the main experiment (all 4 models, 80 runs per cell, shared runs counted once). Script:
`run_main/random_baseline.py` (local, git-ignored).

### F30. Random teams match the LLM teams at depth 1 and beat every LLM condition at depth 2
- Depth 1, 3 distractors (5 items, 15 pairs): one random agent 26% (no-repeat) / 24% (with replacement) vs one LLM
  agent 65%; 6 random agents 82% / 80% vs 6 independent LLM agents 82%, with talk 74% (fully_connected); 3 random
  agents 60% vs the 3-agent star 81% (with or without talk).
- Depth 2, 0 distractors (about 3 items, 6 pairs): one random agent 24% / 19% vs one LLM 30%; 6 random agents 63% /
  55% vs 6 independent LLM agents 41%, with talk 31%; 6 random agents that skip pairs any teammate found invalid 82%.
  One random agent with 17 rounds (what the token-matched agent plays, F19) 79% vs the token-matched LLM 42%.
- Why: the LLMs' knowledge picks a good first pair (depth 1, one agent: 65% vs 26%) but they explore badly: 41% of
  their depth-2 rounds repeat a pair they already tried (20% at depth 1), against the rules' instruction, and agents
  of one model try the same pairs (F6). The outcome is mostly fixed by the target: the 6-agent and 3-agent
  independent teams have the same outcome on 148 of 160 targets.
- Consequences: "teams help through more attempts" (F5, F19, F25) is really "extra agents add the variety a single
  model lacks", and random agents add more. The search spaces are so small that blind enumeration is a strong
  baseline: every result so far must be read against it. Talk can at best recover what herding loses (the random
  team with shared failures shows the head-room: +9 at depth 1, +19 at depth 2). And since all agents see the same
  inventory and target, there is nothing for talk to carry except who tries what. **Acted on**: random agents in the
  framework (`framework/agents/backends/mock.py`, models `mock` and `mock_norepeat`), confirmed on the cluster (F31); a
  task where agents hold different information (`hints`, F33–F36). Part of the depth-2 success comes from targets
  with a shortcut (F32), but random teams still win without them (F31).

---

## Random agents on the cluster: `runs/random_baseline` (2026-11-04, job 74155)

Grid `configs/experiments/random_baseline.yaml`: random agents `mock` (picks two items with replacement) and
`mock_norepeat` (never repeats its own pairs, and skips pairs a teammate reported invalid when it sees them) × 5
protocols (fully_connected, star, fully_connected3, star6, fully_connected_results) × the main grid's 2 settings × 200
replicates (0–19 are the main targets) × base, all_independent, one_agent, one_agent_scaled_steps. 16,000 rows, 9,600
distinct, 0 failed, code `0f59dfd` clean, rtx2080. A random agent plays each target once (its seeds are fixed by the
replicate), so per-target comparisons are noisy. Source: `run_random_baseline/random_baseline/runs.csv` (complete; only
1,092 of the 9,600 run folders were copied).

### F31. The real pipeline confirms F30: random teams match the LLM teams at depth 1 and beat them at depth 2
- 200 targets, with replacement / no-repeat: depth 1, 6 agents 85.0% / 83.5%, 3 agents 57.5 / 53.5, one agent 25.0 /
  25.0; depth 2, 6 agents 67.0 / 64.0, 3 agents 49.5 / 47.5, one agent 25.5 / 27.5. The simulation of F30 gave 82 and
  63 for 6 agents.
- Check: `mock` ignores what it receives, and its base equals its all_independent exactly in every protocol, so the
  pipeline adds nothing of its own.
- Sharing helps random teams: `mock_norepeat` in fully_connected (it skips teammates' failures) vs no links: depth 1
  86.0 vs 83.5, depth 2 **80.5 vs 64.0**; star 63.0 vs 53.5 (depth 1), 46.5 vs 47.5 (depth 2).
- Paired with the LLMs on the main targets (4 models × 20 targets, bootstrap by target), LLM − random: depth 1, 6
  independent agents +2.5 [−16.2, +23.8], 3-agent star +37.5 [+11.2, +61.3], one agent +55.0 [+33.8, +72.5]; depth 2,
  6 independent agents −33.8 [−57.5, −8.8], 6 talking agents −43.8 [−65.0, −21.2], 3-agent star −18.8 [−41.2, +3.8],
  one agent −5.0 [−25.0, +15.0].
- Without the shortcut targets of F32, random teams of 6 still solve 53–56% at depth 2, the LLM teams 31%.
- So the models' knowledge pays off for one agent and for small teams at depth 1; once a team has 6 agents, or the
  task needs two steps, blind trial and error does as well or better.

### F32. A quarter of the depth-2 targets can be made in one step from the starting items
- `sample_targeted` (`framework/tasks/wordcraft/tasks.py`) picks one recipe of the target (and, at depth 2, one of an
  ingredient) but does not check the target's other recipes against the starting items. Over 200 targets: depth 2 with
  0 distractors 51 have a direct recipe, depth 2 with 6 distractors 54; depth 1 with 3 distractors 13 have a second
  direct recipe.
- In the targets used: main depth 2 3 of 20 (replicate 0 'tank' = 'car' + 'metal', 11 'solar system' = 'planet' +
  'planet', 16 'pollen' = 'plant' + 'wind'), hints 4 of 20 (the same three and 10 'spaceship' = 'airplane' + 'space').
  In the hints run of replicate 0, five of six agents made 'tank' from 'car' and 'metal' in round one.
- They carry much of the depth-2 success. Main, 4 models, shortcut targets vs the rest: one agent 91.7% vs 19.1%, 6
  independent agents 100% vs 30.9%, 6 talking agents 75.0% vs 23.5%. Hints without notes: 58–67% vs 0–10%.
- The paired results hold without them: F16 (fully_connected, talk − no links) −6.6 [−11.0, −2.2] on 136 pairs (−9.4
  with them); F18 (star) +1.5 [−4.4, +8.1].
- Consequences: depth-2 success rates overstate how often agents solve two-step tasks; the paired differences stand.
  **Acted on**: `task.allow_shortcuts: false` draws again any task where the target or the intermediate has a recipe
  made only of starting items besides the path's own (`other_route` in `tasks.py`; at depth 1 this also rejects a
  second direct recipe); 0 of 200 drawn tasks have one. The default stays true, so the earlier grids keep their
  targets (checked: 60 of 60 stored tasks reproduced). Used by `hints2`. Earlier depth-2 results should be reported
  without shortcut targets too.

---

## Hints: agents that hold different information, `runs/hints` (2026-11-04 → 2026-11-05)

Grid `configs/experiments/hints.yaml`: qwen3-4b, gemma3-4b, llama3.2-3b and random agents (`mock_norepeat`) × 5
protocols (fully_connected, star, fully_connected sharing only results / only messages / only failures) × depth 2 with
6 distractors (9 items, 45 pairs) × notes split / all / none × 20 replicates × base, all_independent, one_agent. Each
agent has 2 private notes, true recipes shown only to it. With notes split, the two recipes of the target's path go to
two different agents (in star, 2 of 3) and every other note is a recipe off the path (a decoy); with all, every agent
holds both; with none, nobody has notes. A single agent always gets both recipes. 3,600 rows, 1,920 distinct, all
ended on their own, code `0f59dfd` clean. Gates 74158 (rtx2080) and 74161 (L40); full jobs 74162 (Qwen, Llama, random:
rtx2080, float16) and 74163 (Gemma: L40, bfloat16). Every setting, protocol and model of a replicate has the same
target, so comparisons are paired across all of them: 60 pairs per protocol (3 models × 20). Intervals are 95%
bootstraps resampled by target. 4 of the 20 targets have a shortcut (F32); numbers exclude them (48 pairs) unless they
say "all 20". Sources: `run_hints/hints/runs.csv` and the runs' `events.jsonl`.

### F33. Talk does not pool what only some agents know
- Success without shortcut targets (LLMs pooled, 48 runs per cell):
  - no notes: one agent 2%, 6 independent agents 10%, teams with talk 0–8%; random teams 0%;
  - notes split: 6 independent agents 23%, 3 independent agents 23%; with talk fully_connected 27%, results only 29%,
    messages only 17%, failures only 25%, star 29%; **one agent holding both recipes 60%**;
  - every agent holds both: independent teams 90%, with talk 81–88%; one agent 62%.
- Notes split, talk − no links: fully_connected +4.2 [−6.2, +14.6], results only +6.2 [−4.2, +16.7], messages only
  −6.2 [−14.6, +2.1], failures only +2.1 [0.0, +6.2], star +6.2 [−2.1, +14.6]. All 20 targets, the 5 protocols
  pooled: −1.0 [−9.0, +5.3].
- The information is in the team and unused: every agent holding both − notes split (no links) +66.7 [+50.0, +81.2];
  notes split − none +12.5 [0.0, +25.0]. One agent holding both − 6 talking agents with the notes split +33.3 [+14.6,
  +50.0] (fully_connected), +31.2 [+14.6, +50.0] (star).
- Checks: one agent with notes all − with notes split (the same information) +1.7 [−5.0, +10.0]; 6 independent agents
  holding both − one agent holding both +23.3 [+13.3, +33.3] (all 20).
- So when the two halves of the recipe sit with two agents a team solves about a quarter of the tasks, when one agent
  holds both 60%, and talk recovers almost none of the gap. This is the study's first test where talk had something
  to carry, and it carried almost nothing.

### F34. Why: holders seldom pass their notes on or even use them, and the others seldom act on them
- From the logs (notes split, with talk, without shortcut targets, 48 runs per protocol):
  - holders of the intermediate's recipe try it in 44–54% of runs (at once in 23–31%);
  - holders name their path note in a message rarely: the intermediate's recipe in 6–11% of their messages (in 12–26%
    of runs at least once), the target's recipe in 9–32% (23–50% of runs; most in star, where the hub hears it);
  - the others try a recipe that only a teammate holds in 3–18% (intermediate) and 2–14% (target) of cases, against
    4–5% and 2–5% without links (guessing); star is highest (18% and 14%);
  - the target is mostly made by the holder of the target's recipe once it found the intermediate itself (it made the
    intermediate in 17–33% of runs with links, 15–19% without).
- Decoys distract: 51–57% of agents try one of their off-path notes and 27–40% of all rounds go to them, although the
  target is shown.
- Even one agent holding both recipes never tries the intermediate's recipe in 11 of 48 runs, and in 10 more never
  tries the target's.
- Notes bring premature attempts: with every agent holding both (qwen3-4b and gemma3-4b, no links), 275 of 276 refused
  answers try the target's recipe before the intermediate exists; parse failure rate 0.07 → 0.18 (Qwen) and 0.02 →
  0.18 (Gemma) from no notes to all.
- Consequence: it fails at both ends. Holders do not tell (the message slot asks for "what you want to tell the other
  players"; nothing mentions notes) and do not reason "my note makes X, and the target needs X"; receivers rarely act
  on what they hear. **Acted on**: `hints2` (notes without decoys) and `hints2_share` (holders told to tell the others
  what their notes say, `task.share_notes`) separate "do not think of telling" from "cannot use what is told".

### F35. Sharing failures does not help LLM agents: they retry pairs a teammate reported failing
- fully_connected sharing only failures − no links: notes split +2.1 [0.0, +6.2], none −4.2 [−10.4, 0.0] (48 pairs),
  all −3.3 [−8.3, 0.0] (all 20). Random agents gain 16.5 points from the same information (F31, depth 2).
- With a teammate's failure in the inbox, 20% of LLM actions repeat a pair a teammate reported failing (blind adoption
  0.198, all hint settings); repeat rate 0.36 vs 0.35 without links, redundancy 0.62 in both.
- So the head-room of "avoid what failed" (F30, F31) is not reached: the models do not use the failure list. Closes the
  last herding remedy proposed after F26.

### F36. On the 9-item task, talk without notes still hurts, and with full notes it does not help
- All 20 targets, talk − no links: no notes fully_connected −10.0 [−20.0, −1.7], results / messages / failures only
  −3.3 each, star −5.0 [−10.0, 0.0]; everyone holds both: fully_connected −5.0 [−15.0, +5.0], results only −5.0
  [−11.7, +1.7], messages only 0.0, failures only −3.3 [−8.3, 0.0], star −5.0 [−13.3, +3.3].
- Herding as in F17: fully_connected with talk blind adoption 0.16, 2.4 copy events per run; tokens per run 20.1k vs
  13.6k without links; star the cheapest team, 8.9k.
- No setting in this grid has talk significantly helping. With 20 targets the intervals are ±10–15 points: effects
  smaller than that cannot be seen.

---

## Hints 2: why the notes are not pooled, `runs/hints2` and `runs/hints2_share` (2026-11-05)

Grid `configs/experiments/hints2.yaml`: qwen3-4b, gemma3-4b, llama3.2-3b and random agents × fully_connected and star ×
depth 2 with 6 distractors × notes split with decoys, split without decoys (only the two holders have notes, one recipe
each) and all × 40 targets drawn without shortcuts (`allow_shortcuts: false`; checked: 0 of 40 have another route) ×
base, all_independent, one_agent. 2,880 rows, 2,400 distinct. `configs/experiments/hints2_share.yaml`: the split fully
connected teams again (base only), each holder told "The other players do not know your notes: use your message to
tell them what your notes say." (`task.share_notes`); 240 runs with the same tasks and notes as hints2 (checked: 40 of
40). All ended on their own, 0 failed. Gates 74189 (rtx2080) and 74192 (L40); full jobs 74193 (Qwen, Llama, random:
rtx2080, float16), 74194 (Gemma: L40, bfloat16), 74195 and 74196 (share). Code `07adf9f`, recorded as dirty (F41).
Comparisons are paired by target: 120 pairs (3 models × 40), 95% bootstraps resampled by target. Sources:
`run_hints2/hints2/runs.csv`, `run_hints2_share/hints2_share/runs.csv` and the runs' `events.jsonl`.

### F37. On 40 targets without shortcuts, talk still does not pool split notes (F33 confirmed)
- Success, LLMs pooled (120 runs each), notes split with decoys: 6 silenced agents 21.7%, 3 silenced 21.7%, fully
  connected with talk 21.7%, star with talk 25.0%; one agent holding both recipes 70.0%; every agent holding both:
  silenced 90.8% (fully_connected) and 88.3% (star). Random teams 0–2.5% in every setting.
- Talk − silenced, notes split: fully_connected 0.0 [−7.5, +7.5], star +3.3 [−4.2, +10.8]; without decoys −0.8
  [−10.8, +9.2] and 0.0 [−9.2, +9.2].
- Every agent holding both − notes split (silenced): +69.2 [+60.0, +77.5] (fully_connected), +66.7 [+58.3, +75.0]
  (star). One agent holding both − talking team with the notes split: +48.3 [+36.7, +60.0] (fully_connected), +45.0
  [+34.2, +55.8] (star).
- Checks: one agent with notes all vs split (the same information) −0.8 [−2.5, 0.0]; with and without decoys its runs
  are identical (a single agent gets both recipes and no decoys either way).

### F38. Told to share, holders do tell; their teammates still rarely use it
- The instruction raises success: share − talk without it +15.0 [+9.2, +21.7] with decoys (21.7% → 36.7%; Qwen +22,
  Gemma +12, Llama +10) and +13.3 [+4.2, +22.5] without (31.7% → 45.0%); share − silenced +15.0 [+7.5, +22.5] and
  +12.5 [+2.5, +22.5]. It closes about a fifth of the gap: share − every agent holding both −54.2 [−63.3, −45.0]
  (decoys) and −45.8 [−55.8, −35.8]; share − one agent holding both −33.3 [−44.2, −21.7] and −25.0 [−35.8, −14.2].
- From the logs (fully_connected, notes split; talk without the instruction → with it; with decoys / without):
  - the holder of the intermediate's recipe names it in its first message 13% → 63% / 17% → 82% (at least once 24% →
    85% / 28% → 90%); the holder of the target's recipe 13% → 49% / 23% → 65% (at least once 28% → 65% / 38% → 78%);
  - a teammate tries the intermediate's recipe 20% → 20% / 28% → 39% (silenced: 9% / 17%), the target's recipe 7% →
    10% / 10% → 14% (silenced: 4% / 7%);
  - without decoys, a receiver with the intermediate's recipe in its inbox in round 1 plays it in 72 of 457 answers
    (16%);
  - agents that make the target per run 0.57 → 0.68 / 0.73 → 0.97 (of 6); refused answers that try the target's
    recipe before the intermediate exists 3.0 → 5.1 / 2.6 → 3.0 per run.
- In the examples, receivers either ignore a teammate's true note or follow other teammates' unfounded claims ("I've
  discovered the key to little alchemy by combining the alchemist and philosophy"). The rate of such claims was not
  measured (a keyword count cannot tell them from note sharing).
- Consequence: of the two failures of F34, "do not think of telling" is fixed by one sentence; "do not use what is
  told" is not. These models do not act on true information from teammates. **Closed** by F42 (an instruction to the
  receivers changes nothing) and F43 (qwen3-8b does act on it).

### F39. Decoy notes cost about 10 points
- Without decoys − with: 6 silenced agents +10.8 [+2.5, +19.2], 3 silenced +12.5 [+5.0, +20.0], fully_connected with
  talk +10.0 [+0.8, +18.3], star with talk +9.2 [+0.8, +18.3], with the share instruction +8.3 [−1.7, +18.3].
- With the share instruction, holders pass on their decoys too: 57% of messages name a decoy note, so receivers get
  true but useless recipes beside the useful ones (F34: decoys took 27–40% of rounds).

### F40. When every agent holds both recipes, talk hurts; independent attempts still help
- Talk − silenced, every agent holding both: fully_connected −8.3 [−14.2, −2.5], star −6.7 [−12.5, −1.7], each model
  −5 to −10. F36 showed the same direction (−5) on 20 targets; with 120 pairs it is clear.
- 6 silenced agents holding both − one agent holding both +21.7 [+15.0, +28.3]: one agent with the whole recipe still
  fails 30% of the time (it does not follow its notes, F34), and more independent attempts recover most of that.

### F41. The hints2 runs record a dirty checkout
- Every hints2 and hints2_share run records commit `07adf9f` with `dirty: true` (the hints runs on `0f59dfd` were
  clean). The commit holds all the code these grids need. The likeliest cause is the untracked `slurm_hints.txt` and
  `slurm_random_baseline.txt` lists written into the repo root on giano before these jobs started; provenance does not
  record which files differed.
- **Acted on**: `.gitignore` ignores `/slurm_*.txt`. **Closed**: the next runs (hints2_use, hints2_8b,
  hints2_8b_share, on `d01c77a`) all record a clean checkout.

---

## The receiving side and a bigger model: `runs/hints2_use`, `runs/hints2_8b`, `runs/hints2_8b_share` (2026-11-06)

`configs/experiments/hints2_use.yaml`: hints2_share's teams (fully_connected, notes split with / without decoys, base)
with `share_notes` and `use_notes` (every agent is also told "Recipes the other players tell you from their notes are
true: when one helps make the target, use it."), qwen3-4b, gemma3-4b, llama3.2-3b; 240 runs.
`configs/experiments/hints2_8b.yaml`: hints2's cells with qwen3-8b; 720 rows, 600 distinct.
`configs/experiments/hints2_8b_share.yaml`: qwen3-8b, the split fully_connected teams with `share_notes`, `use_notes`
false / true; 160 runs. All on hints2's 40 targets and notes, so everything pairs with hints2 and hints2_share. All
1,000 distinct runs ended on their own, 0 failed, code `d01c77a` clean (999 runs `dirty: false`, one `null`: its
`git status` call failed). hints2_use: jobs 74232, 74234–74236 (Qwen, Llama: rtx2080, float16; Gemma: L40, bfloat16);
qwen3-8b: 74237–74240 (L40, bfloat16). Intervals are 95% bootstraps resampled by target: 120 pairs for the three 4B
models pooled, 40 for qwen3-8b (about ±15–20 points). Sources: `run_hints2_use/`, `run_hints2_8b/`,
`run_hints2_8b_share/` (`runs.csv`, `events.jsonl`); scripts `docs/analysis/k3.py` (rates, contrasts) and `k4.py`
(logs; it reproduces F38's numbers for hints2 and hints2_share).

### F42. Telling receivers to use their teammates' notes changes nothing
- Success, three 4B models pooled, fully_connected, notes split (with decoys / without): talk 21.7 / 31.7%, told to
  share 36.7 / 45.0%, told to share and to use 31.7 / 40.8%. Use − share only: −5.0 [−12.5, +3.3] / −4.2 [−10.8,
  +3.3]; per model −5 to −12 except qwen3-4b without decoys (+8). Use − talk with no instruction +10.0 [+0.8, +19.2] /
  +9.2 [0.0, +19.2] (that is the share instruction's gain, F38). Still far below the same information in one head: −38.3
  [−48.3, −28.3] / −29.2 vs one agent holding both, −59.2 / −50.0 vs every agent holding both.
- From the logs (share → share and use, with decoys / without): a receiver that has just been told the intermediate's
  recipe by its holder plays it at its next action 9% → 10% / 16% → 14%; the target's recipe (when it can) 35% → 35% /
  23% → 17%; non-holders that ever try the intermediate's recipe 20% → 21% / 39% → 36%. Holders tell a little less
  (intermediate's recipe in the first message 63% → 57% / 82% → 79%).
- qwen3-8b: use − share −5.0 [−20.0, +10.0] with and without decoys.
- So the receiving failure of F38 is not a matter of trust: told outright that the recipes are true and to use them,
  the 4B models still play a teammate's recipe about one time in ten. Closes the open part of F38 for these models.

### F43. Qwen3-8B pools split notes through talk: the first setting where talk helps
- qwen3-8b, notes split, talk − silenced: fully_connected +27.5 [+12.5, +45.0] with decoys (20.0% → 47.5%), +15.0
  [0.0, +32.5] without; star +25.0 [+7.5, +42.5] and +27.5 [+10.0, +45.0]. For the 4B models these were 0.0 to +3.3
  (F37).
- With the share instruction it solves as often as one agent holding both recipes: 72.5% (decoys) and 65.0% vs 67.5%;
  share − talk +25.0 [+12.5, +37.5] and +27.5 [+12.5, +42.5]; share and use − one agent holding both 0.0 [−17.5,
  +17.5] and −7.5. Still below every agent holding both (87.5% silenced): −20.0 [−35.0, −2.5] and −27.5.
- Same targets, qwen3-8b − qwen3-4b: one agent holding both −7.5 [−25.0, +10.0], silenced team −2.5 [−17.5, +12.5],
  but talk +17.5 [+2.5, +32.5], talk with the share instruction +20.0 [0.0, +40.0] and +22.5 [+5.0, +40.0]. The bigger
  model does not play better alone; it gains only through communication.
- Why, from the logs (fully_connected, decoys / no decoys; the 4B models in brackets): without any instruction its
  holders name the target's recipe in their first message 57% / 60% [13% / 23%], the intermediate's 25% / 38% [13% /
  17%]; a receiver just told the intermediate's recipe plays it at its next action 40% / 51% [15% / 21%], the target's
  53% / 28% [37% / 20%]; non-holders try the target's recipe in 24% / 30% of cases [7% / 10%]; agents that make the
  target 1.65 / 1.90 of 6 [0.57 / 0.73]. Told to share, its holders always tell (100% of first messages) and 2.5–2.7
  agents per run make the target.
- Decoys: without − with, qwen3-8b, between −10.0 and +7.5, no interval excludes 0 (the 4B models lose ~10, F39).
- Caveats: one model and 40 targets; qwen3-8b ran in bfloat16 on L40 and qwen3-4b in float16 on rtx2080 (outcomes do
  not change with the precision, F23). Whether it is size or this model: **open**, needs a second larger model.

### F44. Agents remake an intermediate they already hold, and fully connected talk locks the team into it
- qwen3-8b, every agent holding both recipes: fully_connected talk − silenced −25.0 [−40.0, −10.0] (87.5% → 62.5%);
  star +10.0 [−2.5, +22.5] (95.0%, the best cell of the study). Repeat rate 0.34 → 0.51, copy events 0 → 2.2.
- In all 11 targets that talk lost, the team's 264 actions are all the intermediate's recipe, 198 of them repeats:
  every agent makes it in round 1, then remakes it for three rounds while the messages say "Let's make explosion
  first!". The prompt lists the intermediate in the agent's inventory (checked, rep 2 round 2); the answer still
  reasons "we need to create 'explosion' first".
- Actions that remake the intermediate after the agent already made it (every agent holding both, silenced → fully
  connected talk): qwen3-8b 60% → 80%, qwen3-4b 29% → 43%, gemma3-4b 17% → 48%, llama3.2-3b 51% → 52%; star 64% → 34%
  (qwen3-8b). With notes split and talk 28–69%.
- So the agents lose track of their own progress and re-execute the first step of the plan in their notes; talk that
  restates the plan ("first make X") makes it worse, while the star's hub moves its spokes on. This explains much of F40
  (talk hurts when everyone holds the notes) and of F34 (one agent holding both fails 30%).
- **Open**: whether a prompt that marks the items an agent made ("you made 'explosion' last round", or new items
  listed apart from the starting inventory) removes the loop; it is a change to the task's observation, so a choice of
  the study.
