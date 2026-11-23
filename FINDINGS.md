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
  from temperature does not help. Variety *between families* does not help either (F49: a silenced mixed team solves
  what its members' agents predict, no more than Llama alone); this entry only shows that models whose agents differ more gain more from a team. Random agents add more
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
- Revised (2026-11-22, F64): the adoption and copy numbers have no silenced baseline (0 by construction). Against the
  inbox a silenced agent would have received, talk adds +16 points of result adoption, +0.5 copy events and no blind
  adoption (+0.8). The herding reading rests on distinct paths, divergence and repeats, which are measured in both.

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
- Revised (2026-11-22, F64): copy events and blind adoption count only what an agent received, so they differ between
  protocols partly by construction (fewer senders, fewer counted copies); compare them against a silenced baseline.

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
- Revised (2026-11-20): with real names the silenced baseline guesses the missing recipe (F52). With made-up names
  silenced teams solve 0% and talk adds +7.5 to +45 points (F61): talk pools, and the leak hid it.

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
  not change with the precision, F23). Size or this model: **closed** by F46 (telling unprompted is Qwen's; a bigger
  Gemma pools once told to share) and confirmed on 80 targets by F47.
- Revised (2026-11-20): without the instruction the gain comes through the broadcast results, not the messages
  (results only +22.5, messages only +2.5, F59); with made-up names Qwen3-4B's talk pools nearly as much (+35 vs +45,
  F61), so the size effect is in using told recipes (share +35.0), much less in unprompted talk.

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
- **Acted on**: `task.mark_made` lists the items an agent made apart from its inventory. It removes most of the loop
  for Qwen (qwen3-8b fully connected talk 62.5% → 92.5%), not for Gemma or Llama: **closed** by F48.

---

## Per-model breakdown of the pooled results (2026-11-08, no new runs)

Recomputed per model from the existing `runs.csv` files (main, followup, pilot, hints, hints2*) and `k4.py`'s log
counts, to replace pooled numbers in the report to the supervisor. 40 paired targets per model in main and followup,
95% bootstraps by target. Script: the session's scratchpad `permodel.py` (not kept; it only filters `runs.csv` by model
and uses `lib.boot`).

### F45. Several pooled results hide opposite or one-model effects
- **Team size is a Llama effect** (refines F25): silenced 6 − 3 agents, fully_connected: llama3.2-3b +15.0 [+5.0,
  +27.5], gemma3-4b +2.5, qwen3-4b 0.0 (same outcome on all 40 targets); star6 − star: +10.0, +2.5, 0.0. With talk
  llama keeps it in fully_connected (+17.5 [+7.5, +30.0]) and loses most in star (+5.0); qwen3-4b −7.5, gemma3-4b
  −2.5. So "talk erases the gain from a bigger team" holds for Qwen and Gemma, which have almost no gain to erase,
  and only in star for Llama.
- **Star talk is not neutral per model** (refines F18): main, talk − silenced: qwen3-4b +10.0 [+2.5, +20.0],
  gemma3-4b +5.0, qwen3-8b 0.0, llama3.2-3b −10.0 [−22.5, 0.0]; the pooled +1.2 averages opposite effects.
- **Herding is a Qwen trait** (refines F17): main, fully_connected with talk, blind adoption qwen3-4b 0.42, qwen3-8b
  0.45, gemma3-4b 0.15, llama3.2-3b 0.12; distinct paths silenced → talk gemma 0.72 → 0.55, qwen3-4b 0.53 → 0.40,
  qwen3-8b 0.53 → 0.45, llama 0.80 → 0.78. Yet the success loss is similar (−12.5, −12.5, −7.5, −5.0).
  Revised (2026-11-22, F64): silenced Qwen agents already repeat a teammate's failed pair as often (qwen3-8b 0.43,
  qwen3-4b 0.31): talk adds +3.7 and +10.9 (neither excludes 0). The blind adoption gap is Qwen agents agreeing
  (F55), not following; the distinct-paths drop stands.
- **Channels hurt different models** (refines F26): results only −12.5 llama, 0 qwen and gemma; messages only −10
  gemma, −5 qwen and llama.
- **The cold temperature sweep hurts only Llama** (refines F21): −12.5 [−22.5, −2.5]; others −2.5 to +2.5.
- **Against random agents** (refines F31), main, 6 silenced agents, depth 2: gemma3-4b 60%, llama3.2-3b 50%, qwen3-8b
  30%, qwen3-4b 25% (random 67%); depth 1: llama 100%, qwen3-4b 80%, gemma and qwen3-8b 75% (random 85%). One agent's
  repeat rate at depth 2: qwen 0.48–0.49, llama 0.46, gemma 0.21 (F30's 41% is their mean).
- **Telling and receiving differ by model** (refines F38, F42): share instruction, decoys, holder names the
  intermediate's recipe in its first message qwen3-4b 97%, gemma3-4b 62%, llama3.2-3b 30% (no decoys 97 / 100 / 47%);
  a receiver just told the intermediate's recipe plays it next 12% / 4% / 8% (no decoys 21 / 4 / 27%). Gemma tells
  but almost never uses what it hears. Decoys cost gemma +18 and llama +12 points silenced, qwen +2.
- Status: **closed** (descriptive). Report per-model numbers alongside pooled ones from now on; the pooled 4B
  averages are not a good summary of any single model.

## The 2026-11-07 grids: a second bigger model, 80 Qwen3-8B targets, the made line, mixed teams (2026-11-11)

Grids `hints2_12b` (600 distinct runs), `hints2_12b_share` (160), `hints2_8b` and `hints2_8b_share` extended to
targets 40–79 (+600, +160), `hints2_made` (1,600), `mixed` (160), `mixed_hints` (320), queued 2026-11-07 (gates
74395–74397, full jobs 74398–74405). All ended on their own: `unfinished.json` empty, every Slurm `.out` with
`0 failed runs`, `events.jsonl` = distinct `run_dir`, code `bd60792` clean (the first 40 Qwen3-8B targets: `d01c77a`,
one run with `dirty` unrecorded). Pairing checked: `hints2_made` has the same task and notes as hints2 / hints2_8b
in 1,600 of 1,600 runs, `mixed_hints` as hints2 in 320 of 320, `mixed` the same targets as main (its `meta.json`
only adds the `path` key). Gemma3-12B, Qwen3-8B and the mixed teams ran on L40 in bfloat16; `hints2_made` ran Qwen3-4B
and Llama on rtx2080 in float16 (F23). 95% bootstraps by target. Scripts: `docs/analysis/k5.py` (rates, contrasts),
`k6.py` (who tells, who uses, from the logs), `k7.py` (made line), `k8.py`, `k9.py` (mixed teams).

### F46. Gemma3-12B does not tell its notes unprompted, but told to share it pools them fully: F43 is partly size, partly Qwen
- gemma3-12b, notes split, talk − silenced: fully_connected +2.5 [−12.5, +20.0] with decoys (50.0% → 52.5%), +5.0
  [−5.0, +17.5] without (75.0% → 80.0%); star +5.0 and 0.0. Like the 4B models (F37), unlike qwen3-8b (F47).
- From the logs: without instruction its holders name a path recipe in their first message 0% (decoys) and 0–3% (no
  decoys) [qwen3-8b 29–64%]; 21–23 messages per run all the same.
- With the share instruction: holders name the intermediate's recipe first 97–100%, the target's 82–88%; a receiver
  just told the target's recipe plays it at its next action 92% / 73% (decoys / none; n = 83 / 113) [same targets
  and instruction: gemma3-4b 40% / 45% (n = 15 / 42), qwen3-4b 57% / 21%, llama3.2-3b 7% / 9%, qwen3-8b 69% / 30%],
  the intermediate's only 17% / 23% [gemma3-4b 4%, qwen3-8b 32–39%]. Success with decoys 52.5% →
  92.5%: share − talk **+40.0 [+25.0, +55.0]**, share − one agent holding both 0.0 [−12.5, +12.5] (87.5%); without
  decoys 80.0% → 82.5% (+2.5). The use instruction adds nothing (−5.0, +7.5).
- Better without talk too: same 40 targets, silenced split fully_connected with decoys gemma3-12b − gemma3-4b +20.0
  [+2.5, +37.5], − qwen3-8b +30.0 [+17.5, +45.0]; one agent holding both 87.5% vs 77.5% and 67.5%. With the share
  instruction − qwen3-8b +20.0 [+7.5, +35.0].
- Every agent holding both: talk − silenced fully_connected −10.0 [−20.0, −2.5] (100% → 90%), star −22.5 [−37.5,
  −10.0] (97.5% → 75.0%). It remakes the intermediate more than any model: 63% / 66% of actions (fully_connected
  silenced / talk), star 44% / 52%, one agent 61%. In the 9 star targets talk lost, the hub (which holds both) writes
  "Let's focus on getting 'explosion' next!" every round and the spokes remake it four times (rep 2).
- So F43's gain has two parts: *telling* unprompted is Qwen's (Gemma 4B and 12B do not); *using* what one is told
  is better in the bigger Gemma for the target's recipe (73–92% vs 40–45%), not for the intermediate's (17–23%),
  and its team still reaches one agent holding both. A bigger model pools once
  someone tells it to share. Caveat: one model per size and family, 40 targets.
- Status: **closes** F43's open question (size or Qwen: both, for different halves).

### F47. On 80 targets Qwen3-8B's gain from talk holds and is larger; so is its loss when everyone holds both
- Targets 40–79 alone, talk − silenced: split fully_connected +37.5 [+22.5, +52.5] (decoys), +35.0 [+17.5, +52.5]
  (none); star +15.0 [+2.5, +27.5], +15.0 [0.0, +30.0]; every agent holding both fully_connected **−52.5 [−67.5,
  −37.5]** (87.5% → 35.0%), star +17.5 [+5.0, +30.0] (80.0% → 97.5%).
- All 80: split fully_connected +32.5 [+22.5, +43.8] (22.5% → 55.0%) and +25.0 [+13.8, +36.2]; star +20.0 [+10.0,
  +31.2] and +21.2 [+10.0, +32.5]. With the share instruction 75.0% / 73.8% (share − talk +20.0 [+11.2, +28.7], +25.0
  [+15.0, +35.0]), one agent holding both 66.2%: share and use − one agent +6.2 [−6.2, +18.8], +2.5. Everyone holding
  both: fully_connected −38.8 [−50.0, −27.5] (87.5% → 48.8%), star +13.8 [+5.0, +22.5] (82.5% → 96.2%).
- Logs, 80 targets, without instruction: holders name the target's recipe first 54% / 64%, the intermediate's 29% /
  35%; told to share 94–100%.
- Status: **confirms** F43 and F44 with intervals about 30% narrower; the report's 40-target numbers stand.

### F48. Marking the items an agent made breaks the remake loop for Qwen, not for Gemma or Llama
- `task.mark_made` (an "Items you made (already in your inventory): ..." line), every agent holding both, made − not
  made, same targets: qwen3-8b fully_connected talk **+30.0 [+17.5, +45.0]** (62.5% → 92.5%; remake share 80% →
  41%), silenced +12.5 [+2.5, +25.0] (87.5% → 100%), star silenced +12.5, star talk +2.5 (95.0% → 97.5%), one agent
  +15.0 [+5.0, +27.5] (67.5% → 82.5%; remakes 57% → 33%). qwen3-4b fully_connected talk +10.0 [+2.5, +20.0] (remakes
  43% → 12%), other cells +0 to +2.5.
- gemma3-4b: remakes 48% → 37% (fully_connected talk), 29% → 37% (star talk); success unchanged (0 to +2.5).
  llama3.2-3b: remakes 51–55% → 44–49%; success −7.5 to +7.5, no interval excludes 0. The three 4B pooled: +0.8 to
  +3.3, none excludes 0.
- Talk − silenced with the line (everyone holds both): qwen3-8b fully_connected −7.5 [−15.0, 0.0] (was −25.0),
  qwen3-4b 0.0 (was −7.5), gemma3-4b −10.0 (was −10.0), llama3.2-3b −15.0 [−30.0, −2.5] (was −7.5).
- Notes split: no change (−7.5 to +5.0 for teams; one agent holding both +12.5 for qwen3-8b, +0 to +5 for the rest):
  there the problem is pooling, not the loop.
- So for Qwen the loop is an observation problem: told what they made, they move on, and most of F40's gap closes for
  qwen3-8b. Gemma and Llama keep remaking with the line in the prompt: for them it is not losing track of the
  inventory (Llama repeats any pair often, F45).
- Status: **closes** F44's open question. The line changes the task's observation; keep it off in comparisons with
  the earlier grids, and decide whether later grids use it.

### F49. Mixing model families adds no variety beyond the best family, but a mixed team herds less when it talks
- Silenced mixed teams (mix-qgl: two agents of each family in fully_connected, one each in star) solve what drawing
  independent agents from the single-model runs predicts: fully_connected depth 1 90.0% vs 97.0% expected, depth 2
  60.0% vs 58.4%; star 90.0% vs 93.3%, 50.0% vs 60.0%; notes split 27.5% vs 23.6%. And no more than the best single
  family: silenced fully_connected, both main settings, mixed − qwen3-4b +22.5 [+7.5, +37.5], − gemma3-4b +7.5
  [−5.0, +20.0], − llama3.2-3b 0.0 [−12.5, +15.0]; star +12.5, +2.5, 0.0. Variety *between* families does not beat
  Llama's variety within the family (F6's hypothesis: **not supported**).
- With talk, fully_connected (main settings pooled): mixed talk − silenced −2.5 [−12.5, +7.5] vs −12.5 (qwen3-4b,
  gemma3-4b) and −5.0 (llama); difference in differences +10 [−2, +22], +10 [−5, +28], +2: suggestive, no interval
  excludes 0. Herding measures with talk: blind adoption 0.20 (qwen3-4b 0.42, gemma 0.15, llama 0.12), distinct paths
  0.78 (0.40, 0.55, 0.78; mixed silenced 0.88), repeat rate 0.28 (0.44, 0.27, 0.32). Revised (2026-11-22, F64): with
  a silenced baseline the mixed team's blind adoption is 0.25 silenced → 0.20 talk (Qwen3-4B 0.31 → 0.42): the
  blind adoption gap is mostly how alike the agents are; distinct paths and repeats stand. A talking mixed team beats talking
  Qwen +32.5 [+17.5, +47.5] and Gemma +17.5 [+2.5, +32.5], ties Llama +2.5.
- Notes (mixed_hints vs hints2, decoys): every agent holding both, fully_connected talk − silenced 0.0 [−7.5, +7.5]
  (97.5% both; single models −7.5 to −10) although it remakes as often (35% → 43%); star −7.5. Notes split: fully
  connected +5.0 [−10.0, +20.0]; **star +17.5 [+7.5, +30.0]** (22.5% → 40.0%; single 4B models +10, 0, 0), with
  qwen3-4b always the hub. By which family holds the path recipes the cells have 8–20 runs: in fully_connected with
  talk 60% when Qwen holds the target's recipe vs 14% when Gemma does (n = 10, 14), in line with F46 (Gemma does not
  tell); too few to separate the star's gain by holder.
- Caveats: 20 targets per main setting, 40 for notes; one mix (no 8B or 12B member); the star's hub is always Qwen.
- Status: **open** for the talk effects (a larger run, or mixes with the hub rotated); **closed** for silenced teams.

---

## Review of 2026-11-11 (no new runs): corrections and new readings of the hints2 logs

From the critical review in `docs/REVIEW_2026-11-11.md`. Recomputed from `runs.csv` and `events.jsonl` of main,
followup, hints2, hints2_share, hints2_8b(_share) and hints2_12b(_share); every earlier headline number reproduced.
Fully connected, notes split, unless stated; "with / without decoys". Scripts were in the session's scratchpad (not
kept): they reuse `lib.boot` and `k4.py`'s `mentions` / holder logic, add an exact sign test on discordant pairs, and
read `llm_call` (every attempt), `msg_delivered` and `env_step`.

### F50. (review) The engine drops an agent's message when its action is refused, mostly the target-recipe holder's
- `engine.run_agent` sets `turn.message` only when an attempt parsed; `apply` returns on a no-op before publishing.
  The holder of the target's recipe often plays it before the intermediate exists ("not in your inventory"), fails
  its attempts, and the message written in those answers is never sent.
- Turns of that holder with no valid action whose discarded message named its recipe, against its published
  messages naming it. Share instruction: qwen3-4b 37 vs 89 / 24 vs 116 (29% / 17% dropped), gemma3-4b 31 vs 27 / 18
  vs 49 (53% / 27%), llama3.2-3b 4 vs 44 / 3 vs 45, qwen3-8b 35 vs 220 / 22 vs 237 (14% / 8%), gemma3-12b 1 vs 124 /
  3 vs 127. Without the instruction: qwen3-4b 16 vs 35 / 8 vs 47, qwen3-8b 13 vs 116 / 10 vs 118.
- Changes: F38's and F45's tell rates count published messages only, so they overstate what reached teammates for
  the 4B models; the loss is larger for smaller models, so it inflates the size contrast of F43 and F46.
- Status: **acted on**: `engine.publish_on_no_op`, `hints2_share_pub`; the drop costs little success (F58).

### F51. (review) The intermediate's recipe reaches teammates mostly through auto-shared results, and even those go unused by Gemma3-4B
- In fully connected teams every success is broadcast ("Other players' valid combinations"). Holder of the target's
  recipe when it first plays the intermediate's recipe, by what it had received (talk, no instruction, with decoys):
  qwen3-8b result only 14, result and message 17, message only 1, nothing 10; gemma3-12b result only 12, nothing 3
  (no message ever named it); qwen3-4b 7 / 5 / 0 / 1.
- Of target holders that received the intermediate's recipe by either channel, those that ever played it (both decoy
  settings): gemma3-4b 4 of 61 (7%), llama3.2-3b 22 of 59 (37%), qwen3-4b 22 of 48 (46%), gemma3-12b 24 of 51 (47%),
  qwen3-8b 61 of 94 (65%). Silenced, they guess it in 5–20% of runs (gemma3-12b without decoys 35%). Not corrected for recipes received in the last
  round.
- Changes: F43's "holders tell" explains the target's recipe, not the intermediate's; `k2.py`, `k4.py` and `k6.py`
  ignore the result channel. The use failure of F38/F42 holds for ground-truth results too, so it is not about
  trusting teammates. With the share instruction the solver is mostly the intermediate's holder (qwen3-4b 15 of 21,
  gemma3-12b 26 of 37 with decoys): the told target recipe is the one that gets used (as in F46).
- Status: **acted on**: `hints2_channels(_share)`; Qwen3-8B's unprompted gain is the results channel (F59).

### F52. (review) The split manipulation leaks: silenced teams solve through the intermediate's holder guessing the target's recipe
- Solver in silenced split teams (target's holder / intermediate's holder / non-holder), with decoys and without:
  gemma3-12b 3 / 14 / 3 of 20 and 7 / 20 / 3 of 30 (50% and 75% solved, silenced); gemma3-4b 1 / 9 / 2 and 2 / 14 /
  3; qwen3-4b 1 / 7 / 1 and 1 / 7 / 2; qwen3-8b 10 / 6 / 2 and 11 / 6 / 2 (of 80); llama3.2-3b 4 / 1 / 0 and 4 / 3 / 3.
- The intermediate's holder makes it from its note and finds the target's recipe without being told: the second
  note is partly in the models' knowledge of the game, more in the bigger Gemma. Random agents solve 0–2.5% (F37).
- Changes: gemma3-12b without decoys has no head-room (75% → 80% → 82.5%); in F43 and F46 size, family and recall of
  Little Alchemy 2 are confounded. Earlier numbers stand.
- Status: **acted on**: `renamed(_share)`; silenced split teams solve 0%, talk and share pool (F61).

### F53. (review) The remake loop echoes the prompt's example
- The prompt is rebuilt each turn without the agent's previous answer; the v2 example's answer is "I make 'robot'
  now and combine it with 'human' in the next round" (step 1 of a two-step plan). One qwen3-8b agent holding both
  recipes (hints2_8b, 80 targets): 92 of its 143 turns after it holds the intermediate remake it; 48 of those answers
  say "next round", 63 "first" / "need to create" (e.g. rep 35: the same combination and reasoning in rounds 2–4).
- Also seen in the prompts: the notes line says "only you were told them" when every agent holds both (false there);
  messages are listed without round numbers and results without deduplication (rep 2, round 3: the same valid
  combination ten times, round-1 messages still listed).
- Changes: F44, F47's −38.8 and F48 may be properties of this prompt. Not tested: no run varies the example.
- Status: **mostly rejected** by `hints2_ex2` (F60): the loop barely moves with the example; stamped / deduplicated
  lists (C5) not tested.

### F54. (review) Statistical notes on F16–F49
- Followup is not an independent replication of main: same 40 targets, 566 of 600 shared fully connected cells with
  the same outcome. F16 pooled is 1 vs 16 discordant pairs (exact p = 0.0003) on 16 distinct targets; per model 1/6,
  0/2, 0/5, 0/3 (p = 0.13, 0.50, 0.06, 0.25).
- Percentile bootstrap intervals that exclude 0 with few discordant pairs fail an exact sign test: qwen3-4b star
  +10.0 [+2.5, +20.0] is 4/0 (p = 0.13, F18/F45); llama share +10.0 [+2.5, +20.0] 4/0 (F38); gemma3-12b everyone
  holds both, fully connected −10.0 [−20.0, −2.5] 0/4 (F46). F40 per model: 2/5, 2/6, 0/3. The large effects hold:
  qwen3-8b +32.5 is 27/1, −38.8 is 1/32, star +13.8 is 13/2 (p = 0.007); gemma3-12b share +40.0 is 16/0, star −22.5
  is 0/9 (p = 0.004); qwen3-4b share +22.5 is 9/0.
- Silenced fully_connected3 is the first three agents of silenced fully_connected (same seeds and prompts: 0 targets
  where 3 solve and 6 do not, 7 the other way), so F25's "6 − 3" is ≥ 0 by construction.
- Star: only the hub has a message slot (1 writer per run), spokes cannot tell their notes; the hub holds a path
  recipe in 32 of 40 targets (53 of 80 for qwen3-8b). Star and fully connected also differ in personas and schedule.
- Share − everyone holding both, silenced (the team's ceiling): qwen3-8b −12.5 [−23.8, −1.2] / −13.8 [−26.2, −1.2];
  gemma3-12b −7.5 / −17.5 [−30.0, −7.5].
- Status: **closed** (descriptive). Report exact tests beside the bootstrap for per-model contrasts; treat F45, F49's
  talk effects and the per-model rows of F39/F40 as exploratory.

---

## Confidence and team size (2026-11-12, no new runs; the tutor's hypothesis)

The tutor's idea: a model that gives the right answer one time in three gains from a bigger team, because one of
six agents finds it; a model that is right or wrong "with confidence" gains little, whether it has 3, 6 or 12
agents. The six agents of a silenced team all get the same first-round prompt, so their six first answers sample
the model's answer distribution, with no confidence logging needed. Source: fully_connected all_independent of
followup (qwen3-4b, gemma3-4b, llama3.2-3b) and main (qwen3-8b), 20 targets per setting. Script:
`docs/analysis/k10.py` (part 1).

### F55. Agents of Qwen and Gemma answer alike (right or wrong together); Llama's agents disagree, which is why only Llama gains from a bigger team
- Depth 1, 3 distractors, first answers of the six agents: share of agents playing the most common pair qwen3-4b
  0.86, gemma3-4b 0.90, qwen3-8b 0.91, llama3.2-3b **0.56**; distinct pairs among six 1.4, 1.4, 1.2, **3.0**. Targets
  where all six are right / all six wrong / mixed: qwen3-4b 7 / 7 / 6, gemma3-4b 2 / 14 / 4, qwen3-8b 10 / 7 / 3,
  llama3.2-3b 1 / 2 / **16** (of 19; one target has fewer than six first actions).
- So the per-agent solve rate varies within a target only for Llama. All four models solve about the same share
  per agent (0.67–0.70), but the team: llama3.2-3b **1.00**, qwen3-4b 0.80, gemma3-4b 0.75, qwen3-8b 0.75. The
  per-target rates predict for 1 / 3 / 6 / 12 independent agents: llama 0.67 / 0.90 / 0.97 / 0.99, qwen3-4b 0.69 /
  0.75 / 0.78 / 0.79, gemma3-4b 0.68 / 0.73 / 0.75 / 0.75, qwen3-8b 0.70 / 0.74 / 0.75 / 0.75.
- Depth 2, 0 distractors: modal share qwen3-4b 0.98, qwen3-8b 0.95, gemma3-4b 0.83, llama3.2-3b 0.80; team − agent
  rate +1, +5, +12, **+28** points.
- This explains F45's "team size is a Llama effect" (6 − 3 silenced agents: llama +15.0, gemma +2.5, qwen 0.0) and
  F6/F30 (Qwen agents are near-copies). It is agreement between samples, a behavioural measure of confidence; a
  model's token probabilities are the direct one (`engine.log_confidence`, `scripts/rescore.py`, 2026-11-12). The
  per-target rate comes from the same six agents, so the "predicted" column only restates the heterogeneity; the
  12-agent grid (`size12`) tests it out of sample.
- Status: **closed** by F57: 6 → 12 adds 0 to +5 points; token probability tracks agreement, not correctness.

---

## The review's and the tutor's grids: `runs/hints2_share_pub`, `hints2_channels(_share)`, `hints2_ex2`, `renamed(_share)`, `chain`, `chain_hints`, `size12`, `confidence_check`, `runs_confidence` (run 2026-11-12 →, analysed 2026-11-20)

All ten grids complete: events = distinct runs = rows of `runs.csv` (240, 160, 80, 480, 960, 160, 320, 320, 120, 4),
`unfinished.json` `[]`, every shard log `0 failed runs`. Runs record a clean checkout: `076fb11` for everything
submitted or started after the pull, `e7ff8e3` for the review grids' rtx2080 blocks (80 + 120 + 240 + 40 runs; only
`engine.log_confidence` differs, off there). Rescore (`scripts/rescore.py`, jobs 74712–74715, test 74700):
qwen3-4b 1,440, llama3.2-3b 1,440, gemma3-4b 1,440, qwen3-8b 1,560 runs of followup, main, hints2 and hints2_8b,
downloaded to `run_confidence/`. All on hints2's targets 0–39 or main's 40 targets, so every contrast is paired by
target. Scripts: `docs/analysis/k10.py` (parts 4–5 added), `k11.py`, `k12.py` (new). "x/y" after a contrast is the
exact sign test's discordant pairs (F54).

### F56. Logging confidence changes nothing that is sampled, and the rescored probabilities match the sampled ones
- `confidence_check` (gemma3-4b, fully connected, depth 1, replicates 0–1, base and all_independent): every answer
  text, environment step and message identical to main's and followup's runs (4 of 4 runs, 54 calls).
- Those same runs were rescored from followup: P(combination) sampled vs rescored r = 0.996, mean |difference| 0.013
  (largest 0.08; the answer is retokenized). The rescore can stand in for logging on the earlier grids.
- Status: **closed**.

### F57. A 12-agent team adds nothing over 6 for Qwen and Gemma; token confidence tracks agreement, not correctness
- `size12` (12 silenced agents, main's 40 targets). Agents 1–6 replay followup's six agents exactly (Qwen and Gemma
  240 of 240 action sequences, Llama 219 of 240), so 12 − 6 is what agents 7–12 add (≥ 0 almost by construction).
  12 − 6: qwen3-4b +0.0 (52.5% both, 0/0), gemma3-4b +2.5 [+0.0, +7.5] (1/0), llama3.2-3b +5.0 [+0.0, +12.5] (2/0).
  12 − 3: qwen3-4b 0.0, gemma3-4b +5.0 (2/0), llama3.2-3b **+20.0 [+7.5, +32.5]** (8/0, p = 0.008).
- F55's prediction from the first six agents' per-target rates holds out of sample: 12 agents predicted / seen,
  depth 1: qwen3-4b 0.79 / 0.80, gemma3-4b 0.75 / 0.80, llama 0.99 / 1.00; depth 2: 0.21 / 0.25, 0.59 / 0.60,
  0.50 / 0.60.
- Token probability of the first combination (12 agents' first answers, logged): right vs wrong qwen3-4b 0.78 / 0.84
  (AUROC 0.44), gemma3-4b 0.79 / 0.76 (0.53), llama3.2-3b 0.47 / 0.34 (0.64). The same on the rescored followup / main
  first answers (qwen3-8b 0.64 / 0.69). The models are as sure, or surer, when wrong: confidence does not say whether
  the pair is right.
- But per target it tracks how much the agents agree (Spearman, P vs modal share of the 12 first pairs): qwen3-4b
  +0.49, gemma3-4b +0.85, llama3.2-3b +0.85; P vs share of agents right −0.08, +0.11, +0.33.
- And the targets where the model is unsure are the ones a big team helps (12 agents − one agent, by median P):
  gemma3-4b unsure +20.0 / sure +5.0 points, llama3.2-3b +50.0 / +31.6, qwen3-4b +0.0 / +5.0 (its "unsure" half still
  has modal share 0.86). n = 18–20 targets per half: descriptive.
- Changes: confirms the tutor's hypothesis in its behavioural form (F55): a team gains in proportion to how much
  its agents disagree, and the model's own token probability measures that disagreement (low P → varied samples →
  gain from more agents), not its accuracy. More agents cannot help a model that is confidently wrong. Closes F55.
- Status: **closed**.

### F58. Publishing the messages of refused turns lets the target's recipe reach everyone, but adds little success
- `hints2_share_pub` (share instruction + `engine.publish_on_no_op`) − `hints2_share` / `hints2_8b_share` (targets
  0–39), with / without decoys: qwen3-4b +2.5 (2/1) / 0.0, gemma3-4b **+7.5 [+0.0, +17.5]** (3/0, p = 0.25) / +2.5
  (1/0), qwen3-8b +2.5 (1/0) / 0.0.
- From the logs, holder of the target's recipe names it in a published message (share of runs) and receivers told
  (of 5), with decoys: qwen3-4b 0.80 → 0.97, 3.98 → 4.88; gemma3-4b 0.45 → 0.65, 2.10 → 3.20; qwen3-8b 0.97 → 1.00,
  4.88 → 5.00. A told receiver that can play the recipe plays it next: qwen3-4b 0.55 → 0.49, gemma3-4b 0.22 → 0.33,
  qwen3-8b 0.65 → 0.64.
- Changes: F50's dropped messages were real, but the 4B models' gap is in using what is told, not in telling: F38 and
  F42 stand, and F43 / F46's size contrast is not an artefact of the drop. Share + pub stays far below everyone
  holding both, silenced: qwen3-4b −35.0 (1/15), gemma3-4b −50.0 (2/22), qwen3-8b −12.5 (3/8).
- Status: **closed** (F50 acted on).

### F59. Qwen3-8B's unprompted gain from talk comes through the shared results, not the messages
- `hints2_channels` (split, decoys, targets 0–39) against hints2_8b / hints2_12b's silenced (all_independent) and
  fully connected (both channels) runs. Success: silenced / results only / messages only / both:
  qwen3-8b 20.0 / **42.5** / 22.5 / 47.5; gemma3-12b 50.0 / 60.0 / 50.0 / 52.5.
- qwen3-8b: results only − silenced **+22.5 [+7.5, +37.5]** (10/1, p = 0.012), messages only − silenced +2.5 (6/5),
  both − results only +5.0 (5/3), both − messages only +25.0 (10/0, p = 0.002). gemma3-12b: results only +10.0
  (7/3, p = 0.34), messages only 0.0.
- How: with results only, the solver is the target's holder in 16 of 17 runs: the intermediate's holder makes it,
  the result is broadcast ("Other players' valid combinations"), the target's holder plays its note. With messages
  only, qwen3-8b holders name the target recipe in 65% of runs and reach 3.1 of 5 receivers, but a told receiver plays
  it next only 10% of the time (it lacks the intermediate); gemma3-12b holders name it in 7% of runs.
- With the share instruction, messages alone carry most of it (`hints2_channels_share`): messages only + share −
  messages only: qwen3-8b +37.5 (17/2), gemma3-12b +35.0 (15/1); against both channels + share −12.5 (3/8) and −7.5
  (2/5). The solver is then mostly the intermediate's holder (qwen3-8b 11 of 24, gemma3-12b 23 of 34).
- Changes: F43's "pools split notes through talk" is, without the instruction, pooling through the broadcast of
  results: the target's holder acts on a teammate's made intermediate. Messages pool only when agents are told to
  share (F38, F46). F51's reading confirmed and quantified.
- Status: **closed** (F51 acted on).

### F60. An example that plays step 2 helps Qwen3-8B's talking team, but the remake loop stays: it is not the example
- `hints2_ex2` (prompts v3), everyone holds both, v3 − v2 (paired by target): qwen3-8b talk **+17.5 [+5.0, +32.5]**
  (8/1, p = 0.04), silenced +7.5 (3/0), one agent +10.0 (4/0); gemma3-4b +5.0 / +2.5 / +7.5; llama3.2-3b +2.5 /
  −5.0 / −2.5; gemma3-12b +5.0 / −2.5 / **−12.5** (0/5, p = 0.06).
- qwen3-8b talk − silenced: v2 −25.0 (1/11), v3 −15.0 (1/7, p = 0.07): smaller but still a loss.
- Remake share (actions remaking an intermediate already held, k7.py) v2 → v3: qwen3-8b talk 80% → 75%, silenced
  60% → 55%, one agent 57% → 54%; gemma3-12b 66% → 62%, 63% → 72%, 61% → 72%; llama 52% → 47%; only gemma3-4b's
  silenced and single agents drop (17% → 2%, 17% → 0%), where its success was already 92.5%.
- Changes: F53's hypothesis is mostly rejected: the loop barely moves with the example, while the "Items you made"
  line halves it (F48: 80% → 41%). The agents do not see that they hold the intermediate; it is state tracking, not
  copying the example. F62 points the same way.
- Status: **closed**. `mark_made` (F48) is the fix; use it in new grids.

### F61. With made-up names the leak is gone, and talk pools split notes: Qwen most, Gemma3-4B not at all
- `renamed` / `renamed_share` (made-up names, prompts v2r, publish_on_no_op; hints2's targets and notes). Silenced
  split teams solve **0 of 40 for every model** (with real names 20–50%, F52), as random agents do (F37).
- Split, success: talk / share instruction / one agent holding both / everyone holds both, silenced:
  qwen3-4b 35.0 / 57.5 / 42.5 / 60.0; gemma3-4b 7.5 / 7.5 / 57.5 / 92.5; qwen3-8b 45.0 / **92.5** / 37.5 / 60.0;
  gemma3-12b 17.5 / **87.5** / 75.0 / 97.5.
- Talk − silenced: qwen3-4b **+35.0** (14/0), qwen3-8b **+45.0** (18/0), gemma3-12b +17.5 (7/0, p = 0.016),
  gemma3-4b +7.5 (3/0). Share − talk: qwen3-4b +22.5 (12/3), qwen3-8b +47.5 (19/0), gemma3-12b +70.0 (28/0),
  gemma3-4b 0.0. qwen3-8b's shared split team beats one agent holding both (+55.0, 22/0) and everyone holding both,
  silenced (+32.5, 16/3).
- Against real names, paired by target: talk qwen3-4b +5.0, qwen3-8b −2.5, gemma3-4b −15.0, gemma3-12b **−35.0**
  (1/15); share qwen3-4b +5.0, qwen3-8b +20.0 (10/2), gemma3-4b −27.5 (2/13), gemma3-12b −5.0. So Qwen's talk solves
  with real names were pooling (they survive renaming), and its talk gain was hidden by the silenced teams' guessing;
  Gemma3-12B's were mostly guesses (its talking team falls with its silenced one).
- Who solves: talk, the target's holder (qwen3-4b 13 of 14, qwen3-8b 17 of 18, gemma3-12b 7 of 7: it got the
  intermediate through the results, F59); share, the intermediate's holder (qwen3-8b 20 of 37, gemma3-12b 31 of 35).
  Holders name the target recipe unprompted in 38% (qwen3-4b), 57% (qwen3-8b), 3% (gemma3-12b), 0% (gemma3-4b) of
  runs; with the instruction 97–100%, except gemma3-4b 30%, whose told receivers play it 17% of the time (gemma3-12b
  79%, qwen3-8b 62%).
- Size within a family, renamed: qwen3-8b − qwen3-4b talk +10.0 (11/7, p = 0.48), share **+35.0** (15/1);
  gemma3-12b − gemma3-4b talk +10.0 (7/3), share **+80.0** (32/0). With real names the Qwen talk gap was +17.5.
- Changes: the cleanest result of the project: when the knowledge cannot be guessed, communication is the only way to
  solve, and it works: unprompted through results and Qwen's messages, fully once agents are told to share, given a
  big enough model. F33's "talk does not pool" was the 4B models with real names, where guessing hid it; F43's
  size effect holds for using told recipes (share), much less for unprompted talk (Qwen 4B pools too).
- Status: **open**: renamed with `mark_made` (F62), and gemma3-4b's failure to tell under the instruction. Being
  tested (2026-11-22): `renamed_made` / `renamed_made_share` add the made line and one agent holding both notes with
  the team's token budget (is qwen3-8b's 92.5% vs 37.5% more than compute?); `scripts/counterfactual.py` scores
  whether a told recipe moves the receiver (gemma3-4b's use failure).

### F62. Without real names Qwen's remake loop gets much worse: it does not recognise the item it made
- One agent holding both recipes, split notes, renamed − real names: qwen3-4b −32.5 (5/18), qwen3-8b −30.0 (4/16),
  gemma3-4b −20.0, gemma3-12b −12.5. Everyone holds both, silenced: qwen3-4b −30.0 (2/14), qwen3-8b −27.5 (4/15);
  talking qwen3-8b **−47.5** (2/21), 15.0% (talk − silenced −45.0, 0/18).
- What the single agent plays (share of its actions, renamed / real names): qwen3-8b remakes the intermediate 59% /
  32%, plays the target recipe 11% / 25%; qwen3-4b 47% / 12%, 14% / 30%. Its first action is the intermediate's
  recipe in 39 of 40 runs: it starts right and then cannot move on. Everyone-holds-both talk, qwen3-8b: 96% of
  actions after holding the intermediate remake it.
- In k7.py's measure (actions after the agent holds the intermediate), one agent, real → made-up names: qwen3-8b
  56% → 85%, qwen3-4b 29% → 77%, gemma3-12b 61% → 57%, gemma3-4b 16% → 19%. The loop worsens for Qwen only; Gemma's
  one-agent loss with renaming comes from elsewhere.
- Changes: with "small" in the inventory a Qwen agent can see that "alchemist + small" is now playable; "zonut"
  carries no such cue. The loop is a failure to match the made item to the next note (F44, F60), which names partly hide. The
  renamed grids' everyone-holds-both and one-agent cells are a lower bound on what the notes allow.
- Status: **open**. `mark_made` on the renamed grid would test it directly: `renamed_made`, running since 2026-11-22.

### F63. A relay chain beats the parallel fully connected team only by talking less; sequential turns do not help
- `chain` (main's 40 targets, base) against followup / main's fully connected parallel team and its silenced twin.
  chain6 (each agent hears only the one before it) − parallel: gemma3-4b +15.0 (6/0, p = 0.03), qwen3-4b +12.5 (5/0),
  qwen3-8b +10.0 (4/0), llama +7.5 (5/2); pooled **+11.2 [+5.0, +18.1]** (20/2, p = 0.0001). chain6 − silenced:
  pooled +1.2 (8/6): the chain only gives back what fully connected talk costs (F16).
- It herds less (copy events per run 1.0 vs 2.2 gemma3-4b, 0.3 vs 1.3 qwen3-4b; blind adoption 0.27 vs 0.40 qwen3-4b)
  and spends 20–30% fewer tokens. **Revised (2026-11-22, F64)**: the logged copies count only items the one agent a
  chain agent hears found first; against every teammate, chain6 copies as much as parallel talk or more (gemma3-4b
  2.77 vs 2.25, llama 2.50 vs 3.48, qwen3-4b 0.93 vs 1.30, qwen3-8b 0.60 vs 0.82; silenced 1.40, 2.33, 0.93, 0.70).
  "Herds less" is not shown by these measures; the success and token results stand.
- Sequential (fully connected, agent k sees agents 0..k−1 of the same round) − parallel: pooled +2.5 (11/7); −
  silenced: pooled **−7.5 [−13.8, −1.9]** (4/16, p = 0.012). Qwen's agents already play the same first pair in parallel
  (agents 1–5 playing agent_00's: 0.88–0.89 parallel, 0.93–0.95 sequential); Gemma's drop (0.75 → 0.47); solve rate by
  turn position is flat (later agents do not solve more).
- Hints2 cells, sequential − parallel: split pooled −0.6 (17/18; llama +15.0, 6/0; qwen3-8b −17.5, 3/10, p = 0.09);
  everyone holds both pooled −4.4, qwen3-8b **−22.5** (0/9) and − silenced −47.5 (0/19): seeing this round's
  "first make X" amplifies the remake lock (F44; repeat rate 0.64 vs 0.51).
- Changes: the order of turns does not create a useful cascade: these agents mostly agree already (F55), and seeing
  more of the others makes them herd. Less contact (chain, star F18) is what avoids the loss; nothing here beats
  silenced agents on the main task.
- Status: **closed**.

---

## Literature check (2026-11-22, no new runs)

The 17 papers the deep-research review (`docs/Communication in LLM-Based Multi-Agent Systems_ ...md`) left unverified
were read (`docs/downloaded_papers/`, notes in `docs/LITERATURE_2026-11-22.md`). Two of them (ICR, arXiv 2610.01042;
"Not All Flips Are Conformity", 2606.00820) measure peer influence against a matched no-message control, because
agents change and agree without peers too. Applied to our herding measures, that check gives F64.

### F64. Most of what the adoption and copy measures counted as herding happens without talk
- `adoption_rate_result`, `blind_adoption_rate` and `copy_events` (`framework/tasks/wordcraft/metrics.py`) count only
  what an agent received, so a silenced team has none by construction, and every comparison so far set talk against 0.
  Baseline: give the silenced team the inbox fully connected would have delivered (each teammate's action of the
  previous round) and count copies against every teammate. On talking runs this reproduces the logged values exactly
  (main and followup, every model), so talk − silenced is what talk adds. Script: `docs/analysis/k13.py`.
- main, fully connected, silenced → talk (bootstrap by target): result adoption pooled 65.2% → 81.6%, **+16.4
  [+10.8, +21.9]** (gemma3-4b +25.4, qwen3-4b +18.1, qwen3-8b +15.6, llama3.2-3b +7.0); blind adoption 26.3% → 27.1%,
  **+0.8 [−2.6, +4.4]** (qwen3-4b +10.9 [−0.8, +24.6], qwen3-8b +3.7, gemma3-4b +2.3, llama3.2-3b **−10.6 [−18.4,
  −3.8]**); copy events 1.31 → 1.77, **+0.46 [+0.14, +0.81]** (llama +1.15, gemma +0.85, qwen3-8b +0.12, qwen3-4b
  −0.28). followup the same (Gemma's runs are main's, F23): result adoption +17.0, blind +1.9, copies +0.79.
- mixed teams (F49): blind adoption 25.0% silenced → 20.2% talk (−4.8 [−12.2, +3.2]); result adoption +22.6.
- By channel (followup, pooled over the three 4B models; "blind" = playing a pair a teammate played and failed in
  the previous round): results only, copies **+1.07 [+0.68, +1.45]** (llama +1.35, gemma +1.38, qwen3-4b +0.50),
  blind −4.9 [−9.3, −0.9] (seeing the failures, agents avoid them); messages only, blind **+9.1 [+4.3, +14.4]**
  (qwen3-4b +16.7 [+5.4, +29.5], gemma +8.3, llama +3.8), copies −0.07. So the two channels act in opposite
  ways: shared results make agents make what teammates made, messages make them retry teammates' suggestions even
  when those failed (F26's repeat rate), and with both channels the failures shown cancel the second effect in
  this measure. Llama's loss with results only (F26, F45) comes with copying, but Gemma copies as much without
  losing.
- chain6 (F63): its logged copies count only what the one agent it hears found first. Against every teammate it
  copies as much as the parallel talking team or more: gemma3-4b 2.77 vs 2.25, llama 2.50 vs 3.48, qwen3-4b 0.93 vs
  1.30, qwen3-8b 0.60 vs 0.82 (silenced 1.40, 2.33, 0.93, 0.70).
- So "blind adoption" (repeating a pair a teammate reported failing) is mostly agents that would play it anyway: Qwen's
  agents are near-copies (F55), so they hit the same failed pairs without being told. Talk does make agents play
  teammates' results (+16) and find items after teammates (+0.5 per run), Gemma and Llama most.
- What stands: F16 (the success loss), and the measures recorded in both conditions (distinct paths, divergence,
  repeat rate, F17, F45, F49). What changes: F45's "herding is a Qwen trait" by blind adoption (a Qwen trait of
  agreeing, not following), F49's "mixed teams follow blindly less" (their agents agree less), F63's "the chain herds
  less" (copies cannot show it), and the report's "Seguiti alla cieca" column and text (report_v2.typ: the table of
  section "Esperimento principale" and its bullet, line 128; "Squadre miste", line 316; "Turni in sequenza e catena",
  line 420).
- Status: **open**. Report talk − silenced with the would-be inbox from now on; a fix in `metrics.py` (log the
  baseline for silenced runs) would make it standard. Whether the herding comes from the messages' content or their
  presence is being scored on main's logs (`scripts/counterfactual.py`: swapped vs removed messages, 2026-11-22). The report v2 is corrected (2026-11-22: its main table's
  last column, the passages above, a new section with the table by model and channel, `make_data_v2.py` key
  `adopt`).
