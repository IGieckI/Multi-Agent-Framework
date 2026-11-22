# Literature check, 2026-11-22: the deep-research review's unverified papers

The deep-research review `docs/Communication in LLM-Based Multi-Agent Systems_ State of the Art, Metrics, and a
Research Agenda.md` (written without knowledge of this project) left 17+ references unverified. The user downloaded
17 of them into `docs/downloaded_papers/`. Each was read (abstract, method and the results the review cites). This
file records, per paper: what it is, whether the review's claim holds, and what it means for this project. Section 4
says what changes in the plan. One new finding came out of it (F64, in FINDINGS.md).

All 17 exist and match the arXiv ID the review gave them. None of the review's specific claims about them was wrong.
Two were thin: 2604.02668 had no title, and DEAR had no description. Not downloaded, so still unverified: the latent,
security and training papers in the review's §9 list (2607.26773, 2606.05711, 2605.28214, 2606.28958, 2605.22786,
2503.09501, 2510.11062, 2603.06859, 2603.21563, MALT), none of them relevant to the next experiments.

## 1. Closest to this project (read first when writing)

**ICR: "Beyond Final Accuracy: Auditing Communication in LLM Multi-Agent Systems"** (Li, Yang et al., USC;
arXiv 2610.01042, 1 Oct 2026).
- Agents answer independently, then one receives a peer's message and revises. Outcomes are split by the
  receiver's and the sender's initial correctness: correction rate CR (wrong receiver, right sender), preservation
  rate PR (right receiver, wrong sender), selectivity SI = (CR + PR) / 2. A matched no-message revision is the
  control.
- Same models as ours: Qwen3-4B (MedQA, ARC-C, GSM8K, GPQA-D), Qwen3-8B on two of them.
- Results: richer messages (full reasoning vs answer only) raise correction and lower preservation on all four
  tasks. MedQA full text: CR 80%, PR 42%. Aggregate accuracy hides this: conditions spanning 0.11 points of accuracy
  differ by 27 points in PR. A structured verification policy at the receiver raises preservation and lowers
  correction.
- Review's claim: listed as a lead only. Exists; it is the closest prior to the review's M7 (correct-flip ratio)
  and to message auditing in general.
- For us: the no-message control is what our adoption metrics lacked (F64). The renamed share cells map onto
  its strata: a told receiver is "corrected" if it plays the true recipe it could not guess. The analogue of
  preservation is an agent on the right path that is not pulled off it by a teammate's plan (the remake lock,
  F44). Cite it for the method; our setting (sequential actions, verified results, private notes) is not covered.

**"Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems"** (Batra, Khandelwal et al.;
arXiv 2610.35930, 18 Oct 2026; not peer-reviewed).
- MAD Bench, 2,400 questions; Qwen3.5-9B, Llama-3.1-8B, Gemma-4-E4B; a round-matched independent-reflection control.
- Message ablations and source relabelling scored as logit margins on the receiver. Labelling the same text as the
  receiver's own shifts the margin 39–45× more than moving it. A stated track record shifts it 1.4–14.3× more than
  professional status; a professor with a poor record is trusted less than no description at all.
- Debate gives a small accuracy gain over the initial majority vote but raises unanimous wrong answers by 3.1–7.4
  points against reflection, and makes agents' correctness more correlated.
- Review's claim: listed as "very recent, not peer-reviewed": correct.
- For us: it already uses the counterfactual scoring the earlier advice proposed as "idea 1" (edit the prompt, score
  the receiver). That idea is a method transfer to an agentic task with private notes, not a new method. Two
  contrasts to discuss: (a) telling receivers that teammates' notes are true changed nothing in our runs (F42),
  while stated track record moves their margins; our measure is the action actually played in a 45-pair space, and
  the receiver often lacks the intermediate (F59). (b) Attribution to "self" weighs most: our agents do not see their
  own previous answer, and the remake loop (F44, F62) is a self-tracking failure.

**"At Equal Inference Cost, Multi-Agent Structure Does Not Beat a Single Frozen Agent"** (Dylan et al.; arXiv
2609.04217 v1, 25 Jun 2026; the PDF carries an IEEE TPAMI header, venue not confirmed).
- MA-Evolve: a Planner→Executor→Critic team of role prompts evolved on one frozen Qwen2.5-7B, compared with an
  evolved single agent at an equal number of LLM calls (iso-call).
- ALFWorld (n = 134): evolving the single executor +0.097 (McNemar p = 0.021); the team 0.769 vs 0.754 (p = 0.80) at
  1.8× the calls. Leave-one-in decomposition: all value is in the executor; the planner and critic prompts evolve
  to empty (influence on the action 0.14 and 0.41). WebShop: null, the team trends worse.
- For us: the same conclusion as F19/F28 (iso-token) in another agentic environment. Supports reporting cost as
  calls as well as tokens.

**"When Do Multi-Agent Systems Help? An Information Bottleneck Perspective"** (Yu, Zhou et al., Texas A&M; arXiv
2607.16133, 17 Jul 2026).
- A MAS with unbounded relays can simulate a single agent. The MAS gain comes from bounded relays: they cut
  redundant context but can lose task-relevant information. A capability parameter β sets the trade-off.
- 18 experiments, ALFWorld, WebShop, WorkBench, WideSearch, TravelPlanner; Qwen2.5-7B, GPT-4o-mini, Qwen3.5-27B.
  MAS helps when relays are near-sufficient, most for weaker models; gains shrink or reverse with information loss
  and for stronger models.
- For us: a theory that fits F20 (fully connected adds about 250 tokens of others' messages per turn), F63 (the
  chain, a bounded relay, recovers the loss) and F48 (the made line, a context aid, helps Qwen). It also predicts the
  unused summary channel (`form: summary`, review D1) should help small models when the summary keeps the note
  recipes. Novelty of the review's C1 (rate–distortion): framing taken; an empirical bound where the minimal
  information is known (renamed: about one recipe) is still ours.

**"Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate"** (Hao, Wu et al.;
arXiv 2606.00820 v2, 18 Oct 2026).
- Five GPT-4o agents, MMLU-Pro disagreement questions, three matched conditions: self-reflection with no peers,
  peer stances only, stances with reasoning. 37% of agent–question pairs change answer with no peer at all. Of
  those stable under that control, 29.3% adopt a peer answer from stances alone. Full reasoning raises individual
  accuracy but adds no majority-vote gain over self-reflection, at 57% more tokens. An answer-agnostic intervention
  suppresses harmful and useful adoption alike.
- Review's claim ("some convergence is legitimate updating"): holds.
- For us: the reason for F64. Change and agreement need a no-peer baseline; ours had none for adoption and copies.

**"The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate"**
(Bertalanič and Fortuna, Jožef Stefan Institute; arXiv 2605.00914, 29 Apr 2026).
- 10 homogeneous agents (Qwen2.5-7B, Llama-3.1-8B, Ministral-3-8B), 3 rounds, GSM-Hard and MMLU-Hard. Debate
  against isolated self-correction and a noise control that injects rationales from unrelated problems.
- Review's numbers hold: modal-answer adoption up to 85.5%, vulnerability of correct reasoning up to 70%, an
  oracle gap up to 32.3 points. Conformity is already high with 2 peers. Debate costs 2.1–3.4× the tokens for
  equal or lower accuracy. The noise control beat debate on GSM-Hard for Qwen2.5-7B (63.2% vs 58.8%).
- For us: the closest analogue in size and homogeneity to F16/F55. Its noise control is the placebo arm proposed
  for the main task: replace teammates' messages with messages of another target's run, to tell herding on
  content from prompt clutter.

## 2. Related methods (useful in discussion or as arms)

**"Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity"** (Zhang, Cui,
Chen et al.; arXiv 2502.08788 v3, Jun 2025). 5 MAD methods × 9 benchmarks × 4 models: MAD often fails to beat CoT and
self-consistency at more compute. Mixing models improves MAD (Heter-SoM +6.4%, Heter-EoT +8.2%), but the bar
is the *average* of the two models' CoT. F49 uses the stricter bar (the best family) and finds no gain for silenced
mixed teams. Discuss the difference when citing it.

**"When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning"** (Choi, Zhu, Li, UW–Madison;
arXiv 2510.07517 v5, Apr 2026; same group as "Debate or Vote"). Debate as an identity-weighted Bayesian update. An
Identity Bias Coefficient measures following a peer vs oneself; sycophancy is far more common than self-bias.
Removing "self" and "peer" markers reduces it. For us: messages are labelled `agent_XX` and an agent's own past
answer is never shown. An anonymized arm is cheap but secondary.

**"Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems"** (Zhou, Song et al.;
arXiv 2601.21742, Jan 2026). Agents get peers' interaction history and build peer-reliability profiles (plus RL).
Qwen3-4B with it beats a history-agnostic Qwen3-30B at picking reliable peers. For us: a track-record line ("agent_02's
reported recipes: 3 of 3 valid") is an alternative to F42's flat "notes are true" instruction. It is also the basis of
the review's A3, which is therefore less novel than claimed.

**"Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems"** (Kasprova, Parulekar et
al., UIUC; arXiv 2604.02668 v2, Jul 2026; **the review gave no title**). Six open models (Qwen2.5 3B–32B) given peer
sycophancy rankings, static or updated online: less influence of sycophancy-prone peers, fewer error cascades,
+10.5 points accuracy. Same family as ECL: reliability priors on peers.

**"Free-MAD: Consensus-Free Multi-Agent Debate"** (Cui et al.; arXiv 2509.11035, Sep 2025). Scores the whole debate
trajectory instead of a final-round vote, adds an anti-conformity prompt, and needs one round. It reports gains on
8 benchmarks and robustness to attacks. An anti-conformity prompt is a cheap arm against fully connected herding.

**"Group Perspective Matters: Regulating Debate Relationships Can Mitigate Blind Conformity in Multi-Agent Debate"**
(DEAR; Wu et al., Beijing Jiaotong; arXiv 2608.03648 v2, Oct 2026; **the review gave no description**). Two RL agents
choose which peers each agent reads and how it generates; consensus and divergence serve as group evidence. It
argues confidence and perplexity are poor reliability signals (agrees with F57: AUROC 0.44–0.64). Needs training.

**"From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation"** (Wang et al., AWS and HSBC;
arXiv 2604.07667 v2, Oct 2026). Pools verbalized probabilities and calibrates them with split conformal prediction.
Non-singleton sets escalate to a human, which flags 81.9% of wrong-consensus cases at α = 0.05. Its premise,
"agreement is not evidence of correctness", is the same as F57's.

**"The Six Sigma Agent: Achieving Enterprise-Grade Reliability in LLM Systems Through Consensus-Driven Decomposed
Execution"** (Patel et al., Lyzr Research; arXiv 2601.22290, Jan 2026). Industry paper: atomic sub-tasks, each run
n times across diverse LLMs with voting. Error O(p^⌈n/2⌉) assumes *independent* errors. F55/F57 show that assumption
fails within one small model (modal share 0.86–0.91; 12 − 6 agents adds 0–5 points). Cite as the optimistic case
that independence decides.

**AnyMAC: "Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction"** (Wang, Tan et al.; arXiv
2506.17784 v2). Sequential pipelines with learned next-agent and next-context selection instead of a fixed graph.
Loosely related to the sequential protocol (F63); trained, not an arm for us.

**"Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems"**
(TraceElephant; Chen et al., ISCAS; arXiv 2604.22708, Apr 2026). Full execution traces (inputs and context) improve
attribution accuracy by up to 76%. The review's "orchestrator/planner agents … non-negligible (18–29%)" is quoted
verbatim. For us: our logs keep full prompts, which is the setting it recommends.

**Agent-in-the-Middle: "Red-Teaming LLM Multi-Agent Systems via Communication Attacks"** (He et al., MSU; arXiv
2502.14847 v2, Jun 2025). An LLM adversary intercepts and rewrites messages, with reflection. Background for a
possible liar-teammate arm.

## 3. Corrections to the review

- Footnotes [47], [52], [53] do not support the sentences that cite them. [47] (2606.14672, latent-space synthesis
  for parallel branches) is cited for KVComm. [52] (2603.21574, an adaptive robust estimator for MARL) and [53]
  (2605.02801, RL through orchestration traces) are cited for MALT, ReMA and Stronger-MAS. These were not downloaded.
- 2604.02668 is "Too Polite to Disagree" (sycophancy propagation, peer sycophancy rankings).
- 2608.03648 (DEAR) is RL-based regulation of which peers each agent reads.
- Novelty claims to soften:
  - M2 (content-pairing effect): text debate already has a noise control with unrelated rationales (2605.00914).
  - M7 (correct-flip ratio): ICR's CR/PR/SI with a no-message control and the "Not All Flips" decomposition precede
    it.
  - A3 (track-record weighting): ECL, Who Convinced Whom and the sycophancy rankings test track-record signals,
    though not against a saboteur in an agentic task.
  - B3 (anti-conformity protocol): its components are now all published (anonymization, Free-MAD, sycophancy priors,
    DEAR); only the combination with a stopping rule remains.
  - C1 (rate–distortion): the information-bottleneck framing is published (2607.16133).

## 4. What changes for this project

1. **F64 (new):** adoption and copy measures need a silenced baseline. Talk adds result adoption (+16) and a few
   copies (+0.5), not blind adoption (+0.8). F45's "Qwen follows blindly", F49's "mixed teams follow blindly less"
   and F63's "the chain herds less" do not survive. F16 and the distinct-paths reading do. The report's "Seguiti
   alla cieca" column needs the baseline beside it.
2. **The earlier "idea 1" (counterfactual scoring of messages)** is precedented (Who Convinced Whom; ICR's
   no-message control). It is still the cheapest causal test here, and the agentic setting with private notes and
   verified results is new. Cite those papers for the method.
3. **Placebo messages on the main task** have a precedent (the noise control of 2605.00914, where noise beat
   debate). Adding that arm would replicate it in an agentic task and tell herding on content from clutter.
4. **Track record instead of "notes are true"**: ECL, the sycophancy rankings and Who Convinced Whom agree that
   reliability evidence about a peer moves receivers more than flat assertions. A cheap prompt arm against the use
   failure of Gemma3-4B (F45, F51, F61).
5. **Context reduction**: the information-bottleneck result predicts a summary relay or a deduplicated blackboard
   (review C5, D1) helps the 3–4B models. Pair it with `mark_made`.
6. **Writing**: the closest work for the discussion is Cost of Consensus (same size class, homogeneous, conformity
   at K = 2), Not All Flips and ICR (baselines for change), Equal Inference Cost (iso-call, agentic), and the
   information-bottleneck paper (when relays help). F49 against Heter-MAD needs the bar difference spelled out.
