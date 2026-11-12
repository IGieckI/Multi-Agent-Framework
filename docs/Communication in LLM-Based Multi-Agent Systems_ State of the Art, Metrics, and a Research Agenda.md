# Communication in LLM-Based Multi-Agent Systems: State of the Art, Metrics, and a Research Agenda (MULTI_AGENT_COMMUNICATION_REVIEW.md)

Inter-agent communication in LLM multi-agent systems (MAS) usually adds less than reported: once compute is matched and confounds are controlled, most of the gain comes from extra sampling, voting, or verification, not from the content of the messages. Communication is clearly worth it in a narrower set of cases: tasks with real information asymmetry, where one agent holds facts another needs; tasks that parallelize and need breadth; and settings with an explicit verification step that stops errors from spreading. The field's main problem is measurement. Almost nobody measures whether a given message mattered.

## TL;DR

- **The strongest recent evidence is deflationary.** On seven benchmarks (Arithmetics, GSM8K, MMLU Pro Medicine, MMLU Formal Logic, HellaSwag, CommonsenseQA, HH-RLHF) with Qwen2.5-7B and Llama3.1-8B, Choi, Zhu & Li (NeurIPS 2025) find "Majority Voting alone accounts for most of the performance gains typically attributed to MAD," and debate itself behaves like a martingale. At equal thinking-token budgets, single agents match or beat five MAS architectures on multi-hop reasoning (Tran & Kiela, 2026). Anthropic reports that token usage alone explains 80% of performance variance in its multi-agent research system.
- **Where communication does help, it fails in predictable ways.** Measured failure modes include conformity and sycophancy cascades, "integrative compromise" that drowns out experts (teams fall 6.3–41.1% below their best member in Pappu et al.'s ICML 2026 version), unchecked error amplification (17.2× trace-level for independent agents vs 4.4× with a central orchestrator), and self-replicating prompt injection. Latent (KV-cache) channels cut tokens by 70–84%, but a 2026 causal audit finds their gains on standard benchmarks carry no detectable example-specific content.
- **Recommended agenda:** treat message-level causal metrics as mandatory: message necessity (ablation), a content-pairing control (swapping in another example's message), correct-vs-incorrect flip asymmetry, relay faithfulness, and compute-matched efficiency frontiers. Then test protocols that block conformity and pass on only decision-relevant information (independent-first and anonymized rounds, track-record-weighted aggregation, typed claim/evidence messages, value-of-information gating) on tasks built to require communication. The first one-week experiment is a compute-matched message-ablation audit on hidden-profile tasks.

---

## 1. Executive summary

1. **Debate ≈ voting plus noise in the canonical setup.** Choi et al. (NeurIPS 2025, arXiv 2508.17536) model debate as a Bayesian belief process in which each agent's belief in the correct answer forms a martingale, so "debate alone does not improve expected correctness."\[1\] Empirically, majority voting matches or beats debate across seven benchmarks (Arithmetics, GSM8K, MMLU Pro Medicine, MMLU Formal Logic, HellaSwag, CommonsenseQA, HH-RLHF), confirming the earlier warning from Smit et al. (2023, arXiv 2311.17371) that debate "do[es] not reliably outperform" self-consistency and ensembling.
2. **Matching compute removes much of the MAS advantage.** Tran & Kiela (2026, arXiv 2604.02460) give a Data Processing Inequality argument and experiments on Qwen3, DeepSeek-R1-Distill-Llama and Gemini 2.5: single agents consistently match or outperform MAS when reasoning tokens are held equal.\[2\] Anthropic's June 2025 engineering post reports that its multi-agent research system (Opus 4 lead + Sonnet 4 subagents) uses "about 15× more tokens than chats" and "outperformed single-agent Claude Opus 4 by 90.2%," with token usage alone explaining 80% of variance on BrowseComp.
3. **Architecture determines whether errors propagate.** Kim et al. (Google Research/MIT, arXiv 2512.08296, v3) ran 260 configurations spanning six agentic benchmarks, five canonical architectures and three LLM families. Independent agents show 17.2× trace-level error amplification, centralized ones 4.4×, and coordination returns turn negative once single-agent accuracy exceeds about 45%. The authors' project page warns that these factors measure extra work caused by coordination failures, not final-answer error rates.\[3\]
4. **Failures are mostly at the seams.** The MAST taxonomy (Cemri et al., NeurIPS 2025 D&B, arXiv 2503.13657) was derived from "rigorous analysis of 150 traces" (κ = 0.88) and applied to "1600+ annotated traces collected across 7 popular MAS frameworks," sorted into 14 failure modes. About a third are inter-agent misalignment: withheld information, ignored input, derailment, failing to ask for clarification.
5. **Conformity is the dominant pathology of free-form discussion.** Agents flip from correct to incorrect answers under peer pressure (Wynn, Satija & Hadfield, 2025), sycophancy collapses disagreement too early (Yao et al., 2025), and self-organizing teams average expert and non-expert views instead of deferring to the expert (Pappu et al., 2026).\[4\]\[5\]\[6\]
6. **Latent communication is the most active new format, and its evidence is the most contested.** LatentMAS reports up to 14.6% higher accuracy, 70.8–83.7% fewer output tokens and 4–4.3× faster inference; Cache-to-Cache (C2C) reports +3.1–5.4% over text exchange with about 2.5× lower latency.\[7\]\[8\] A causal audit (Cheng, Das & Ramnath, 2026) finds the relayed cache's example-specific content adds nothing detectable on standard benchmarks (equivalence within ±2.8 points), and C2C's released projector shows no detected example-specific transfer.\[9\]
7. **Efficiency claims for learned topologies are fragile.** Zhang et al. (EMNLP 2026, arXiv 2609.05933) show many reported pruning gains come from structural collapse, disabled tool pathways, or starting systems where random pruning already preserves accuracy.\[10\]
8. **Training for communication works but is under-validated.** Optima and MAPoRL show that RL or iterative fine-tuning yields shorter messages and more productive interaction,\[11\]\[12\] but comparisons against a single agent RL-trained with the same budget are rare.
9. **Protocols (MCP, A2A, ACP, ANP) standardize transport, not meaning.** They make discovery and tool access interoperable but do not address faithfulness, calibration or conformity, and they widen the attack surface; self-replicating prompt infection and steganographic collusion have both been demonstrated.\[13\]\[14\]
10. **The best near-term research bet** is measurement: message-level causal metrics plus compute-matched baselines. Most published "communication gains" cannot currently be told apart from ensembling.

---

## 2. Taxonomy of the field

| Dimension | Design options | Representative work | Key open tension |
|---|---|---|---|
| **Content/format** | Free natural language; structured/JSON; summaries/compressed; latent (embeddings, hidden states, KV caches); emergent codes | AutoGen, CAMEL; Optima; LatentMAS, C2C, KVComm | Bandwidth and latency vs interpretability, auditability, security |
| **Structure** | Chain, star/orchestrator, fully connected, hierarchical, sparse, learned/dynamic graphs, blackboard/shared memory | Li et al. 2024; G-Designer; AgentPrune; AgentDropout; AGP; Anthropic orchestrator-worker | Error containment vs diversity; token cost vs coverage |
| **Policy** | Always broadcast vs gated; addressing; rounds and stopping; agent dropout | AgentPrune, AgentDropout, RGA-Designer; MAD-Conformist, Free-MAD | Stopping criteria are almost always fixed round counts |
| **Interaction paradigm** | Debate, critique/verification, role teams, orchestrator-worker, negotiation, voting | Du et al. 2023; Choi et al. 2025; MAST; Anthropic Research | Interaction gains vs aggregation gains |
| **Failure modes** | Hop-wise information loss, error cascades, conformity/sycophancy, expert dilution, redundancy, injection spread, collusion, cost blow-up | MAST; Kim et al.; Wynn et al.; Pappu et al.; Prompt Infection; Motwani et al. | Is the failure in the agent, the message, or the topology? |
| **Training** | Prompting only; SFT/DPO on MAS traces; multi-agent RL with team or counterfactual rewards | Optima; MAPoRL; MALT; ReMA; C3 | Credit assignment; collusion/steganography under optimization |
| **Protocols** | MCP (agent↔tool), A2A (delegation via Agent Cards), ACP (REST/MIME), ANP (DID-based networks); KV-cache serving | Ehtesham et al. 2025; KVFlow | Syntactic interoperability vs semantic reliability |

---

## 3. State of the art by area

### 3.1 Interaction paradigms: debate, voting, critique

**Foundational claim.** Du et al. (2023, arXiv 2305.14325) showed that several LLM instances debating over rounds improve factuality and reasoning over single-model chain-of-thought and reflection on six tasks, launching the "society of minds" line of work.\[15\]

**Established since then:**
- *Aggregation does most of the work.* Choi et al. separate debate into majority voting and inter-agent debate. Voting alone accounts for most of the gains, and mean agent accuracy across rounds stays flat, as the martingale model predicts.\[16\]\[17\] Debate helps only when the update rule is biased toward correctness, as in their MAD-Conformist variant (keep an answer if it matches the previous majority).\[18\]
- *Debate can make things worse.* Wynn, Satija & Hadfield (2025, arXiv 2509.05396) find that "debate can lead to a decrease in accuracy over time — even in settings where stronger … models outnumber their weaker counterparts," with models "favoring agreement over challenging flawed reasoning." Agents most often abandon correct answers when no peer initially agreed with them.\[5\]\[19\]
- *Sycophancy is measurable.* Yao et al. (2025, arXiv 2509.23055) define inter-agent sycophancy, introduce the Disagreement Collapse Rate, Negative Agreement Rate and Sycophancy Score, and show sycophancy collapses debates into premature consensus.\[4\]\[20\] A 2026 study of unguided homogeneous debate ("The Cost of Consensus," arXiv 2605.00914) reports modal-answer adoption of up to 85.5%, with isolated self-correction beating debate.\[21\]
- *Sparser talk can be better.* Li et al. (2024, arXiv 2406.11776): neighbor-connected debate gives +2% on MATH, keeps GSM8K accuracy, and cuts average input tokens by over 40% vs fully connected debate.\[22\]

**Contested:** Whether debate beats voting with *heterogeneous* agents, tool-verifiable tasks, or a strong judge. Smit et al. found that tuning agreement levels can lift some debate protocols above all non-debate baselines; that hyperparameter sensitivity itself suggests fragile gains.\[23\] "Not All Flips Are Conformity" (arXiv 2606.00820) argues some convergence is legitimate updating, so simple flip-rate metrics overstate the problem.

**My assessment:** The popular claim that "debate improves reasoning" rests mostly on experiments without vote-matched and compute-matched controls. Treat it as unproven except for heterogeneous teams or tasks with verifiable intermediate claims.

### 3.2 Team structure, expertise and organizational failure

- **Experts get diluted.** Pappu, El, Cao, di Nolfo, Sun, Cao & Zou (2026, arXiv 2602.01011) find self-organizing LLM teams fail to match their best member even when told who the expert is. Leveraging, not identifying, the expert is the bottleneck; the mechanism is "integrative compromise," and the gap grows with team size.\[6\]\[24\] The v4 abstract (ICML 2026, PMLR 306) reports "performance losses of up to 41.1% on ML benchmarks," with Table 2 ranging "from 6.3% (MMLU Pro) to 41.1% (HLE Text-Only)"; the earlier 37.6% figure no longer appears. Consensus-seeking does protect teams against a saboteur.\[24\]
- **MAST** (Cemri et al.) attributes failures to system design, inter-agent misalignment and task verification. Category shares differ across versions and write-ups (41.8/36.9/21.3% vs 44.2/32.3/23.5%),\[25\]\[26\] so do not over-interpret the split; the inter-annotator agreement (κ = 0.88) is the more robust contribution.\[27\]
- **Failure attribution is hard.** Zhang et al. (ICML 2025, arXiv 2505.00212) introduced Who&When, failure logs labelled with the responsible agent and decisive error step;\[28\] attributing the *step* is far harder than the agent. TraceElephant (arXiv 2604.22708) finds orchestrator/planner agents cause a "non-negligible (18–29%)" share of failures, which then propagate.\[29\]

### 3.3 Topology and structure

- **Scaling principles.** Kim et al. (arXiv 2512.08296) fit a mixed-effects model across single, independent, decentralized, centralized and hybrid architectures. Relative to a single agent, multi-agent performance "ranges from +80.8% on decomposable financial reasoning to -70.0% on sequential planning"; decentralized helps dynamic web navigation (+9.2%). A tool-coordination trade-off (β = −0.267) and capability saturation (β ≈ −0.40) predict when coordination hurts. The v3 cross-validated R² is 0.373, leaving most of the variance unexplained.
- **Learned and pruned graphs.** G-Designer (arXiv 2410.11782) builds task-adaptive topologies with a graph autoencoder and reports beating DyLAN on SVAMP by 4% with 23.7% of its tokens.\[30\] AgentPrune (arXiv 2410.02506) prunes the spatial-temporal message graph in one shot; AgentDropout (ACL 2025) removes redundant agents; AGP (arXiv 2506.02951) prunes edges and nodes jointly.\[31\]\[32\]\[33\] RGA-Designer (arXiv 2608.20099) reports 20.5% fewer tokens than ARG-Designer at statistically indistinguishable accuracy.\[34\]
- **The evaluation-validity critique.** Zhang et al. (EMNLP 2026, arXiv 2609.05933) re-run these methods with a shared backbone, agent registry and runtime and find many gains setup-dependent: AgentPrune adds a finalizer node, and AGP on HumanEval collapses the graph so tokens *and* accuracy fall. Many successes occur where "a single agent or a randomly pruned system can already preserve strong performance."\[10\]\[35\]\[36\] **Implication:** topology papers must report random-pruning and single-agent baselines on tasks that need multiple agents.
- **Theory.** CAPE (Shen et al., EMNLP 2025, arXiv 2505.23352) measures the counterfactual causal effect of one agent's output on the final answer under a given graph and finds moderately sparse topologies suppress error propagation while keeping beneficial diffusion.\[37\]\[38\] It is one of the few metrics aimed at communication itself.

### 3.4 Content and format: natural language, structured, compressed, latent

- **Trained concise NL.** Optima (Chen et al., 2024, arXiv 2410.08115) iterates generate → rank → select → train with a reward balancing task performance, token efficiency and readability; messages first grow, then shrink while accuracy rises.\[12\]\[39\] Its information-exchange tasks are among the few benchmarks *requiring* communication.\[40\]
- **Latent communication.**
  - *LatentMAS* (Zou et al., arXiv 2511.20639; PMLR 2026) is training-free: agents reason in last-layer hidden states and pass layer-wise KV caches, reporting up to 14.6% higher accuracy, 70.8–83.7% fewer output tokens and 4–4.3× speedup across 9 benchmarks.\[41\]\[42\]
  - *Cache-to-Cache* (Fu et al., ICLR 2026, arXiv 2510.03215) learns a projector and gated fuser between *different* models' caches, reporting +6.4–14.2% over individual models, +3.1–5.4% over text exchange and about 2.5× lower latency (arXiv v1 reported 8.5–10.5%, 3.0–5.0% and 2.0×).\[8\]\[43\]\[44\]\[45\]\[46\]
  - *KVComm* names two systems: Ye et al. (NeurIPS 2025, cross-context KV reuse) and Shi et al. (2026, selective layer sharing).\[9\]\[47\]
- **Contested: does the latent channel carry content?** Cheng, Das & Ramnath (2026, arXiv 2608.04893) swap the relayed cache for another example's, a zeroed, or a moment-matched random cache.\[9\]
  - *Calibrated regime* (receiver needs the sender's private facts): aligned relay reaches about 100% vs 23–25% for answer-irrelevant relays, replicated across 5 checkpoints from 3 families.\[9\]
  - *Natural benchmarks* (GSM8K, ARC-Challenge, MedQA): the pairing effect is equivalent to zero within ±2.8 points (pre-registered, Holm-corrected TOST, 5 seeds × 500 items). In one cell zeroing the cache costs 14.7 points while a mismatched cache costs 0.4: the relay helps *generically*, not by transmitting the example's "thoughts."\[9\]
  - *Across systems:* LatentMAS's native relay reads at ceiling, KVComm's layer subset shows partial transfer, and C2C's released projector shows no detected example-specific transfer.\[9\]
  - The authors conclude: "Benchmark deltas do not by themselves establish latent-thought transmission."\[9\] I consider this the most important methodological paper of the period for this subfield.
- **Latent channels are a new attack surface.** "Out of Sight, Not Out of Mind" (arXiv 2605.28214) and "When Latent Agents Lie" (arXiv 2606.28958) show tampered KV states can steer a coordinator while the visible message looks benign;\[48\]\[49\] LCGuard (arXiv 2605.22786) proposes guarded KV sharing.

### 3.5 Training agents to communicate

- **MAPoRL** (Park et al., ACL 2025, arXiv 2502.18439) co-trains multiple LLMs with multi-agent RL, a verifier scoring answers and discussion, and collaboration-shaped rewards. The authors report that "training single LLMs solely is insufficient for encouraging collaboration"; training reduced Correct→Incorrect transitions and transferred to unseen domains.\[11\]\[50\]
- **MALT** (Motwani et al., COLM 2025) trains generator–verifier–refiner roles; **ReMA** (arXiv 2503.09501) trains a meta-thinker and executor; **Stronger-MAS** (arXiv 2510.11062), C3 (arXiv 2603.06859, leave-one-out message credit) and CCPO (arXiv 2603.21563) address credit assignment.\[51\]\[52\]\[53\]
- **Assessment:** Gains are real relative to *untrained multi-agent* baselines, but few papers compare against a single agent trained with the same RL compute, so "learned collaboration" and "more RL" remain confounded. Optimization pressure also carries risk: Motwani et al. (NeurIPS 2024, arXiv 2402.07510) argue steganographic collusion can emerge from misspecified objectives; in their prompted group capability evaluation (Table 3), GPT-4 covertext steganography "can evade overseer to coordinate ca. 26% of the time," a capability measure rather than collusion emerging from optimization.

### 3.6 Protocols and infrastructure

The 2025 protocol wave (MCP for tool/context access, A2A for task delegation via Agent Cards, ACP for REST-native MIME messaging, ANP for open-network discovery via W3C DIDs and JSON-LD) is surveyed by Ehtesham et al. (2025, arXiv 2505.02279), who propose phased adoption (MCP → ACP → A2A → ANP) and catalogue lifecycle threats such as Agent Card spoofing.\[54\]\[55\]

**Research implications:** these protocols fix the *envelope* (identity, discovery, task state) and leave *payload semantics* in free text, so faithfulness, calibration and provenance remain unsolved. They make cross-organization agent networks realistic, raising the stakes of injection and collusion research, and offer a natural place for typed claim/evidence schemas and signed provenance (idea B1).

### 3.7 Security and adversarial communication

- **Prompt Infection** (Lee & Tiwari, 2024, arXiv 2410.07283): malicious prompts self-replicate across agents like a virus, even when not all communications are shared; "LLM Tagging" mitigates spread only when combined with other defenses.\[13\]\[56\]
- **Agent-in-the-Middle** attacks intercept and rewrite messages (He et al., 2025, arXiv 2502.14847).
- **Steganography.** Rippin (2026, arXiv 2606.28425) argues that with tool use the barrier to covert channels is no longer capability but *coordination* on schemes and keys; monitors help when agents cannot pre-coordinate.\[57\]\[58\]
- Robustness claims of pruning methods (AgentPrune, G-Designer, M³Prune) are mostly tested with handcrafted attacks.

---

## 4. Evaluation and metrics

### 4.1 Current practice

| Metric family | Typical operationalization | Main weaknesses / ways it can be gamed |
|---|---|---|
| Task success | Accuracy, pass@1, success rate | Rarely compute-matched; contamination on GSM8K/MMLU/HumanEval; ceiling effects; silent on *which* message mattered |
| Token/cost efficiency | Prompt + completion tokens, $, latency | Hidden reasoning tokens excluded (Tran & Kiela found Gemini 2.5's reported thinking tokens inflated about 4.7× vs visible thought);\[59\] "savings" via graph collapse; inconsistent accounting |
| Rounds to consensus / agreement | Rounds until agreement | Rewards conformity; premature consensus scores *well* |
| Error amplification | E_MAS/E_SAS or extra trace work (Kim et al.) | Trace-level factors misread as answer-error rates |
| Failure taxonomies | MAST distributions, Who&When accuracy | LLM-judge annotators; shares unstable across versions |
| Debate dynamics | Flip rates, DCR/NAR/Sycophancy Score | Flips mix legitimate updating with conformity unless conditioned on correctness |
| Robustness | Accuracy drop with a malicious agent | Handcrafted, non-adaptive attackers |
| Causal influence | CAPE; cache derangement | Rarely reported; expensive; unstandardized |

**Core gaps:** (1) almost no paper tests whether removing or swapping a message changes the outcome; (2) baselines are rarely compute-matched (same total tokens, samples, verifier); (3) benchmarks mostly let one agent solve the task alone, so communication has nothing to transmit; (4) agreement is never checked for calibration; (5) few seeds and rare confidence intervals.

### 4.2 Proposed communication-centric metrics

Notation: a run produces messages m₁…m_K, final answer Y, gold Y*; Acc(·) is expected correctness over decoding seeds.

**M1. Message Necessity (MN).** *Definition:* MN(mₖ) = Acc(full) − Acc(run with mₖ removed or nulled, downstream recomputed); system MN is the mean plus the fraction of messages with MN significantly > 0. *Compute:* replay to mₖ, drop it, regenerate downstream with S ≥ 5 seeds; sample messages or ablate message classes in long traces. *Reveals:* redundant chatter and load-bearing messages; targets pruning. *Limitations:* O(K·S) replays; removal is out-of-distribution; misses interactions (see M3).

**M2. Content-Pairing Effect (CPE)** (adapted from Cheng et al.). *Definition:* Acc(true message) − Acc(message from another example, same position). Separates example-specific content from "any well-formed message helps." *Compute:* within-batch derangement; works for text and latent channels. *Reveals:* whether communication transmits information or acts as a generic prior. *Limitations:* needs comparable examples; near zero by construction when the receiver can solve alone, so pair with a calibrated private-information task.

**M3. Message Shapley Influence (MSI).** *Definition:* Shapley value of each message or agent in v(S) = Acc(run with only messages in S). *Compute:* Monte-Carlo permutation sampling (about 50–200 permutations) or a leave-one-out approximation (CAPE-style). *Reveals:* who changed the outcome, interactions, whether influence tracks expertise. *Limitations:* expensive; sensitive to the null-message choice.

**M4. Information Gain per Message/Token (IG).** *Definition:* log p_R(Y* | context + mₖ) − log p_R(Y* | context), normalized by message tokens. *Compute:* receiver log-probs (open weights) or sampled answer frequencies (API). *Reveals:* channel efficiency; persuasive-but-wrong messages have negative IG. *Limitations:* needs gold labels; miscalibrated log-probs; many samples for API models.

**M5. Redundancy Ratio (RR).** *Definition:* fraction of a message's atomic claims the receiver already had or could predict. *Compute:* LLM claim extraction plus entailment against prior context, or receiver perplexity on the message. *Reveals:* echo chambers, compressibility. *Limitations:* noisy extraction; some repetition is useful.

**M6. Relay Faithfulness and Half-Life (RF).** *Definition:* precision/recall after h hops of facts injected at hop 0; half-life is hops to 50% recall; plus hallucinated-fact rate per hop. *Compute:* procedurally generated (contamination-free) fact registries relayed through chains/trees under different formats. *Reveals:* "telephone" loss and which formats preserve information. *Limitations:* synthetic facts may understate loss on nuanced content.

**M7. Correct-Flip Ratio (CFR).** *Definition:* P(incorrect→correct) / P(correct→incorrect) over agent-round transitions, also conditioned on persuader correctness; healthy deliberation has CFR ≫ 1. A conformity index is flip probability toward the majority minus toward the correct minority. *Reveals:* whether discussion tracks truth or is merely social. *Limitations:* needs gold labels; misses reasoning changes without answer flips.

**M8. Agreement Calibration (AC).** *Definition:* calibration error of consensus strength against accuracy, plus AUROC of consensus strength for error detection. *Reveals:* whether unanimity means anything; conformity makes unanimity overconfident. *Limitations:* needs many items per bin.

**M9. Diversity Trajectory (DT).** *Definition:* per-round entropy of answers (or rationale-embedding dispersion); premature collapse is entropy ≈ 0 while accuracy is below the round-0 "any agent correct" rate. *Reveals:* echo chambers; gives a stopping signal (B3). *Limitations:* low entropy is fine when all are right; read with M7.

**M10. Compute-Matched Gain and Efficiency Frontier.** *Definition:* CMG = Acc(MAS) − Acc(best single-agent method at equal total tokens including hidden reasoning); frontier hypervolume in (tokens, accuracy) space vs self-consistency and long-thinking frontiers. *Reveals:* whether a protocol is a better use of compute, not just more compute. *Limitations:* needs budget sweeps and honest accounting; report wall-clock latency separately.

**M11. Adversarial Breakdown Point and Infection R₀.** *Definition:* smallest adversarial fraction that drops the system below the single-agent baseline; R₀ = mean downstream agents compromised per infected agent. *Compute:* sweep adversary fraction and strength with *adaptive* attackers. *Reveals:* robustness as a curve comparable to Byzantine thresholds. *Limitations:* attack-space coverage; sensitivity to attacker strength.

**M12. Expertise Leverage Ratio (ELR).** *Definition:* (team − mean member) / (best member − mean member); ELR ≥ 1 is strong synergy; report with expert identified and unidentified. *Reveals:* integrative compromise. *Limitations:* estimate "best member" out-of-sample.

**How these can be gamed:** MN and CPE can be inflated by making the receiver artificially ignorant, so always report the full-information single agent. IG can be inflated by messages that simply state the answer, so pair it with RF and sender correctness. Frontier metrics can be gamed by hiding tokens, so require all-in, provider-reported accounting.

---

## 5. Open problems

1. **When is communication necessary?** No theory or benchmark suite reliably separates tasks needing information exchange (asymmetry, parallel breadth) from tasks where ensembling suffices.
2. **Breaking the martingale.** Which update rules or incentives make debate correctness-increasing in expectation without a verifier that already solves the task?
3. **Expertise weighting.** How can LLM teams defer to the right member without being told who it is, and without becoming vulnerable to confident saboteurs?
4. **Faithful compression.** What is the quality-vs-bits frontier for messages, and how close are NL summaries, typed messages and latent channels to it?
5. **Verifiable latent communication.** How can latent channels be audited for content-specific transfer, integrity and covert payloads?
6. **Credit assignment for training communication** without inducing collusion or steganography.
7. **Security at network scale.** Prompt-infection dynamics in heterogeneous A2A/MCP networks, and certified bounds on spread.
8. **Stopping and routing.** When to stop talking, and when to escalate to a single stronger model.
9. **Reproducibility.** Results vary with prompts and model versions, and headline results (debate gains, pruning savings, latent gains) have repeatedly shrunk or failed to replicate.

---

## 6. Ranked research ideas

Cost assumes API access to mid-tier models (about $0.2–$3 per million tokens) or open-weight 7–14B models on 1–4 GPUs. "Low" is under about $1k or a few GPU-days; "Medium" is $1–10k or about 2 GPU-weeks; "High" is RL training at scale.

### Tier A: High-value open questions

**A1. The information-asymmetry dose-response of communication value** *(rank 1)*
- *Hypothesis:* The causal value of messages (M1, M2) is near zero when every agent has the needed information and rises monotonically with the fraction of answer-critical facts held privately by others; compute-matched gain (M10) is positive only above a threshold.
- *Design:* Tasks: procedurally generated hidden-profile registries (as in Cheng et al.); split-evidence HotpotQA/MuSiQue; Optima's information-exchange tasks. Manipulate private-fact fraction (0/25/50/75/100%), agents (2/4/8), format (NL vs typed JSON). Baselines: full-information single agent, single agent with its own share, self-consistency at matched tokens, voting without talk. Metrics: accuracy, MN, CPE, IG/token, CMG. 5 seeds × 500 items per cell, 2 model families.
- *Expected:* Sigmoid relation between asymmetry and CPE/CMG; CMG ≤ 0 at 0% asymmetry.
- *Falsified if:* CMG > 0 at 0% asymmetry in both families, which would indicate real interaction benefits beyond ensembling.
- *Confounds:* The oracle single agent has a longer context; distribute distractors equally across arms.
- *Cost:* Low–medium (API).
- *Novelty:* Cheng et al. used a binary need/no-need contrast for latent channels only, and Pappu et al. study expertise, not information distribution. A graded dose-response across formats and paradigms appears new.

**A2. Can debate break the martingale with heterogeneity and verifiable claims?**
- *Hypothesis:* Debate beats vote-matched baselines only when agents come from different model families, or when messages contain checkable claims (code, citations, arithmetic) that receivers verify with tools.
- *Design:* MATH-500/AIME-style held-out problems, GPQA-Diamond, LiveCodeBench. 2×2×2 factorial: homogeneous vs heterogeneous; free text vs claim+evidence; with vs without tool verifier. Baselines: majority vote over the same samples, self-consistency at equal tokens, MAD-Conformist. Metrics: accuracy, CFR, DT, CMG. 5 seeds, 3 rounds, ≥300 items per task.
- *Expected:* Debate − vote > 0 only in heterogeneous + verifier cells; CFR > 1.5 there, < 1 elsewhere.
- *Falsified if:* No cell beats vote-matched baselines with 95% CIs excluding 0.
- *Confounds:* Heterogeneous teams are also better ensembles; add a heterogeneous vote baseline.
- *Cost:* Low–medium.
- *Novelty:* Choi et al. and Wynn et al. mostly use homogeneous agents; heterogeneity has been argued for (arXiv 2502.08788), but factorial compute-matched tests are rare.

**A3. Expertise identification vs leveraging under realistic uncertainty**
- *Hypothesis:* Teams reach ELR ≥ 1 only if influence is set by *track record* (calibrated past accuracy), not self-reported confidence, which confident saboteurs can exploit.
- *Design:* Pappu et al.'s tasks via their released code; Moon Survival-style ranking; domain-split MMLU-Pro with specialist agents. Manipulate influence signal (none / self-reported confidence / track record / hidden oracle) × saboteur (0/1). Metrics: ELR, MSI, breakdown point. 5 seeds × 200 items per cell.
- *Expected:* Track record gives the highest ELR and resists saboteurs; self-confidence helps without saboteurs but collapses with one.
- *Falsified if:* Track record yields ELR < 1 even with the expert identified.
- *Confounds:* Expert identity leaking through writing style.
- *Cost:* Low.
- *Novelty:* Extends Pappu et al. and epistemic-trust work (arXiv 2601.21742); the saboteur × influence-signal interaction appears untested.

**A4. Information half-life across hops, by message format**
- *Hypothesis:* Relay recall decays roughly geometrically per hop with free NL summaries; typed claim lists with provenance flatten the decay; latent relays preserve facts only between identical checkpoints.
- *Design:* Chains of 1–8 hops; registries of 5–40 facts plus distractors. Formats: free NL, "summarize for next agent," typed JSON claims with source IDs, verbatim pass-through, KV relay (open models). Metrics: RF(h), half-life, hallucinated-fact rate, tokens. 5 seeds × 300 chains per cell.
- *Expected:* NL half-life about 3–5 hops; typed format ≥ 2× longer; hallucinations concentrated at summarization hops.
- *Falsified if:* No format difference or no decay.
- *Confounds:* Message length; add a length-matched NL arm.
- *Cost:* Low.
- *Novelty:* MAST names handoff context loss qualitatively;\[60\] quantitative half-life by format appears unmeasured.

### Tier B: Methods to improve communication

**B1. Typed claim–evidence–confidence protocol (TCEC)**
- *Hypothesis:* Requiring messages as lists of {claim, evidence pointer, confidence, provenance} improves relay faithfulness and CFR and lowers conformity vs free text at equal tokens.
- *Design:* A4 relay tasks; split-evidence QA; orchestrator-worker coding on a SWE-bench Verified subset. Baselines: free NL; JSON without confidence/provenance; Optima-style trained NL where available. Metrics: RF, CFR, AC, accuracy, tokens, MAST frequencies. 5 seeds.
- *Expected:* RF +15–30 points at 4+ hops; higher CFR; small token overhead.
- *Falsified if:* No RF or CFR gain at equal tokens.
- *Confounds:* Schema-following varies by model; report schema-validity rates.
- *Cost:* Low.
- *Novelty:* A2A/ACP type the envelope, not epistemic content; typed *epistemic* fields evaluated with causal metrics appear new.

**B2. Value-of-information gated messaging**
- *Hypothesis:* Sending only when estimated IG for the receiver exceeds a threshold matches full communication at 30–60% of tokens and beats learned pruners under controlled evaluation.
- *Design:* The sender predicts the receiver's current answer distribution and sends only if its evidence would change it (prompted gate vs small trained classifier). Baselines: full broadcast; random pruning at matched message rate; AgentPrune and G-Designer under the Zhang et al. (2026) harness. Tasks: A1 at 50% asymmetry plus GSM8K as a no-need control. Metrics: M10 frontier, MN of retained messages, accuracy. 5 seeds.
- *Expected:* Frontier dominance over random pruning on asymmetric tasks; near-silence and parity on GSM8K.
- *Falsified if:* Indistinguishable from random pruning at a matched rate.
- *Confounds:* Gate tokens; count them.
- *Cost:* Low (prompted) to medium (trained).
- *Novelty:* Pruners learn static or task-level masks; per-message VoI gating tested against random pruning is underexplored.

**B3. Anti-conformity deliberation: independent-first, anonymized, track-record-weighted, entropy-stopped**
- *Hypothesis:* A protocol that (i) collects independent answers first, (ii) shows peers only anonymized claim+evidence, never vote counts, (iii) aggregates with track-record weights, and (iv) stops when answer entropy stops falling or CFR drops below 1 beats both vanilla debate and majority voting at equal tokens.
- *Design:* 2⁴ (or fractional) factorial ablating each component. Tasks: GPQA-Diamond, held-out MMLU-Pro subjects, AIME 2025/2026. Baselines: majority vote, MAD, MAD-Conformist, Free-MAD. Metrics: accuracy, CFR, AC, DT, CMG. 5 seeds, 3 model families.
- *Expected:* +2–5 points over voting on hard items with split initial votes; no gain on easy items.
- *Falsified if:* No improvement over voting with matched samples.
- *Confounds:* Weighting alone may explain gains; the factorial isolates it.
- *Cost:* Low.
- *Novelty:* Components exist separately (identity anonymization, arXiv 2510.07517; Free-MAD; confidence calibration); the Delphi-style combination with principled stopping and causal metrics is untested.

**B4. Counterfactual message-credit RL**
- *Hypothesis:* Rewarding each message's counterfactual effect on the receiver's correctness (M1/M4 as reward) yields higher CFR and shorter messages than team-reward-only MARL (MAPoRL-style), and beats single-agent RL with the same compute.
- *Design:* Qwen3-8B-class models, GRPO; A1 asymmetric tasks plus math. Arms: team reward; team + counterfactual credit; single-agent GRPO with the same rollout budget. Metrics: accuracy, CFR, IG/token, steganography probes (paraphrase-robustness). 3 training seeds.
- *Expected:* Better sample efficiency and fewer correct→incorrect flips; gain over single-agent RL only on asymmetric tasks.
- *Falsified if:* Single-agent RL matches it on all tasks.
- *Confounds:* Rollout count and context length; match total generated tokens.
- *Cost:* High (about 2–6 GPU-weeks).
- *Novelty:* C3 and CCPO introduce counterfactual credit; the compute-matched single-agent RL control plus steganography monitoring is new.

**B5. Hybrid latent+text with an audit channel**
- *Hypothesis:* Latent relays with a mandatory short text commitment and a periodic derangement self-check keep LatentMAS-like savings while detecting tampered or non-informative caches.
- *Design:* LatentMAS and HyLaT-style hybrids on Qwen3 4B/8B; integrity check via consistency between commitment and decoded cache probes. Tasks: calibrated registry, GSM8K, MedQA, plus KV-tampering attacks from arXiv 2606.28958. Metrics: CPE, detection AUROC, tokens. 5 seeds × 500 items.
- *Expected:* AUROC > 0.9 for nonsemantic tampering with ≥ 50% token savings vs text MAS.
- *Falsified if:* Attacks evade the check at matched savings.
- *Cost:* Medium (open-weight GPUs).
- *Novelty:* LCGuard and integrity papers exist; combining a causal content audit with integrity checking is new.

### Tier C: Outside the box

**C1. Rate–distortion frontiers for agent communication (information theory)**
- *Hypothesis:* Protocols can be placed on an empirical rate–distortion curve (rate = bits or tokens per decision, distortion = task loss); current protocols sit far from the frontier, and distance from it predicts how much compression helps.
- *Design:* Rate as token count and compressed size under a reference LM; distortion as error vs the full-information oracle. Sweep compression (k-token summaries, k ∈ {8…512}; top-k claims; latent dimensions) on A1 registries at 50–100% asymmetry, where the minimal needed information is *known by construction*. 5 seeds.
- *Expected:* Free NL uses 10–100× the minimal bits; typed protocols get within about 3×.
- *Falsified if:* Free NL is already near the bound.
- *Confounds:* Tokens are not bits; report both.
- *Cost:* Low.
- *Novelty:* Information-bottleneck framings exist (arXiv 2607.16133, not examined in depth); empirical R(D) curves against a known minimal-information bound appear new.

**C2. Gricean / Rational Speech Act senders (linguistics and pragmatics)**
- *Hypothesis:* Senders that model the receiver's current beliefs and send only what is informative relative to them (Grice's quantity and relevance; RSA; akin to predictive coding, transmitting only prediction error) reduce redundancy and raise IG/token.
- *Design:* Arms: sender sees a receiver-belief summary plus "send only what would change their answer" vs standard; a two-pass RSA variant that simulates the receiver and revises. Tasks: A4 and A1. Metrics: RR, IG/token, RF, accuracy. 5 seeds.
- *Expected:* RR down 30–50% with no accuracy loss.
- *Falsified if:* Receiver modelling does not change RR or harms RF.
- *Confounds:* Prompt length; use a length-matched control.
- *Cost:* Low.
- *Novelty:* RSA is established in computational pragmatics, but systematic tests in LLM MAS with causal metrics appear absent [unverified].

**C3. Byzantine-quorum deliberation (distributed systems)**
- *Hypothesis:* BFT structure (n ≥ 3f+1 agents, provenance-tagged messages, propose → echo → accept only with a quorum of independently derived support) yields a breakdown point near f < n/3 against adaptive adversaries and prompt infection, whereas vanilla debate breaks at a single confident adversary.
- *Design:* n ∈ {4, 7, 10}; adversary fraction 0–40%, adaptive persuasion and infection-style. Tasks: GPQA, factual QA, tool-using agent tasks. Baselines: debate, voting, AgentPrune robustness mode. Metrics: breakdown point, R₀, accuracy, tokens. 5 seeds.
- *Expected:* Breakdown point about 0.3 vs about 0.1 for debate, at 1.5–2× tokens.
- *Falsified if:* No improvement, or agents "echo" without independent derivation.
- *Confounds:* Correlated honest-agent errors violate BFT independence; measure error correlation.
- *Cost:* Low–medium.
- *Novelty:* Consensus-driven decomposition (arXiv 2601.22290) and conformal social choice (arXiv 2604.07667) exist; BFT quorum tested against adaptive LLM adversaries with breakdown curves appears new.

**C4. Hidden-profile and transactive-memory paradigms (organizational science)**
- *Hypothesis:* As in Stasser & Titus's hidden-profile experiments with humans, LLM teams over-discuss shared information and under-surface unique information; a transactive-memory directory of who-knows-what on a blackboard fixes this.
- *Design:* Hidden-profile decision tasks (candidate selection with distributed attributes) and A1 registries. Arms: free discussion; plus directory; plus a "share your unique facts first" turn. Metrics: unique facts surfaced, accuracy, ELR, MSI. 5 seeds × 200 tasks.
- *Expected:* Shared-information bias in free discussion; the directory raises unique-fact surfacing by more than 30%.
- *Falsified if:* No bias appears, which would itself show LLM teams differ from humans here.
- *Cost:* Low.
- *Novelty:* Social-psychology framings of LLM MAS exist; a direct hidden-profile replication with causal metrics appears untested.

**C5. Proper-scoring and peer-prediction incentives (mechanism design)**
- *Hypothesis:* "Surprisingly popular" aggregation (Prelec-style: each agent gives its answer *and* a prediction of others' answers) counteracts conformity better than majority voting, and telling agents they are scored by a proper scoring rule improves calibration.
- *Design:* Elicit answer, predicted peer distribution and confidence; aggregate by majority, confidence-weighted majority, or surprisingly-popular. Tasks: adversarially filtered majority-wrong items (GPQA/TruthfulQA-style) plus standard items. Metrics: accuracy on majority-wrong items, AC, CFR. 5 seeds, 3 families.
- *Expected:* Recovers 10–30% of majority-wrong items.
- *Falsified if:* Peer predictions are too correlated with agents' own answers to carry signal.
- *Confounds:* Homogeneous agents share beliefs and meta-beliefs; heterogeneity matters.
- *Cost:* Low.
- *Novelty:* Surprisingly-popular aggregation appears sporadically for LLM ensembles; its use against MAS conformity appears unexplored [unverified].

**C6. Reframing: communication as test-time compute allocation**
- *Hypothesis:* The right question is not "multi-agent or single agent?" but the marginal accuracy per token of a message token vs a thinking token vs a sampling token; optimal systems spend on messages only when information is distributed or a verifier exists.
- *Design:* Fit marginal-value curves per token type across budgets on A1 tasks at varying asymmetry; derive an allocation policy and test on held-out tasks against fixed MAS and long-thinking single agents. Metric: frontier hypervolume. 3 seeds × budget grid.
- *Expected:* A predictable allocation crossover at moderate asymmetry.
- *Falsified if:* Curves are too task-specific to transfer.
- *Cost:* Medium.
- *Novelty:* Builds on Tran & Kiela and Kim et al. but casts the problem as explicit marginal allocation.

---

## 7. Top 5 recommendations and a first experiment

**Top 5 (impact × feasibility × novelty):**

1. **A1: information-asymmetry dose-response with M1/M2/M10.** It answers the central question of when communication matters, needs only API calls, and becomes the benchmark for every other idea.
2. **B3: anti-conformity deliberation.** Conformity is the best-documented failure; the intervention is cheap, its components have partial support, and the combined test with causal metrics is new.
3. **C1: rate–distortion frontiers.** A principled yardstick for the efficiency literature, directly addressing the validity crisis exposed by Zhang et al. (2026) and Cheng et al. (2026).
4. **A4 + B1: relay half-life and typed claim–evidence protocol.** Handoff loss drives a large share of MAST failures, and a typed schema can ship in A2A/MCP payloads immediately.
5. **C3: Byzantine-quorum deliberation against adaptive adversaries.** Security risk grows with protocol adoption, and robustness curves (breakdown point, R₀) are missing from the literature.

B4 (counterfactual-credit RL) has high impact but lower feasibility; run it after A1 identifies which tasks actually need communication.

**First experiment, startable within a week: "Does this message matter?" audit**
- **Day 1–2:** Generate 500 hidden-profile items per asymmetry level (0/50/100%): 4-entity registries plus 10 distractor facts spread across 3 agents. Implement four arms: (a) single agent with all facts; (b) self-consistency ×3 at matched tokens; (c) 3-agent round-robin discussion, 2 rounds, final answer from a random agent; (d) voting without talk.
- **Day 3–4:** Run 2 model families (one open-weight, one API), 5 seeds, logging all tokens including reasoning tokens. For arm (c), compute MN by leave-one-message-out replay on 100 items and CPE by within-batch message derangement.
- **Day 5:** Compute CFR, DT, AC and CMG with bootstrap 95% CIs and paired tests across seeds.
- **Decision rule:** If CPE and CMG are about 0 at 0% asymmetry and positive at 100%, the measurement stack is validated; proceed to B3 and C1 with it.
- **Budget:** about 3–6M tokens per family, well under $500.

---

## 8. Methodology pitfalls checklist

- [ ] **No compute-matched single-agent baseline.** Compare against self-consistency and long-thinking single agents at equal *total* tokens, including hidden reasoning tokens; API thinking-token accounting can be badly off (Tran & Kiela).\[59\]
- [ ] **No vote-matched baseline for debate.** Always include majority voting over the same initial samples (Choi et al.).
- [ ] **Benchmarks that don't need communication.** On GSM8K/MMLU message content cannot matter by construction; add information-asymmetric tasks.
- [ ] **No content control.** Any claim that "communication transmits X" needs a message-derangement control and an ablation (Cheng et al.).
- [ ] **Missing random-pruning and structural-collapse checks** for efficiency methods (Zhang et al., EMNLP 2026).
- [ ] **Too few runs.** Use ≥5 seeds, item-level paired tests, bootstrap CIs, TOST for null claims, and multiple-comparison correction.
- [ ] **Prompt sensitivity.** Report variance across ≥3 prompt paraphrases including role descriptions; debate results depend heavily on agreement-level hyperparameters (Smit et al.).\[23\]
- [ ] **Contamination.** Prefer procedurally generated or post-cutoff items (AIME 2025/2026, LiveCodeBench).
- [ ] **Homogeneous agents described as "diverse."** Measure inter-agent error correlation; persona prompts rarely create independent errors.
- [ ] **LLM-as-judge in the loop and in evaluation.** Judges share debaters' biases; use gold labels or human audits and report judge–human agreement.
- [ ] **Flip-counting without conditioning on correctness.** Use CFR, not raw flip rates (cf. arXiv 2606.00820).
- [ ] **Misreading amplification factors.** Trace-level overhead is not the final-error ratio (Kim et al. project page).\[3\]
- [ ] **Version drift.** Pin model versions and dates; headline numbers differ across arXiv versions (MAST shares; Pappu et al.'s abstract moving from 37.6% to 41.1% in v4; C2C v1 vs ICLR).
- [ ] **Handcrafted attacks only.** Robustness claims need adaptive attackers and adversary-fraction sweeps.
- [ ] **Ignoring latency.** Report wall-clock time alongside tokens; MAS can win on latency while losing on tokens.
- [ ] **In-sample "best member"** when computing synergy (ELR); estimate it out-of-sample.

---

## 9. References: verification notes

Papers are cited inline by first author, year and arXiv ID or venue; each arXiv ID resolves at arxiv.org/abs/<ID>.

**Verified (abstract, paper text or proceedings page checked):** Choi et al. 2025 (NeurIPS; 2508.17536); Kim et al. 2025 (2512.08296; full author list not checked); Cemri et al. 2025 (2503.13657); Zhang S. et al. 2025 (ICML; 2505.00212); Zou et al. (2511.20639; PMLR v306); Fu et al. (ICLR 2026; 2510.03215); Cheng, Das & Ramnath 2026 (2608.04893); Wynn, Satija & Hadfield 2025 (2509.05396); Yao et al. 2025 (2509.23055); Smit et al. 2023 (2311.17371); Du et al. 2023 (2305.14325); Li Y. et al. 2024 (2406.11776); Tran & Kiela 2026 (2604.02460); Pappu et al. 2026 (2602.01011); G-Designer (2410.11782); AgentPrune (2410.02506); AgentDropout (ACL 2025); AGP (2506.02951); RGA-Designer (2608.20099); Zhang J. et al. 2026 (EMNLP; 2609.05933); Shen et al. 2025 (CAPE; 2505.23352); Chen W. et al. Optima (2410.08115); Park et al. MAPoRL (ACL 2025; 2502.18439); Ehtesham et al. 2025 (2505.02279); Lee & Tiwari 2024 (2410.07283); Motwani et al. 2024 (2402.07510); Rippin 2026 (2606.28425); Anthropic, "How we built our multi-agent research system" (June 2025).

**Seen in search results or citations, not read in full:** 2605.00914; 2606.00820; 2604.02668; DEAR (2608.03648); Free-MAD (2509.11035); 2605.28214; 2606.28958; LCGuard (2605.22786); "Beyond Tokens" survey (2606.05711); Stronger-MAS (2510.11062); MALT (COLM 2025); ReMA (2503.09501); C3 (2603.06859, author list unverified); CCPO (2603.21563); Agent-in-the-Middle (2502.14847); TraceElephant (2604.22708); AnyMAC (2506.17784); "Who Convinced Whom?" (2610.35930, very recent, not peer-reviewed); "At Equal Inference Cost, Multi-Agent Structure Does Not Beat a Single Frozen Agent" (2609.04217).

**[unverified] leads:** "When Do Multi-Agent Systems Help? An Information Bottleneck Perspective" (2607.16133); "Beyond Final Accuracy: Auditing Communication in LLM Multi-Agent Systems" (2610.01042); "Do Latent Channels Actually Communicate?" (2607.26773). Re-run the Section 6 novelty checks that depend on these before submission.

## Sources

1. [Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://arxiv.org/pdf/2508.17536)
2. [Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460)
3. [Project Overview ‹ When do AI agents benefit from collaboration?](https://www.media.mit.edu/projects/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/overview/)
4. [Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate](https://arxiv.org/abs/2509.23055)
5. <https://arxiv.org/pdf/2509.05396>
6. [Multi-Agent Teams Hold Experts Back (arXiv:2602.01011)](https://paperswithcode.co/paper/2602.01011)
7. [\[2511.20639\] Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)
8. [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://www.alphaxiv.org/abs/2510.03215)
9. <https://arxiv.org/pdf/2608.04893>
10. [Replanteamiento de la evaluación de los métodos de eficiencia para sistemas multiagente](https://www.alphaxiv.org/es/abs/2609.05933)
11. [\[Quick Review\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://liner.com/review/maporl-multiagent-postcotraining-for-collaborative-large-language-models-with-reinforcement)
12. [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://arxiv.org/pdf/2410.08115)
13. [\[2410.07283\] Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems](https://arxiv.org/abs/2410.07283)
14. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://arxiv.org/pdf/2402.07510)
15. [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/html/2305.14325v1)
16. [Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://www.alphaxiv.org/abs/2508.17536)
17. [Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? — Lacuna](https://lacuna.tiptreesystems.com/work/debate-or-vote-which-yields-better-decisions-in-multi-agent-large-language/wrk_2bfc2cda1ed52e4985174af838032d24)
18. [\[Literature Review\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://www.themoonlight.io/en/review/debate-or-vote-which-yields-better-decisions-in-multi-agent-large-language-models)
19. [Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate](https://www.alphaxiv.org/abs/2509.05396)
20. [Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate](https://arxiv.org/html/2509.23055)
21. [The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate](https://arxiv.org/html/2605.00914v1)
22. [Improving Multi-Agent Debate with Sparse Communication Topology](https://arxiv.org/html/2406.11776v1)
23. [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371)
24. [Multi-Agent Teams Hold Experts Back](https://alphaxiv.org/overview/2602.01011v1)
25. [MAST](https://sites.google.com/berkeley.edu/mast)
26. [Why Do Multi-Agent LLM Systems Fail? — Lacuna](https://lacuna.tiptreesystems.com/work/why-do-multi-agent-llm-systems-fail/wrk_fc5c016428e16426fc809ea7fc9e523e)
27. [\[2503.13657\] Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
28. [\[PDF\] Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems](https://www.semanticscholar.org/paper/Which-Agent-Causes-Task-Failures-and-When-On-of-LLM-Zhang-Yin/a0d37ec77dc2acddb223d9a7ac4f23ca10e2908f)
29. [Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](https://arxiv.org/pdf/2604.22708)
30. [G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/pdf/2410.11782)
31. [Adaptive Graph Pruning for Multi-Agent Communication](https://arxiv.org/pdf/2506.02951)
32. [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://www.alphaxiv.org/abs/2410.02506)
33. [AgentDropout: Dynamic Agent Elimination for Token- ...](https://aclanthology.org/2025.acl-long.1170.pdf)
34. [Reward-Guided Autoregressive Graph Generation for Efficient Multi-Agent Communication Topology Design](https://arxiv.org/pdf/2608.20099)
35. [Rethinking the Evaluation of Efficiency Methods for Multi-Agent Systems](https://arxiv.org/pdf/2609.05933)
36. [Rethinking the Evaluation of Efficiency Methods for Multi-Agent Systems](https://arxiv.org/html/2609.05933)
37. [Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems](https://arxiv.org/html/2505.23352)
38. [Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems](https://arxiv.org/abs/2505.23352)
39. [\[2410.08115\] Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://arxiv.org/abs/2410.08115)
40. [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://openreview.net/forum?id=c4w1TqcSi0)
41. [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/html/2511.20639v1)
42. [Latent Collaboration in Multi-Agent Systems](https://proceedings.mlr.press/v306/zou26k.html)
43. [\[2510.03215\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
44. [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://mlanthology.org/iclr/2026/fu2026iclr-cachetocache/)
45. [Cache-to-Cache(C2C): Direct Semantic Communication Between Large Language Models via KV-Cache Fusion - MarkTechPost](https://www.marktechpost.com/2025/11/04/cache-to-cachec2c-direct-semantic-communication-between-large-language-models-via-kv-cache-fusion/)
46. [\[2510.03215v1\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215v1)
47. [Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows](https://arxiv.org/pdf/2606.14672)
48. [When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration](https://arxiv.org/html/2606.28958v1)
49. [\[2606.28958\] When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration](https://arxiv.org/abs/2606.28958)
50. [MAPoRL2: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning - ACL Anthology](https://aclanthology.org/2025.acl-long.1459/)
51. [Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration](https://arxiv.org/html/2603.06859v1)
52. [Adaptive Robust Estimator for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2603.21574)
53. [Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces](https://arxiv.org/pdf/2605.02801)
54. [A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)](https://www.alphaxiv.org/overview/2505.02279v2)
55. [A Survey of Agent Interoperability Protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)](https://arxiv.org/html/2505.02279v1)
56. [\[Literature Review\] Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems](https://www.themoonlight.io/en/review/prompt-infection-llm-to-llm-prompt-injection-within-multi-agent-systems)
57. [Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems](https://arxiv.org/html/2606.28425v1)
58. [Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems](https://arxiv.org/pdf/2606.28425)
59. [Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://chatpaper.com/paper/264450)
60. [Why do multi agent LLM systems fail (and how to fix)- 2026 Guide](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)
