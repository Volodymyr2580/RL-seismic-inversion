# Peer Review: Introduction of "Reinforcement Learning Framework for Seismic Full-Waveform Inversion with Progressive Reward Design"

**Reviewer:** Anonymous
**Discipline:** Exploration Geophysics / Computational Seismology
**Date:** 2026-05-24
**Target File:** cited.md

---

## PART 1 — STRUCTURED REVIEW

---

### 1. SUMMARY

This manuscript proposes a reinforcement learning (RL) framework for full-waveform inversion (FWI) of seismic data, treating the forward simulation as a non-differentiable black box and using physics-based reward signals (travel-time misfit followed by waveform misfit) in a progressive curriculum to train a policy network. The authors position this as the first application of modern policy gradient methods to seismic FWI. The introduction follows a well-executed funnel structure: classical FWI → its limitations (cycle skipping, differentiability requirement) → deep learning approaches and their limitations → RL as an underexplored alternative → the identified gap → the proposed modular framework → enumerated contributions → paper outline. The narrative is generally clear, the motivation is sensible, and the gap identification is directionally correct.

However, the introduction contains several issues that range from **fatal** (incorrect citation-to-claim mapping for the GRPO/GDPO lineage that forms the theoretical backbone of the proposed method) to **major** (overclaims regarding the RL-to-FWI mapping, imprecise analogy between progressive reward and frequency continuation, and reliance on an unverified prior work as the sole gap benchmark). These must be resolved before the manuscript is suitable for journal submission.

---

### 2. STRENGTHS

1. **Well-structured narrative arc.** The introduction moves logically from the well-known problem (FWI nonlinearity) through existing attempted solutions (multi-scale, optimal transport, DL methods) to the proposed paradigm (RL), creating a clear and compelling funnel. The reader is never lost.

2. **Clear gap identification.** The authors correctly observe that RL has been applied to complex PDE-governed systems (tokamak control, [32]) but is virtually absent from seismic inversion. This is a genuine gap, and the framing — RL as a way to bypass both the differentiability requirement of classical FWI and the data-hunger of supervised DL — is intellectually coherent.

3. **Modular framework presentation.** The three interchangeable components (parameterization, reward function, optimization algorithm) are concisely introduced and effectively communicate the flexibility of the proposed approach without drowning the reader in implementation details.

4. **Appropriate classical FWI coverage.** Citations [1]–[10] cover the adjoint-state foundation, multi-scale strategies, alternative misfit functions, and wavefield-reconstruction inversion. The selection is authoritative and representative of the field through ~2016.

5. **Effective use of the tokamak analogy.** Reference [32] (Degrave et al., 2022, *Nature*) provides a compelling existence proof that model-free RL can optimize PDE-constrained physical systems — a strategically important citation for motivating the proposed work.

---

### 3. WEAKNESSES

Weaknesses are classified as **FATAL** (must be corrected before publication; would lead to rejection if unaddressed), **MAJOR** (substantially weaken the argument or evidence base; require significant revision), or **MINOR** (do not threaten the paper's core claims but should be addressed for scholarly precision).

#### FATAL

**F1. Incorrect citation-to-claim mapping for GRPO and GDPO (Paragraph 4, lines 9–10).**

The introduction states:

> "More recently, Group Relative Policy Optimization (GRPO) [25] and its generalization to Group Direct Policy Optimization (GDPO) [26] have eliminated the need for a learned value function…"

- Reference [25] is Shao et al. (2024), *DeepSeekMath*. This paper indeed introduces GRPO, but **solely in the context of large language model (LLM) fine-tuning with discrete token-level actions**. GRPO has not been validated on continuous-control tasks, let alone PDE-constrained optimization of velocity models. Citing it as the foundation for an FWI framework — without acknowledging this domain gap — is misleading. The introduction further claims GRPO is "particularly well-suited to problems where comparing multiple candidate solutions against a physics-based objective is more natural than learning a scalar value function over high-dimensional state spaces," appending the citation markers [25, 26] as though both references substantiate this claim. Neither reference addresses physics-based optimization or continuous control.

- Reference [26] is Meng et al. (2024), *SimPO: Simple Preference Optimization with a Reference-Free Reward*. **This paper does not describe GDPO.** SimPO is a direct preference optimization (DPO) variant for LLM alignment. "Group Direct Policy Optimization" as a named algorithm does not appear in the cited source. The citation is factually incorrect.

- If GDPO exists as a distinct algorithm, the authors must provide its correct bibliographic reference. If it does not, the claim of a "generalization" from GRPO to GDPO must be removed or substantiated.

- More fundamentally: the introduction builds its RL narrative as Policy Gradient Theorem → TRPO → PPO → GRPO → GDPO, implicitly suggesting this is the maturation arc of RL for continuous control. This is historically inaccurate. The standard continuous-control lineage would be DDPG → TRPO → PPO → SAC → TD3. GRPO and SimPO are LLM-alignment innovations from 2024 that have no established track record in continuous control. The manuscript's reliance on them as the algorithmic backbone, presented without caveat, would raise immediate skepticism from any reviewer familiar with the RL literature.

**F2. Unsupported "first" priority claim undermined by DeepWaveRL uncertainty (Paragraph 5, line 11).**

The manuscript claims:

> "No existing study has applied modern policy gradient methods such as PPO or GRPO to recover subsurface velocity models from observed seismic waveforms."

This is an absolute priority claim. Yet the introduction itself acknowledges DeepWaveRL as "the sole work that tangentially addresses RL for geophysical inversion," and the flagged note (line 106 of cited.md) states: "A definitive bibliographic reference for DeepWaveRL was not available at the time of this review." If DeepWaveRL cannot be definitively characterized, the authors cannot claim with certainty that no existing study has done what they propose. A priority claim built on an unverifiable negative is methodologically unsound. The authors must either (a) locate and cite DeepWaveRL precisely, characterizing exactly what it does and how the present work differs, or (b) remove the absolute "first" framing and replace it with a qualified statement (e.g., "to the best of our knowledge").

#### MAJOR

**M1. Overclaimed "natural mapping" of seismic inversion to the RL formalism (Paragraph 5, line 11).**

The introduction asserts:

> "seismic inversion—a sequential decision process of iteratively adjusting a subsurface model to better explain observed data—maps naturally onto the RL formalism."

This is a philosophical claim presented as a factual observation. Classical FWI is not "naturally" an RL problem: gradient-based FWI is a deterministic continuous optimization, not a sequential decision process under uncertainty with a stochastic policy. The "mapping" requires the authors' *choice* to reformulate model perturbation as a stochastic action, to define a Markov state from wavefield residuals, and to treat the forward solver as an environment. Whether this reformulation is *fruitful* is precisely what the paper must demonstrate — it should not be assumed in the introduction. Presenting the mapping as "natural" preempts the very debate the paper should invite. The authors should recast this as "can be formulated within the RL framework" and briefly justify the formulation choices.

**M2. Imprecise analogy between progressive reward and frequency continuation (Paragraph 6, lines 15–16).**

The introduction claims:

> "This progressive reward curriculum stabilizes training in the same way that low-to-high frequency multi-scale inversion stabilizes classical FWI [5], but it is implemented entirely through the reward function without modifying the observed data."

The analogy is conceptually appealing but technically imprecise. Multi-scale FWI [5] uses progressively higher-frequency *data*, exploiting the fact that low-frequency waveforms are less prone to cycle skipping because their phase wraps less rapidly with velocity perturbations. The proposed method switches between travel-time misfit and waveform misfit — two fundamentally different physical quantities, not two frequency bands of the same quantity. Travel-time tomography is quasi-linear in velocity but discards amplitude information; FWI is nonlinear but resolves finer structure. Switching between them is a *scientific-information curriculum*, not a frequency-continuation strategy. The claim of equivalence ("in the same way") oversells the analogy and risks misleading readers about what the method actually does. The authors should precisely delineate how the two strategies are analogous and where they differ.

**M3. Computational cost figure (Paragraph 2, line 5).**

The introduction states that a single 3D elastic FWI iteration "can demand hundreds to thousands of wave-equation solves, with per-iteration costs reaching O(10^4) core-hours [7]." Reference [7] (Virieux & Operto, 2009) discusses the computational challenges of 3D elastic FWI, but O(10^4) core-hours *per iteration* is an extreme upper bound at the very largest scales circa 2009 and is likely an order-of-magnitude overstatement for a typical 3D acoustic FWI iteration on modern hardware. The figure risks undermining credibility with computational geophysicists. The authors should verify whether [7] characterizes this as per-iteration or per-run cost, and either cite a more precise source or soften the claim with "at the time of writing" or "for industrial-scale 3D elastic problems."

**M4. Missing key DL-for-inversion references and incomplete DL critique (Paragraph 3).**

The DL section focuses on end-to-end methods (InversionNet, VelocityGAN, Araya-Polo et al.) and hybrid methods (Sun & Demanet, Rizzuti et al.). However, it omits several important lines of work that would strengthen the gap argument:
- Physics-informed neural networks (PINNs) for FWI (e.g., Rasht-Behesht et al., 2022, *Geophysics*; Song et al., 2022)
- Loop-unrolled / learned iterative methods (e.g., Adler & Öktem, 2017; Hauptmann et al., 2020)
- Generative-model-based inversion with MCMC (e.g., Laloy et al., 2018; Mosser et al., 2020, already in the research file as [20] but unused in the introduction)

Including these would demonstrate that the authors are aware of the full spectrum of DL-based inversion approaches and that RL fills a gap that even these advanced methods do not address (notably, the black-box forward solver compatibility).

**M5. The RL literature lineage is skewed toward LLM-alignment work (Paragraph 4, lines 9–10).**

As noted under F1, the RL narrative jumps from PPO (2017, continuous control) directly to GRPO and GDPO (2024, LLM alignment), skipping the entire continuous-control RL literature from 2017–2024: SAC (Haarnoja et al., 2018), TD3 (Fujimoto et al., 2018), and model-based RL approaches that have been applied to physical systems. The research file acknowledges SAC [27], so the authors are aware of it. The introduction's omission of SAC (which explicitly addresses exploration in continuous action spaces — directly relevant to the claim that stochastic policies help escape local minima) while prominently featuring LLM-alignment algorithms creates an unbalanced narrative. At minimum, the introduction should mention SAC as the state-of-the-art continuous-control baseline and explain why GRPO/GDPO are preferred despite their LLM origins.

#### MINOR

**m1. "Carbon capture monitoring" anachronism (Paragraph 1, line 3).**

The manuscript lists "carbon capture monitoring" as an FWI application supported by [7] (Virieux & Operto, 2009). CCS monitoring as a major FWI application emerged predominantly after 2009. The claim is not wrong — FWI can certainly be applied to CCS monitoring — but [7] does not substantiate it as an established application at the time of writing. Either provide a more recent citation or rephrase to "subsurface monitoring" (which [7] does support).

**m2. Citation [8] for cycle skipping definition (Paragraph 2, line 5).**

The joint citation [7, 8] for the definition of cycle skipping is acceptable, but as flagged in the cited report, [7] alone is the more authoritative source for the phenomenon's definition. Warner et al. (2013) [8] primarily contributes AWI as a *solution* to cycle skipping. The authors should consider citing only [7] for the definition and reserving [8] for the AWI mention later in the same paragraph.

**m3. Literature search results claims in main text vs. appendix (Paragraph 5, line 11).**

The claim "fewer than 50 publications" is presented without date range, database, or query specification in the main text. These details appear only in Appendix A. For a claim that underpins the entire "gap" argument, the methodology should be summarized in a parenthetical or footnote in the main text, or the appendix should be explicitly referenced at this point.

**m4. Missing definition of "RL formalism" components (Paragraph 5, line 11).**

The sentence "the subsurface model parameterization defines the action space, the forward simulator is the environment, and the misfit between simulated and observed data defines the reward" is a helpful mapping but omits the state space. What does the agent observe? The seismic data residuals? The current velocity model? Both? A one-sentence clarification of the state would complete the formalism mapping and help readers assess the claim of a "natural" mapping.

**m5. Contribution (4) overpromises before results (Paragraph 7, line 17).**

Contribution (4) states: "we show that our RL-based approach can produce high-quality velocity models competitive with traditional FWI while removing the requirement for differentiable forward solvers." This is a results claim placed in the introduction's contribution list. It is acceptable for a contribution list to preview results, but the phrasing "can produce" (present tense, assertoric) rather than "we demonstrate that" or "we find that" makes it sound like an accomplished fact rather than a finding to be evaluated. This is a stylistic preference, but Geophysics reviewers will note it.

---

### 4. QUESTIONS FOR THE AUTHORS

1. **On GRPO/GDPO applicability (F1, M5):** What evidence exists that GRPO — developed for discrete token-level actions in LLM fine-tuning — transfers effectively to continuous velocity model parameterizations with potentially 10^3–10^6 parameters? Have you conducted preliminary experiments validating this transfer? If GRPO proves unsuitable, does the modular framework design (which advertises swappable algorithms) genuinely protect the paper's contribution, or would the failure of the primary algorithm undermine the claimed advance?

2. **On DeepWaveRL (F2):** Can you provide a definitive bibliographic reference for DeepWaveRL? If it is a preprint, working paper, or unpublished manuscript, please characterize its formulation precisely so that reviewers can assess whether it constitutes prior art. If it cannot be located, how do you plan to support the "no existing study" priority claim?

3. **On GDPO (F1):** Is Group Direct Policy Optimization (GDPO) a distinct published algorithm? If so, please provide the correct citation. If the intended reference is to a different paper, or if GDPO is an internal/unpublished extension of GRPO, this must be stated explicitly. The current citation [26] (SimPO) does not describe GDPO.

4. **On progressive reward (M2):** How do you handle the transition between travel-time and waveform reward phases? Is there a hard switch, a weighted blending, or an automated criterion for when travel-time misfit has "converged"? The two-phase structure is clearly described conceptually, but the transition mechanism has implications for whether the method is truly "progressive" versus a two-stage pipeline.

5. **On computational cost (M3):** The introduction argues that RL-based FWI removes the differentiability requirement but does not address the sample-complexity cost. Even if each forward solve is "cheap" (black-box, possibly GPU-accelerated), RL typically requires orders of magnitude more environment interactions than gradient-based optimization. How many forward solves does the proposed method require relative to classical FWI? A brief acknowledgment of the sample-complexity trade-off would strengthen the motivation.

6. **On the state space (m4):** What is the observation that the policy receives at each step? Does the agent observe the full seismic data residual, the current velocity model, both, or a learned embedding? The introduction maps action, environment, and reward to FWI concepts but leaves the state undefined.

---

### 5. VERDICT

**Revise and resubmit (major revision required).**

The introduction successfully identifies a genuine gap and tells a coherent story. The modular framework concept is appealing, and the progressive reward design is a sensible idea. However, the introduction cannot be published in its current form because (a) the algorithmic foundation (GRPO/GDPO) rests on citations that do not support the claimed properties, and the RL literature lineage presented to a Geophysics audience is misleadingly skewed; (b) the "first-ever" priority claim is undermined by the acknowledged uncertainty about the sole competing work; and (c) several conceptual claims (the "natural mapping" to RL, the frequency-continuation analogy) are presented with more confidence than the evidence warrants.

The fatal issues (F1, F2) are individually sufficient to require revision. The major issues (M1–M5) collectively weaken the introduction's authority. I am optimistic that all issues are addressable: the GRPO/GDPO discussion can be reframed around established continuous-control RL algorithms (PPO, SAC) with GRPO presented as an *experimental* alternative rather than the culmination of the RL lineage; the DeepWaveRL uncertainty can be resolved or the priority claim qualified; and the conceptual claims can be tempered.

---

### 6. REVISION SUGGESTIONS

1. **Restructure the RL narrative (addresses F1, M5).** Replace the current lineage (Policy Gradient → TRPO → PPO → GRPO → GDPO) with one grounded in continuous control: Policy Gradient Theorem → TRPO → PPO → SAC. Introduce GRPO/GDPO in a separate sentence as "recent group-relative advantage formulations developed in the LLM alignment literature that we explore as an alternative to value-function-based methods." This separates the established foundation from the experimental exploration and is more honest to the RL literature.

2. **Correct the GDPO citation (addresses F1).** Either provide the correct reference for GDPO or remove references to GDPO as a distinct named algorithm. If GDPO is the authors' own generalization of GRPO (not yet published elsewhere), state this explicitly and describe it in the methods section.

3. **Resolve or qualify the DeepWaveRL uncertainty (addresses F2).** Either (a) locate the definitive reference and characterize DeepWaveRL's scope precisely, explaining how the present work differs, or (b) replace "No existing study has applied..." with "To the best of our knowledge, no study has applied..." and move the DeepWaveRL discussion to a "related work" paragraph with appropriate caveats.

4. **Temper the "natural mapping" claim (addresses M1).** Replace "maps naturally onto the RL formalism" with "can be formulated within the RL framework" and briefly enumerate the key design choices (stochastic policy over velocity perturbations, physics-based reward, forward solver as environment). This is more precise and invites rather than forecloses debate.

5. **Refine the progressive-reward / frequency-continuation analogy (addresses M2).** Instead of "in the same way," write "This progressive reward curriculum is conceptually analogous to the frequency-continuation strategy of multi-scale FWI [5], in that a well-behaved, globally informative signal (travel-time misfit / low-frequency data) constrains the long-wavelength structure before a more sensitive but nonlinear signal (waveform misfit / high-frequency data) refines the details. However, unlike frequency continuation, our curriculum operates through the reward function and does not require modifying or filtering the observed seismic data."

6. **Verify and contextualize the O(10^4) core-hours figure (addresses M3).** Check whether [7] characterizes this as per-iteration or per-run, and update the text accordingly. Consider adding a modern reference on computational cost.

7. **Expand the DL coverage (addresses M4).** Add brief mentions of PINN-based FWI and loop-unrolled methods to demonstrate awareness of the broader DL-for-inversion landscape. This bibliography exists in the research file — use it.

8. **Define the state space (addresses m4).** Add a sentence clarifying what the policy observes (e.g., "The policy receives as input the current velocity model and a representation of the seismic data residual, forming the state of the Markov decision process").

9. **Cite Appendix A in the main text (addresses m3).** When stating the literature search results, add a parenthetical "(see Appendix A for search methodology)."

10. **Temper contribution (4) (addresses m5).** Change "we show that our RL-based approach can produce" to "we demonstrate that our RL-based approach produces" or "we find that our approach yields."

---

## PART 2 — INLINE ANNOTATIONS

Quoted passages are verbatim from cited.md. Weakness IDs reference Section 3 above.

---

### Paragraph 1 (line 3)

> "The method has become indispensable in applications ranging from hydrocarbon reservoir characterization and carbon capture monitoring to crustal-scale tectonic imaging [7]."

**Annotation [m1]:** "Carbon capture monitoring" is an application that emerged predominantly after the 2009 publication of [7]. The reference supports "subsurface monitoring" generically but does not specifically address CCS. Either replace with "subsurface monitoring" (which maps cleanly to [7]) or provide a post-2015 citation that establishes CCS monitoring as an established FWI application.

---

### Paragraph 2 (line 5)

> "...a phenomenon known as cycle skipping [7, 8]."

**Annotation [m2]:** Reference [7] (Virieux & Operto, 2009) is the authoritative source for the definition of cycle skipping. Reference [8] (Warner et al., 2013) primarily contributes Adaptive Waveform Inversion as a *remedy* for cycle skipping. The joint citation is not incorrect but dilutes the attribution. Consider citing only [7] here and reserving [8] for "adaptive waveform criteria" later in the same sentence.

---

> "...a single 3D elastic FWI run can demand hundreds to thousands of wave-equation solves, with per-iteration costs reaching O(10^4) core-hours [7]."

**Annotation [M3]:** O(10^4) core-hours *per iteration* is an extreme upper bound and may represent total-run rather than per-iteration cost in the source. Even at thousands of shots with a 3D elastic solver, this figure would imply approximately 10,000 cores × 1 hour per iteration, which strains credibility for a single gradient step. The authors should verify the precise characterization in [7] and adjust the text accordingly.

---

### Paragraph 3 (line 7)

> "End-to-end methods such as InversionNet [15], VelocityGAN [14], and the pioneering work of Araya-Polo et al. [13] train deep convolutional networks to directly map seismic gathers to velocity models."

**Annotation [M4]:** This is a fair characterization of end-to-end methods, but the paragraph would benefit from a brief mention of other DL paradigms that also claim to address the physics-violation concern: PINN-based FWI and loop-unrolled methods. The absence of these approaches weakens the claim that the introduction has comprehensively surveyed the DL-for-inversion landscape and makes the gap for RL appear less carefully substantiated.

---

### Paragraph 4 (lines 9–10)

> "Over the past decade, policy gradient methods have advanced from the foundational policy gradient theorem [22] through trust-region approaches (TRPO) [23] to Proximal Policy Optimization (PPO) [24], which has become the de facto standard for continuous control tasks..."

**Annotation [M5]:** This sentence correctly characterizes PPO as the de facto standard for continuous control. However, the omission of SAC (Soft Actor-Critic, Haarnoja et al., 2018) — which is widely regarded as state-of-the-art for continuous control alongside PPO and explicitly addresses exploration via maximum-entropy objectives — is a notable gap. SAC's entropy-regularized exploration is directly relevant to the claim in point (ii) that "stochastic policies can explore the model space probabilistically."

---

> "More recently, Group Relative Policy Optimization (GRPO) [25] and its generalization to Group Direct Policy Optimization (GDPO) [26] have eliminated the need for a learned value function by computing advantages relative to the mean reward within a batch of sampled trajectories—a formulation particularly well-suited to problems where comparing multiple candidate solutions against a physics-based objective is more natural than learning a scalar value function over high-dimensional state spaces [25, 26]."

**Annotation [F1] — CRITICAL.** This passage contains multiple problems:

(1) **GRPO was developed for LLM fine-tuning, not continuous control.** Reference [25] (Shao et al., 2024, DeepSeekMath) introduces GRPO for training LLMs on mathematical reasoning tasks with discrete token-level actions. No evidence is cited — and to the reviewer's knowledge, none exists — that GRPO is effective for continuous control or PDE-constrained optimization. Presenting GRPO as a natural successor to PPO in the continuous-control lineage is misleading.

(2) **Reference [26] does not describe GDPO.** Meng et al. (2024) is SimPO (Simple Preference Optimization), a DPO variant for LLM alignment. It does not introduce "Group Direct Policy Optimization." This citation is factually incorrect. If GDPO exists as a distinct algorithm, the correct reference must be provided. If it does not, the "generalization" claim must be removed.

(3) **The claim of "particularly well-suited" is unsupported.** Neither [25] nor [26] discusses physics-based objectives, PDE-constrained optimization, or seismic inversion. The "well-suited" assertion is the authors' speculation and must be labeled as such, not presented as a finding of the cited work.

(4) **The citation markers [25, 26] appended at the end of the sentence** imply that both references substantiate the "well-suited" claim. They do not.

---

> "The key properties that make DRL compelling for seismic inversion are: (i) the forward model is treated as a black-box environment with no differentiability requirement, (ii) stochastic policies can explore the model space probabilistically, potentially escaping the local minima that trap deterministic gradient methods, and (iii) reward functions can encode arbitrary physics-based objectives without needing ground-truth models..."

**Annotation:** Points (i) and (iii) are well-stated and represent genuine advantages. Point (ii) is plausible but should be tempered: the claim that stochastic policy exploration "potentially" escapes local minima is a hypothesis to be tested, not an established property. Gradient-based methods with stochastic perturbations (e.g., stochastic gradient Langevin dynamics) also explore probabilistically. The authors should acknowledge that stochasticity alone does not guarantee escape from local minima — the specific exploration mechanism matters.

---

### Paragraph 5 (line 11)

> "Yet, a systematic review of the literature reveals a striking gap: reinforcement learning has been almost entirely absent from seismic inversion research."

**Annotation [m3]:** "Systematic review" is a strong methodological claim. The Appendix A search methodology describes keyword searches on Google Scholar — a reasonable approach, but not a systematic review by the standards of PRISMA or similar frameworks. Consider replacing "systematic review" with "targeted literature search" (the phrase used in Appendix A itself).

---

> "A targeted literature search combining 'reinforcement learning' with 'seismic inversion' returns fewer than 50 publications [Appendix A], and among these, DeepWaveRL stands as the sole work that tangentially addresses RL for geophysical inversion [Appendix A]."

**Annotation [F2]:** See detailed discussion under F2 in Part 1. The DeepWaveRL reference is flagged as uncertain ("if it exists; literature on this is sparse") in the research file. The introduction presents it as a concrete, known prior work. The authors must either (a) provide a definitive bibliographic reference for DeepWaveRL, characterizing its scope precisely, or (b) qualify the claim and acknowledge the uncertainty. An absolute priority claim cannot rest on an unverifiable negative.

---

> "This gap is particularly surprising given that seismic inversion—a sequential decision process of iteratively adjusting a subsurface model to better explain observed data—maps naturally onto the RL formalism: the subsurface model parameterization defines the action space, the forward simulator is the environment, and the misfit between simulated and observed data defines the reward."

**Annotation [M1]:** "Maps naturally" is an overstatement. Classical FWI is a deterministic gradient-based optimization, not a sequential decision process under uncertainty. The mapping to RL is a *design choice* by the authors, not an intrinsic property of seismic inversion. Furthermore, the state space is not defined — what does the agent observe? A complete RL formalism requires (S, A, P, R), not just (A, environment, R). See M1 and m4 in Part 1.

---

### Paragraph 6 (lines 15–16)

> "This progressive reward curriculum stabilizes training in the same way that low-to-high frequency multi-scale inversion stabilizes classical FWI [5], but it is implemented entirely through the reward function without modifying the observed data, and it is compatible with black-box forward simulators."

**Annotation [M2]:** "In the same way" is too strong. Multi-scale FWI uses the same physical quantity (waveform misfit) evaluated at different frequency bands; the proposed method switches between qualitatively different physical quantities (travel-time misfit vs. waveform misfit). These are different mechanisms, and the analogy should be presented as conceptual rather than mechanistic. See detailed discussion under M2 in Part 1.

---

### Paragraph 7 (line 17)

> "(4) Through numerical experiments on synthetic and benchmark datasets, we show that our RL-based approach can produce high-quality velocity models competitive with traditional FWI while removing the requirement for differentiable forward solvers."

**Annotation [m5]:** The assertive phrasing "we show that our approach can produce" (present tense, outcome stated as fact) is stylistically forward for an introduction. Consider "we demonstrate that our approach produces" or "we find that our approach yields velocity models competitive with..." The current phrasing reads as a certainty rather than a finding to be evaluated by the reader.

---

### Appendix A (line 31)

> "Key observation: the query 'reinforcement learning' 'seismic' 'inversion' returned fewer than 50 results, and only DeepWaveRL is directly relevant."

**Annotation [F2, continued]:** The specificity "fewer than 50 results" is potentially misleading without temporal bounds. A Google Scholar search without date restriction for 'reinforcement learning' 'seismic' 'inversion' conducted in May 2026 returns results that may include papers from 2020–2026 that the authors might have missed. The search should specify the date of the query and ideally be supplemented with searches on Scopus, Web of Science, or SEG Digital Library for completeness.

---

### Removed Unsupported Claims (lines 44–48)

The flagged items in the cited report are correctly identified. In addition to the CCS monitoring issue (already annotated at Paragraph 1), I note:

**Flag 3 (Citation [8] for cycle skipping):** Addressed under m2 above. Acceptable as-is but [7] alone is preferred for the definitional claim.

---

## SUMMARY TABLE OF ISSUES

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| F1 | FATAL | Para 4, lines 9–10 | GRPO/GDPO citations do not support claimed properties; RL lineage skewed |
| F2 | FATAL | Para 5, line 11 | "No existing study" claim undermined by DeepWaveRL uncertainty |
| M1 | MAJOR | Para 5, line 11 | "Maps naturally onto RL formalism" is an overstatement |
| M2 | MAJOR | Para 6, lines 15–16 | Frequency-continuation analogy is imprecise |
| M3 | MAJOR | Para 2, line 5 | O(10^4) core-hours/iteration figure needs verification |
| M4 | MAJOR | Para 3 | Missing PINN / loop-unrolled / generative DL approaches |
| M5 | MAJOR | Para 4, lines 9–10 | SAC and continuous-control RL 2017–2024 omitted |
| m1 | MINOR | Para 1, line 3 | CCS monitoring anachronism vs. [7] |
| m2 | MINOR | Para 2, line 5 | Citation [8] for cycle skipping definition |
| m3 | MINOR | Para 5, line 11 | "Systematic review" claim and unqualified search results |
| m4 | MINOR | Para 5, line 11 | State space undefined in RL formalism mapping |
| m5 | MINOR | Para 7, line 17 | Contribution (4) phraseology |

---

*End of review.*
