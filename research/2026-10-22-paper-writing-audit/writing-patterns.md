# Writing a Web Conference paper: synthesis from the supplied corpus

Date: 2026-10-22. This report synthesizes close readings of 23 distinct supplied works (26 files, 304 distinct-file PDF pages including references and appendices). It studies argument construction and evidence presentation. It is not a systematic sample of the venue, a comparison of acceptance probabilities, or a new literature novelty audit. Some supplied versions are preprints, a technical report, or a NeurIPS paper.

Confidence convention: [Certain] = directly inspected source fact or first-principles implication. [Likely] = interpretive synthesis/recommendation. Claims of performance remain claims reported by the papers; no experimental results were reproduced. Reading coverage is documented in the four companion reports and manifest.

Companion analyses:

- [Recommendation, compression, and explanations: 5 papers](D:/web_2/research/2026-10-22-paper-writing-audit/core-recommendation-papers.md)
- [Retrieval, ranking, and sequential recommendation: 7 papers](D:/web_2/research/2026-10-22-paper-writing-audit/retrieval-papers.md)
- [Safety, prediction, fairness, and robustness: 6 papers](D:/web_2/research/2026-10-22-paper-writing-audit/safety-fairness-papers.md)
- [Social measurement, accessibility, and human evaluation: 5 papers](D:/web_2/research/2026-10-22-paper-writing-audit/social-measurement-papers.md)

## 1. The common pattern is an argument with obligations

[Likely] A useful paper does more than fill Introduction, Method, and Experiments slots. It creates a chain in which each important claim incurs a specific obligation, and subsequent sections discharge that obligation:

    Web setting and consequence
        -> precise unresolved question or failure
        -> reason the obvious approaches are insufficient
        -> insight, measurement, or design principle
        -> contribution derived from that principle
        -> evidence matched to the contribution
        -> bounded finding and implication.

[Certain] The supplied papers implement this chain in different ways. SPRec puts empirical/theoretical diagnosis ahead of the method. G-Refer explicitly names C1/C2 and assigns components to them. GraphHash separates effectiveness from method analysis. The accessibility paper begins with participant evidence. Government Bias places the dataset and labeling construct at the center. The crisis paper organizes results by evaluation question. None of these facts establishes a compulsory section order.

[Likely] The most useful planning question for every section is: **Which unresolved reader objection does this section answer?** An equation that does not justify a design, an ablation that does not test a claim, or a figure whose interpretation never affects the argument may be expendable.

## 2. Five narrative families, with hybrids

This classification is interpretive rather than a claim about official tracks.

| Family | Main uncertainty resolved | Examples | Middle of the paper |
|---|---|---|---|
| Diagnosis followed by intervention | Why does a reasonable existing practice fail? | SPRec; PEAR; parts of BACO | Phenomenon → explanation/hypothesis → intervention → targeted tests |
| Representation or pipeline design | How should available information enter a constrained system? | GraphHash; G-Refer; HtmlRAG; ED2; ImageScope | Task → representation/design → components → performance and component tests |
| Constrained optimization or systems trade-off | Can quality be retained while reducing a concrete cost? | CCE; TourRank; MA4DIV; BACO | Constraint → reformulation → algorithm → quality/cost frontier |
| Measurement and empirical explanation | What is happening, for whom, and under what conditions? | Behavioral Homophily; Hate Spreaders; Crisis Robustness | Construct/RQs → sampling and measurement → findings → alternative explanations/limits |
| Dataset, operationalization, or human-centered validation | Is the current definition or benchmark missing something essential? | Inclusive Mobile Web; Government Bias; Gender Fairness; Counterspeech; Cross-Modal Transfer | Missing construct → definition/data or study design → validity checks → downstream evaluation |

[Likely] These are not mutually exclusive. MIMoE uses a taxonomy to motivate a prediction architecture; Gender Fairness moves from a measurement blind spot to a regularizer; Counterspeech studies both generation strategies and evaluation validity. Choose the family matching the central new knowledge, rather than the paper's most complicated component.

## 3. The exact paper-by-paper writing move to learn

| Work | Distinctive move | What your version would need |
|---|---|---|
| SPRec | Turns an improvement (DPO) into the source of a new diagnosed failure | Direct evidence of the failure and tests of the proposed mechanism |
| GraphHash | Replaces random sharing with graph-informed sharing, then explains interactions | A controlled resource comparison and analysis beyond an aggregate win |
| G-Refer | Names two challenges and maps modules to them | Challenge→component→ablation traceability |
| BACO | Converts two structural observations into an objective and solver | Correct algebra, fair baseline variants, and separate cost accounting |
| CCE | Bridges two useful method families and explains why combining them is hard | A real obstacle, a mechanism overcoming it, and separated theory/practice claims |
| HtmlRAG | A better representation creates a new length/noise problem, which motivates the system | Evidence for both representation benefit and processing necessity |
| ImageScope | Unifies distinct input tasks through a shared representation and verification stages | Cross-task coverage plus per-stage and cost analysis |
| LLM4Rerank | Makes multiple reranking objectives explicit and traceable through decisions | Separate objective metrics and evidence for claimed personalization |
| MA4DIV | Changes a sequential ranking formulation into cooperative document decisions | Evidence for both ranking quality and computational consequences |
| PEAR | Identifies a targeted internal behavior, modifies it, and checks preserved capability | Intervention-specific evidence, overhead definition, and non-target checks |
| TourRank | Converts input-length, order, and cost challenges into testable design requirements | One experiment family corresponding to each requirement |
| ED2 | Presents coupled information failures, then tests indexing and user components separately | Clear coupling mechanism and ablations that isolate its parts |
| Responsible Diffusion | Evaluates harmful-content reduction together with ordinary-generation preservation | A measured safety–utility trade-off, not only a harm-count reduction |
| IC-Mamba | Uses the observation process and prediction horizon to structure the task | Temporal data boundaries and results for each claimed observation window |
| Cross-Modal Transfer | Diagnoses incompatible labels before evaluating transfer | Label harmonization evidence; distinct substitution and augmentation comparisons |
| MIMoE-FND | Builds a modality-interaction taxonomy, then routes by it | Tests of the taxonomy/gating mechanism and a valid selection protocol |
| Gender Fairness | Shows that aggregate metrics miss category-level disparities | A justified construct, discriminating metric, and utility–fairness evaluation |
| Crisis Robustness | Uses measurement slices, not a new architecture, as the contribution | Clear sampling, tasks, model versions, and slice-level uncertainty |
| Behavioral Homophily | Defines behavioral similarity separately from topic similarity | Validation of the operationalization and evidence of what it adds |
| Inclusive Mobile Web | Lets participant difficulties determine task and dataset definitions | Traceability from observed need to labels, model, and evaluation |
| Hate Spreaders | Studies heterogeneity in associations rather than only prediction accuracy | Confounding assumptions, subgroup estimates, and bounded causal language |
| Government Bias | Establishes a domain-specific labeling resource and tests lexical generalization | Annotation validity and an evaluation split that probes shortcut learning |
| Counterspeech | Treats disagreement between automatic and human evaluation as a finding | A defensible human protocol and a clear account of the outcome actually rated |

## 4. The abstract should expose the research decision

[Likely] A method abstract can often use six functions: setting; precise limitation; insight; proposed mechanism; evaluation; principal result with scope. A measurement abstract substitutes the measurement design and main finding for the proposed mechanism. These are rhetorical functions, not a mandatory sentence count.

Template for a method:

> In [Web setting], [task] must satisfy [meaningful constraint]. Existing [closest approach] addresses [part of the problem] but fails to [specific capability] under [condition]. We identify [mechanistic or structural insight] and introduce [contribution], which [core operation]. We evaluate [comparisons] on [data and relevant conditions]. Results show [measured effect against a named reference] at [cost], with [material boundary if needed]. This establishes [narrow implication].

Template for a diagnosis or measurement study:

> Existing work assumes/measures [X], leaving [specific question] unresolved. We study this question in [Web setting] using [data, measurement, and controls]. We find [principal pattern], distinguish it from [alternative explanation], and show [heterogeneity or intervention result]. The finding holds within [scope] and implies [consequence for evaluation or system design].

[Certain] The corpus gives reasons to keep the result qualified. SPRec's Goodreads accuracy does not lead the table. GraphHash's headline gain is against other compressed methods, not the Full model. CCE's main theoretical guarantee has a linear-model setting. [Likely] Do not let the abstract combine the largest improvement, the cheapest regime, and the broadest applicability into a single sentence if no one experiment establishes that combination.

## 5. Introduction: a sequence of decisions, not a background essay

[Likely] A strong first draft should contain the following moves, merged or split according to the paper:

1. **Locate the problem.** Name the Web system, the task, and the consequence. Avoid spending a paragraph on generic model popularity.
2. **Give the closest existing answer.** Explain what it already solves; this prevents a straw-man gap.
3. **Identify the unresolved condition.** State exactly when or why the answer is incomplete. Use a motivating observation, example, measurement, or derivation when necessary.
4. **Explain the obstacle.** Why does the obvious fix not suffice? CCE's permanently merged IDs and HtmlRAG's token/noise problem make this concrete.
5. **State the insight.** Describe a relationship or design principle, not a list of technologies.
6. **Derive the contribution.** Explain why each major part follows from the obstacle or insight.
7. **Preview bounded findings.** State what was learned and define the comparison. End with contributions that are supported by identifiable evidence later in the paper.

An introduction contribution bullet should name new knowledge or capability. 'We run extensive experiments' is an activity; 'we establish that the effect persists under X but fails under Y' is a finding. A method bullet should state what the design enables, not just give its acronym.

## 6. Related Work should make the comparison precise

[Certain] The corpus varies the placement of Related Work. Its function remains locating the proposed work relative to the closest alternatives. CCE uses a common matrix representation to compare methods; GraphHash groups literature by hashing, clustering, and graph use; G-Refer contrasts implicit embeddings with explicit retrieved evidence.

[Likely] Organize related work along the distinction your contribution depends on: information available, timing of an operation, representation, objective, assumption, or evaluation regime. For each group write: what the group solves; what assumption/mechanism it shares; why that leaves your question open. Merely saying every predecessor is limited does not establish novelty.

Do not infer a present-day novelty claim from a paper's historical 'first' statement. This review analyzed the supplied works as writing examples; it did not search every later paper.

## 7. Formulation and method must make the information flow auditable

[Likely] A formulation should identify inputs and outputs, unit of observation, train/validation/test boundaries, objective or estimand, assumptions, resource constraints, and what information is actually available when the system acts. Include only preliminaries needed for your argument.

For each method component, use this order:

    Purpose -> available inputs -> operation/equation -> output
            -> how the output enters the next stage -> cost/assumption.

[Certain] G-Refer repeats local motivation before major components. CCE provides a lookup routine and a separate clustering routine. BACO distinguishes assignments from codebook parameters. [Likely] Those choices help a reader understand execution, not merely architecture names.

When adding theory, specify the statement's domain before its consequence. A guarantee for a dense linear surrogate is not a guarantee for a nonlinear practical model. A random-walk analogy is not equality of learned representations. A lemma's existence is not evidence that an omitted algebraic step is valid. The narrow BACO check in the companion report illustrates the last point without claiming a complete proof audit.

## 8. Experiments should answer the contribution's questions

[Likely] Build the following table for your own paper before producing figures:

| Claim | Necessary comparison | Controls or definition | Result location | Remaining alternative |
|---|---|---|---|---|
| The phenomenon exists | Original system versus relevant condition | Fixed data/evaluation; show frequency/severity | Diagnostic figure | Noise or confounding |
| The explanation is plausible | Intervention on proposed mechanism | Preserve other inputs and budget | Mechanism test | Another coupled factor |
| The method improves quality | Closest strong baseline | Same permitted data and fair tuning | Main table | Extra compute or supervision |
| A component matters | Removal or meaningful replacement | Comparable training and capacity where relevant | Ablation | Optimization/capacity change |
| It is efficient | Quality at comparable cost and/or cost at comparable quality | Explicit denominator and included stages | Frontier plot | Hidden preprocessing/storage |
| It is robust | Deliberate perturbation or held-out regime | Define the variation and outcome | Robustness analysis | Untested shifts |
| People benefit | Study of the claimed human outcome | Population, sample, tasks, randomization/analysis | Human results | Perceived quality versus actual behavior |

[Certain] An ablation can show that removing X changes performance; that alone does not uniquely establish why X helps. Likewise, preferences between explanations do not establish causally faithful reasoning. [Likely] Use alternative explanations to choose additional contrasts instead of automatically adding a larger benchmark list.

Test-set model selection appears explicitly in two supplied artifacts, Cross-Modal Transfer (§4.1.3) and MIMoE-FND (Appendix A.2). [Certain] Selecting a maximum over test scores makes that test set part of the selection procedure; it is no longer an untouched final assessment. [Likely] For your work, select models and hyperparameters on validation data and reserve test results for the fixed final comparison. This recommendation follows the independence requirement of holdout evaluation, not the publication status of an example.

## 9. Figure and table roles are contribution-dependent

[Likely] Choose the smallest set that resolves the main uncertainties:

| Artifact | Question answered | Corpus examples |
|---|---|---|
| Concrete motivating example | What exactly goes wrong? | SPRec movie repetition; HtmlRAG lost page structure |
| Diagnostic measurement | Is the motivating problem observed? | SPRec popularity distribution; BACO clustering statistics |
| System/algorithm overview | How does information move? | G-Refer pipeline; ImageScope stages; CCE iterative cycle |
| Main comparison table | What improves under the stated protocol? | GraphHash; G-Refer; MIMoE |
| Component contrast | Which design decision contributes? | ED2 indexing/user studies; SPRec negatives; MIMoE routing |
| Cost–quality plot | What is the operating trade-off? | CCE parameter curves; TourRank cost/latency analysis |
| Subgroup/condition plot | Where is the aggregate misleading? | GraphHash users; Crisis event/information slices; Gender categories |
| Human/qualitative evidence | How does the outcome look to the relevant people? | Counterspeech study; accessibility study; G-Refer preference/cases |

A caption should identify the setting and comparison, define axes/metrics and units, explain uncertainty bars where present, and state the principal supported observation. Label a diagram as conceptual when it is not measured data. Mark synthetic demonstrations separately from real-data results.

## 10. Results paragraphs should make a bounded inference

[Likely] Use this five-part structure:

> To test [question], we compare [conditions], holding [controls] fixed. [Table/Figure] shows [observation, including scale and uncertainty where measured]. This supports [narrow interpretation]. The result differs for [exception or subgroup]. Thus the evidence establishes [conclusion], while leaving [unmeasured claim] unresolved.

Do not narrate every cell. Explain why the comparison matters, which rival explanation it weakens, and how an exception changes the claim. Keep statements of statistical significance attached to an actual statistical procedure. Distinguish percentage points from relative percentage changes, total memory from parameter storage, preprocessing time from end-to-end latency, and perceived usefulness from measured behavioral impact.

## 11. Limitations should alter the reader's interpretation

[Likely] A useful limitation identifies a boundary and its consequence: one fixed backbone restricts generalization; a transductive split excludes unseen-entity behavior; an expert panel measures expert preference; a snapshot dataset does not test drift; a detector operationalizes only part of harmful output. Generic 'more datasets in future' language is less useful unless the missing regime is specified.

[Certain] CCE discusses training-budget differences and failed variants. SPRec acknowledges SFT bias and long-horizon alignment as unfinished. G-Refer discusses sparse-graph limits and future training-free retrieval. The social-study papers explicitly discuss measurement and population limits. [Likely] Put a limitation in the main result discussion when it changes the headline conclusion; do not rely on an appendix to reverse an overbroad abstract.

## 12. A practical drafting sequence

[Likely] Write in this order, even if the final paper has a different reading order:

1. Define one central research question and the scope of its answer.
2. Make a claim-to-evidence ledger; remove or label claims lacking support.
3. Choose the narrative family matching the central contribution.
4. Draft captions and result takeaways from verified artifacts.
5. Draft formulation, protocol, and method so those results are interpretable.
6. Draft the introduction as the reason that this evidence was needed.
7. Write the abstract and title to match the result's actual scope.
8. Audit every broad noun/adjective: efficient relative to what; fair by which definition; robust to which variation; interpretable for whom; general across what.

For a diagnosis paper, a defensible skeleton is Introduction → Related Work → Setting and Protocol → Observed Failure → Mechanism and Discriminating Tests → Intervention, Benefits, and Costs → Limitations → Conclusion. For a new-method paper, use Introduction → Related Work → Problem → Method → Evaluation → Limitations → Conclusion. For a measurement/resource paper, make construct validity and dataset/study design central sections.

The paper's first page must justify why the question belongs to the Web. For current formatting, consult the [official 2027 research-track instructions](https://www2027.thewebconf.org/research-track-papers/), checked earlier in this conversation. The long-paper call specifies eight main pages, twelve total, anonymous double-column ACM format. These requirements are separate from the observed writing patterns and should be rechecked if the target year or track changes.
