# Proposed next direction after closing the compression proposal

Status: recommendation and research-design brief, 16 October 2026. No new dataset download, API experiment, implementation, novelty validation, or submission has been performed for this direction.

## Decision and user constraint

The KuaiRand count-sharing proposal has failed its agreed diagnostic gates. Preserve those records and close that proposal. The user reports that only they are available to review outputs. A new qualitative reference built by two independent coders is therefore unavailable.

Recommended adaptation: investigate order-dependent loss of human-labeled key points in bounded argument memory used by an agent for Web opinion analysis. Use existing human annotations as the primary reference and the user's blinded inspection as a limited audit. This is a candidate question, not a certified novel contribution.

## Exact candidate question

When an agent processes the same opinion arguments in different batch histories under the same memory budget, does its retained evidence lose supported key points beyond ordinary repeated-run variation, and does the effect remain after comparison with simple memory-selection baselines?

The initial object being scored is a retained set of original argument IDs/texts, not a free-form summary. Existing argument-to-key-point labels can measure whether a supported key point has representative evidence in memory. Retaining evidence does not establish that a later generated summary communicates the viewpoint correctly.

## Human-grounded source

The IBM KPA-2021 shared-task repository describes expert-written key points and crowd annotations relating arguments to key points, including matches, nonmatches and unresolved cases. Arguments may match several key points or none. These are crowd-elicited opinion arguments, not a natural chronological social-media stream; any batch ordering is a controlled intervention.

- Official repository: https://github.com/IBM/KPA_2021_shared_task
- Task and annotation protocol: https://aclanthology.org/2021.argmining-1.16/

Before using data, pin its repository revision, inspect the actual license and split files, count usable positive/negative/ambiguous labels, and verify that proposed corpus units have sufficient support. Do not silently treat unresolved or absent labels as negative. Keep the official evaluation topics reserved and use development topics for the pilot.

## Why this is not automatically novel

- LLooM already revisits uncovered and overly generic examples. Adding a revisit pass is not sufficient novelty: https://doi.org/10.1145/3613904.3642830
- DeliberationBank already studies minority-opinion representation and discusses input order: https://arxiv.org/abs/2510.05154
- Codebook Reduction and Saturation explicitly considers sequence/repetition variability: https://arxiv.org/abs/2503.04859
- Streaming diversity selection, representative sampling, and key-point analysis are established neighboring problems. The exact memory-history question still needs a targeted closest-work screen.

The possible contribution must be an important measured failure that survives these controls, a precise account of where it occurs, and an effective correction beyond simple baselines. Neither an abstract mechanism nor a favorable pilot certifies this.

## Next concrete deliverable

A short novelty-and-data gate, before implementing an agent or launching API jobs:

1. Closest-paper table: what each paper already tests, exact candidate distinction, and whether an accessible baseline can be reproduced.
2. Dataset contract: pinned release, license, topic/stance units, usable human-label coverage, and untouched evaluation split.
3. Frozen pilot specification: primary metric, memory budget, ordering manipulation, repeat controls, candidate baselines, uncertainty unit and stop rule.
4. User audit pack: a small blinded sample of argument/label cases and retained evidence states. One reviewer can supply a documented audit, but cannot establish inter-rater agreement.

## Smallest informative pilot, conditional on the gate

Use three development corpus units with roughly 100-150 arguments each if actual data support permits. Preserve the complete argument multiset within each paired intervention. Use four prespecified order/layout conditions and three repeated executions per condition with one model. Freeze all settings before reserved topics.

Controls: unchanged-order repeats; simple recency and uniform reservoir memory; an embedding-diversity selection baseline if accessible and justified; and a direct/global view when the full input fits, labeled as having a different information-access pattern. An LLooM-style residual-coverage comparator is required before asserting a new repair advantage.

The inference system must not see human key-point labels or the oracle placement rule. Report actual memory and exposure budgets; repeated source text and extra calls count. A label-informed selection oracle may be reported only as an explicitly labeled upper bound.

Primary component metric: fraction of supported human key points with at least one positively matched argument retained in memory, together with support-weighted coverage and matched-argument mass. Validate that evidence IDs refer to intact retained text. Minority definitions must be fixed from development labels, not selected after results. Report unmatched/ambiguous records separately. Do not claim these component metrics establish final-summary semantic quality.

Continue only if between-history differences are larger than within-history stochastic variation, repeat across development corpus units, and survive simple baselines. If simple memory selection explains or removes the effect, do not manufacture a complex controller. A successful pilot still requires independent evaluation topics, another model, a novelty check, and a useful correction before paper claims.

## Resource and submission position

Preserve the available Kaggle and NVIDIA resources until the data/novelty gate and a measured endpoint-throughput check are complete. API keys are access credentials, not verified independent throughput or unlimited capacity.

Planning recommendation: target the November WWW 2027 short-paper schedule if a sound contribution emerges. The official dates list abstracts 9 November and papers 16 November, 23:59 AoE, corresponding to 10 and 17 November at 17:29 IST. This is a planning recommendation, not a readiness or acceptance claim.

Official dates: https://www2027.thewebconf.org/important-dates/
