# NexusCTI Final Evaluation Report

**Evaluation date:** 2026-10-30  
**Scope:** Built-in benchmark, graph evidence checks, source-citation screening,
component ablation, and independent-dataset overlap audit.

## Executive summary

On the built-in 12-case benchmark, the hybrid system achieved 100% top-1
accuracy. Its 32 returned trail edges all existed in the graph with the
requested relationship type and a non-empty citation, and all 11 cases
designated as multi-hop had connected, valid paths. These are checks against
the current graph—not verification that the cited documents substantiate each
edge.

The benchmark has extensive knowledge-source overlap: all 12 anchors already
had graph paths to their expected actors and overlapped both actor-labeled
reports and reports containing their anchors. In the separate 65-edge
citation-catalog audit, only 4 citation labels matched a catalog record;
none resolved to a passage containing both endpoints. No semantic checker was
provided, so semantic support was not evaluated for any edge. Consequently,
the perfect benchmark scores are exploratory and do not establish performance
on unseen incidents or independent source verification.

## Methodology

The built-in evaluation uses 12 curated cases from the NexusCTI benchmark.
Prediction methods receive each case's query and anchor IOC; ground-truth
labels, case descriptions, and expected-hop counts are not passed to the
predictors. The same cases are used to compare dense Vector RAG, the standalone
GNN, and Hybrid NexusCTI.

Reported ranking measures are top-1, top-3, and top-5 accuracy over all cases
and mean reciprocal rank (MRR), with absent actors assigned zero reciprocal
rank. **Multi-hop attribution accuracy** is top-1 actor accuracy for the 11
cases with `expected_hops >= 2`; it does not require a valid evidence path.
**Multi-hop evidence validity** is reported separately: a case passes when
its returned trail has at least two connected edges, each present in the graph
with the stated relationship type and a matching, non-empty citation. The
citation check does not verify the cited source's content. The evaluator's
hallucination rate is a case-level proxy, not a count of independently
verified factual claims.

The five-way ablation uses the same built-in cases: dense vector retrieval,
BM25-only retrieval, GNN, NetworkX unweighted shortest-path distance on an
undirected view of the graph from the known anchor to an actor, and Hybrid
NexusCTI. Its GNN initialization is seeded with 42.
BM25 scores are aggregated using actor labels attached to retrieved report
chunks; graph reasoning has access to the benchmark anchor and existing graph.
The ablation runner obtains the singleton through `get_service()` once and
reuses that fully initialized service across all five prediction methods.
Service construction builds the graph, trains the GNN, and initializes the
retrieval indexes before scoring. The comparisons isolate selected
prediction-time calls only; they are not independently initialized,
process-isolated, or retrained component-removal experiments.

## Built-in benchmark results

Results from `python run_benchmark.py`, saved to
[`artifacts/rigorous_benchmark_results.json`](../artifacts/rigorous_benchmark_results.json):

| System | Top-1 | Top-3 | Top-5 | MRR | Multi-hop attribution accuracy | Multi-hop evidence validity | Hallucination proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector RAG | 41.67% | 83.33% | 83.33% | 0.5694 | 45.45% | N/A (no graph trail) | 58.33% |
| Standalone GNN | 8.33% | 25.00% | 83.33% | 0.3000 | 9.09% | N/A (no returned evidence trail) | 91.67% |
| Hybrid NexusCTI | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% | 100.00% (11/11) | 0.00% |

The GNN is initialized stochastically in the existing benchmark runner; its
score can vary between runs. The separate seeded ablation result is reported
below rather than treated as the same run.

### Historical Phase 2 multi-hop attribution results

The Phase 2 scientific audit summary reports **90.91%** hybrid multi-hop
accuracy, while its detailed benchmark report records **100.00% (11/11)** for
the full hybrid. These are attribution-ranking figures, not evidence-validity
figures: the summary does not provide evidence-path scoring details, and
neither result should be used to infer multi-hop evidence validity. Both
concern the 12-case, closed-world transductive benchmark; the detailed report
defines multi-hop as `expected_hops >= 2` and evaluates 11 cases. The summary
does not provide a run configuration or case-level result that explains its
different figure. These results remain unreconciled and should be cited
separately with their source and conditions, not selected between or
presented as equivalent. The 100.00% table above is the later
`run_benchmark.py` result on the built-in benchmark and does not resolve the
historical discrepancy. All such benchmark scores are exploratory because
the anchors and expected actor paths overlap the graph and retrieval corpus.

## Evidence and citation verification

### Predicted hybrid trails

The current rigorous benchmark returned 32 trail edges across 12 cases:

| Check | Result | Interpretation |
|---|---:|---|
| Evidence validity | 100% (32/32) | Edge and relationship type were present in the current graph |
| Citation coverage | 100% (32/32) | Matching graph edges had a non-empty citation field |
| Multi-hop evidence validity | 100% (11/11) | Required multi-hop paths were connected and passed relationship and citation-field checks; the single-hop case is not applicable |
| Unsupported evidence rate | 0% (0/32) | No trail edge was missing or relation-mismatched in the graph |

Citation coverage here measures **citation presence**, not whether a source
exists or supports its associated relationship. Case 02 is a one-edge,
single-hop case: its graph edge is valid, but multi-hop path evaluation is
`not_applicable`, not a failed path. It is excluded from the multi-hop
denominator.

### Citation-to-source audit

The citation verifier audited the 65 authored graph edges (excluding generated
reverse edges) against the six-report default corpus as its source catalog:

| Audit finding | Count |
|---|---:|
| Citation labels matching a catalog record | 4 / 65 |
| Unique document identities matched | 4 |
| Relevant passages resolved with both endpoints | 0 / 65 |
| Relationships semantically supported | Not evaluated (no semantic checker supplied) |
| Citations classified as verified | 0 / 65 |
| Citations with semantic support not evaluated | 65 / 65 |
| Citation labels reused across multiple relationships | 9 |

The catalog is limited to the default corpus. An unmatched citation is
not found in that catalog, not proof that its source does not exist. Passage
resolution is a retrieval check, not an entailment judgment. The lexical
overlap field is diagnostic only and never establishes semantic support. A
caller must provide a semantic-support checker (or a human review workflow)
before any relationship can be marked semantically supported or verified.

## Component ablation

The seeded ablation results are in
[`experiments/rigorous_evaluation/results/ablation_study_results.json`](../experiments/rigorous_evaluation/results/ablation_study_results.json);
method and interpretation details are in
[`experiments/rigorous_evaluation/ablation_study.md`](../experiments/rigorous_evaluation/ablation_study.md).

| Component | Top-1 | Top-3 | Top-5 | MRR | Multi-hop ranking success |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 41.67% | 83.33% | 83.33% | 0.5694 | 45.45% |
| BM25 Retrieval | 91.67% | 100.00% | 100.00% | 0.9583 | 90.91% |
| GNN (seed 42) | 66.67% | 100.00% | 100.00% | 0.8056 | 63.64% |
| Knowledge Graph Reasoning | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |
| Hybrid NexusCTI | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |

Graph-only reasoning ties the hybrid on this benchmark; no top-k or MRR
improvement from fusion is measurable here. BM25 is also strong, but its
aggregation uses report actor labels. These results do not isolate causal
contributions because the benchmark overlaps both the graph and retrieval
corpus.

## Data separation and leakage

The built-in benchmark leakage audit found:

- **0/12** prediction inputs with a direct actor-label match.
- **12/12** benchmark anchors with a pre-existing graph path to the expected
  actor.
- **12/12** cases overlapping actor-labeled reports and reports containing
  their anchors.
- Actor labels in **10/12** case names and **9/12** descriptions; these
  metadata fields are not sent to predictors.

The independent four-case dataset has separate prediction inputs and
ground-truth files. Its audit found zero actor-label leaks, anchor matches,
expected relationships in the current graph, or expected-relationship text
overlaps in the current default corpus; all four source records were present.
The cases include two known-actor campaigns and two open-set actors. The
campaigns were cataloged in earlier experiment artifacts, so this is held out
from the current graph/index but is **not a newly blinded prospective
dataset**. Open-set cases should be assessed for abstention, not only
closed-set actor-ranking accuracy. No predictions were run on this holdout.
See the audit at
[`experiments/rigorous_evaluation/independent_test/overlap_audit.json`](../experiments/rigorous_evaluation/independent_test/overlap_audit.json)
and the dataset scope in
[`experiments/rigorous_evaluation/independent_test/manifest.json`](../experiments/rigorous_evaluation/independent_test/manifest.json).

### Follow-up frozen open-set evaluation

A separate two-case split was added using CISA AA23-335A and AA23-352A,
covering CyberAv3ngers/Unitronics PLC activity and Play ransomware/CVE-2022-41040.
The default graph and corpus were frozen before prediction:

| Frozen asset | Version | SHA-256 |
|---|---|---|
| Knowledge graph | `nexuscti-default-graph-2026-10-30-v1` | `016a61dddfee7696610495ef5ae17e64da47dfc5d9ce21c987f84d1cbf143910` |
| Retrieval corpus | `nexuscti-default-corpus-2026-10-30-v1` | `5abca5fd18373e62af3414e10438fcba61393342a0402d21cb0c84a56d67068a` |

The split audit found zero actor-label leaks, anchor matches, exact expected
edges in the frozen graph, or expected relationship text overlaps in the
corpus. Both source records were present in the separate source registry.
Predictions were generated from `test_cases.json` only; the prediction artifact
records `ground_truth_loaded: false`. The later, separate scoring step found
**2/2 correct open-set abstentions (100%)**. Closed-set top-1 accuracy is not
applicable because neither actor exists in the six-actor candidate graph. This
small test is a workflow validation, not a statistically reliable estimate or
proof that public reports were never encountered outside this repository.
Details are in [`data_split.json`](../experiments/rigorous_evaluation/data_split.json),
the [overlap audit](../experiments/rigorous_evaluation/independent_test/frozen_overlap_audit.json),
the [blind predictions](../experiments/rigorous_evaluation/independent_test/predictions.json),
and [scored results](../experiments/rigorous_evaluation/independent_test/evaluation_results.json).

## Limitations

1. **Small and curated benchmark:** 12 cases are insufficient for robust
   generalization claims, statistical comparisons, or representative sampling.
   Some indicators and scenarios are synthetic or illustrative.
2. **Train/test knowledge overlap:** Expected actor paths are already in the
   graph, and all benchmark cases overlap labeled or anchor-matching corpus
   reports. The perfect hybrid and graph-only scores should not be read as
   unseen-incident accuracy.
3. **Citation verification is limited:** The default source catalog does not
   match most graph citation strings. Graph citation presence and a lexical
   screen do not prove source provenance or substantive support.
4. **Ablation is exploratory:** It uses a known-anchor graph baseline and a
   BM25 predictor that reads actor labels from chunks, without controlled
   component removal and retraining.
5. **Stochastic GNN:** GNN results differ between the current benchmark run and
   the seeded ablation; repeat runs and uncertainty estimates are absent.
6. **Metric scope:** The evaluator's hallucination rate is a proxy based on
   misattribution/cross-validation, not claim-level fact checking. Reported
   confidence is not shown to be calibrated.
7. **Independent-set limitations:** The additional fresh split has only two
   open-set cases and cannot establish broad generalization; prior exposure to
   public reporting outside this repository cannot be ruled out by the audit.
8. **Attribution uncertainty:** Shared infrastructure, tools, copied
   indicators, planted evidence, and conflicting reporting can make actor
   attribution ambiguous. Results should support analyst investigation, not
   replace corroboration.

## Future work

- Build a larger, source-reviewed, blinded dataset from independently collected
  incidents, with labels and evidence kept outside prediction inputs.
- Evaluate known-actor and open-set cases separately; report abstention and
  calibration alongside ranking metrics.
- Use time-based and entity/relationship-disjoint splits; rerun overlap audits
  against the exact graph, index, and model used for each experiment.
- Resolve citations to stable source identifiers and manually verify whether
  cited passages support each relationship; publish unresolved and conflicting
  evidence separately.
- Run controlled ablations with consistent data access, training procedures,
  and multiple seeds; report per-case results, uncertainty intervals, and
  significance tests where appropriate.
- Measure claim-level support and unsupported-claim rates with a documented
  annotation procedure and independent reviewers.
- Freeze graph, corpus, model configuration, and random seeds per run, and
  record hashes, environment, and run metadata for reproducibility.

## Reproducibility artifacts

- Main benchmark: `python run_benchmark.py`
- Ablation: `python experiments\rigorous_evaluation\run_ablation_study.py`
- Independent overlap audit: `python experiments\rigorous_evaluation\independent_test\validate_independence.py`
- Metric definitions: [`docs/evaluation_metrics.md`](./evaluation_metrics.md)
- Evaluation protocol: [`experiments/rigorous_evaluation/evaluation_protocol.md`](../experiments/rigorous_evaluation/evaluation_protocol.md)
- Known limitations: [`docs/limitations.md`](./limitations.md)
