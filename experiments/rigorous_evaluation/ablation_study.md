# NexusCTI Component Ablation

## Reproduce

From the project root, run:

```powershell
python experiments\rigorous_evaluation\run_ablation_study.py
```

The runner writes the machine-readable predictions and summary to
[`results/ablation_study_results.json`](./results/ablation_study_results.json).
It uses the same built-in 12-case benchmark as the existing evaluator and
does not run predictions on the independent test set. GNN initialization is
seeded with `42` and the seed is recorded in the results file.

## Compared systems

- **Vector RAG:** existing dense retriever and actor-score aggregation.
- **BM25 Retrieval:** BM25-only retrieval; retrieved chunks are grouped by their
  `target_actor` labels and their scores are aggregated.
- **GNN:** existing standalone GNN attribution model.
- **Knowledge Graph Reasoning:** actors are ranked by shortest directed path
  from the known anchor IOC in the current graph; shorter paths rank first.
- **Hybrid NexusCTI:** existing fusion engine combining graph inference,
  dense/sparse RAG, and cross-validation.

Every system is scored against the same cases and ground-truth labels. Top-k
accuracy and top-1 error use all 12 cases. MRR is the mean reciprocal rank
(ranks 1–10; absent labels receive zero). **Multi-hop attribution accuracy**
is top-1 actor accuracy among the 11 cases with `expected_hops >= 2`; it does
not require a valid evidence path. This ablation runner does not calculate
multi-hop evidence validity. That separate metric requires a returned trail
with at least two connected edges, each present in the graph with the stated
relationship type and a matching, non-empty citation. Citation-field checks
do not establish that the cited source substantively supports the edge.

## Result

| Component | Top-1 accuracy | Top-3 accuracy | Top-5 accuracy | MRR | Multi-hop attribution accuracy |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 41.67% | 83.33% | 83.33% | 0.5694 | 45.45% |
| BM25 Retrieval | 91.67% | 100.00% | 100.00% | 0.9583 | 90.91% |
| GNN | 16.67% | 50.00% | 91.67% | 0.3931 | 18.18% |
| Knowledge Graph Reasoning | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |
| Hybrid NexusCTI | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |

On this benchmark, graph-only reasoning reaches the same measured ranking
scores as the hybrid, so these results do not show a measurable top-k or MRR
gain from fusion. BM25 also scores highly, but it uses actor labels attached
to the report chunks; it is not an unlabeled document retrieval experiment.

## Shared initialization and isolation scope

The Phase 3 runner calls `get_service()` once and reuses the returned singleton
for all five configuration predictors. Constructing that service builds the
graph, trains the GNN, loads the corpus, and initializes dense and BM25
retrieval indexes before any configuration is scored. The configurations
therefore share one fully initialized service state; they are not separately
initialized or process-isolated runs.

The comparisons select different prediction methods over that shared state.
For example, the vector-only predictor does not call graph/GNN inference, and
the graph-only predictor does not call retrieval, but those components were
still initialized in the process. Results support a comparison of prediction
paths under shared setup, not a claim that excluded components were absent or
unavailable throughout each experiment.

## Interpretation and limitations

Treat these results as a plumbing and component comparison, not evidence of
generalization or causal component contribution. The built-in benchmark's
anchors and expected actor paths overlap the graph, and its reports overlap
the retrieval corpus. The graph-only ceiling therefore reflects known graph
paths. The benchmark has only 12 curated cases, some with synthetic
indicators, and the ablation does not remove components from a fixed trained
hybrid model under controlled retraining. A larger, blind, independently
collected benchmark and repeated training runs would be needed to estimate
uncertainty and isolate marginal gains.
