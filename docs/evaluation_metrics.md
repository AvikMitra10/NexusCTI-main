# Evaluation Metrics

Let `N` be the number of test cases, and let each case have one ground-truth actor and a ranked prediction list. A missing prediction counts as an incorrect actor prediction and has reciprocal rank zero. Report metric counts as well as rates where possible.

## Actor ranking

| Metric | Definition and formula | Denominator |
|---|---|---|
| Top-1 accuracy (actor accuracy) | `100 * cases whose rank-1 actor equals the ground truth / N` | All test cases |
| Top-3 accuracy | `100 * cases whose ground-truth actor appears in the first 3 predictions / N` | All test cases |
| Top-5 accuracy | `100 * cases whose ground-truth actor appears in the first 5 predictions / N` | All test cases |
| Mean Reciprocal Rank (MRR) | `sum(1 / rank_i) / N`, where `rank_i` is the ground-truth actor's first rank; use zero if it is absent | All test cases |

The current evaluator caps reciprocal-rank credit at rank 10: ranks 1–10 receive `1 / rank`, and an absent actor (stored internally as rank 99) receives zero. Since its top-1/top-3/top-5 metrics test only the ground-truth actor's presence in the ranked list, their denominators remain all test cases.

## Evidence and path quality

For these metrics, an **evidence item** is a claimed graph relationship: source node, relation, and target node. The current validator separately checks that the graph contains the edge with the requested relation type and whether that matching relationship has a non-empty citation field. Citation coverage measures citation presence only; it does not independently verify the cited material.

| Metric | Definition and formula | Denominator |
|---|---|---|
| Evidence validity rate | `100 * predicted edges present with the stated relation / all predicted trail edges` | All predicted trail edges in hybrid predictions |
| Citation coverage | `100 * predicted edges with a matching relation and non-empty citation / all predicted trail edges` | All predicted trail edges in hybrid predictions |
| Multi-hop evidence validity | `100 * multi-hop cases whose returned trail has at least 2 connected edges, each present in the graph with the stated relation and a matching non-empty citation / all multi-hop cases` | Test cases designated as requiring multi-hop reasoning |
| Unsupported evidence rate | `100 * predicted trail edges missing or relation-mismatched / all predicted trail edges` | All predicted trail edges in hybrid predictions |

These graph-evidence metrics are calculated for the hybrid model; baseline values are `null` because those baselines do not produce graph trails. Predictions with no trail edges contribute no edge observations; if the aggregate contains no edges, the edge-rate metrics are reported as `0.0`. Disconnected paths, relationship mismatches, and missing or mismatched citations fail multi-hop evidence validity. This evidence metric is independent of actor-ranking correctness: a case can have a valid trail without ranking the ground-truth actor first, or rank the correct actor first without a valid trail. Cases with `expected_hops < 2` are not applicable to multi-hop path evaluation: their per-case `multihop_path_status` is `"not_applicable"` and `multihop_path_valid` is `null`, while their individual edge validity remains reported.

The evaluator's citation check establishes only that a non-empty trail citation matches a citation field on the corresponding graph edge. It does **not** establish that the source exists, that a passage supports the relationship, or that the relationship is true. The separate citation audit in [citation_verifier.py](../nexuscti/eval/citation_verifier.py) reports source-catalog existence, relevant-passage resolution, and semantic support separately. Its lexical overlap screen is diagnostic only; semantic support is `not_evaluated` unless a caller supplies a semantic checker.

## Unsupported claims (hallucination)

For a rigorous evidence-based measure, define an unsupported claim as a factual attribution or evidence claim that is not supported by the validated graph path or retrieved source material:

`Hallucination rate = 100 * unsupported claims / all claims evaluated`

The denominator is all extracted attribution and evidence claims evaluated, not only incorrect actor predictions. Specify the claim-extraction procedure and report the count of evaluated claims. If claims cannot be extracted consistently, use the case-level proxy below and label it as such rather than calling it evidence-based hallucination.

**Existing evaluator proxy:** For the dense and GNN baselines, a case is counted as a hallucination when a non-empty prediction list has a wrong top actor. For the hybrid, a case is counted when a top actor exists and either its actor differs from ground truth or its cross-validation score is below `0.35`. The reported rate divides that count by all test cases; empty prediction lists are not counted as hallucinations. This is a coarse benchmark proxy, not a measurement that checks whether individual claims are unsupported.

## Data leakage warnings

| Metric | Definition and formula | Denominator |
|---|---|---|
| Leakage warning rate | `100 * prediction inputs in which one or more ground-truth actor labels are detected / prediction inputs checked` | All prediction inputs checked |

Also report the number of inputs with warnings and the labels detected. This lexical check is a warning, not proof of leakage: it may flag legitimate discussion and may miss aliases, misspellings, or indirect exposure. Keep ground-truth labels out of prediction inputs regardless of the warning result.

The standalone [`audit_benchmark_leakage`](../nexuscti/eval/leakage_checker.py) additionally compares benchmark query/anchor inputs with graph actor labels, graph paths, and labeled corpus reports. It reports direct input exposure separately from knowledge-source overlap. A pre-existing path or labeled report is an overlap warning, not proof that the evaluator explicitly passed the ground-truth field.

## Benchmark interpretation

`actor_accuracy` duplicates top-1 accuracy for clarity. The existing `multihop_reasoning_success` metric is **Multi-Hop Attribution Accuracy**: correct top-1 actor predictions among cases whose `expected_hops >= 2`. In the built-in benchmark, `expected_hops` now matches the directed shortest-path edge count from anchor to ground-truth actor; `category` denotes scenario archetype, not hop count. The existing `multihop_path_validity` field is **Multi-Hop Evidence Validity**: cases in that same multi-hop subset whose returned trail passes connectivity, relationship, and citation-field checks. They are separate metrics and must not be combined into a single “multi-hop success” figure. The evaluator does not currently calculate lexical leakage warnings. `run_benchmark.py` writes the updated output to `artifacts/rigorous_benchmark_results.json`, leaving the historical `artifacts/benchmark_results.json` untouched.
