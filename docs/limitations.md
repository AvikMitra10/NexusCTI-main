# Limitations

## Small benchmark

The built-in benchmark contains 12 curated test cases. This is too small to support strong claims about generalization, statistical reliability, or performance across the diversity of real-world incidents. A few cases can materially change the reported percentages, and the benchmark does not establish representative sampling.

## Synthetic indicators and scenarios

Some benchmark scenarios use constructed or illustrative indicators, domains, hashes, and infrastructure narratives rather than independently collected incident telemetry. Synthetic examples are useful for controlled tests but may not reflect real indicator distributions, reporting quality, adversary behavior, or operational ambiguity. Keep synthetic and real-world cases clearly separated and document the provenance of each.

## Potential data leakage

Actor names and labels are stored in benchmark case objects alongside their prediction queries. The current evaluator passes the query and anchor indicator to prediction methods, not the ground-truth field directly; however, actor names may still appear in query text, report chunks, or other inputs. Leakage should be checked explicitly, and labels must be withheld from prediction inputs in any rigorous evaluation.

The same curated reports and indicators may also inform both the retrieval corpus and evaluation scenarios. Without independent split construction and a documented overlap audit, retrieval of a known report or indicator can make results look stronger than generalization to unseen incidents.

The built-in benchmark's leakage audit found no actor label in prediction queries or anchor indicators, and confirms that the evaluator does not pass ground-truth fields, case descriptions, or expected-hop counts to prediction calls. However, all 12 test anchors resolve to a pre-existing graph path to their ground-truth actor, and every case overlaps an actor-labeled report and a report containing its anchor. Since retrieval uses report actor labels to score predictions, these are substantial train/test knowledge-source overlaps; the built-in benchmark should not be treated as an unseen-data generalization test.

The component comparison in [`ablation_study.md`](../experiments/rigorous_evaluation/ablation_study.md) also uses those same 12 cases. Its perfect graph-only and hybrid scores are consistent with the known-path overlap and do not establish that the components generalize or that fusion adds value on unseen cases. The BM25-only baseline aggregates actor labels stored with retrieved chunks, so its score likewise depends on labeled-corpus overlap.

## Possible graph memorization

GNN training labels are derived from which entities are reachable from each actor in the same graph used for inference. Benchmark indicators and expected attribution paths may also be represented in that graph. A high score can therefore reflect graph familiarity or topology memorization rather than the ability to infer attribution for genuinely unseen relationships. Evaluate on held-out entities or relationships and report graph overlap before making generalization claims.

## Limits of automated attribution

The engine ranks actors using a curated graph, a fixed corpus, heuristic retrieval and fusion, and graph-derived supervision. Its dense retriever uses hash-based word and character-trigram vectors rather than a pretrained semantic model. The cross-validation score is a heuristic based on text mentions of trail endpoints, not independent source verification. Current benchmark hallucination values are proxies and do not inspect each factual claim. Confidence and verdicts are not demonstrated to be calibrated.

Graph trails are also constrained by graph coverage, edge direction, and the path search limit. A missing path does not prove that no relationship exists, and a present path does not prove that its source is accurate or that the relationship is causally meaningful.

The citation verifier matches citation strings against supplied catalog metadata and screens report text for endpoint and relation-cue co-occurrence. Citation naming differences can produce unresolved references, while lexical co-occurrence can miss paraphrases or incorrectly suggest support. It does not retrieve external publications or replace manual source review.

## Uncertainty in threat actor attribution

Threat actor attribution is inherently uncertain. Multiple groups may share tools, infrastructure, techniques, or access; indicators can be reused, copied, or deliberately planted; and reporting sources may disagree or later revise assessments. Actor names and aliases can also vary between sources. An automated ranking should be treated as an investigative lead with traceable evidence and uncertainty—not as a definitive identity claim or a substitute for analyst review and corroboration.
