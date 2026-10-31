# NexusCTI Rigorous Evaluation Protocol

## 1. Objective
Evaluate the accuracy, evidence quality, multi-hop reasoning, and reliability of the NexusCTI threat attribution engine.

## 2. Data separation

- Separate training, validation, and test cases.
- Keep ground-truth actor labels outside prediction inputs.
- Check whether test relationships already exist in the graph or retrieval index.
- Document synthetic and real-world test cases separately.
- Store independent test inputs separately from labels and evidence annotations.
- Run the independent-set overlap audit against the exact graph and retrieval corpus used for evaluation.

The authoritative frozen evaluation split is recorded in [`data_split.json`](./data_split.json). It freezes the default graph and corpus with canonical SHA-256 hashes, and uses the label-free `independent_test/test_cases.json`; labels and source evidence are in separate files. The earlier four-case dataset in `independent_test/prediction_inputs.jsonl` and `ground_truth.jsonl` is historical only: its campaigns were previously cataloged in experiment artifacts, so it is not the fresh split.

The new two-case test uses CISA advisory AA23-335A (CyberAv3ngers / Unitronics PLCs) and AA23-352A (Play ransomware / CVE-2022-41040). Repository searches found no references to these campaign identifiers or anchors in the frozen corpus, graph source data, or prior experiment artifacts. Both actors are open-set relative to the six-actor graph. See `source_registry.json` and the frozen overlap audit for provenance and exact graph/corpus relationship checks.

Run the staged workflow in order:

```powershell
python experiments\rigorous_evaluation\independent_test\audit_frozen_split.py
python experiments\rigorous_evaluation\independent_test\run_independent_predictions.py
python experiments\rigorous_evaluation\independent_test\score_independent_predictions.py
```

The prediction runner reads only the test input, frozen hashes, and passing overlap audit. It does not load ground truth or source annotations. It persists predictions before the separate scorer opens ground truth. If either frozen hash changes, audit and prediction execution stop. Since both actors are open-set, evaluate abstention; closed-set actor-ranking accuracy is not applicable. A clean repository audit does not rule out prior external model exposure or establish the truth of source claims.

## 3. Evaluation metrics

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Mean Reciprocal Rank (MRR)
- Evidence validity
- Multi-hop attribution accuracy: correct actor ranked first among cases with `expected_hops >= 2`; does not require a valid trail.
- Multi-hop evidence validity: percentage of those cases whose returned trail has at least two connected edges, each present with the stated relation and a matching, non-empty citation. This does not depend on whether the correct actor ranked first.
- Hallucination rate
- Data leakage warnings

## 4. Evidence validation
Every predicted evidence relationship must be checked against the knowledge graph and its source information. The path validator's citation check verifies only a matching non-empty citation field on the graph edge; it does not verify that a source resolves or substantively supports the relationship.

## 5. Reproducibility
Record the dataset version, graph version, model configuration, random seed, and evaluation date.

## 6. Reporting
Report the original benchmark and rigorous benchmark separately. Never replace historical results with newly calculated results.

## 7. Component comparison
Run the five-way component comparison with
[`run_ablation_study.py`](./run_ablation_study.py) and review
[`ablation_study.md`](./ablation_study.md) before interpreting the results.
The current results use the built-in benchmark and are explicitly exploratory;
they are not an independent generalization estimate.
