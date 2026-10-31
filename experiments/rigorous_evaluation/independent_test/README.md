# Independent Test Sets

The primary frozen holdout is `test_cases.json` with labels isolated in
`ground_truth.json`. Source metadata and claim excerpts are in
`source_registry.json`.

This v2 split contains two additional public CISA advisories which were checked
against the frozen graph, corpus, and repository experiment history. The
earlier `prediction_inputs.jsonl` / `ground_truth.jsonl` files are retained as a
legacy dataset; those campaigns had prior repository experiment exposure and
must not be treated as newly held out.

## Files and label separation

- `test_cases.json` contains only case IDs, queries, and anchors.
- `ground_truth.json` contains expected actors and relationships.
- `source_registry.json` records the public CISA source metadata and claim
  excerpts.
- `frozen_overlap_audit.json` records label, anchor, relationship, and snapshot
  hash checks.
- `predictions.json` is written without loading the label file.
- `evaluation_results.json` is written later by the separate scorer.

Do not pass ground truth or source annotations into the prediction runner.
Both v2 cases are open-set for the current six-actor model; evaluate
abstention, not closed-set actor-ranking accuracy.

## Reproducible staged evaluation

From the project root:

```powershell
python experiments\rigorous_evaluation\independent_test\audit_frozen_split.py
python experiments\rigorous_evaluation\independent_test\run_independent_predictions.py
python experiments\rigorous_evaluation\independent_test\score_independent_predictions.py
```

`data_split.json` freezes the graph and corpus SHA-256 values. The audit and
prediction runner stop if either changes. A clean repository audit does not
prove source claims, rule out prior external exposure, or establish that two
cases are representative.
