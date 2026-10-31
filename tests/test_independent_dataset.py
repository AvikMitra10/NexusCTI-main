import json
import runpy
from pathlib import Path


DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "rigorous_evaluation"
    / "independent_test"
)


def test_independent_dataset_has_separated_inputs_and_a_clean_overlap_audit():
    namespace = runpy.run_path(str(DATASET_DIR / "validate_independence.py"))
    report = namespace["audit"]()

    assert report["case_count"] == 4
    assert report["independence_checks_passed"]
    assert report["prediction_inputs_actor_label_leaks"] == 0
    assert report["anchors_resolving_in_graph"] == 0
    assert report["anchors_found_exactly_in_retrieval_corpus"] == 0
    assert report["expected_relationships_found_in_graph"] == 0
    assert report["expected_relationships_with_text_overlap_in_corpus"] == 0
    assert report["missing_source_records"] == 0


def test_independent_dataset_keeps_labels_out_of_prediction_inputs():
    inputs = [
        json.loads(line)
        for line in (DATASET_DIR / "prediction_inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ground_truth = [
        json.loads(line)
        for line in (DATASET_DIR / "ground_truth.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {case["case_id"] for case in inputs} == {
        case["case_id"] for case in ground_truth
    }
    assert all(
        set(case) == {"case_id", "task", "query", "anchor_ioc"}
        for case in inputs
    )
    assert all("expected_actor" in case for case in ground_truth)
