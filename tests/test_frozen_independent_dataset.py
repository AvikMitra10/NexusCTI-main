import json
from pathlib import Path

from experiments.rigorous_evaluation.independent_test.audit_frozen_split import audit
from experiments.rigorous_evaluation.independent_test.snapshot import frozen_snapshot


ROOT = Path(__file__).resolve().parents[1]
RIGOROUS_DIR = ROOT / "experiments" / "rigorous_evaluation"
DATASET_DIR = RIGOROUS_DIR / "independent_test"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_dataset_separates_labels_and_has_no_checked_overlap():
    inputs = _read_json(DATASET_DIR / "test_cases.json")
    truths = _read_json(DATASET_DIR / "ground_truth.json")

    input_ids = {case["case_id"] for case in inputs["cases"]}
    truth_ids = {case["case_id"] for case in truths["cases"]}
    assert input_ids == truth_ids
    assert all(
        set(case) == {"case_id", "task", "query", "anchor_ioc"}
        for case in inputs["cases"]
    )
    for prediction_case, truth_case in zip(inputs["cases"], truths["cases"]):
        prediction_text = (
            f"{prediction_case['query']} {prediction_case['anchor_ioc']}"
        ).casefold()
        assert truth_case["expected_actor_name"].casefold() not in prediction_text
        assert all(
            alias.casefold() not in prediction_text
            for alias in truth_case.get("expected_actor_aliases", [])
        )

    report = audit()
    assert report["independence_checks_passed"]
    assert report["case_count"] == 2
    assert report["prediction_inputs_label_leaks"] == 0
    assert report["anchors_in_frozen_graph"] == 0
    assert report["anchors_in_frozen_corpus"] == 0
    assert report["expected_edges_in_frozen_graph"] == 0
    assert report["expected_relationship_text_overlaps"] == 0


def test_prediction_artifact_is_label_blind_and_matches_frozen_data():
    split = _read_json(RIGOROUS_DIR / "data_split.json")
    predictions = _read_json(DATASET_DIR / "predictions.json")

    assert predictions["ground_truth_loaded"] is False
    assert predictions["random_seed"] == split["random_seed"]
    assert predictions["frozen_training_data"] == frozen_snapshot()
    assert len(predictions["predictions"]) == 2


def test_scored_results_evaluate_open_set_abstention_separately():
    results = _read_json(DATASET_DIR / "evaluation_results.json")

    assert results["case_count"] == 2
    assert results["open_set_case_count"] == 2
    assert results["closed_set_top1_accuracy"] is None
    assert results["open_set_correct_abstention_rate"] == 100.0
    assert all(case["correct_open_set_abstention"] for case in results["cases"])
