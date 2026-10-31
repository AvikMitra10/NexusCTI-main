"""Score saved blind predictions against the separately stored labels."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICTIONS_PATH = HERE / "predictions.json"
GROUND_TRUTH_PATH = HERE / "ground_truth.json"
OUTPUT_PATH = HERE / "evaluation_results.json"


def main() -> None:
    predictions_doc = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    truth_doc = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    if predictions_doc["dataset_id"] != truth_doc["dataset_id"]:
        raise ValueError("Prediction and ground-truth dataset IDs do not match.")
    predictions = {item["case_id"]: item for item in predictions_doc["predictions"]}
    truths = {item["case_id"]: item for item in truth_doc["cases"]}
    if len(predictions) != len(predictions_doc["predictions"]):
        raise ValueError("Prediction case IDs must be unique.")
    if len(truths) != len(truth_doc["cases"]) or predictions.keys() != truths.keys():
        raise ValueError("Predictions and ground truth must contain the same unique case IDs.")

    records = []
    for case_id, prediction in predictions.items():
        truth = truths[case_id]
        is_open_set = bool(truth["open_set"])
        top_actor_correct = prediction["predicted_top1"] == truth["expected_actor"]
        correct_abstention = is_open_set and prediction["abstained"]
        records.append({
            "case_id": case_id,
            "open_set": is_open_set,
            "expected_actor": truth["expected_actor"],
            "predicted_top1": prediction["predicted_top1"],
            "top_actor_correct": top_actor_correct,
            "abstained": prediction["abstained"],
            "correct_open_set_abstention": correct_abstention,
            "confidence": prediction["confidence"],
            "verdict": prediction["verdict"],
        })

    open_set_records = [record for record in records if record["open_set"]]
    count = len(records)
    results = {
        "dataset_id": predictions_doc["dataset_id"],
        "prediction_artifact": "predictions.json",
        "ground_truth_artifact": "ground_truth.json",
        "case_count": count,
        "open_set_case_count": len(open_set_records),
        "closed_set_top1_accuracy": None,
        "open_set_correct_abstention_rate": (
            round(
                sum(record["correct_open_set_abstention"] for record in open_set_records)
                / len(open_set_records)
                * 100,
                2,
            )
            if open_set_records else None
        ),
        "cases": records,
        "interpretation": (
            "This dataset contains open-set actors only. Actor-ranking accuracy "
            "against the closed six-actor label set is not applicable; report "
            "abstention separately."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
