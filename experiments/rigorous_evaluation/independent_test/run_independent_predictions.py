"""Run predictions using only the frozen split's unlabeled input file."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexuscti.service import NexusCTIService
from experiments.rigorous_evaluation.independent_test.snapshot import frozen_snapshot


HERE = Path(__file__).resolve().parent
SPLIT_PATH = PROJECT_ROOT / "experiments" / "rigorous_evaluation" / "data_split.json"
AUDIT_PATH = HERE / "frozen_overlap_audit.json"
INPUT_PATH = HERE / "test_cases.json"
OUTPUT_PATH = HERE / "predictions.json"


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    cases = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not audit.get("independence_checks_passed"):
        raise ValueError("Run a passing frozen-split audit before predictions.")
    if audit["frozen_training_data"] != frozen_snapshot():
        raise ValueError("Frozen graph/corpus fingerprint changed after the audit.")

    seed = int(split["random_seed"])
    torch.manual_seed(seed)
    service = NexusCTIService()
    predictions = []
    for case in cases["cases"]:
        result = service.analyze(
            query=case["query"],
            ioc_override=case["anchor_ioc"],
        )
        predictions.append({
            "case_id": case["case_id"],
            "predicted_top1": result.top_actor.actor_id if result.top_actor else None,
            "top_actor_name": result.top_actor.actor_name if result.top_actor else None,
            "confidence": result.top_actor.fused_confidence if result.top_actor else 0.0,
            "verdict": result.top_actor.verdict if result.top_actor else "NO_CANDIDATE",
            "abstained": (
                result.top_actor is None
                or result.top_actor.verdict == "INCONCLUSIVE"
            ),
            "candidates": [
                {
                    "actor_id": candidate.actor_id,
                    "confidence": candidate.fused_confidence,
                    "verdict": candidate.verdict,
                }
                for candidate in result.all_candidates
            ],
            "evidence_path": [
                {
                    "source": hop.source_id,
                    "relation": hop.relation,
                    "target": hop.target_id,
                    "citation": hop.citation,
                }
                for hop in result.graph_trail.hops
            ] if result.graph_trail else [],
        })

    output = {
        "dataset_id": cases["dataset_id"],
        "prediction_input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "ground_truth_loaded": False,
        "random_seed": seed,
        "frozen_training_data": frozen_snapshot(),
        "predictions": predictions,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(predictions)} label-blind predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
