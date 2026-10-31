"""Run the five-component ablation on the built-in benchmark dataset."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexuscti.service import get_service

RANDOM_SEED = 42


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    service = get_service()
    results = service.evaluator.run_ablation_study()
    results["random_seed"] = RANDOM_SEED

    output_path = (
        PROJECT_ROOT
        / "experiments"
        / "rigorous_evaluation"
        / "results"
        / "ablation_study_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Ablation results ({results['total_test_cases']} cases)")
    print(
        f"{'Component':<30} {'Top-1':>8} {'Top-3':>8} "
        f"{'Top-5':>8} {'MRR':>8} {'Multi-hop':>11}"
    )
    for component, metrics in results["summary"].items():
        print(
            f"{component:<30} "
            f"{metrics['top1_accuracy']:>7.2f}% "
            f"{metrics['top3_accuracy']:>7.2f}% "
            f"{metrics['top5_accuracy']:>7.2f}% "
            f"{metrics['mrr']:>8.4f} "
            f"{metrics['multihop_reasoning_success']:>10.2f}%"
        )
    print(f"\nSaved results to {output_path}")
    print(results["interpretation_note"])


if __name__ == "__main__":
    main()
