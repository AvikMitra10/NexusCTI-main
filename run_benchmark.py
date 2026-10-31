"""
NexusCTI Empirical Benchmark Evaluation Runner.
Executes comparative evaluation across all 12 test cases for:
1. Naive Vector-only RAG
2. Standalone GNN Classifier
3. NexusCTI Hybrid (GNN + RAG)
Saves results to artifacts/benchmark_results.json and prints markdown summary.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from nexuscti.service import get_service
from nexuscti.config import ARTIFACTS_DIR


def main():
    print("=" * 70)
    print("[*] NEXUS-CTI: EMPIRICAL BENCHMARK EVALUATION")
    print("Comparing Structural GNN, Naive Vector RAG, and Synergistic Hybrid")
    print("=" * 70)

    service = get_service()
    results = service.run_benchmark()

    summary = results["summary"]

    print("\n📊 COMPARATIVE PERFORMANCE METRICS SUMMARY:")
    print("-" * 75)
    header = f"{'Evaluation Dimension':<30} | {'Vector RAG':<12} | {'Standalone GNN':<14} | {'NexusCTI Hybrid':<15}"
    print(header)
    print("-" * 75)

    v_res = summary["Naive Vector-only RAG"]
    g_res = summary["Standalone GNN Classifier"]
    h_res = summary["NexusCTI Hybrid (GNN + RAG)"]

    rows = [
        ("Top-1 Attribution Accuracy", f"{v_res['top1_accuracy']}%", f"{g_res['top1_accuracy']}%", f"[+] {h_res['top1_accuracy']}%"),
        ("Top-3 Attribution Accuracy", f"{v_res['top3_accuracy']}%", f"{g_res['top3_accuracy']}%", f"[+] {h_res['top3_accuracy']}%"),
        ("Top-5 Attribution Accuracy", f"{v_res['top5_accuracy']}%", f"{g_res['top5_accuracy']}%", f"[+] {h_res['top5_accuracy']}%"),
        ("Multi-Hop Attribution Accuracy", f"{v_res['multihop_reasoning_success']}%", f"{g_res['multihop_reasoning_success']}%", f"[+] {h_res['multihop_reasoning_success']}%"),
        ("Hallucination / Misattribution", f"{v_res['hallucination_rate']}%", f"{g_res['hallucination_rate']}%", f"[*] {h_res['hallucination_rate']}%"),
        ("Mean Reciprocal Rank (MRR)", f"{v_res['mrr']:.4f}", f"{g_res['mrr']:.4f}", f"[+] {h_res['mrr']:.4f}"),
    ]

    for label, v, g, h in rows:
        print(f"{label:<30} | {v:<12} | {g:<14} | {h:<15}")
    print("-" * 75)

    print("\nRIGOROUS HYBRID EVIDENCE METRICS:")
    print("-" * 40)
    rigorous_rows = [
        ("Actor Accuracy", h_res["actor_accuracy"]),
        ("Evidence Validity Rate", h_res["evidence_validity_rate"]),
        ("Citation Coverage", h_res["citation_coverage"]),
        ("Multi-Hop Evidence Validity", h_res["multihop_path_validity"]),
        ("Unsupported Evidence Rate", h_res["unsupported_evidence_rate"]),
    ]
    for label, value in rigorous_rows:
        formatted_value = f"{value}%" if value is not None else "N/A"
        print(f"{label:<30} | {formatted_value}")
    print("-" * 40)

    out_file = ARTIFACTS_DIR / "rigorous_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Rigorous benchmark records saved separately to: {out_file}\n")


if __name__ == "__main__":
    main()
