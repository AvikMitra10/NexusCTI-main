from types import SimpleNamespace

import networkx as nx

from nexuscti.data.ontology import NodeType
from nexuscti.eval.benchmark_cases import BenchmarkTestCase
from nexuscti.eval.evaluator import AttributionEvaluator


def _case(case_id, anchor, actor, expected_hops=1):
    return BenchmarkTestCase(
        id=case_id,
        name=case_id,
        query=f"query for {case_id}",
        anchor_ioc=anchor,
        ground_truth_actor=actor,
        expected_hops=expected_hops,
        category="test",
        description="A test case",
    )


def test_ablation_compares_five_components_using_common_ranking_metrics():
    cases = [
        _case("case-1", "ioc-1", "actor-a", expected_hops=2),
        _case("case-2", "ioc-2", "actor-b"),
    ]
    graph = nx.MultiDiGraph()
    graph.add_edge("ioc-1", "actor-a")
    graph.add_edge("ioc-2", "actor-b")
    actor_nodes = [
        SimpleNamespace(id="actor-a", type=NodeType.THREAT_ACTOR),
        SimpleNamespace(id="actor-b", type=NodeType.THREAT_ACTOR),
    ]
    tkg = SimpleNamespace(
        nx_graph=graph,
        raw_nodes=actor_nodes,
        find_node_by_indicator=lambda indicator: (
            SimpleNamespace(id=indicator) if indicator in {"ioc-1", "ioc-2"} else None
        ),
    )
    bm25 = SimpleNamespace(
        search=lambda query, top_k: [
            (SimpleNamespace(target_actor="actor-a"), 2.0),
        ]
    )
    hybrid_candidates = {
        "ioc-1": [SimpleNamespace(actor_id="actor-a", fused_confidence=0.9)],
        "ioc-2": [SimpleNamespace(actor_id="actor-b", fused_confidence=0.8)],
    }
    fusion_engine = SimpleNamespace(
        retriever=SimpleNamespace(bm25=bm25),
        tkg=tkg,
        analyze_incident=lambda query, ioc_override: SimpleNamespace(
            all_candidates=hybrid_candidates[ioc_override]
        ),
    )
    vector_baseline = SimpleNamespace(
        predict=lambda query: [("actor-a", 1.0)]
    )
    gnn_predictions = {
        "ioc-1": [("actor-a", 1.0)],
        "ioc-2": [("actor-b", 1.0)],
    }
    gnn_baseline = SimpleNamespace(
        predict=lambda anchor: gnn_predictions[anchor]
    )
    evaluator = AttributionEvaluator(
        fusion_engine=fusion_engine,
        vector_baseline=vector_baseline,
        gnn_baseline=gnn_baseline,
        test_cases=cases,
    )

    results = evaluator.run_ablation_study()

    assert list(results["summary"]) == [
        "Vector RAG",
        "BM25 Retrieval",
        "GNN",
        "Knowledge Graph Reasoning",
        "Hybrid NexusCTI",
    ]
    assert results["total_test_cases"] == 2
    assert results["summary"]["Knowledge Graph Reasoning"]["top1_accuracy"] == 100.0
    assert results["summary"]["Vector RAG"]["top1_accuracy"] == 50.0
    assert results["summary"]["BM25 Retrieval"]["top1_accuracy"] == 50.0
    assert results["summary"]["Hybrid NexusCTI"]["top1_accuracy"] == 100.0
    assert results["summary"]["GNN"]["multihop_reasoning_success"] == 100.0
    assert len(results["detailed_case_evaluations"]) == 10


def test_bm25_ablation_returns_empty_for_zero_score_results():
    evaluator = AttributionEvaluator(
        fusion_engine=SimpleNamespace(
            retriever=SimpleNamespace(
                bm25=SimpleNamespace(
                    search=lambda query, top_k: [
                        (SimpleNamespace(target_actor="actor-a"), 0.0),
                    ]
                )
            )
        ),
        vector_baseline=None,
        gnn_baseline=None,
        test_cases=[],
    )

    assert evaluator._predict_bm25("unmatched query") == []
