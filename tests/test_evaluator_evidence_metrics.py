from types import SimpleNamespace

import networkx as nx

from nexuscti.eval.benchmark_cases import BenchmarkTestCase
from nexuscti.eval.evaluator import AttributionEvaluator


def _trail(edges):
    return SimpleNamespace(
        hops=[
            SimpleNamespace(
                source_id=source,
                relation=relation,
                target_id=target,
                citation=citation,
            )
            for source, relation, target, citation in edges
        ]
    )


def test_hybrid_benchmark_reports_separate_evidence_metrics():
    graph = nx.MultiDiGraph()
    graph.add_edge("ioc", "malware", key="uses", rel_type="uses", citation="source 1")
    graph.add_edge(
        "malware",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source 2",
    )

    candidate = SimpleNamespace(
        actor_id="actor",
        cross_validation_score=0.9,
        fused_confidence=0.9,
    )
    results = [
        SimpleNamespace(
            all_candidates=[candidate],
            top_actor=candidate,
            graph_trail=_trail([
                ("ioc", "uses", "malware", "source 1"),
                ("malware", "attributed_to", "actor", "source 2"),
            ]),
            multi_hop_depth=2,
            hallucination_risk="LOW",
        ),
        SimpleNamespace(
            all_candidates=[candidate],
            top_actor=candidate,
            graph_trail=_trail([
                ("missing", "attributed_to", "actor", "unverified source"),
            ]),
            multi_hop_depth=1,
            hallucination_risk="LOW",
        ),
    ]
    engine = SimpleNamespace(
        tkg=SimpleNamespace(nx_graph=graph),
        analyze_incident=lambda **kwargs: results.pop(0),
    )
    cases = [
        BenchmarkTestCase(
            id=f"case-{index}",
            name=f"Case {index}",
            query=f"query-{index}",
            anchor_ioc=f"ioc-{index}",
            ground_truth_actor="actor",
            expected_hops=2,
            category="test",
            description="Test case",
        )
        for index in (1, 2)
    ]
    evaluator = AttributionEvaluator(
        fusion_engine=engine,
        vector_baseline=None,
        gnn_baseline=None,
        test_cases=cases,
    )

    metrics, records = evaluator._evaluate_hybrid()

    assert metrics["actor_accuracy"] == 100.0
    assert metrics["evidence_validity_rate"] == 66.67
    assert metrics["citation_coverage"] == 66.67
    assert metrics["multihop_path_validity"] == 50.0
    assert metrics["unsupported_evidence_rate"] == 33.33
    assert len(records[0]["evidence_path"]) == 2
    assert records[1]["invalid_evidence"][0]["reason"] == "Edge not found"


def test_correct_actor_prediction_does_not_make_invalid_evidence_valid():
    candidate = SimpleNamespace(
        actor_id="actor",
        cross_validation_score=0.9,
        fused_confidence=0.9,
    )
    result = SimpleNamespace(
        all_candidates=[candidate],
        top_actor=candidate,
        graph_trail=_trail([
            ("missing-ioc", "attributed_to", "actor", "unverified source"),
        ]),
        multi_hop_depth=1,
        hallucination_risk="LOW",
    )
    engine = SimpleNamespace(
        tkg=SimpleNamespace(nx_graph=nx.MultiDiGraph()),
        analyze_incident=lambda **kwargs: result,
    )
    case = BenchmarkTestCase(
        id="correct-actor-invalid-evidence",
        name="Correct actor with unsupported trail",
        query="investigate indicator",
        anchor_ioc="indicator",
        ground_truth_actor="actor",
        expected_hops=1,
        category="test",
        description="Test evidence separately from actor ranking.",
    )
    evaluator = AttributionEvaluator(
        fusion_engine=engine,
        vector_baseline=None,
        gnn_baseline=None,
        test_cases=[case],
    )

    metrics, records = evaluator._evaluate_hybrid()

    assert metrics["actor_accuracy"] == 100.0
    assert metrics["evidence_validity_rate"] == 0.0
    assert metrics["citation_coverage"] == 0.0
    assert metrics["unsupported_evidence_rate"] == 100.0
    assert records[0]["is_correct"]
    assert not records[0]["path_valid"]
    assert records[0]["invalid_evidence"][0]["reason"] == "Edge not found"


def test_single_hop_case_is_not_applicable_to_multihop_path_metric():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "malware",
        "actor",
        key="rev_uses",
        rel_type="rev_uses",
        citation="source",
    )
    candidate = SimpleNamespace(
        actor_id="actor",
        cross_validation_score=0.9,
        fused_confidence=0.9,
    )
    result = SimpleNamespace(
        all_candidates=[candidate],
        top_actor=candidate,
        graph_trail=_trail([
            ("malware", "rev_uses", "actor", "source"),
        ]),
        multi_hop_depth=1,
        hallucination_risk="LOW",
    )
    evaluator = AttributionEvaluator(
        fusion_engine=SimpleNamespace(
            tkg=SimpleNamespace(nx_graph=graph),
            analyze_incident=lambda **kwargs: result,
        ),
        vector_baseline=None,
        gnn_baseline=None,
        test_cases=[
            BenchmarkTestCase(
                id="single-hop",
                name="Single-hop case",
                query="query",
                anchor_ioc="malware",
                ground_truth_actor="actor",
                expected_hops=1,
                category="test",
                description="Single-hop attribution.",
            )
        ],
    )

    metrics, records = evaluator._evaluate_hybrid()

    assert records[0]["path_valid"]
    assert records[0]["multihop_path_status"] == "not_applicable"
    assert records[0]["multihop_path_valid"] is None
    assert metrics["multihop_path_validity"] == 0.0
