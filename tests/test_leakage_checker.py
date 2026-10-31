import networkx as nx

from nexuscti.eval.leakage_checker import (
    audit_benchmark_leakage,
    check_actor_label_leakage,
)


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "actor_apt28",
        type="ThreatActor",
        name="APT28 (Fancy Bear / STRONTIUM)",
        description="Threat actor description.",
        metadata={"aliases": ["Forest Blizzard"]},
    )
    graph.add_node("ioc_domain", type="Indicator", name="example.test")
    graph.add_node("malware", type="Malware", name="Zebrocy")
    graph.add_edge("ioc_domain", "malware", rel_type="communicates_with")
    graph.add_edge("malware", "actor_apt28", rel_type="rev_uses")
    return graph


def test_actor_label_leakage_detects_names_and_aliases_as_whole_words():
    result = check_actor_label_leakage(
        "APT28 (Fancy Bear) used this indicator.",
        ["APT28", "Fancy Bear", "STRONTIUM", "APT2"],
    )

    assert result == {
        "possible_leakage": True,
        "detected_labels": ["APT28", "Fancy Bear"],
    }


def test_flags_preexisting_graph_relationship_as_evaluation_contamination():
    graph = _graph()
    case = {
        "id": "case-1",
        "name": "Generic indicator investigation",
        "query": "Investigate the suspicious domain.",
        "anchor_ioc": "example.test",
        "ground_truth_actor": "actor_apt28",
        "expected_hops": 2,
        "category": "multi-hop",
        "description": "Domain -> Zebrocy -> APT28.",
    }
    corpus = [
        {
            "id": "report-1",
            "target_actor": "actor_apt28",
            "title": "APT28 threat report",
            "text": "APT28 used Zebrocy to communicate with example.test.",
        }
    ]

    result = audit_benchmark_leakage([case], graph, corpus)
    finding = result["findings"][0]

    assert result["total_cases"] == 1
    assert result["direct_actor_label_exposures"] == 0
    assert result["actor_labels_in_benchmark_descriptions"] == 1
    assert result["preexisting_graph_paths_to_ground_truth"] == 1
    assert result["cases_with_actor_labeled_retrieval_reports"] == 1
    assert result["cases_with_anchor_text_in_retrieval_corpus"] == 1
    assert finding["ground_truth_metadata_passed_to_predictor"] is False
    assert finding["expected_hops_passed_to_predictor"] is False
    assert finding["expected_relationships_available_in_graph"] is True
    assert finding["retrieval_corpus_exposes_actor_labels"] is True


def test_flags_actor_label_in_prediction_input_as_possible_leakage():
    case = {
        "id": "case-2",
        "name": "Known actor test",
        "query": "APT28 activity observed at example.test",
        "anchor_ioc": "example.test",
        "ground_truth_actor": "actor_apt28",
        "expected_hops": 2,
        "category": "multi-hop",
        "description": "Expected chain.",
    }

    result = audit_benchmark_leakage([case], _graph(), [])

    assert result["direct_actor_label_exposures"] == 1
    assert result["findings"][0]["detected_actor_labels"] == ["APT28"]
