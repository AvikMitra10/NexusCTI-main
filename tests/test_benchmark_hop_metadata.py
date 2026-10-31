import networkx as nx

from nexuscti.data.knowledge_base import build_default_knowledge_graph_data
from nexuscti.eval.benchmark_cases import get_benchmark_dataset
from nexuscti.gnn.graph_builder import ThreatKnowledgeGraph


def test_expected_hops_match_current_shortest_actor_paths():
    nodes, edges = build_default_knowledge_graph_data()
    tkg = ThreatKnowledgeGraph(nodes, edges)
    cases = get_benchmark_dataset()

    assert len(cases) == 12
    for case in cases:
        anchor = tkg.find_node_by_indicator(case.anchor_ioc)
        assert anchor is not None, case.id
        distance = nx.shortest_path_length(
            tkg.nx_graph,
            anchor.id,
            case.ground_truth_actor,
        )
        assert case.expected_hops == distance, case.id

    assert sum(case.expected_hops >= 2 for case in cases) == 11
    assert {case.category for case in cases} == {
        "direct_observation",
        "stager_pivot",
        "infrastructure_pivot",
        "deep_chain",
    }
