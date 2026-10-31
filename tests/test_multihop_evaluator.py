import networkx as nx

from nexuscti.eval.multihop_evaluator import evaluate_multihop_path


def _graph_with_cited_path():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "ioc",
        "malware",
        key="uses",
        rel_type="uses",
        citation="source one",
    )
    graph.add_edge(
        "malware",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source two",
    )
    return graph


def test_accepts_connected_valid_multihop_path():
    result = evaluate_multihop_path(
        _graph_with_cited_path(),
        [
            ("ioc", "uses", "malware"),
            ("malware", "attributed_to", "actor"),
        ],
    )

    assert result["is_multihop"]
    assert result["path_connected"]
    assert result["path_valid"]
    assert result["success"]
    assert result["valid_edges"] == 2
    assert result["citation_edges"] == 2


def test_rejects_disconnected_graph_path_even_when_each_edge_is_valid():
    graph = _graph_with_cited_path()
    graph.add_edge(
        "other",
        "actor",
        key="uses",
        rel_type="uses",
        citation="source three",
    )

    result = evaluate_multihop_path(
        graph,
        [
            ("ioc", "uses", "malware"),
            ("other", "uses", "actor"),
        ],
    )

    assert not result["path_connected"]
    assert not result["path_valid"]
    assert not result["success"]
    assert result["valid_edges"] == 2
    assert len(result["disconnected_edges"]) == 1


def test_rejects_invalid_relationship_and_paths_below_hop_threshold():
    graph = _graph_with_cited_path()

    invalid_edge = evaluate_multihop_path(
        graph,
        [
            ("ioc", "wrong_relation", "malware"),
            ("malware", "attributed_to", "actor"),
        ],
    )
    short_path = evaluate_multihop_path(
        graph,
        [("ioc", "uses", "malware")],
    )

    assert not invalid_edge["path_valid"]
    assert not invalid_edge["success"]
    assert not short_path["is_multihop"]
    assert short_path["path_valid"]
    assert not short_path["success"]
