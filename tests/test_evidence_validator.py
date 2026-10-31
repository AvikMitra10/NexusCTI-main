import networkx as nx

from nexuscti.eval.evidence_validator import validate_evidence_path


def test_validates_edge_relationship_and_citation():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "report",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source report",
    )

    result = validate_evidence_path(
        graph,
        [("report", "attributed_to", "actor")],
    )

    assert result["valid"]
    assert result["valid_edges"] == 1
    assert result["relationship_edges"] == 1
    assert result["citation_edges"] == 1
    assert result["total_edges"] == 1
    assert result["invalid_edges"] == []


def test_rejects_missing_edge_relationship_and_citation():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "report",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source report",
    )
    graph.add_edge(
        "tool",
        "actor",
        key="uses",
        rel_type="uses",
        citation="",
    )

    result = validate_evidence_path(
        graph,
        [
            ("missing", "uses", "actor"),
            ("report", "uses", "actor"),
            ("tool", "uses", "actor"),
        ],
    )

    assert not result["valid"]
    assert result["valid_edges"] == 0
    assert result["relationship_edges"] == 1
    assert result["citation_edges"] == 0
    assert [edge["reason"] for edge in result["invalid_edges"]] == [
        "Edge not found",
        "Relationship type mismatch",
        "Missing citation",
        "Disconnected consecutive hops",
    ]


def test_empty_path_is_invalid():
    result = validate_evidence_path(nx.MultiDiGraph(), [])

    assert not result["valid"]
    assert result["valid_edges"] == 0
    assert result["relationship_edges"] == 0
    assert result["citation_edges"] == 0
    assert result["total_edges"] == 0
    assert not result["continuity_valid"]
    assert result["reason"] == "Empty evidence path"


def test_rejects_trail_citation_that_does_not_match_graph_edge():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "report",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source report",
    )

    result = validate_evidence_path(
        graph,
        [("report", "attributed_to", "actor", "different source")],
    )

    assert not result["valid"]
    assert result["relationship_edges"] == 1
    assert result["citation_edges"] == 0
    assert result["invalid_edges"][0]["reason"] == (
        "Trail citation does not match graph edge"
    )


def test_rejects_disconnected_consecutive_hops():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "report",
        "tool",
        key="uses",
        rel_type="uses",
        citation="source one",
    )
    graph.add_edge(
        "other",
        "actor",
        key="attributed_to",
        rel_type="attributed_to",
        citation="source two",
    )

    result = validate_evidence_path(
        graph,
        [
            ("report", "uses", "tool"),
            ("other", "attributed_to", "actor"),
        ],
    )

    assert not result["valid"]
    assert not result["continuity_valid"]
    assert result["valid_edges"] == 2
    assert result["invalid_edges"][-1]["reason"] == (
        "Disconnected consecutive hops"
    )
