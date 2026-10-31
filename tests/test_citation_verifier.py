import networkx as nx

from nexuscti.eval.citation_verifier import (
    audit_graph_citations,
    verify_edge_citation,
)


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node("actor_apt28", name="APT28", metadata={"aliases": ["Fancy Bear"]})
    graph.add_node("mal_zebrocy", name="Zebrocy")
    graph.add_node("cve_2023_23397", name="CVE-2023-23397")
    graph.add_edge(
        "actor_apt28",
        "mal_zebrocy",
        key="uses",
        rel_type="uses",
        citation="Vendor Threat Report",
    )
    graph.add_edge(
        "actor_apt28",
        "mal_zebrocy",
        key="rev_uses",
        rel_type="rev_uses",
        citation="Vendor Threat Report",
    )
    graph.add_edge(
        "actor_apt28",
        "cve_2023_23397",
        key="targets",
        rel_type="targets",
        citation="Vendor Threat Report",
    )
    return graph


def _catalog():
    return [
        {
            "id": "report-1",
            "title": "Vendor Threat Report",
            "source": "Vendor",
            "text": "APT28 (Fancy Bear) uses the Zebrocy malware.",
        }
    ]


def test_separates_source_existence_and_passage_resolution_from_semantics():
    result = verify_edge_citation(
        _graph(),
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        _catalog(),
    )

    assert result["source_exists"]
    assert result["citation_existence_status"] == "exists"
    assert result["citation_exists"] is True
    assert result["document_identity_verified"]
    assert result["citation_resolution_status"] == "resolved"
    assert result["source_resolved"] is True
    assert result["resolved_passage"] == "APT28 (Fancy Bear) uses the Zebrocy malware."
    assert result["lexical_support_status"] == "lexically_supported"
    assert result["semantic_support_status"] == "not_evaluated"
    assert result["semantic_support"] == "unverified"
    assert result["verification_method"] == "pending"
    assert result["reason"] == (
        "Citation retrieved, but claim support not yet assessed"
    )
    assert result["supports_relationship"] is None
    assert not result["verified"]
    assert result["source_document_ids"] == ["report-1"]


def test_matches_reference_label_with_intervening_title_words():
    graph = _graph()
    graph["actor_apt28"]["mal_zebrocy"]["uses"]["citation"] = "CISA AA20-352A"
    catalog = [
        {
            "id": "cisa-report",
            "title": "CISA Advisory AA20-352A",
            "text": "APT28 uses the Zebrocy malware.",
        }
    ]

    result = verify_edge_citation(
        graph,
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        catalog,
    )

    assert result["document_identity_verified"]
    assert result["citation_resolution_status"] == "resolved"
    assert result["semantic_support_status"] == "not_evaluated"
    assert not result["verified"]


def test_rejects_citation_not_in_source_catalog():
    graph = _graph()

    unknown_source = verify_edge_citation(
        graph,
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        [],
    )

    assert not unknown_source["source_exists"]
    assert unknown_source["citation_exists"] is False
    assert unknown_source["source_resolved"] is False
    assert unknown_source["semantic_support"] == "unverified"
    assert unknown_source["verification_method"] == "pending"
    assert unknown_source["citation_existence_status"] == "not_found"
    assert not unknown_source["document_identity_verified"]
    assert not unknown_source["verified"]


def test_rejects_catalogued_citation_that_does_not_support_edge():
    graph = _graph()

    unrelated_claim = verify_edge_citation(
        graph,
        "actor_apt28",
        "targets",
        "cve_2023_23397",
        _catalog(),
    )

    assert unrelated_claim["source_exists"]
    assert unrelated_claim["citation_resolution_status"] == "unresolved"
    assert unrelated_claim["semantic_support_status"] == "not_evaluable"
    assert unrelated_claim["supports_relationship"] is None
    assert not unrelated_claim["verified"]
    assert unrelated_claim["reason"] == (
        "Relevant passage could not be resolved"
    )


def test_flags_reused_citation_for_review_without_claiming_support():
    result = audit_graph_citations(_graph(), iter(_catalog()))

    assert result["total_authored_edges"] == 2
    assert result["reused_citations"] == 1
    assert result["verified_citations"] == 0
    assert result["unverified_citations"] == 2
    assert result["relationships_semantically_supported"] == 0
    assert result["semantic_support_not_evaluated"] == 2

    by_relation = {record["relation"]: record for record in result["records"]}
    assert by_relation["uses"]["reuse_assessment"] == "review_required"
    assert by_relation["targets"]["reuse_assessment"] == "review_required"
    assert by_relation["targets"]["reused_across_relationships"]
    assert not by_relation["targets"]["verified"]


def test_semantic_support_requires_explicit_checker():
    result = verify_edge_citation(
        _graph(),
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        _catalog(),
        semantic_support_checker=lambda passage, subject, relation, obj: True,
    )

    assert result["citation_existence_status"] == "exists"
    assert result["citation_resolution_status"] == "resolved"
    assert result["semantic_support_status"] == "supported"
    assert result["semantic_support"] == "supported"
    assert result["verification_method"] == (
        "caller_provided_semantic_support_checker"
    )
    assert result["supports_relationship"] is True
    assert result["verified"]
    assert result["support_method"] == "caller_provided_semantic_support_checker"


def test_semantic_checker_can_reject_or_defer_a_resolved_passage():
    rejected = verify_edge_citation(
        _graph(),
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        _catalog(),
        semantic_support_checker=lambda passage, subject, relation, obj: False,
    )
    inconclusive = verify_edge_citation(
        _graph(),
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        _catalog(),
        semantic_support_checker=lambda passage, subject, relation, obj: None,
    )

    assert rejected["citation_resolution_status"] == "resolved"
    assert rejected["semantic_support_status"] == "not_supported"
    assert rejected["semantic_support"] == "unsupported"
    assert rejected["supports_relationship"] is False
    assert not rejected["verified"]
    assert inconclusive["semantic_support_status"] == "inconclusive"
    assert inconclusive["semantic_support"] == "inconclusive"
    assert inconclusive["supports_relationship"] is None
    assert not inconclusive["verified"]


def test_custom_passage_resolver_is_separate_from_semantic_check():
    calls = []

    def resolve(document, graph, subject, relation, obj):
        calls.append((document["id"], subject, relation, obj))
        return "A manually retrieved report passage."

    result = verify_edge_citation(
        _graph(),
        "actor_apt28",
        "uses",
        "mal_zebrocy",
        _catalog(),
        passage_resolver=resolve,
    )

    assert calls == [("report-1", "actor_apt28", "uses", "mal_zebrocy")]
    assert result["citation_resolution_status"] == "resolved"
    assert result["resolved_passage"] == "A manually retrieved report passage."
    assert result["semantic_support_status"] == "not_evaluated"
    assert not result["verified"]


def test_graph_audit_can_use_explicit_semantic_checker():
    result = audit_graph_citations(
        _graph(),
        _catalog(),
        semantic_support_checker=(
            lambda passage, subject, relation, obj: relation == "uses"
        ),
    )

    by_relation = {record["relation"]: record for record in result["records"]}
    assert by_relation["uses"]["semantic_support_status"] == "supported"
    assert by_relation["uses"]["verified"]
    assert by_relation["targets"]["semantic_support_status"] == "not_evaluable"
    assert result["relationships_semantically_supported"] == 1
    assert result["verified_citations"] == 1
