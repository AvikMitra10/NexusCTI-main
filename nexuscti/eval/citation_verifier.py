"""Audit graph citations against an explicit catalog of source documents."""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Callable


_RELATION_CUES = {
    "uses": ("use", "uses", "used", "using", "deploy", "deployed", "employ", "operated"),
    "executes": ("execute", "executes", "conduct", "performs", "technique"),
    "targets": ("target", "targets", "targeted", "attack", "attacks", "victim"),
    "controls": ("control", "controls", "controlled", "operates"),
    "communicates_with": (
        "communicate", "communicates", "communication", "connect", "beacon",
        "command and control", "c2",
    ),
    "exploits": ("exploit", "exploits", "exploited", "exploitation"),
    "maps_to": ("map", "maps", "mapped", "mapping", "technique"),
    "resolves_to": ("resolve", "resolves", "resolved", "resolution", "dns"),
    "shares_infra": (
        "share", "shares", "shared", "infrastructure", "overlap", "same subnet",
        "co located",
    ),
    "associated_with": ("associate", "associated", "association", "linked", "attributed"),
}


def _read_field(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _catalog_records(source_catalog: Iterable[Any]) -> list[Any]:
    if isinstance(source_catalog, Mapping):
        if any(field in source_catalog for field in ("id", "title", "source")):
            return [source_catalog]
        records = []
        for alias, document in source_catalog.items():
            if isinstance(document, Mapping):
                record = dict(document)
                aliases = record.get("citation_aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                record["citation_aliases"] = [*aliases, str(alias)]
            else:
                record = {
                    "citation_aliases": [str(alias)],
                    "title": str(_read_field(document, "title", "")),
                    "source": str(_read_field(document, "source", "")),
                    "id": str(_read_field(document, "id", "")),
                    "text": str(_read_field(document, "text", "")),
                }
            records.append(record)
        return records
    return list(source_catalog)


def _citation_source_matches(citation: str, source_catalog: Iterable[Any]) -> list[Any]:
    normalized_citation = _normalize(citation)
    if not normalized_citation:
        return []
    citation_terms = set(normalized_citation.split())

    matches = []
    for document in _catalog_records(source_catalog):
        identifiers = [
            _read_field(document, field)
            for field in ("id", "title", "source")
        ]
        aliases = _read_field(document, "citation_aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        identifiers.extend(aliases or [])
        normalized_identifiers = [
            _normalize(str(identifier))
            for identifier in identifiers
            if identifier
        ]
        if any(
            normalized_citation in identifier
            or citation_terms.issubset(set(identifier.split()))
            for identifier in normalized_identifiers
        ):
            matches.append(document)
    return matches


def _node_terms(graph: Any, node_id: str) -> list[str]:
    if node_id not in graph:
        return []

    node_data = graph.nodes[node_id]
    names = [str(node_id), str(node_data.get("name", ""))]
    metadata = node_data.get("metadata", {})
    if isinstance(metadata, Mapping):
        aliases = metadata.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        names.extend(str(alias) for alias in aliases or [])

    terms = set()
    for name in names:
        for variant in re.split(r"[()/,;|]", name):
            normalized = _normalize(variant)
            if normalized:
                terms.add(normalized)
    return sorted(terms, key=len, reverse=True)


def _contains_any(text: str, terms: list[str]) -> bool:
    normalized_text = f" {_normalize(text)} "
    return any(f" {term} " in normalized_text for term in terms if term)


def _resolve_relevant_passage(
    document: Any,
    graph: Any,
    source_node: str,
    target_node: str,
) -> str | None:
    """Return a sentence containing both endpoints, without judging support."""
    document_text = str(_read_field(document, "text", ""))
    if not document_text.strip():
        return None

    source_terms = _node_terms(graph, source_node)
    target_terms = _node_terms(graph, target_node)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", document_text.strip())
    for sentence in sentences:
        if (
            _contains_any(sentence, source_terms)
            and _contains_any(sentence, target_terms)
        ):
            return sentence.strip()
    return None


def _verification_result(
    *,
    edge_exists: bool,
    citation: str,
    reason: str,
) -> dict:
    return {
        "edge_exists": edge_exists,
        "citation": citation,
        "citation_exists": False,
        "source_resolved": False,
        "semantic_support": "unverified",
        "verification_method": "pending",
        "source_exists": False,
        "citation_existence_status": "not_checked",
        "document_identity_verified": False,
        "source_document_ids": [],
        "citation_resolution_status": "not_attempted",
        "resolved_passage": None,
        "lexical_support_status": "not_evaluated",
        "semantic_support_status": "not_evaluated",
        "supports_relationship": None,
        "support_method": None,
        "verified": False,
        "reason": reason,
    }


def verify_edge_citation(
    graph: Any,
    source_node: str,
    relation: str,
    target_node: str,
    source_catalog: Iterable[Any],
    citation: str | None = None,
    *,
    passage_resolver: Callable[[Any, Any, str, str, str], str | None] | None = None,
    semantic_support_checker: Callable[[str, str, str, str], bool | None] | None = None,
) -> dict:
    """Check citation existence, resolve a passage, and assess semantic support.

    The catalog is the source-existence authority for this check. By default,
    passage resolution selects a sentence containing both graph endpoints.
    Semantic support is only assessed when a caller supplies a checker; lexical
    overlap is reported separately and is never treated as semantic support.

    ``passage_resolver``, when supplied, receives the document, graph, subject,
    relation, and object, and returns a relevant passage or ``None``.
    ``semantic_support_checker`` receives that passage and the claim and
    returns ``True`` (supported), ``False`` (not supported), or ``None``
    (inconclusive).
    """
    edge_data = graph.get_edge_data(source_node, target_node) or {}
    matching_edges = [
        data
        for data in edge_data.values()
        if data.get("rel_type") == relation
        and (citation is None or data.get("citation", "") == citation)
    ]

    if not graph.has_edge(source_node, target_node):
        return _verification_result(
            edge_exists=False,
            citation=citation or "",
            reason="Graph edge not found",
        )
    if not matching_edges:
        return _verification_result(
            edge_exists=True,
            citation=citation or "",
            reason="Relationship type or cited edge not found",
        )

    selected_citation = citation
    if selected_citation is None:
        citations = {str(data.get("citation", "")) for data in matching_edges}
        if len(citations) != 1:
            return _verification_result(
                edge_exists=True,
                citation="",
                reason="Multiple matching edges have different citations",
            )
        selected_citation = next(iter(citations))

    cited_edges = [
        data for data in matching_edges
        if str(data.get("citation", "")) == selected_citation
    ]
    source_matches = _citation_source_matches(selected_citation, source_catalog)
    source_exists = bool(source_matches)
    identity_verified = len(source_matches) == 1

    if not selected_citation:
        reason = "Citation is empty"
    elif not source_exists:
        reason = "Citation does not match a source in the catalog"
    elif not identity_verified:
        reason = "Citation matches multiple source documents"
    else:
        reason = ""

    passage = None
    if identity_verified:
        document = source_matches[0]
        if passage_resolver is None:
            passage = _resolve_relevant_passage(
                document,
                graph,
                source_node,
                target_node,
            )
        else:
            passage = passage_resolver(
                document,
                graph,
                source_node,
                relation,
                target_node,
            )
        if passage is not None and not isinstance(passage, str):
            raise TypeError("passage_resolver must return str or None")

    resolution_status = (
        "resolved" if passage and passage.strip() else
        "unresolved" if identity_verified else
        "not_attempted"
    )
    if resolution_status != "resolved":
        passage = None

    lexical_supported = False
    if passage:
        source_mentioned = _contains_any(passage, _node_terms(graph, source_node))
        target_mentioned = _contains_any(passage, _node_terms(graph, target_node))
        relation_mentioned = _contains_any(
            passage,
            list(_RELATION_CUES.get(relation, ())),
        )
        lexical_supported = (
            source_mentioned and target_mentioned and relation_mentioned
        )

    if not identity_verified:
        semantic_status = "not_evaluated"
        semantic_support = None
    elif not passage:
        semantic_status = "not_evaluable"
        semantic_support = None
    elif semantic_support_checker is None:
        semantic_status = "not_evaluated"
        semantic_support = None
    else:
        semantic_support = semantic_support_checker(
            passage,
            source_node,
            relation,
            target_node,
        )
        if semantic_support is True:
            semantic_status = "supported"
        elif semantic_support is False:
            semantic_status = "not_supported"
        elif semantic_support is None:
            semantic_status = "inconclusive"
        else:
            raise TypeError(
                "semantic_support_checker must return bool or None"
            )

    if not reason and resolution_status != "resolved":
        reason = "Relevant passage could not be resolved"
    elif not reason and semantic_status == "not_evaluated":
        reason = "Citation retrieved, but claim support not yet assessed"
    elif not reason and semantic_status == "not_evaluable":
        reason = "Semantic support could not be evaluated without a passage"
    elif not reason and semantic_status == "inconclusive":
        reason = "Semantic support assessment was inconclusive"
    elif not reason and semantic_status == "not_supported":
        reason = "Resolved passage does not support the relationship"

    verification_method = (
        "caller_provided_semantic_support_checker"
        if semantic_support_checker is not None and passage
        else "pending"
    )
    semantic_support_result = {
        "supported": "supported",
        "not_supported": "unsupported",
        "inconclusive": "inconclusive",
    }.get(semantic_status, "unverified")

    return {
        "edge_exists": True,
        "citation": selected_citation,
        "citation_exists": source_exists,
        "source_resolved": resolution_status == "resolved",
        "semantic_support": semantic_support_result,
        "verification_method": verification_method,
        "source_exists": source_exists,
        "citation_existence_status": (
            "empty" if not selected_citation else
            "not_found" if not source_exists else
            "exists" if identity_verified else
            "ambiguous"
        ),
        "document_identity_verified": identity_verified,
        "source_document_ids": [
            str(_read_field(document, "id", ""))
            for document in source_matches
        ],
        "citation_resolution_status": resolution_status,
        "resolved_passage": passage,
        "lexical_support_status": (
            "lexically_supported" if lexical_supported else
            "not_lexically_supported" if passage else
            "not_evaluated"
        ),
        "semantic_support_status": semantic_status,
        "supports_relationship": semantic_support,
        "support_method": verification_method if passage else None,
        "verified": bool(
            cited_edges
            and source_exists
            and identity_verified
            and resolution_status == "resolved"
            and semantic_support is True
        ),
        "reason": reason,
    }


def audit_graph_citations(
    graph: Any,
    source_catalog: Iterable[Any],
    *,
    passage_resolver: Callable[[Any, Any, str, str, str], str | None] | None = None,
    semantic_support_checker: Callable[[str, str, str, str], bool | None] | None = None,
) -> dict:
    """Audit authored graph edges and report citations reused across relationships.

    Generated reverse edges (``rev_*``) are excluded because they are graph
    traversal artifacts, not separately sourced relationship claims.
    """
    source_catalog = _catalog_records(source_catalog)
    authored_edges = []
    citation_relationships: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

    for source, target, _key, data in graph.edges(keys=True, data=True):
        relation = str(data.get("rel_type", ""))
        if relation.startswith("rev_"):
            continue
        citation = str(data.get("citation", ""))
        edge = {
            "source": source,
            "relation": relation,
            "target": target,
            "citation": citation,
        }
        authored_edges.append(edge)
        if citation:
            citation_relationships[_normalize(citation)].add(
                (str(source), relation, str(target))
            )

    records = []
    for edge in authored_edges:
        result = verify_edge_citation(
            graph,
            edge["source"],
            edge["relation"],
            edge["target"],
            source_catalog,
            citation=edge["citation"],
            passage_resolver=passage_resolver,
            semantic_support_checker=semantic_support_checker,
        )
        related_claims = sorted(
            citation_relationships[_normalize(edge["citation"])]
        ) if edge["citation"] else []
        reused = len(related_claims) > 1
        reuse_assessment = (
            "review_required" if reused else "not_reused"
        )
        records.append({
            **edge,
            **result,
            "reused_across_relationships": reused,
            "reuse_assessment": reuse_assessment,
            "relationships_using_citation": [
                {"source": source, "relation": relation, "target": target}
                for source, relation, target in related_claims
            ],
        })

    return {
        "total_authored_edges": len(authored_edges),
        "source_references_found": sum(record["source_exists"] for record in records),
        "document_identities_verified": sum(
            record["document_identity_verified"] for record in records
        ),
        "passages_resolved": sum(
            record["citation_resolution_status"] == "resolved"
            for record in records
        ),
        "relationships_lexically_supported": sum(
            record["lexical_support_status"] == "lexically_supported"
            for record in records
        ),
        "relationships_semantically_supported": sum(
            record["semantic_support_status"] == "supported"
            for record in records
        ),
        "semantic_support_not_evaluated": sum(
            record["semantic_support_status"] in {"not_evaluated", "not_evaluable"}
            for record in records
        ),
        "verified_citations": sum(record["verified"] for record in records),
        "unverified_citations": sum(not record["verified"] for record in records),
        "reused_citations": len({
            _normalize(record["citation"])
            for record in records
            if record["reused_across_relationships"]
        }),
        "records": records,
        "semantic_support_limitation": (
            "Semantic support is only evaluated when a caller provides a "
            "semantic_support_checker; source existence and lexical overlap "
            "are not evidence of semantic support."
        ),
    }
