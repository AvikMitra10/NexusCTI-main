def validate_evidence_path(graph, path):
    """
    Validate graph relationships and citation consistency
    against a NetworkX MultiDiGraph.

    Each path item:
        (source_node, relationship_type, target_node)

    An optional fourth item provides the citation from the trail.
    """

    if not path:
        return {
            "valid": False,
            "valid_edges": 0,
            "relationship_edges": 0,
            "citation_edges": 0,
            "total_edges": 0,
            "continuity_valid": False,
            "invalid_edges": [],
            "reason": "Empty evidence path"
        }

    valid_edges = 0
    relationship_edges = 0
    citation_edges = 0
    invalid_edges = []

    for item in path:
        source, relation, target = item[:3]
        trail_citation = item[3] if len(item) > 3 else None

        if not graph.has_edge(source, target):
            invalid_edges.append({
                "source": source,
                "relation": relation,
                "target": target,
                "reason": "Edge not found"
            })
            continue

        edge_data = graph.get_edge_data(source, target)

        matching_edges = [
            data for data in edge_data.values()
            if data.get("rel_type") == relation
        ]

        if not matching_edges:
            invalid_edges.append({
                "source": source,
                "relation": relation,
                "target": target,
                "reason": "Relationship type mismatch"
            })
            continue

        relationship_edges += 1

        if trail_citation:
            cited_edges = [
                data for data in matching_edges
                if data.get("citation") == trail_citation
            ]
        else:
            cited_edges = [
                data for data in matching_edges
                if data.get("citation")
            ]

        if cited_edges:
            citation_edges += 1
        else:
            invalid_edges.append({
                "source": source,
                "relation": relation,
                "target": target,
                "reason": (
                    "Trail citation does not match graph edge"
                    if trail_citation
                    else "Missing citation"
                )
            })
            continue

        valid_edges += 1

    continuity_valid = all(
        path[index][2] == path[index + 1][0]
        for index in range(len(path) - 1)
    )

    if not continuity_valid:
        invalid_edges.append({
            "reason": "Disconnected consecutive hops"
        })

    return {
        "valid": len(invalid_edges) == 0 and continuity_valid,
        "valid_edges": valid_edges,
        "relationship_edges": relationship_edges,
        "citation_edges": citation_edges,
        "total_edges": len(path),
        "continuity_valid": continuity_valid,
        "invalid_edges": invalid_edges
    }
