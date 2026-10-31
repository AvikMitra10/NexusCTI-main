from .evidence_validator import validate_evidence_path


def evaluate_multihop_path(
    graph,
    predicted_path,
    expected_min_hops=2
):
    validation = validate_evidence_path(
        graph,
        predicted_path
    )

    actual_hops = len(predicted_path)
    disconnected_edges = []
    for previous_item, item in zip(predicted_path, predicted_path[1:]):
        previous_target = previous_item[2]
        source, relation, target = item[:3]
        if previous_target != source:
            disconnected_edges.append({
                "source": source,
                "relation": relation,
                "target": target,
                "reason": "Path is disconnected from the previous edge"
            })

    path_connected = validation["continuity_valid"]
    path_valid = validation["valid"] and path_connected

    return {
        "is_multihop": actual_hops >= expected_min_hops,
        "path_connected": path_connected,
        "path_valid": path_valid,
        "valid_edges": validation["valid_edges"],
        "relationship_edges": validation["relationship_edges"],
        "citation_edges": validation["citation_edges"],
        "total_edges": validation["total_edges"],
        "success": (
            actual_hops >= expected_min_hops
            and path_valid
        ),
        "invalid_edges": validation["invalid_edges"],
        "disconnected_edges": disconnected_edges
    }
