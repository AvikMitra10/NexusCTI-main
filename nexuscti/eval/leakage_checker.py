"""Leakage checks for evaluation data and protocols."""

import re
from collections.abc import Iterable
from typing import Any

import networkx as nx


def check_actor_label_leakage(
    prediction_text: str,
    actor_labels: list[str]
) -> dict:
    """
    Detect actor names appearing in prediction input.
    This is a warning tool, not proof of leakage.
    """

    text = prediction_text.lower()
    found = []

    for actor in actor_labels:
        if not actor:
            continue

        pattern = r"(?<!\w)" + re.escape(actor.lower()) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(actor)

    return {
        "possible_leakage": len(found) > 0,
        "detected_labels": found
    }


def _normalize(text: str) -> str:
    normalized = text.lower().replace("[.]", ".").replace("[:]", ":")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _actor_labels(graph: Any) -> list[str]:
    labels = set()
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "ThreatActor":
            continue
        labels.add(str(node_id))
        if data.get("name"):
            names = re.split(r"[()/,;|]", str(data["name"]))
            labels.update(name.strip() for name in names if name.strip())
        metadata = data.get("metadata", {})
        aliases = metadata.get("aliases", []) if isinstance(metadata, dict) else []
        if isinstance(aliases, str):
            aliases = [aliases]
        labels.update(str(alias) for alias in aliases if alias)
    return sorted(labels, key=str.casefold)


def _resolve_graph_node(graph: Any, indicator: str) -> str | None:
    normalized_indicator = _normalize(indicator)
    if not normalized_indicator:
        return None

    for node_id, data in graph.nodes(data=True):
        candidates = [str(node_id), str(data.get("name", ""))]
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            aliases = metadata.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            candidates.extend(str(alias) for alias in aliases)

        for candidate in candidates:
            normalized_candidate = _normalize(candidate)
            if normalized_indicator == normalized_candidate:
                return str(node_id)
            # Names in test anchors can include a parenthetical malware alias.
            if len(normalized_indicator) >= 8 and normalized_indicator in normalized_candidate:
                return str(node_id)
    return None


def _report_text(report: Any) -> str:
    if isinstance(report, dict):
        return " ".join(
            str(report.get(field, ""))
            for field in ("title", "source", "summary", "text")
        )
    return " ".join(
        str(getattr(report, field, ""))
        for field in ("title", "source", "summary", "text")
    )


def _report_field(report: Any, field: str, default: Any = None) -> Any:
    return report.get(field, default) if isinstance(report, dict) else getattr(report, field, default)


def _case_field(case: Any, field: str, default: Any = "") -> Any:
    return case.get(field, default) if isinstance(case, dict) else getattr(case, field, default)


def audit_benchmark_leakage(
    test_cases: Iterable[Any],
    graph: Any,
    corpus: Iterable[Any],
) -> dict:
    """Audit benchmark labels and expected facts against predictor inputs and data.

    The current evaluator sends only ``query`` and ``anchor_ioc`` to the
    prediction methods. This audit separately checks direct lexical exposure,
    benchmark metadata overlap, graph connectivity, and matching corpus reports.
    """
    cases = list(test_cases)
    reports = list(corpus)
    labels = _actor_labels(graph)
    case_results = []

    for case in cases:
        query = str(_case_field(case, "query"))
        anchor = str(_case_field(case, "anchor_ioc"))
        truth = str(_case_field(case, "ground_truth_actor"))
        description = str(_case_field(case, "description"))
        prediction_input = f"{query}\n{anchor}"
        label_scan = check_actor_label_leakage(prediction_input, labels)

        actor_node_exists = bool(truth and truth in graph)
        anchor_node = _resolve_graph_node(graph, anchor)
        graph_path = False
        if actor_node_exists and anchor_node:
            try:
                graph_path = nx.has_path(graph, anchor_node, truth)
            except nx.NetworkXError:
                graph_path = False

        normalized_anchor = _normalize(anchor)
        matching_reports = []
        actor_reports = []
        for report in reports:
            report_id = str(_report_field(report, "id", ""))
            report_actor = str(_report_field(report, "target_actor", ""))
            text = _report_text(report)
            normalized_report = _normalize(text)
            if truth and report_actor == truth:
                actor_reports.append(report_id)
            if normalized_anchor and normalized_anchor in normalized_report:
                matching_reports.append(report_id)

        description_labels = check_actor_label_leakage(description, labels)
        name_labels = check_actor_label_leakage(
            str(_case_field(case, "name")),
            labels,
        )
        case_results.append({
            "case_id": str(_case_field(case, "id")),
            "prediction_input_fields": ["query", "anchor_ioc"],
            "benchmark_metadata_fields_not_passed": [
                "name",
                "ground_truth_actor",
                "expected_hops",
                "category",
                "description",
            ],
            "ground_truth_metadata_passed_to_predictor": False,
            "actor_label_in_prediction_input": label_scan["possible_leakage"],
            "detected_actor_labels": label_scan["detected_labels"],
            "actor_label_in_benchmark_name": name_labels["possible_leakage"],
            "actor_label_in_benchmark_description": description_labels["possible_leakage"],
            "ground_truth_actor_in_graph": actor_node_exists,
            "ground_truth_actor_description_available_in_graph": bool(
                actor_node_exists and graph.nodes[truth].get("description")
            ),
            "anchor_resolved_in_graph": anchor_node is not None,
            "anchor_graph_node": anchor_node,
            "preexisting_graph_path_to_ground_truth": graph_path,
            "expected_relationships_available_in_graph": graph_path,
            "actor_labeled_corpus_reports": actor_reports,
            "corpus_reports_containing_anchor": matching_reports,
            "retrieval_corpus_exposes_actor_labels": bool(actor_reports),
            "benchmark_description_passed_to_predictor": False,
            "expected_hops_passed_to_predictor": False,
        })

    count = len(case_results)
    direct_label_exposures = sum(
        result["actor_label_in_prediction_input"] for result in case_results
    )
    graph_path_exposures = sum(
        result["preexisting_graph_path_to_ground_truth"] for result in case_results
    )
    corpus_overlaps = sum(
        bool(result["actor_labeled_corpus_reports"] or result["corpus_reports_containing_anchor"])
        for result in case_results
    )
    benchmark_name_exposures = sum(
        result["actor_label_in_benchmark_name"] for result in case_results
    )
    benchmark_description_exposures = sum(
        result["actor_label_in_benchmark_description"] for result in case_results
    )
    actor_labeled_reports = sum(
        bool(result["actor_labeled_corpus_reports"]) for result in case_results
    )
    anchor_corpus_matches = sum(
        bool(result["corpus_reports_containing_anchor"]) for result in case_results
    )
    expected_graph_paths = sum(
        result["expected_relationships_available_in_graph"] for result in case_results
    )

    return {
        "total_cases": count,
        "direct_actor_label_exposures": direct_label_exposures,
        "actor_labels_in_benchmark_names": benchmark_name_exposures,
        "actor_labels_in_benchmark_descriptions": benchmark_description_exposures,
        "preexisting_graph_paths_to_ground_truth": graph_path_exposures,
        "expected_relationships_available_in_graph": expected_graph_paths,
        "cases_overlapping_actor_or_anchor_corpus_data": corpus_overlaps,
        "cases_with_actor_labeled_retrieval_reports": actor_labeled_reports,
        "cases_with_anchor_text_in_retrieval_corpus": anchor_corpus_matches,
        "ground_truth_metadata_passed_to_predictor": False,
        "prediction_inputs": ["query", "anchor_ioc"],
        "findings": case_results,
        "interpretation": (
            "Labels and benchmark-only metadata are not passed directly to the "
            "prediction calls. Actor identities, graph relationships, and some "
            "test indicators may nevertheless already be represented in the "
            "knowledge graph or report corpus; this is data overlap and can "
            "inflate benchmark performance."
        ),
    }
