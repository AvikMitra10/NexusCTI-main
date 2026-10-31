"""Audit this dataset against NexusCTI's current default graph and corpus."""

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from nexuscti.data.corpus import get_default_threat_corpus
from nexuscti.data.knowledge_base import build_default_knowledge_graph_data
from nexuscti.eval.leakage_checker import check_actor_label_leakage
from nexuscti.gnn.graph_builder import ThreatKnowledgeGraph


DATASET_DIR = Path(__file__).resolve().parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(value: str) -> str:
    value = value.lower().replace("[.]", ".").replace("[:]", ":")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _all_document_text(corpus: list[Any]) -> str:
    return "\n".join(
        " ".join(
            str(getattr(report, field, ""))
            for field in ("id", "title", "source", "summary", "text")
        )
        for report in corpus
    )


def _actor_labels(graph: Any) -> list[str]:
    labels = set()
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "ThreatActor":
            continue
        labels.add(str(node_id))
        names = re.split(r"[()/,;|]", str(data.get("name", "")))
        labels.update(name.strip() for name in names if name.strip())
        aliases = data.get("metadata", {}).get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        labels.update(str(alias) for alias in aliases if alias)
    return sorted(labels, key=str.casefold)


def _resolve_node(graph: Any, label: str) -> str | None:
    normalized_label = _normalize(label)
    if not normalized_label:
        return None

    for node_id, data in graph.nodes(data=True):
        candidates = [str(node_id), str(data.get("name", ""))]
        metadata = data.get("metadata", {})
        aliases = metadata.get("aliases", []) if isinstance(metadata, dict) else []
        if isinstance(aliases, str):
            aliases = [aliases]
        candidates.extend(str(alias) for alias in aliases or [])
        for candidate in candidates:
            normalized_candidate = _normalize(candidate)
            if normalized_label == normalized_candidate:
                return str(node_id)
            if len(normalized_label) >= 8 and normalized_label in normalized_candidate:
                return str(node_id)
    return None


def audit() -> dict[str, Any]:
    inputs = _read_jsonl(DATASET_DIR / "prediction_inputs.jsonl")
    truths = _read_jsonl(DATASET_DIR / "ground_truth.jsonl")
    source_registry = json.loads(
        (DATASET_DIR / "sources.json").read_text(encoding="utf-8")
    )["sources"]

    input_by_id = {record["case_id"]: record for record in inputs}
    truth_by_id = {record["case_id"]: record for record in truths}
    source_by_id = {record["source_id"]: record for record in source_registry}
    if len(input_by_id) != len(inputs) or set(input_by_id) != set(truth_by_id):
        raise ValueError("Prediction inputs and ground-truth case IDs must be unique and match.")

    nodes, edges = build_default_knowledge_graph_data()
    graph = ThreatKnowledgeGraph(nodes, edges).nx_graph
    corpus = get_default_threat_corpus()
    corpus_text = _normalize(_all_document_text(corpus))
    labels = _actor_labels(graph)
    ground_truth_labels = [
        str(truth["expected_actor"])
        for truth in truths
    ]
    all_scan_labels = sorted(set(labels + ground_truth_labels), key=str.casefold)
    findings = []

    for case_id, case_input in input_by_id.items():
        truth = truth_by_id[case_id]
        prediction_text = f"{case_input['query']}\n{case_input['anchor_ioc']}"
        label_scan = check_actor_label_leakage(prediction_text, all_scan_labels)
        normalized_anchor = _normalize(case_input["anchor_ioc"])
        anchor_in_index = bool(
            normalized_anchor and f" {normalized_anchor} " in f" {corpus_text} "
        )
        anchor_node = _resolve_node(graph, case_input["anchor_ioc"])
        relationship_checks = []

        for relationship in truth["expected_relationships"]:
            subject_node = _resolve_node(graph, relationship["subject"])
            object_node = _resolve_node(graph, relationship["object"])
            relation = relationship["predicate"]
            edge_exists = False
            if subject_node and object_node:
                edge_data = graph.get_edge_data(subject_node, object_node) or {}
                edge_exists = any(
                    data.get("rel_type") == relation
                    for data in edge_data.values()
                )

            subject_term = _normalize(relationship["subject"])
            object_term = _normalize(relationship["object"])
            relation_term = _normalize(relation)
            in_index = bool(
                subject_term
                and object_term
                and relation_term
                and all(term in corpus_text for term in (subject_term, object_term, relation_term))
            )
            relationship_checks.append({
                **relationship,
                "subject_resolves_in_graph": subject_node is not None,
                "object_resolves_in_graph": object_node is not None,
                "edge_exists_in_graph": edge_exists,
                "relationship_text_overlap_in_index": in_index,
            })

        source_ids = {
            relation["source_id"]
            for relation in truth["expected_relationships"]
        }
        missing_sources = [
            source_id
            for source_id in source_ids
            if source_id not in source_by_id
            or not source_by_id[source_id].get("url")
            or not source_by_id[source_id].get("supports")
        ]
        findings.append({
            "case_id": case_id,
            "direct_actor_label_leakage": label_scan["possible_leakage"],
            "detected_actor_labels": label_scan["detected_labels"],
            "anchor_resolves_in_graph": anchor_node is not None,
            "anchor_exact_text_in_retrieval_corpus": anchor_in_index,
            "ground_truth_actor_in_candidate_graph": truth["expected_actor"] in graph,
            "expected_relationships": relationship_checks,
            "missing_source_records": missing_sources,
        })

    graph_manifest = sorted(
        (
            str(source),
            str(target),
            str(data.get("rel_type", "")),
            str(data.get("citation", "")),
        )
        for source, target, data in graph.edges(data=True)
        if not str(data.get("rel_type", "")).startswith("rev_")
    )
    corpus_manifest = sorted(
        (
            str(report.id),
            str(report.title),
            str(report.target_actor),
            hashlib.sha256(str(report.text).encode("utf-8")).hexdigest(),
        )
        for report in corpus
    )

    report = {
        "dataset_id": "nexuscti-independent-test-v1",
        "case_count": len(inputs),
        "prediction_inputs_actor_label_leaks": sum(
            item["direct_actor_label_leakage"] for item in findings
        ),
        "anchors_resolving_in_graph": sum(
            item["anchor_resolves_in_graph"] for item in findings
        ),
        "anchors_found_exactly_in_retrieval_corpus": sum(
            item["anchor_exact_text_in_retrieval_corpus"] for item in findings
        ),
        "expected_relationships_found_in_graph": sum(
            relationship["edge_exists_in_graph"]
            for item in findings
            for relationship in item["expected_relationships"]
        ),
        "expected_relationships_with_text_overlap_in_corpus": sum(
            relationship["relationship_text_overlap_in_index"]
            for item in findings
            for relationship in item["expected_relationships"]
        ),
        "missing_source_records": sum(
            len(item["missing_source_records"]) for item in findings
        ),
        "graph_sha256": hashlib.sha256(
            json.dumps(graph_manifest, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "retrieval_corpus_sha256": hashlib.sha256(
            json.dumps(corpus_manifest, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "findings": findings,
        "scope": (
            "Exact anchor and relationship-overlap checks against the current "
            "default in-repository graph and report corpus; no guarantee about "
            "prior model exposure or external data stores."
        ),
    }
    report["independence_checks_passed"] = all((
        report["prediction_inputs_actor_label_leaks"] == 0,
        report["anchors_resolving_in_graph"] == 0,
        report["anchors_found_exactly_in_retrieval_corpus"] == 0,
        report["expected_relationships_found_in_graph"] == 0,
        report["expected_relationships_with_text_overlap_in_corpus"] == 0,
        report["missing_source_records"] == 0,
    ))
    if not report["independence_checks_passed"]:
        raise ValueError("Independent test-set overlap audit failed.")
    return report


if __name__ == "__main__":
    report = audit()
    output_path = DATASET_DIR / "overlap_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in report.items()
        if key not in {"findings"}
    }, indent=2))
