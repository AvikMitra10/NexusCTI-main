"""Verify split separation and frozen graph/corpus fingerprints."""

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexuscti.data.corpus import get_default_threat_corpus
from nexuscti.data.knowledge_base import build_default_knowledge_graph_data
from nexuscti.eval.leakage_checker import check_actor_label_leakage
from nexuscti.gnn.graph_builder import ThreatKnowledgeGraph
from experiments.rigorous_evaluation.independent_test.snapshot import frozen_snapshot


HERE = Path(__file__).resolve().parent
SPLIT_PATH = PROJECT_ROOT / "experiments" / "rigorous_evaluation" / "data_split.json"
RELATION_CUES = {
    "targets": ("target", "targeting", "targets", "targeted"),
    "exploits": ("exploit", "exploits", "exploiting", "exploited"),
}


def _normalize(value: str) -> str:
    value = value.lower().replace("[.]", ".").replace("[:]", ":")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_node(graph: Any, label: str) -> str | None:
    normalized_label = _normalize(label)
    if not normalized_label:
        return None
    for node_id, data in graph.nodes(data=True):
        names = [str(node_id), str(data.get("name", ""))]
        metadata = data.get("metadata", {})
        aliases = metadata.get("aliases", []) if isinstance(metadata, dict) else []
        if isinstance(aliases, str):
            aliases = [aliases]
        names.extend(str(alias) for alias in aliases or [])
        if normalized_label in {_normalize(name) for name in names}:
            return str(node_id)
    return None


def audit() -> dict[str, Any]:
    split = _read_json(SPLIT_PATH)
    inputs_doc = _read_json(HERE / "test_cases.json")
    truth_doc = _read_json(HERE / "ground_truth.json")
    sources_doc = _read_json(HERE / "source_registry.json")
    inputs = inputs_doc["cases"]
    truths = truth_doc["cases"]
    sources = {source["source_id"]: source for source in sources_doc["sources"]}

    input_ids = [case["case_id"] for case in inputs]
    truth_ids = [case["case_id"] for case in truths]
    if len(set(input_ids)) != len(input_ids) or set(input_ids) != set(truth_ids):
        raise ValueError("Prediction and ground-truth case IDs must be unique and match.")
    if inputs_doc["dataset_id"] != truth_doc["dataset_id"]:
        raise ValueError("Prediction inputs and labels must identify the same dataset.")

    snapshot = frozen_snapshot()
    frozen = split["frozen_training_data"]
    if snapshot != {
        key: frozen[key]
        for key in ("graph", "retrieval_corpus")
    }:
        raise ValueError("The graph or retrieval corpus differs from the frozen snapshot.")

    nodes, edges = build_default_knowledge_graph_data()
    graph = ThreatKnowledgeGraph(nodes, edges).nx_graph
    corpus_text = _normalize("\n".join(
        " ".join(
            str(getattr(report, field, ""))
            for field in ("id", "title", "source", "summary", "text")
        )
        for report in get_default_threat_corpus()
    ))
    graph_actor_labels = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "ThreatActor":
            graph_actor_labels.append(str(node_id))
            graph_actor_labels.extend(
                part.strip()
                for part in re.split(r"[()/,;|]", str(data.get("name", "")))
                if part.strip()
            )
            aliases = data.get("metadata", {}).get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            graph_actor_labels.extend(str(alias) for alias in aliases if alias)

    findings = []
    for case_input in inputs:
        truth = next(case for case in truths if case["case_id"] == case_input["case_id"])
        label_scan = check_actor_label_leakage(
            f"{case_input['query']}\n{case_input['anchor_ioc']}",
            graph_actor_labels + [
                truth["expected_actor_name"],
                *truth.get("expected_actor_aliases", []),
            ],
        )
        normalized_anchor = _normalize(case_input["anchor_ioc"])
        anchor_in_corpus = bool(
            normalized_anchor and f" {normalized_anchor} " in f" {corpus_text} "
        )
        anchor_in_graph = _resolve_node(graph, case_input["anchor_ioc"]) is not None
        relation_checks = []

        for claim in truth["expected_relationships"]:
            subject = _resolve_node(graph, claim["subject"])
            object_node = _resolve_node(graph, claim["object"])
            edge_exists = False
            if subject and object_node:
                edge_data = graph.get_edge_data(subject, object_node) or {}
                edge_exists = any(
                    data.get("rel_type") == claim["predicate"]
                    for data in edge_data.values()
                )
            normalized_subject = _normalize(claim["subject"])
            normalized_object = _normalize(claim["object"])
            normalized_corpus = f" {corpus_text} "
            endpoint_text_overlap = bool(
                normalized_subject
                and normalized_object
                and f" {normalized_subject} " in normalized_corpus
                and f" {normalized_object} " in normalized_corpus
            )
            relation_cues = RELATION_CUES.get(
                claim["predicate"],
                (claim["predicate"],),
            )
            relation_text_overlap = any(
                f" {_normalize(cue)} " in normalized_corpus
                for cue in relation_cues
            )
            relation_checks.append({
                "subject": claim["subject"],
                "predicate": claim["predicate"],
                "object": claim["object"],
                "edge_exists_in_frozen_graph": edge_exists,
                "subject_and_object_text_in_frozen_corpus": endpoint_text_overlap,
                "relation_cue_in_frozen_corpus": relation_text_overlap,
                "relationship_text_overlap_in_frozen_corpus": (
                    endpoint_text_overlap and relation_text_overlap
                ),
                "source_record_exists": claim["source_id"] in sources,
            })

        findings.append({
            "case_id": case_input["case_id"],
            "actor_label_leak": label_scan["possible_leakage"],
            "detected_actor_labels": label_scan["detected_labels"],
            "anchor_in_frozen_graph": anchor_in_graph,
            "anchor_in_frozen_corpus": anchor_in_corpus,
            "expected_relationships": relation_checks,
        })

    summary = {
        "dataset_id": inputs_doc["dataset_id"],
        "case_count": len(inputs),
        "prediction_inputs_label_leaks": sum(item["actor_label_leak"] for item in findings),
        "anchors_in_frozen_graph": sum(item["anchor_in_frozen_graph"] for item in findings),
        "anchors_in_frozen_corpus": sum(item["anchor_in_frozen_corpus"] for item in findings),
        "expected_edges_in_frozen_graph": sum(
            claim["edge_exists_in_frozen_graph"]
            for item in findings
            for claim in item["expected_relationships"]
        ),
        "expected_relationship_text_overlaps": sum(
            claim["relationship_text_overlap_in_frozen_corpus"]
            for item in findings
            for claim in item["expected_relationships"]
        ),
        "all_sources_registered": all(
            claim["source_record_exists"]
            for item in findings
            for claim in item["expected_relationships"]
        ),
        "frozen_training_data": snapshot,
        "findings": findings,
    }
    summary["independence_checks_passed"] = all((
        summary["prediction_inputs_label_leaks"] == 0,
        summary["anchors_in_frozen_graph"] == 0,
        summary["anchors_in_frozen_corpus"] == 0,
        summary["expected_edges_in_frozen_graph"] == 0,
        summary["expected_relationship_text_overlaps"] == 0,
        summary["all_sources_registered"],
    ))
    if not summary["independence_checks_passed"]:
        raise ValueError("Frozen-split independence audit failed.")
    return summary


if __name__ == "__main__":
    report = audit()
    output = HERE / "frozen_overlap_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, indent=2))
