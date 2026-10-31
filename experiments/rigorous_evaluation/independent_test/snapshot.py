"""Canonical fingerprints for the frozen NexusCTI graph and report corpus."""

import hashlib
import json
from typing import Any

from nexuscti.data.corpus import get_default_threat_corpus
from nexuscti.data.knowledge_base import build_default_knowledge_graph_data


GRAPH_VERSION = "nexuscti-default-graph-2026-10-30-v1"
CORPUS_VERSION = "nexuscti-default-corpus-2026-10-30-v1"


def _sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def frozen_snapshot() -> dict[str, Any]:
    nodes, edges = build_default_knowledge_graph_data()
    graph_payload = {
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [edge.model_dump(mode="json") for edge in edges],
    }
    reports = get_default_threat_corpus()
    corpus_payload = [
        report.model_dump(mode="json")
        for report in reports
    ]
    return {
        "graph": {
            "version": GRAPH_VERSION,
            "sha256": _sha256(graph_payload),
            "node_count": len(nodes),
            "authored_edge_count": len(edges),
        },
        "retrieval_corpus": {
            "version": CORPUS_VERSION,
            "sha256": _sha256(corpus_payload),
            "document_count": len(reports),
        },
    }
