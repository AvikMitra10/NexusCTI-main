# NexusCTI Methodology

NexusCTI combines structured threat-intelligence relationships with narrative retrieval to rank candidate threat actors and present supporting context. Its output is a model-assisted attribution hypothesis, not a proof of attribution.

## Knowledge graph reasoning

The knowledge graph represents threat actors, malware, indicators, techniques, vulnerabilities, and their relationships. It is loaded as a directed, multi-edge NetworkX graph; each source relationship is also represented in reverse for message passing and graph traversal. A graph trail extractor searches for a best path from the incident indicator to each candidate actor, up to five hops, and includes the nodes, relationship types, and citations recorded on those edges.

The graph is curated data, not a live or exhaustive source of threat intelligence. A previously unknown indicator may be added as a temporary node, but this alone does not establish a relationship to an actor.

## Graph neural network predictions

The structural model is a two-layer relational graph convolutional network. It propagates node features over relation-specific graph edges and learns actor affinity from labels assigned to entities reachable from actors within three graph hops. At inference, it combines latent embedding affinity with graph shortest-path proximity, then normalizes the resulting actor scores.

This makes the GNN dependent on the graph's topology and its derived supervision labels. Its scores should not be interpreted as calibrated probabilities or as independent confirmation of graph relationships.

## BM25 and dense retrieval

Threat reports are split into overlapping text chunks. Two retrieval methods search these chunks:

- **BM25 sparse retrieval** tokenizes text while preserving CTI-like tokens, including defanged domains, IP addresses, CVEs, ATT&CK identifiers, and hashes. BM25 scores exact lexical overlap, with document-length normalization.
- **Dense retrieval** encodes words and character trigrams into deterministic, hash-based vectors and ranks chunks by cosine similarity. This implementation uses a fixed random projection, not a pretrained language model; its similarity is a lightweight lexical/morphological signal rather than deep semantic understanding.

## Hybrid ranking and evidence fusion

The hybrid retriever takes up to twice the requested number of results from each retriever, then merges their rankings with Reciprocal Rank Fusion (RRF):

`RRF(chunk) = 0.6 / (60 + sparse_rank) + 0.4 / (60 + dense_rank)`

Ranks are one-based. The top fused chunks are returned as narrative proofs with their source, date, actor label, text, matched indicators, and retrieval score. For text-based actor ranking, proof scores are summed by actor and exact indicator matches receive a 1.5x boost before normalization.

The fusion engine combines structural confidence, textual confidence, and a cross-validation heuristic:

`fused_score = 0.45 * structural_confidence + 0.35 * textual_confidence + 0.20 * cross_validation_score`

The cross-validation heuristic checks whether retrieved actor-specific passages mention endpoints in the graph trail. Candidates are ranked by fused score; configured thresholds assign descriptive verdicts. The rigorous evaluator separately checks trail-edge existence, relation types, and path connectivity. [citation_verifier.py](../nexuscti/eval/citation_verifier.py) reports citation existence in a supplied catalog, passage resolution, and semantic support as separate levels. Its default passage resolver finds a passage containing both endpoints, while lexical overlap is only a diagnostic. Semantic support remains `not_evaluated` unless a caller provides a semantic-support checker; the verifier never treats document existence or lexical overlap alone as semantic support.

Run the citation audit by passing the NetworkX graph and a catalog of documents with `id`, `title`, `source`, and `text` fields. The bundled report corpus can be used as an initial, incomplete catalog. To assess semantic support, provide a `semantic_support_checker` to `verify_edge_citation`; it receives the resolved passage and the subject–relationship–object claim and must return `True`, `False`, or `None` for inconclusive:

Each edge result includes `citation_exists`, `source_resolved`,
`semantic_support`, `verification_method`, and `reason`. Without a semantic
checker, a retrieved citation reports `semantic_support: "unverified"` and
`verification_method: "pending"`; source existence and passage retrieval alone
do not establish claim support.

```python
from nexuscti.data.corpus import get_default_threat_corpus
from nexuscti.eval.citation_verifier import audit_graph_citations
from nexuscti.service import get_service

service = get_service()
audit = audit_graph_citations(
    service.tkg.nx_graph,
    get_default_threat_corpus(),
)
```

Unmatched citations should be treated as unresolved, not as evidence that a source does not exist; add an authoritative source record or citation alias to the catalog to check them.

## Evaluation scope

The existing benchmark compares dense-only retrieval, a standalone GNN, and the hybrid engine. The hybrid result exposes its trail and narrative proofs, but the existing detailed benchmark records do not store those evidence items. See [evaluation_metrics.md](./evaluation_metrics.md) for the distinction between currently reported benchmark metrics and evidence-based measurements.
