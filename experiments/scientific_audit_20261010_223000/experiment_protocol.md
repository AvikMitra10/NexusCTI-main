# Controlled Ablation Experiment Protocol

**Experiment ID:** `scientific_audit_20261010_223000`  
**Date:** October 10, 2026  
**Evaluation Dataset:** 12 Standardized Multi-Hop Benchmark Cases (`nexuscti/eval/benchmark_cases.py`)  
**Evaluation Scope:** Closed-world APT attribution across 6 primary threat actors (`APT28`, `APT29`, `Lazarus Group`, `Sandworm Team`, `FIN7`, `Volt Typhoon`).

---

## 1. Experimental Objectives

To isolate and quantify the specific performance contributions of each architectural module in **NexusCTI**:
1. Does multi-hop graph structure compensate for the known failure mode of vector retrieval on isolated indicators?
2. Does a learned Relational Graph Convolutional Network (R-GCN) outperform a non-neural shortest-path graph baseline?
3. Does synergistic cross-validation fusion (GNN + RAG + consistency validation) outperform simple heuristic combinations?

---

## 2. The Five Evaluated Configurations

| Configuration | Designation | Active Modules | Inactive / Bypassed Modules | Mathematical Scoring Formulation |
| :--- | :--- | :--- | :--- | :--- |
| **Config A** | **Vector-only RAG** | Dense text retrieval only: `DenseRetriever.search` embeds the query and corpus chunks with `SemanticTextEncoder`, then ranks chunks by cosine similarity; retrieved chunk scores are aggregated by actor | NetworkX/graph topology, graph-adjacency tensor operations, GNN forward pass, BM25/hybrid retrieval, cross-validation | $S_A(a) = \sum_{c \in \text{top-5 chunks}(q)} \max(0, \text{Sim}_{\cos}(E_q, E_c))$ |
| **Config B** | **Graph-only (Topological)** | NetworkX unweighted shortest-path calculation: convert the knowledge graph to an undirected view, then call `nx.shortest_path_length` from the known IOC to each actor | Text chunks, dense or sparse retrieval, embedding lookups, R-GCN neural weights | $S_B(a) = \frac{1.0}{1.0 + d(u, a)}$ where $d$ is the unweighted shortest-path length |
| **Config C** | **GNN-only (Neural)** | Cached `HeteroThreatGNN` node embeddings and neural actor-affinity scores only; no NetworkX distance term | Text corpus, vector retriever, BM25 keyword index, NetworkX shortest-path scoring | $S_C(a) = \text{Softmax}(2.5 \cdot \cos(z_u, z_a))$ |
| **Config D** | **Graph + RAG (Non-Neural)** | Symbolic graph distance $S_B(a)$ + Text retrieval score $S_A(a)$ | Learned R-GCN neural embeddings, consistency matrix | $S_D(a) = 0.5 \cdot S_B(a) + 0.5 \cdot S_A(a)$ |
| **Config E** | **Full Hybrid (NexusCTI)** | GNN neural structural score $S_C(a)$ + RAG hybrid text score $S_A(a)$ + Cross-Validation $C(a)$ | None (Complete unified framework) | $S_E(a) = 0.45 \cdot S_C(a) + 0.35 \cdot S_A(a) + 0.20 \cdot C_{\text{score}}(a)$ |

---

## 3. Strict Verification of Component Isolation

### Shared service initialization and trace scope

The Phase 3 runner obtains its service through `get_service()`, which returns a
module-level singleton. One `NexusCTIService` is constructed before the
configuration predictions and reused across all five configurations. That
construction builds the NetworkX graph, trains the GNN, loads the corpus,
constructs the dense/BM25 retrievers and their indexes, and initializes
fusion. Consequently, the full experiment is **not process-level or
initialization-level isolated**: each configuration runs with access to this
shared, fully initialized service state, and setup operations are not repeated
or attributed separately per configuration.

The allowed/forbidden operation lists below describe **per-case prediction
calls only**, after this shared setup. They do not establish that forbidden
components were absent from the process, unavailable to the predictor, or
unused during initialization. Any trace-based isolation claim must separate
service construction from the prediction method and must not describe these
comparisons as independently initialized configurations.

All 12 built-in anchor IOCs resolve to existing graph nodes. Config C now
rejects an anchor that is absent from the trained graph instead of adding a
temporary IOC node, so Config C cannot mutate graph state that later
configurations observe. This restriction applies to the ablation predictor;
it does not change the service's general novel-IOC behavior in other workflows.

- **Config A Prediction Trace:** Must show `DenseRetriever.search` query encoding, dense-vector similarity, and retrieval over text chunks. During the prediction call, must show zero NetworkX calls, graph reads, graph-adjacency tensor operations, BM25 retrieval, and GNN forward passes. PyTorch operations used internally for dense embeddings/similarity are allowed; they are not graph-adjacency operations. The shared service initialization still constructs the graph and trains the GNN.
- **Config B Prediction Trace:** Must show construction of the undirected NetworkX graph view and `nx.shortest_path_length` calls for the IOC-to-actor distances. This is NetworkX's unweighted shortest-path API (breadth-first distance), not Dijkstra. During the prediction call, must show zero text-chunk access, retrieval calls, embedding lookups, and GNN forward passes. The shared service initialization still loads and indexes the text corpus and trains the GNN.
- **Config C Prediction Trace:** Must show actor affinity computed only from cached GNN node embeddings, with zero NetworkX path-distance calls, text queries, or retrieval calls during prediction. The embeddings were produced by GNN message passing over tensor features and relation adjacency lists during shared service initialization; this is not an independently trained configuration. Unknown anchors must be rejected without adding nodes or rebuilding graph tensors.
- **Config D Prediction Trace:** Uses only scalar topological graph distance and scalar dense-text similarity; no neural GNN forward pass is invoked during prediction. Both graph and retriever state are shared from service initialization.
- **Config E Prediction Trace:** Must show the full `service.analyze` pipeline: GNN attribution, hybrid text retrieval, graph-trail extraction through NetworkX simple-path search, and topological-textual cross-validation/fusion. No component exclusions apply.

---

## 4. Evaluation Metrics & Standards

- **Top-1 Accuracy:** $\frac{\sum_{i=1}^N \mathbb{I}(\text{rank}_i = 1)}{N} \times 100\%$
- **Top-3 Accuracy:** $\frac{\sum_{i=1}^N \mathbb{I}(\text{rank}_i \le 3)}{N} \times 100\%$
- **Mean Reciprocal Rank (MRR):** $\frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$ (where $\text{rank}_i > 10 \implies 0$)
- **Multi-Hop Attribution Accuracy:** Percentage of cases with `expected_hops >= 2` (11 cases) where the ground-truth actor is ranked first. A valid graph trail or citation is not required for this ranking metric. `expected_hops` records the shortest directed path edge count from the anchor to its ground-truth actor in the benchmark graph; the scenario `category` labels the case archetype and is not a hop-count field.
- **Multi-Hop Evidence Validity:** Percentage of those same multi-hop cases for which the returned trail has at least two edges, is connected in sequence, and every edge exists in the graph with the stated relationship type and a matching, non-empty citation. This is measured independently of whether the ground-truth actor ranked first; report the count and denominator. Citation validity here means citation-field presence/match only, not that the source was resolved or substantively supports the relationship.
- **Hallucination / Misattribution Rate:** Percentage of test cases where the top candidate is incorrect.
- **Sample Size Context:** $N = 12$ cases. Each case represents $\approx 8.33\%$ accuracy.
