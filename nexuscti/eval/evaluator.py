"""
Empirical Benchmark Evaluation Harness for NexusCTI.
Compares Hybrid GNN+RAG against Naive Vector RAG and Standalone GNN baselines.
"""

from typing import List, Dict, Any, Tuple, Callable
from collections import defaultdict

import networkx as nx

from .benchmark_cases import BenchmarkTestCase, get_benchmark_dataset
from .baselines import NaiveVectorRAGBaseline, StandaloneGNNBaseline
from .multihop_evaluator import evaluate_multihop_path
from ..data.ontology import NodeType
from ..fusion.engine import SynergisticFusionEngine


class BenchmarkResults(dict):
    """Container for benchmark metrics."""
    pass


class AttributionEvaluator:
    """
    Executes standardized empirical benchmarking across all evaluation dimensions.
    """

    def __init__(
        self,
        fusion_engine: SynergisticFusionEngine,
        vector_baseline: NaiveVectorRAGBaseline,
        gnn_baseline: StandaloneGNNBaseline,
        test_cases: List[BenchmarkTestCase] = None
    ):
        self.fusion_engine = fusion_engine
        self.vector_baseline = vector_baseline
        self.gnn_baseline = gnn_baseline
        self.test_cases = test_cases or get_benchmark_dataset()

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs complete benchmark evaluation comparing the 3 architectures.
        """
        models = {
            "Naive Vector-only RAG": self._evaluate_vector_rag,
            "Standalone GNN Classifier": self._evaluate_standalone_gnn,
            "NexusCTI Hybrid (GNN + RAG)": self._evaluate_hybrid
        }

        benchmark_summary: Dict[str, Any] = {}
        detailed_records: List[Dict[str, Any]] = []

        for model_name, eval_fn in models.items():
            metrics, records = eval_fn()
            benchmark_summary[model_name] = metrics
            for rec in records:
                rec["model"] = model_name
                detailed_records.append(rec)

        return {
            "summary": benchmark_summary,
            "detailed_case_evaluations": detailed_records,
            "total_test_cases": len(self.test_cases)
        }

    def run_ablation_study(self) -> Dict[str, Any]:
        """Compare retrieval and reasoning components on the same test cases."""
        models = {
            "Vector RAG": lambda case: self.vector_baseline.predict(case.query),
            "BM25 Retrieval": lambda case: self._predict_bm25(case.query),
            "GNN": lambda case: self.gnn_baseline.predict(case.anchor_ioc),
            "Knowledge Graph Reasoning": self._predict_knowledge_graph,
            "Hybrid NexusCTI": self._predict_hybrid,
        }

        summary: Dict[str, Any] = {}
        detailed_records: List[Dict[str, Any]] = []
        for model_name, predictor in models.items():
            metrics, records = self._evaluate_ablation_predictions(predictor)
            summary[model_name] = metrics
            for record in records:
                record["model"] = model_name
                detailed_records.append(record)

        return {
            "experiment": "component_ablation",
            "dataset": "built-in benchmark cases",
            "total_test_cases": len(self.test_cases),
            "summary": summary,
            "detailed_case_evaluations": detailed_records,
            "interpretation_note": (
                "Exploratory component comparison on the built-in benchmark. "
                "Its cases and labels overlap the graph and report corpus; this "
                "is not an independent generalization estimate."
            ),
        }

    def _predict_bm25(self, query: str) -> List[Tuple[str, float]]:
        """Rank actors by BM25-only retrieval scores, without dense fusion."""
        retriever = self.fusion_engine.retriever.bm25
        results = retriever.search(query, top_k=5)
        actor_scores: Dict[str, float] = defaultdict(float)
        for chunk, score in results:
            if score > 0:
                actor_scores[chunk.target_actor] += score

        total_score = sum(actor_scores.values())
        if not total_score:
            return []

        ranked = [
            (actor, round(score / total_score, 4))
            for actor, score in actor_scores.items()
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def _predict_knowledge_graph(
        self,
        case: BenchmarkTestCase
    ) -> List[Tuple[str, float]]:
        """Rank actors by shortest directed path from a known anchor IOC."""
        tkg = self.fusion_engine.tkg
        anchor = tkg.find_node_by_indicator(case.anchor_ioc)
        if anchor is None:
            return []

        graph = tkg.nx_graph
        actor_distances = []
        for node in tkg.raw_nodes:
            if node.type != NodeType.THREAT_ACTOR:
                continue
            try:
                distance = nx.shortest_path_length(
                    graph,
                    source=anchor.id,
                    target=node.id,
                )
            except nx.NetworkXNoPath:
                continue
            if distance > 0:
                actor_distances.append((node.id, distance))

        actor_distances.sort(key=lambda item: (item[1], item[0]))
        scores = [(actor, 1.0 / distance) for actor, distance in actor_distances]
        total_score = sum(score for _, score in scores)
        if not total_score:
            return []
        return [
            (actor, round(score / total_score, 4))
            for actor, score in scores
        ]

    def _predict_hybrid(self, case: BenchmarkTestCase) -> List[Tuple[str, float]]:
        result = self.fusion_engine.analyze_incident(
            query=case.query,
            ioc_override=case.anchor_ioc,
        )
        return [
            (candidate.actor_id, candidate.fused_confidence)
            for candidate in result.all_candidates
        ]

    def _evaluate_ablation_predictions(
        self,
        predictor: Callable[
            [BenchmarkTestCase],
            List[Tuple[str, float]],
        ],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        multihop_hits = 0
        multihop_total = 0
        reciprocal_ranks = []
        records = []

        for case in self.test_cases:
            predictions = predictor(case)
            pred_actors = [actor for actor, _ in predictions]
            is_top1 = bool(pred_actors) and pred_actors[0] == case.ground_truth_actor
            is_top3 = case.ground_truth_actor in pred_actors[:3]
            is_top5 = case.ground_truth_actor in pred_actors[:5]
            rank = (
                pred_actors.index(case.ground_truth_actor) + 1
                if case.ground_truth_actor in pred_actors
                else 99
            )
            reciprocal_ranks.append(1.0 / rank if rank <= 10 else 0.0)

            top1_hits += int(is_top1)
            top3_hits += int(is_top3)
            top5_hits += int(is_top5)
            if case.expected_hops >= 2:
                multihop_total += 1
                multihop_hits += int(is_top1)

            records.append({
                "case_id": case.id,
                "ground_truth": case.ground_truth_actor,
                "predicted_top1": pred_actors[0] if pred_actors else None,
                "is_correct": is_top1,
                "rank": rank,
                "confidence": predictions[0][1] if predictions else 0.0,
            })

        case_count = len(self.test_cases)
        metrics = {
            "top1_accuracy": round(top1_hits / case_count * 100, 2) if case_count else 0.0,
            "actor_accuracy": round(top1_hits / case_count * 100, 2) if case_count else 0.0,
            "top3_accuracy": round(top3_hits / case_count * 100, 2) if case_count else 0.0,
            "top5_accuracy": round(top5_hits / case_count * 100, 2) if case_count else 0.0,
            "mrr": round(sum(reciprocal_ranks) / case_count, 4) if case_count else 0.0,
            "multihop_reasoning_success": (
                round(multihop_hits / multihop_total * 100, 2)
                if multihop_total
                else 0.0
            ),
            "top1_error_rate": (
                round((case_count - top1_hits) / case_count * 100, 2)
                if case_count
                else 0.0
            ),
        }
        return metrics, records

    def _evaluate_vector_rag(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        multihop_hits = 0
        multihop_total = 0
        hallucinations = 0
        reciprocal_ranks = []
        records = []

        for case in self.test_cases:
            preds = self.vector_baseline.predict(case.query)
            pred_actors = [p[0] for p in preds]

            is_top1 = len(pred_actors) > 0 and pred_actors[0] == case.ground_truth_actor
            is_top3 = case.ground_truth_actor in pred_actors[:3]
            is_top5 = case.ground_truth_actor in pred_actors[:5]

            rank = (pred_actors.index(case.ground_truth_actor) + 1) if case.ground_truth_actor in pred_actors else 99
            rr = 1.0 / rank if rank <= 10 else 0.0
            reciprocal_ranks.append(rr)

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if is_top5:
                top5_hits += 1

            if case.expected_hops >= 2:
                multihop_total += 1
                if is_top1:
                    multihop_hits += 1

            # Hallucination metric: if top prediction is confident but totally incorrect
            if preds and preds[0][0] != case.ground_truth_actor:
                hallucinations += 1

            records.append({
                "case_id": case.id,
                "case_name": case.name,
                "category": case.category,
                "ground_truth": case.ground_truth_actor,
                "predicted_top1": pred_actors[0] if pred_actors else "None",
                "is_correct": is_top1,
                "rank": rank,
                "confidence": preds[0][1] if preds else 0.0
            })

        n = len(self.test_cases)
        metrics = {
            "top1_accuracy": round((top1_hits / n) * 100, 2),
            "actor_accuracy": round((top1_hits / n) * 100, 2),
            "top3_accuracy": round((top3_hits / n) * 100, 2),
            "top5_accuracy": round((top5_hits / n) * 100, 2),
            "multihop_reasoning_success": round((multihop_hits / multihop_total) * 100, 2) if multihop_total else 0.0,
            "hallucination_rate": round((hallucinations / n) * 100, 2),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "evidence_validity_rate": None,
            "citation_coverage": None,
            "multihop_path_validity": None,
            "unsupported_evidence_rate": None
        }
        return metrics, records

    def _evaluate_standalone_gnn(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        multihop_hits = 0
        multihop_total = 0
        hallucinations = 0
        reciprocal_ranks = []
        records = []

        for case in self.test_cases:
            preds = self.gnn_baseline.predict(case.anchor_ioc)
            pred_actors = [p[0] for p in preds]

            is_top1 = len(pred_actors) > 0 and pred_actors[0] == case.ground_truth_actor
            is_top3 = case.ground_truth_actor in pred_actors[:3]
            is_top5 = case.ground_truth_actor in pred_actors[:5]

            rank = (pred_actors.index(case.ground_truth_actor) + 1) if case.ground_truth_actor in pred_actors else 99
            rr = 1.0 / rank if rank <= 10 else 0.0
            reciprocal_ranks.append(rr)

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if is_top5:
                top5_hits += 1

            if case.expected_hops >= 2:
                multihop_total += 1
                if is_top1:
                    multihop_hits += 1

            if preds and preds[0][0] != case.ground_truth_actor:
                hallucinations += 1

            records.append({
                "case_id": case.id,
                "case_name": case.name,
                "category": case.category,
                "ground_truth": case.ground_truth_actor,
                "predicted_top1": pred_actors[0] if pred_actors else "None",
                "is_correct": is_top1,
                "rank": rank,
                "confidence": preds[0][1] if preds else 0.0
            })

        n = len(self.test_cases)
        metrics = {
            "top1_accuracy": round((top1_hits / n) * 100, 2),
            "actor_accuracy": round((top1_hits / n) * 100, 2),
            "top3_accuracy": round((top3_hits / n) * 100, 2),
            "top5_accuracy": round((top5_hits / n) * 100, 2),
            "multihop_reasoning_success": round((multihop_hits / multihop_total) * 100, 2) if multihop_total else 0.0,
            "hallucination_rate": round((hallucinations / n) * 100, 2),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "evidence_validity_rate": None,
            "citation_coverage": None,
            "multihop_path_validity": None,
            "unsupported_evidence_rate": None
        }
        return metrics, records

    def _evaluate_hybrid(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        multihop_hits = 0
        multihop_total = 0
        multihop_path_hits = 0
        hallucinations = 0
        evidence_edges = 0
        valid_evidence_edges = 0
        cited_evidence_edges = 0
        unsupported_evidence_edges = 0
        reciprocal_ranks = []
        records = []

        for case in self.test_cases:
            result = self.fusion_engine.analyze_incident(
                query=case.query,
                ioc_override=case.anchor_ioc
            )

            pred_actors = [c.actor_id for c in result.all_candidates]

            is_top1 = len(pred_actors) > 0 and pred_actors[0] == case.ground_truth_actor
            is_top3 = case.ground_truth_actor in pred_actors[:3]
            is_top5 = case.ground_truth_actor in pred_actors[:5]

            rank = (pred_actors.index(case.ground_truth_actor) + 1) if case.ground_truth_actor in pred_actors else 99
            rr = 1.0 / rank if rank <= 10 else 0.0
            reciprocal_ranks.append(rr)

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if is_top5:
                top5_hits += 1

            if case.expected_hops >= 2:
                multihop_total += 1
                if is_top1:
                    multihop_hits += 1

            trail = result.graph_trail
            evidence_path = [
                (hop.source_id, hop.relation, hop.target_id, hop.citation)
                for hop in trail.hops
            ] if trail else []
            path_validation = evaluate_multihop_path(
                self.fusion_engine.tkg.nx_graph,
                evidence_path,
                expected_min_hops=2
            )
            total_path_edges = path_validation["total_edges"]
            evidence_edges += total_path_edges
            valid_evidence_edges += path_validation["relationship_edges"]
            cited_evidence_edges += path_validation["citation_edges"]
            unsupported_evidence_edges += (
                total_path_edges - path_validation["relationship_edges"]
            )
            if case.expected_hops >= 2 and path_validation["success"]:
                multihop_path_hits += 1
            multihop_path_status = (
                "not_applicable"
                if case.expected_hops < 2
                else "valid" if path_validation["success"] else "invalid"
            )

            # Hallucination is strictly controlled: if cross-validation was below threshold or top is wrong
            if result.top_actor and (result.top_actor.actor_id != case.ground_truth_actor or result.top_actor.cross_validation_score < 0.35):
                hallucinations += 1

            records.append({
                "case_id": case.id,
                "case_name": case.name,
                "category": case.category,
                "ground_truth": case.ground_truth_actor,
                "predicted_top1": pred_actors[0] if pred_actors else "None",
                "is_correct": is_top1,
                "rank": rank,
                "confidence": result.top_actor.fused_confidence if result.top_actor else 0.0,
                "hops": result.multi_hop_depth,
                "hallucination_risk": result.hallucination_risk,
                "evidence_path": [
                    {
                        "source": hop.source_id,
                        "relation": hop.relation,
                        "target": hop.target_id,
                        "citation": hop.citation
                    }
                    for hop in trail.hops
                ] if trail else [],
                "evidence_validity": (
                    round(path_validation["relationship_edges"] / total_path_edges * 100, 2)
                    if total_path_edges else 0.0
                ),
                "citation_coverage": (
                    round(path_validation["citation_edges"] / total_path_edges * 100, 2)
                    if total_path_edges else 0.0
                ),
                "path_connected": path_validation["path_connected"],
                "path_valid": path_validation["path_valid"],
                "multihop_path_status": multihop_path_status,
                "multihop_path_valid": (
                    path_validation["success"]
                    if case.expected_hops >= 2
                    else None
                ),
                "unsupported_evidence_rate": (
                    round(
                        (total_path_edges - path_validation["relationship_edges"])
                        / total_path_edges * 100,
                        2,
                    )
                    if total_path_edges else 0.0
                ),
                "invalid_evidence": path_validation["invalid_edges"],
                "disconnected_path_edges": path_validation["disconnected_edges"]
            })

        n = len(self.test_cases)
        metrics = {
            "top1_accuracy": round((top1_hits / n) * 100, 2),
            "actor_accuracy": round((top1_hits / n) * 100, 2),
            "top3_accuracy": round((top3_hits / n) * 100, 2),
            "top5_accuracy": round((top5_hits / n) * 100, 2),
            "multihop_reasoning_success": round((multihop_hits / multihop_total) * 100, 2) if multihop_total else 0.0,
            "hallucination_rate": round((hallucinations / n) * 100, 2),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "evidence_validity_rate": (
                round(valid_evidence_edges / evidence_edges * 100, 2)
                if evidence_edges else 0.0
            ),
            "citation_coverage": (
                round(cited_evidence_edges / evidence_edges * 100, 2)
                if evidence_edges else 0.0
            ),
            "multihop_path_validity": (
                round((multihop_path_hits / multihop_total) * 100, 2)
                if multihop_total else 0.0
            ),
            "unsupported_evidence_rate": (
                round(unsupported_evidence_edges / evidence_edges * 100, 2)
                if evidence_edges else 0.0
            )
        }
        return metrics, records
