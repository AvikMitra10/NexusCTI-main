from types import SimpleNamespace

import pytest
import torch

from nexuscti.gnn.trainer import GNNAttributionTrainer
from nexuscti.eval.baselines import StandaloneGNNBaseline
from experiments.scientific_audit_20261010_223000.run_ablations import (
    AblationExperimentSuite,
)


def _trained_trainer():
    tkg = SimpleNamespace(
        x=torch.zeros((3, 8)),
        relation_to_idx={"relation": 0},
        actor_indices=[1, 2],
        node_id_to_idx={"ioc": 0},
        idx_to_node_id={1: "actor-a", 2: "actor-b"},
        actor_idx_to_name={1: "Actor A", 2: "Actor B"},
    )
    trainer = GNNAttributionTrainer(tkg)
    trainer.cached_embeddings = torch.randn((3, 64))
    trainer._is_trained = True
    return trainer


def test_neural_affinity_does_not_calculate_networkx_distances(monkeypatch):
    trainer = _trained_trainer()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Neural affinity must not calculate graph distances.")

    monkeypatch.setattr("nexuscti.gnn.trainer.nx.shortest_path_length", fail_if_called)

    predictions = trainer.predict_attribution_gnn_only("ioc")

    assert {prediction["actor_id"] for prediction in predictions} == {
        "actor-a",
        "actor-b",
    }
    assert sum(
        prediction["structural_confidence"] for prediction in predictions
    ) == pytest.approx(1.0, abs=0.001)


def test_neural_affinity_rejects_nodes_without_cached_embeddings():
    trainer = _trained_trainer()
    trainer.tkg.node_id_to_idx["uncached"] = 3

    with pytest.raises(RuntimeError, match="no cached GNN embedding"):
        trainer.predict_neural_affinity("uncached")


def test_config_c_returns_empty_for_unknown_anchor_without_mutating_graph():
    tkg = SimpleNamespace(find_node_by_indicator=lambda indicator: None)
    suite = AblationExperimentSuite.__new__(AblationExperimentSuite)
    suite.service = SimpleNamespace(tkg=tkg)
    suite.actor_ids = ["actor-a"]

    assert suite.predict_config_c("new-indicator") == []


def test_config_c_uses_neural_affinity_not_mixed_predictor():
    gnn = SimpleNamespace(
        predict_attribution_gnn_only=lambda node_id: [
            {"actor_id": "actor-a", "structural_confidence": 1.0}
        ],
        predict_attribution=lambda node_id: (_ for _ in ()).throw(
            AssertionError("Config C must not use the mixed predictor.")
        ),
    )
    suite = AblationExperimentSuite.__new__(AblationExperimentSuite)
    suite.service = SimpleNamespace(
        tkg=SimpleNamespace(
            find_node_by_indicator=lambda indicator: SimpleNamespace(id="ioc")
        ),
        gnn_trainer=gnn,
    )
    suite.actor_ids = ["actor-a"]

    assert suite.predict_config_c("ioc") == [("actor-a", 1.0)]


def test_gnn_only_attribution_refreshes_cached_embeddings_when_graph_changes():
    trainer = _trained_trainer()
    trainer.tkg.x = torch.zeros((4, 8))
    trainer.tkg.rel_edge_indices = {}
    trainer.tkg.node_id_to_idx["new-ioc"] = 3
    trainer.tkg.idx_to_node_id[3] = "new-ioc"

    predictions = trainer.predict_attribution_gnn_only("new-ioc")

    assert {prediction["actor_id"] for prediction in predictions} == {
        "actor-a",
        "actor-b",
    }
    assert trainer.cached_embeddings.size(0) == 4


def test_standalone_gnn_baseline_uses_neural_affinity_only():
    gnn = SimpleNamespace(
        tkg=SimpleNamespace(
            find_node_by_indicator=lambda indicator: SimpleNamespace(id="ioc")
        ),
        predict_neural_affinity=lambda node_id: [
            {"actor_id": "actor-a", "structural_confidence": 0.8}
        ],
        predict_attribution=lambda node_id: (_ for _ in ()).throw(
            AssertionError("The mixed GNN/topology predictor must not be used.")
        ),
    )

    assert StandaloneGNNBaseline(gnn).predict("ioc") == [("actor-a", 0.8)]
