import pytest

from src import experiments
from src.benchmark import graph_split


@pytest.mark.parametrize("topology", ["karate", "lesmis"])
def test_graph_models_use_topology_size(tmp_path, monkeypatch, topology):
    pytest.importorskip("networkx")
    pytest.importorskip("scipy")

    def small_split(transition, faction, order, n_nodes, n, split_seed, mode):
        return graph_split(transition, faction, order, n_nodes, 8, split_seed, mode)

    monkeypatch.setattr(experiments, "graph_split", small_split)
    result = experiments.run_graph(
        {"graph": topology, "seeds": 1, "out": str(tmp_path / "graph.json")},
        {"device": "cpu"},
    )

    assert len(result) == 10
    for measurement in result.values():
        assert len(measurement["values"]) == 1
        assert 0.0 <= measurement["mean"] <= 1.0
