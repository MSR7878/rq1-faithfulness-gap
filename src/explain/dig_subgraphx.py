"""
SubgraphX wrapper (DIG's implementation: MCTS + Shapley-value node scoring).

SubgraphX is not in ``torch_geometric.explain``.  DIG (``dive-into-graphs``) is
installed ``--no-deps`` because its package tree eagerly imports a model zoo that
needs ``torch_sparse`` / ``captum`` / ``pgmpy`` etc.  We sidestep that by loading
ONLY ``dig/xgraph/method/{shapley,subgraphx}.py`` -- neither of which imports
anything beyond torch / torch_geometric / scipy / rdkit / networkx / matplotlib
(all already present) -- via a stub parent package, so
``dig/xgraph/method/__init__.py`` (the part that pulls the unrelated modules)
never runs.

DIG's SubgraphX calls the model two ways: ``model(x, edge_index)`` positionally
and ``model(data=Batch)``.  ``_DIGAdapter`` bridges both to our
``model(x, edge_index, edge_attr, batch)`` signature and re-supplies the real
bond features (DIG's zero-filling keeps topology, so edge_attr just tiles over
the batch).
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .common import ExplanationResult


# --------------------------------------------------------------------------- #
#  Load dig.xgraph.method.{shapley,subgraphx} without executing method/__init__
# --------------------------------------------------------------------------- #
def _load_dig_subgraphx():
    import dig.xgraph  # dig/__init__.py and dig/xgraph/__init__.py are both empty

    name = "dig.xgraph.method"
    if name not in sys.modules:
        method_dir = os.path.join(os.path.dirname(dig.xgraph.__file__), "method")
        pkg = types.ModuleType(name)
        pkg.__path__ = [method_dir]
        pkg.__package__ = name
        sys.modules[name] = pkg  # pre-seed so submodule import skips __init__.py

    sx = importlib.import_module("dig.xgraph.method.subgraphx")
    return sx.SubgraphX, sx.find_closest_node_result


SubgraphX, find_closest_node_result = _load_dig_subgraphx()


class _DIGAdapter(nn.Module):
    """Accept both ``forward(x, edge_index)`` and ``forward(data=Batch)`` and
    forward to the real D-MPNN with the correct (tiled) bond features."""

    def __init__(self, model: nn.Module, edge_attr: torch.Tensor | None, edge_index_ref: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_edge_attr", edge_attr if edge_attr is not None else None, persistent=False)
        self._E = edge_index_ref.size(1)

    def forward(self, *args, data=None, **kwargs):
        if data is not None:
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)
            n_graphs = int(batch.max()) + 1 if batch is not None else 1
        else:
            x, edge_index = args[0], args[1]
            batch, n_graphs = None, 1

        edge_attr = None
        if self._edge_attr is not None:
            if edge_index.size(1) == self._E * n_graphs:      # zero-filling keeps every edge
                edge_attr = self._edge_attr.repeat(n_graphs, 1)
            elif edge_index.size(1) == self._E:
                edge_attr = self._edge_attr
            # else: 'split' building changed the topology -> fall back to zeros
        return self.model(x, edge_index, edge_attr, batch)


class SubgraphXWrapper:
    name = "subgraphx"

    def __init__(self, model, device, *, num_classes: int = 2, num_hops: int = 3,
                 rollout: int = 15, min_atoms: int = 3, c_puct: float = 10.0,
                 expand_atoms: int = 12, sample_num: int = 50,
                 reward_method: str = "mc_l_shapley", node_frac: float = 0.25):
        self.model = model.to(device).eval()
        self.device = device
        self.num_classes = num_classes
        self.node_frac = node_frac
        self._kw = dict(num_hops=num_hops, rollout=rollout, min_atoms=min_atoms,
                        c_puct=c_puct, expand_atoms=expand_atoms, sample_num=sample_num,
                        reward_method=reward_method, subgraph_building_method="zero_filling")

    def explain(self, data: Data, target: int | None = None) -> ExplanationResult:
        data = data.to(self.device)
        x, edge_index = data.x.float(), data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        num_nodes = data.num_nodes

        adapter = _DIGAdapter(self.model, edge_attr, edge_index).to(self.device).eval()
        with torch.no_grad():
            logits = adapter(x, edge_index)
            pred = int(logits.reshape(-1, logits.shape[-1])[0].argmax())
        tgt = pred if target is None else int(target)

        sx = SubgraphX(adapter, num_classes=self.num_classes, device=self.device,
                       explain_graph=True, verbose=False, vis=False, **self._kw)
        max_nodes = max(self._kw["min_atoms"], round(self.node_frac * num_nodes))
        results, related = sx.explain(x, edge_index, label=tgt, max_nodes=max_nodes)
        best = find_closest_node_result(sx.read_from_MCTSInfo_list(results), max_nodes=max_nodes)
        coalition = list(best.coalition)

        node_imp = torch.zeros(num_nodes, dtype=torch.float)
        node_imp[coalition] = 1.0
        return ExplanationResult(
            node_importance=node_imp, edge_importance=None, target=tgt,
            meta={"explainer": self.name, "coalition": coalition,
                  "max_nodes": max_nodes, "related_pred": related},
        )
