"""
GNNExplainer and PGExplainer wrappers over ``torch_geometric.explain``.

Both drive the trained D-MPNN through PyG's native ``Explainer`` API.  The
model's edge->node aggregation runs through a ``MessagePassing`` layer
(see src/train/dmpnn.py), which is what lets PyG's ``set_masks`` (edge masks)
and ``get_embeddings`` (PGExplainer's edge-MLP inputs) hook the model.

Graph-level, multiclass (2-way) head, raw logits.  Explanations are generated
toward the model's own predicted class so Fidelity/GEF measure the explainer,
not label noise.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer

from .common import ExplanationResult, node_importance_from_edges

_MODEL_CONFIG = dict(mode="multiclass_classification", task_level="graph", return_type="raw")


def _predicted_class(model, data: Data) -> int:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    return int(logits.reshape(-1, logits.shape[-1])[0].argmax())


class GNNExplainerWrapper:
    name = "gnnexplainer"

    def __init__(self, model, device, epochs: int = 200, lr: float = 0.01):
        self.model = model.to(device).eval()
        self.device = device
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",              # explain the model's own decision
            node_mask_type="object",               # one scalar per node
            edge_mask_type="object",               # one scalar per directed edge
            model_config=_MODEL_CONFIG,
        )

    def explain(self, data: Data, target: int | None = None) -> ExplanationResult:
        data = data.to(self.device)
        tgt = _predicted_class(self.model, data) if target is None else int(target)
        expl = self.explainer(
            data.x, data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            target=torch.tensor([tgt], device=self.device),
        )
        node_imp = expl.node_mask.detach().cpu().float().reshape(-1)
        edge_imp = expl.edge_mask.detach().cpu().float().reshape(-1) if "edge_mask" in expl else None
        return ExplanationResult(node_importance=node_imp, edge_importance=edge_imp,
                                 target=tgt, meta={"explainer": self.name})


class PGExplainerWrapper:
    name = "pgexplainer"

    def __init__(self, model, device, epochs: int = 30, lr: float = 0.003):
        self.model = model.to(device).eval()
        self.device = device
        self.epochs = epochs
        self.algorithm = PGExplainer(epochs=epochs, lr=lr)
        self.explainer = Explainer(
            model=self.model,
            algorithm=self.algorithm,
            explanation_type="phenomenon",         # PGExplainer only supports phenomenon
            edge_mask_type="object",
            node_mask_type=None,                    # edge-only method
            model_config=_MODEL_CONFIG,
        )
        self._trained = False

    def train(self, train_data: list[Data]) -> list[float]:
        """Train the edge-mask MLP over a set of graphs (targets = model preds)."""
        losses = []
        for epoch in range(self.epochs):
            ep_loss = 0.0
            for data in train_data:
                data = data.to(self.device)
                tgt = _predicted_class(self.model, data)
                loss = self.algorithm.train(
                    epoch, self.model, data.x, data.edge_index,
                    target=torch.tensor([tgt], device=self.device),
                    edge_attr=getattr(data, "edge_attr", None),
                )
                ep_loss += float(loss)
            losses.append(ep_loss / max(len(train_data), 1))
        self._trained = True
        return losses

    def explain(self, data: Data, target: int | None = None) -> ExplanationResult:
        if not self._trained:
            raise RuntimeError("PGExplainerWrapper.train(...) must be called before explain(...)")
        data = data.to(self.device)
        tgt = _predicted_class(self.model, data) if target is None else int(target)
        expl = self.explainer(
            data.x, data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            target=torch.tensor([tgt], device=self.device),
        )
        edge_imp = expl.edge_mask.detach().cpu().float().reshape(-1)
        node_imp = node_importance_from_edges(edge_imp, data.edge_index, data.num_nodes)
        return ExplanationResult(node_importance=node_imp, edge_importance=edge_imp,
                                 target=tgt, meta={"explainer": self.name})
