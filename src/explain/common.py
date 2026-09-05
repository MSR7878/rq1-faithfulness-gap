"""Shared types + helpers for the explainer wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data


@dataclass
class ExplanationResult:
    node_importance: torch.Tensor            # [num_nodes] float, higher = more important
    edge_importance: torch.Tensor | None = None   # [num_edges] float, or None
    target: int | None = None               # class index the explanation is for
    meta: dict = field(default_factory=dict)

    def topk_node_mask(self, k_frac: float) -> torch.Tensor:
        """Boolean [num_nodes] mask of the top ``k_frac`` fraction of nodes."""
        n = self.node_importance.numel()
        k = max(1, int(round(k_frac * n)))
        idx = torch.topk(self.node_importance, k).indices
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m


def node_importance_from_edges(edge_importance: torch.Tensor, edge_index: torch.Tensor,
                               num_nodes: int) -> torch.Tensor:
    """Fold an edge mask into per-node importance: each node gets the mean of
    its incident (directed) edge scores.  Used for edge-only methods
    (PGExplainer) so their output can feed the node-based metrics harness."""
    imp = torch.zeros(num_nodes, dtype=torch.float)
    cnt = torch.zeros(num_nodes, dtype=torch.float)
    ei = edge_index.cpu()
    e = edge_importance.detach().cpu().float()
    for end in (0, 1):
        imp.index_add_(0, ei[end], e)
        cnt.index_add_(0, ei[end], torch.ones_like(e))
    return imp / cnt.clamp_min(1.0)


def binarize_mean(importance: torch.Tensor) -> torch.Tensor:
    """GraphXAI-style thresholding: keep entries above the mean importance."""
    imp = importance.detach().cpu().float()
    return imp > imp.mean()


def to_cpu_data(data: Data) -> Data:
    return Data(
        x=data.x.detach().cpu().float(),
        edge_index=data.edge_index.detach().cpu(),
        edge_attr=None if getattr(data, "edge_attr", None) is None else data.edge_attr.detach().cpu().float(),
    )
