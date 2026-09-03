"""
Masking references for RQ1 (see configs/rq1_metric_spec.md, Section 4).

Build order, as locked with advisor: R3 -> R2 -> R1.
Each function takes a full molecular graph (PyG Data) and a boolean/float
importance mask over nodes or edges, and returns a *masked* graph to feed
back through the model for Fidelity/GEF computation.

All three references must eventually be applied uniformly to every
(dataset, explainer) pair -- this file starts with R3 only; R2/R1 are
stubbed so the shared harness in fidelity.py / gef.py can be written once
against a common `mask_graph(data, importance, reference=...)` interface.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def mask_r3_distribution_aware(
    data: Data,
    node_importance: torch.Tensor,
    feature_means: torch.Tensor,
    keep_top_k: float = 0.25,
) -> Data:
    """
    R3 -- Distribution-aware masking (build first).

    Keeps graph topology intact. Non-explanation node features are
    replaced with training-set marginal statistics (per-feature mean),
    rather than zeroed -- this avoids handing the model an out-of-
    distribution all-zero atom, which Limitation-2 of the proposal
    argues conflates "importance" with "perturbation artifact".

    Args:
        data: full molecular graph.
        node_importance: (num_nodes,) importance scores from an explainer.
        feature_means: (num_features,) training-set per-feature mean,
            precomputed once per dataset and passed in (do not recompute
            per-call -- expensive and leaks test statistics if computed
            per-split incorrectly).
        keep_top_k: fraction of nodes (by importance) to leave untouched.

    Returns:
        A new Data object with masked node features; edge_index unchanged.
    """
    x = data.x.clone().float()
    num_keep = max(1, int(keep_top_k * node_importance.numel()))
    keep_idx = torch.topk(node_importance, num_keep).indices
    mask_idx = torch.ones(x.size(0), dtype=torch.bool)
    mask_idx[keep_idx] = False

    x[mask_idx] = feature_means.to(x.dtype)

    return Data(x=x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None))


def mask_r2_zero_fill(data: Data, node_importance: torch.Tensor, keep_top_k: float = 0.25) -> Data:
    """
    R2 -- Zero-filled soft masking (build second).
    Topology kept; non-explanation node features zeroed.
    Matches GraphXAI's default node-masking behaviour (see metrics_graph.py).
    """
    raise NotImplementedError("Build after R3 is validated -- see spec Section 4.")


def mask_r1_hard_removal(data: Data, node_importance: torch.Tensor, keep_top_k: float = 0.25) -> Data:
    """
    R1 -- Hard/discrete masking (build third).
    Explanation nodes/edges removed entirely; breaks topology.
    Most aggressive perturbation -- treated as the upper-bound reference.
    """
    raise NotImplementedError("Build after R2 is validated -- see spec Section 4.")


MASKING_REFERENCES = {
    "R3": mask_r3_distribution_aware,
    "R2": mask_r2_zero_fill,
    "R1": mask_r1_hard_removal,
}
