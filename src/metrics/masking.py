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


def training_fill_vector(x_all: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
    """Build an R3 fill vector from stacked training-set node features.

    * ``"mean"`` -- per-feature marginal mean (the spec's default R3). For
      one-hot categoricals this is a *blended* vector, e.g. 0.70*C + 0.10*N + ...
    * ``"mode"`` -- per-COLUMN most-frequent value, i.e. a real "typical atom"
      rather than a fractional blend. Column-wise (not a single global one-hot)
      so it's correct for both one-hot datasets (MUTAG: exactly one column is
      1 more than half the time, so the other columns' modes are all 0 -- this
      reduces to the intuitive "most common category" one-hot) and datasets
      whose node features are several independently-scaled raw integer columns
      (BBBP/Tox21's [atomic_num, chirality, degree, charge, numH, ...] from PyG
      MoleculeNet) -- there, a single global one-hot of the loudest raw column
      (e.g. atomic_num) would zero every other column instead of taking each
      one's own typical value. (v4: was a global one-hot of ``mean.argmax()``;
      fixed before it produced a degenerate "atomic_num~1, everything else 0"
      fill on BBBP. Provably identical to the old formula on one-hot data, so
      MUTAG/mutag_graphxai's already-reported mode-fill numbers stand.)
    """
    x_all = x_all.float()
    mean = x_all.mean(dim=0)
    if strategy == "mean":
        return mean
    if strategy == "mode":
        return torch.mode(x_all, dim=0).values
    raise ValueError(f"unknown fill strategy {strategy!r} (use 'mean' or 'mode')")


def mask_r3_distribution_aware(
    data: Data,
    node_importance: torch.Tensor,
    fill_vector: torch.Tensor,
    keep_top_k: float = 0.25,
) -> Data:
    """
    R3 -- Distribution-aware masking (build first).

    Keeps graph topology intact. Non-explanation node features are replaced
    with a training-set fill vector (``training_fill_vector``: "mean" per the
    spec, or "mode"), rather than zeroed -- this avoids handing the model an
    out-of-distribution all-zero atom, which Limitation-2 of the proposal
    argues conflates "importance" with "perturbation artifact".

    Args:
        data: full molecular graph.
        node_importance: (num_nodes,) importance scores from an explainer.
        fill_vector: (num_features,) training-set fill, precomputed once per
            dataset and passed in (do not recompute per-call -- expensive and
            leaks test statistics if computed per-split incorrectly). Was named
            ``feature_means`` before v4; any (num_features,) vector works.
        keep_top_k: fraction of nodes (by importance) to leave untouched.

    Returns:
        A new Data object with masked node features; edge_index unchanged.
    """
    x = data.x.clone().float()
    num_keep = max(1, int(keep_top_k * node_importance.numel()))
    keep_idx = torch.topk(node_importance, num_keep).indices
    mask_idx = torch.ones(x.size(0), dtype=torch.bool)
    mask_idx[keep_idx] = False

    x[mask_idx] = fill_vector.to(x.dtype)

    return Data(x=x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None))


def mask_r2_zero_fill(data: Data, node_importance: torch.Tensor, keep_top_k: float = 0.25) -> Data:
    """
    R2 -- Zero-filled soft masking (build second).

    Same top-k selection as R3 (keep the ``keep_top_k`` fraction of nodes with
    the highest importance, mask the rest), but non-explanation node features
    are set to ZERO rather than to a training-set statistic. Topology
    (edge_index / edge_attr) is untouched -- only ``x`` rows change. This is
    GraphXAI's default node-masking behaviour (metrics_graph.py), and the
    aggressive counterpart to R3: Phase 3 showed R3's distribution-aware fill
    barely perturbs a saturated categorical model, whereas an all-zero atom is
    out-of-distribution and does move predictions (probed: MUTAG 1.00 -> 0.00).
    """
    x = data.x.clone().float()
    num_keep = max(1, int(keep_top_k * node_importance.numel()))
    keep_idx = torch.topk(node_importance, num_keep).indices
    mask_idx = torch.ones(x.size(0), dtype=torch.bool)
    mask_idx[keep_idx] = False

    x[mask_idx] = 0.0

    return Data(x=x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None))


def mask_r1_hard_removal(data: Data, node_importance: torch.Tensor, keep_top_k: float = 0.25) -> Data:
    """
    R1 -- Hard/discrete masking (build third).

    The masked-out nodes are DELETED (not filled): keep the ``keep_top_k``
    fraction with the highest importance, drop the rest along with every edge
    incident to a dropped node, and reindex the survivors to 0..K-1. The most
    aggressive reference -- topology is genuinely broken, no OOD fill artifact,
    but the graph the model sees is a different object.

    Edge cases (handled here explicitly; see spec Phase 3 R1 findings):

    * Empty result -- CANNOT happen. ``num_keep = max(1, int(keep_top_k * N))``,
      so at least one node always survives. For the paired use in the harness
      (E_i keeps top 25%, G\\E_i keeps bottom 75%), the smallest possible E_i is
      1 node on a >=4-node graph, or 1 node on a 3-node graph
      (int(0.75*3)=2 -> keep 2; int(0.25*3)=0 -> max(1,0)=1).

    * 1-node / few isolated nodes / ZERO edges -- returned as-is (x present,
      edge_index shape [2, 0]). The D-MPNN forward tolerates this: with no
      edges, ``h0`` is [0, H], the message-passing loop is a no-op, the
      per-node message aggregate is all-zeros, and the readout is
      ReLU(W_o[x ; 0]) pooled over the surviving nodes -> a well-defined
      "features-only, no propagation" prediction. NOT zero-filled, NOT skipped.

    * Disconnected components -- returned as-is. The D-MPNN aggregates messages
      per node via scatter and pools globally, so message passing simply does
      not cross component boundaries (correct: the graph really is
      disconnected) and the graph vector still spans all survivors.

    * Reverse edges -- an undirected bond survives R1 iff BOTH endpoints
      survive, so both of its directed edges are kept together and
      ``_reverse_edge_index`` in dmpnn.py stays consistent.

    Returns a new Data with ``x`` [K, F], ``edge_index`` [2, E'] reindexed to
    the survivors, and ``edge_attr`` [E', *] filtered to the surviving edges.
    """
    n = data.x.size(0)
    num_keep = max(1, int(keep_top_k * n))
    keep_idx = torch.topk(node_importance, num_keep).indices
    keep_idx, _ = torch.sort(keep_idx)

    remap = torch.full((n,), -1, dtype=torch.long, device=data.edge_index.device)
    remap[keep_idx.to(remap.device)] = torch.arange(keep_idx.numel(), device=remap.device)

    x = data.x[keep_idx].clone().float()

    ei = data.edge_index
    edge_keep = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    new_ei = remap[ei[:, edge_keep]]

    ea = getattr(data, "edge_attr", None)
    new_ea = None if ea is None else ea[edge_keep]

    return Data(x=x, edge_index=new_ei.contiguous(), edge_attr=new_ea)


MASKING_REFERENCES = {
    "R3": mask_r3_distribution_aware,
    "R2": mask_r2_zero_fill,
    "R1": mask_r1_hard_removal,
}
