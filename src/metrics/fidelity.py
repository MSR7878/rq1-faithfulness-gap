"""
Fidelity+/- (see configs/rq1_metric_spec.md, Section 3).

Fid+ measures necessity: drop in predicted probability for the true class
when the explanation subgraph is REMOVED from the input.
Fid- measures sufficiency: drop when ONLY the explanation subgraph is KEPT.

    Fid+ = (1/N) * sum_i [ f(G_i)_yi - f(G_i \\ E_i)_yi ]
    Fid- = (1/N) * sum_i [ f(G_i)_yi - f(E_i)_yi ]

Both are computed under whichever masking reference (R1/R2/R3) produced
G_i \\ E_i and E_i -- always report which reference was used, since Fid+/-
values are not comparable across references (this is the whole point of
RQ1's rank-correlation analysis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch_geometric.data import Data


@dataclass
class FidelityResult:
    fid_plus: float
    fid_minus: float
    masking_reference: str


@torch.no_grad()
def compute_fidelity(
    model: torch.nn.Module,
    data: Data,
    y_true: int,
    explanation_removed: Data,   # G_i \ E_i, i.e. explanation subgraph removed
    explanation_only: Data,      # E_i, i.e. only explanation subgraph kept
    masking_reference: str,
) -> FidelityResult:
    """
    Args:
        model: trained D-MPNN, callable as model(x, edge_index, ...) -> logits.
        data: the original full graph G_i.
        y_true: index of the true class for G_i.
        explanation_removed: G_i with explanation nodes/edges masked out,
            produced by one of the masking.py reference functions.
        explanation_only: only the explanation subgraph, rest masked,
            produced by one of the masking.py reference functions.
        masking_reference: one of "R1", "R2", "R3" -- recorded for
            downstream rank-correlation analysis, not used in the formula.
    """
    prob_full = _softmax_prob(model, data, y_true)
    prob_removed = _softmax_prob(model, explanation_removed, y_true)
    prob_only = _softmax_prob(model, explanation_only, y_true)

    fid_plus = (prob_full - prob_removed).item()
    fid_minus = (prob_full - prob_only).item()

    return FidelityResult(fid_plus=fid_plus, fid_minus=fid_minus, masking_reference=masking_reference)


def _softmax_prob(model: torch.nn.Module, data: Data, y_true: int) -> torch.Tensor:
    logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    # Model returns graph-level logits shaped [1, C] (or [C]); collapse the
    # optional batch dim before indexing the class, so `.item()` downstream
    # sees a scalar rather than a length-C row.
    probs = torch.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)[0]
    return probs[y_true]
