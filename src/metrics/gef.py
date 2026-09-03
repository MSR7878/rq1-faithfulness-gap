"""
GEF -- Graph Explanation Faithfulness (see configs/rq1_metric_spec.md, Section 3).

Formula VERIFIED against GraphXAI source (graphxai/metrics/metrics_graph.py,
`graph_exp_faith_graph`), not just the paper text -- the paper/derivative
citations describe this loosely as "KL divergence"; the actual
implementation is a bounded transform:

    GEF = 1 - exp( -KL( f(G_i) || f(E_i) ) )

which keeps GEF in [0, 1), lower = more faithful.

GraphXAI's own reference implementation only ever applies ONE masking
scheme per importance type (zero-fill for node/feature masks, hard edge
removal for edge masks) -- it does not vary the masking reference. Our
contribution for RQ1 is computing GEF under all three of our masking
references (R1/R2/R3), which is why `masking_reference` is a required,
explicitly recorded argument here rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class GEFResult:
    gef: float
    masking_reference: str


@torch.no_grad()
def compute_gef(
    model: torch.nn.Module,
    data: Data,
    masked_data: Data,
    masking_reference: str,
) -> GEFResult:
    """
    Args:
        model: trained D-MPNN.
        data: original full graph G_i.
        masked_data: G_i with explanation-guided masking applied under the
            given `masking_reference` (R1/R2/R3), produced by masking.py.
        masking_reference: "R1", "R2", or "R3" -- recorded for analysis.
    """
    full_logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    full_softmax = F.softmax(full_logits, dim=-1)

    masked_logits = model(masked_data.x, masked_data.edge_index, getattr(masked_data, "edge_attr", None))
    masked_softmax = F.softmax(masked_logits, dim=-1)

    kl = F.kl_div(full_softmax.log(), masked_softmax, reduction="sum")
    gef = (1 - torch.exp(-kl)).item()

    return GEFResult(gef=gef, masking_reference=masking_reference)
