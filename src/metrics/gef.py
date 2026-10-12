"""
GEF -- Graph Explanation Faithfulness (see configs/rq1_metric_spec.md, Section 3).

What was checked against the GraphXAI source, and what was not:

1. The bounded transform and the zero-fill match
   graphxai/metrics/metrics_graph.py, `graph_exp_faith_graph` -- the
   paper/derivative citations describe GEF loosely as "KL divergence"; the
   implementation is

       GEF = 1 - exp( -KL( f(E_i) || f(G_i) ) )

   which keeps GEF in [0, 1), lower = more faithful.

2. WHICH nodes are hidden follows graphxai/formal/metrics.py,
   `graph_exp_faith` (explanation kept, the rest zeroed; lines 176-185) and
   the paper's definition -- NOT metrics_graph.py line 192, which builds
   `rem_nodes` from the nodes that ARE in the top-k and so zeroes the
   explanation itself, despite its comments ("Removing the unimportant nodes
   by masking"). Here `masked_data` is always the explanation-only graph E_i.

3. The KL argument order is that of `F.kl_div(log q, p)`, which computes
   KL(p || q): with q = f(G_i) and p = f(E_i) this is KL( f(E_i) || f(G_i) ),
   the same as GraphXAI's code. Earlier text in this file and in the spec
   wrote KL( f(G_i) || f(E_i) ), which is reversed relative to the code.

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
    masked_logits = model(masked_data.x, masked_data.edge_index, getattr(masked_data, "edge_attr", None))

    # Clamp before log: a near-perfect model (B-XAIC, AUROC ~1.0) drives some
    # softmax entries to exactly 0 in fp32, so F.kl_div(log p, q) hits log(0)
    # and 0*log(0) -> NaN. eps=1e-12 leaves any non-degenerate value untouched.
    eps = 1e-12
    full_softmax = F.softmax(full_logits, dim=-1).clamp_min(eps)
    masked_softmax = F.softmax(masked_logits, dim=-1).clamp_min(eps)

    kl = F.kl_div(full_softmax.log(), masked_softmax, reduction="sum")
    gef = (1 - torch.exp(-kl)).item()

    return GEFResult(gef=gef, masking_reference=masking_reference)
