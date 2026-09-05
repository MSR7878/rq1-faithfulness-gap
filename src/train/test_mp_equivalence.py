"""
Equivalence check: the MessagePassing refactor of DMPNN's edge->node
aggregation must be numerically identical to the previous raw-``scatter``
implementation.

Phase 2 did not persist model checkpoints, so this runs the check with freshly
seeded *random* weights (several seeds) rather than one trained point -- a
strictly stronger test: it must hold for arbitrary parameters, not just the
ones a particular training run landed on.  BBBP / Tox21 / B-XAIC Phase-2 numbers
therefore stand without retraining (computation-preserving change).

    python -m src.train.test_mp_equivalence
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from ..data import load_mutag_graphxai
from .dmpnn import DMPNN, DMPNNConfig, _reverse_edge_index


def _legacy_scatter_forward(model: DMPNN, x, edge_index, edge_attr=None, batch=None):
    """The pre-refactor forward: identical maths, edge->node sum via raw scatter."""
    cfg = model.cfg
    x = x.float()
    num_nodes = x.size(0)
    E = edge_index.size(1)
    src, dst = edge_index[0], edge_index[1]

    if edge_attr is None:
        edge_attr = x.new_zeros((E, cfg.edge_in))
    else:
        edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

    rev = _reverse_edge_index(edge_index, num_nodes)

    h0 = F.relu(model.W_i(torch.cat([x[src], edge_attr], dim=-1)))
    h = h0
    for _ in range(cfg.depth - 1):
        node_msg = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
        m = node_msg[src] - h[rev]
        h = F.relu(h0 + model.W_h(m))
        h = model.dropout(h)

    node_msg = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
    h_v = F.relu(model.W_o(torch.cat([x, node_msg], dim=-1)))
    h_v = model.dropout(h_v)

    if batch is None:
        batch = x.new_zeros(num_nodes, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1
    graph = scatter(h_v, batch, dim=0, dim_size=num_graphs, reduce=cfg.pool)
    return model.ffn(graph)


def main() -> int:
    dl, meta = load_mutag_graphxai()
    b_single = dl[0]
    b_batch = next(iter(DataLoader(dl[:37], batch_size=37)))

    worst = 0.0
    for seed in range(5):
        torch.manual_seed(seed)
        model = DMPNN(DMPNNConfig(
            node_in=meta.num_node_features, edge_in=meta.num_edge_features,
            hidden=128, depth=3, dropout=0.0, num_classes=2,
        )).eval()

        cases = {
            "single, edge_attr":      (b_single.x, b_single.edge_index, b_single.edge_attr, None),
            "single, edge_attr=None": (b_single.x, b_single.edge_index, None, None),
            "batch=37, edge_attr":    (b_batch.x, b_batch.edge_index, b_batch.edge_attr, b_batch.batch),
        }
        for name, (x, ei, ea, bt) in cases.items():
            with torch.no_grad():
                new = model(x, ei, ea, bt)
                old = _legacy_scatter_forward(model, x, ei, ea, bt)
            d = (new - old).abs().max().item()
            worst = max(worst, d)
            ok = torch.allclose(new, old, rtol=1e-5, atol=1e-6)
            print(f"seed {seed}  {name:24s}  max_absdiff={d:.2e}  allclose={ok}")
            assert ok, f"MP refactor diverges from scatter: {name}, seed {seed}, max_absdiff={d}"

    print(f"\nPASS -- MessagePassing refactor == raw-scatter forward (worst max_absdiff = {worst:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
