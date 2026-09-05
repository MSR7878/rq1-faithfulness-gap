"""
PyG-native D-MPNN (directed message passing neural network).

Faithful to Yang et al. 2019, "Analyzing Learned Molecular Representations for
Property Prediction" (the Chemprop model), but implemented directly on
``torch_geometric`` tensors so the trained model is callable as

    model(x, edge_index, edge_attr=None, batch=None) -> logits [num_graphs, C]

which is exactly the signature ``src/metrics/fidelity.py`` and the explainer
wrappers expect.  Messages live on *directed* edges; the update for edge (v->w)
sums incoming edge states at v and excludes the reverse edge (w->v):

    h_e^0      = ReLU(W_i [x_v ; e_vw])
    m_e^{t+1}  = ( sum_{e': dst(e')=v} h_{e'}^t )  -  h_{rev(e)}^t
    h_e^{t+1}  = ReLU(h_e^0 + W_h m_e^{t+1})           (residual, shared W_h)
    m_v        = sum_{e': dst(e')=v} h_{e'}^T
    h_v        = ReLU(W_o [x_v ; m_v])
    graph      = pool_v h_v   ->  FFN  ->  logits

The edge->node sum ( sum_{e': dst(e')=v} h_{e'} ) runs through a tiny
``MessagePassing`` layer (``_EdgeToNodeAggr``) rather than a raw ``scatter``, so
PyG's ``torch_geometric.explain`` stack works natively against this model:
``set_masks`` finds the layer and multiplies each per-bond message by the
learned edge mask (GNNExplainer), and ``get_embeddings`` hooks its output for
PGExplainer.  Equivalence to the previous raw-``scatter`` implementation is
checked in ``src/train/test_mp_equivalence.py`` (bit-level, arbitrary weights).
Note: the explicit ``- h_{rev(e)}`` term is applied outside the layer, so an
edge mask does not attenuate that second-order reverse-edge subtraction -- the
standard trade-off for DMPNN-in-PyG explainer setups.

Assumes each undirected bond appears as both directed edges exactly once
(true for TUDataset / MoleculeNet / the B-XAIC loader).  Topology-preserving
masking references (R3, R2) keep that property, so ``rev`` is recomputed per
forward call and the signature stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


@dataclass
class DMPNNConfig:
    node_in: int
    edge_in: int
    hidden: int = 300
    depth: int = 3            # number of message-passing steps
    ffn_hidden: int = 300
    ffn_layers: int = 2
    dropout: float = 0.0
    num_classes: int = 2
    pool: str = "sum"        # "sum" | "mean"


def _reverse_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Index `rev` such that edge_index[:, rev[e]] == edge_index[[1, 0], e].

    Falls back to self (no reverse subtraction) for any edge whose reverse is
    absent, so an odd graph degrades gracefully instead of crashing.
    """
    key = edge_index[0].to(torch.long) * num_nodes + edge_index[1].to(torch.long)
    rev_key = edge_index[1].to(torch.long) * num_nodes + edge_index[0].to(torch.long)
    order = torch.argsort(key)
    sorted_key = key[order]
    pos = torch.searchsorted(sorted_key, rev_key).clamp_(max=key.numel() - 1)
    cand = order[pos]
    ok = key[cand] == rev_key
    rev = torch.arange(key.numel(), device=edge_index.device)
    rev[ok] = cand[ok]
    return rev


class _EdgeToNodeAggr(MessagePassing):
    """sum_{e': dst(e')=v} h_{e'} -- a plain additive gather of per-directed-edge
    states to their target node.  As a ``MessagePassing`` layer so PyG's
    explainer ``set_masks`` / ``get_embeddings`` machinery can hook it; the
    message *is* the edge state, so ``explain_message`` multiplies it by the
    per-bond edge mask exactly as intended.
    """

    def __init__(self) -> None:
        super().__init__(aggr="add")  # flow='source_to_target': aggregate at edge_index[1]

    def forward(self, edge_index: torch.Tensor, edge_state: torch.Tensor, num_nodes: int) -> torch.Tensor:
        return self.propagate(edge_index, edge_state=edge_state, size=(num_nodes, num_nodes))

    def message(self, edge_state: torch.Tensor) -> torch.Tensor:  # [E, hidden], one row per directed edge
        return edge_state


class DMPNN(nn.Module):
    def __init__(self, cfg: DMPNNConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        self.W_i = nn.Linear(cfg.node_in + cfg.edge_in, h, bias=False)
        self.W_h = nn.Linear(h, h, bias=False)
        self.W_o = nn.Linear(cfg.node_in + h, h)
        self.dropout = nn.Dropout(cfg.dropout)
        self.aggr_edges = _EdgeToNodeAggr()

        ffn: list[nn.Module] = []
        d = h
        for _ in range(max(cfg.ffn_layers - 1, 0)):
            ffn += [nn.Linear(d, cfg.ffn_hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.ffn_hidden
        ffn += [nn.Linear(d, cfg.num_classes)]
        self.ffn = nn.Sequential(*ffn)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = x.float()
        num_nodes = x.size(0)
        E = edge_index.size(1)
        dst = edge_index[1]
        src = edge_index[0]

        if edge_attr is None:
            edge_attr = x.new_zeros((E, self.cfg.edge_in))
        else:
            edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        rev = _reverse_edge_index(edge_index, num_nodes)

        # h_e^0 from source-atom features + bond features
        h0 = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0
        for _ in range(self.cfg.depth - 1):
            # sum of edge states arriving at each node, gathered back to edges by
            # the edge's *source*, then remove the reverse edge's contribution.
            node_msg = self.aggr_edges(edge_index, h, num_nodes)
            m = node_msg[src] - h[rev]
            h = F.relu(h0 + self.W_h(m))
            h = self.dropout(h)

        node_msg = self.aggr_edges(edge_index, h, num_nodes)
        h_v = F.relu(self.W_o(torch.cat([x, node_msg], dim=-1)))
        h_v = self.dropout(h_v)

        if batch is None:
            batch = x.new_zeros(num_nodes, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1
        graph = scatter(h_v, batch, dim=0, dim_size=num_graphs, reduce=self.cfg.pool)

        return self.ffn(graph)
