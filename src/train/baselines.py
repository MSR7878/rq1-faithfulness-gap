"""
Node-based baseline architectures for the architecture check (plain PyG).

The D-MPNN (src/train/dmpnn.py) is edge-based.  ``GIN`` here is the standard
node-based counterpart: ``depth`` x ``GINConv`` (2-layer MLP each), sum pooling
and the same FFN head as the D-MPNN.  No dropout by default and no BatchNorm,
to match the D-MPNN.  Bond features are accepted and IGNORED (plain GINConv
has no edge-feature input), so the call signature stays

    model(x, edge_index, edge_attr=None, batch=None) -> logits

and the explainers / masking / metrics run unchanged.  Every conv is a PyG
``MessagePassing`` layer, so ``set_masks`` (GNNExplainer) applies the edge mask
in ALL ``depth`` layers and ``get_embeddings`` (PGExplainer) reads the last
conv's node embeddings -- unlike the D-MPNN, where the mask acts on the single
edge->node aggregation layer.

``build_model`` / ``make_cfg`` are the arch switch used by train.py and
run_phase3.py; the default arch is the D-MPNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import scatter

from .dmpnn import DMPNN, DMPNNConfig


@dataclass
class GINConfig:
    node_in: int
    edge_in: int = 0          # recorded for the checkpoint; not used by the model
    hidden: int = 212         # 3 x GINConv at 212 ~= the D-MPNN's parameter count at 300
    depth: int = 3            # number of GINConv layers
    ffn_hidden: int = 212
    ffn_layers: int = 2
    dropout: float = 0.0
    num_classes: int = 2
    pool: str = "sum"        # "sum" | "mean"


class GIN(nn.Module):
    def __init__(self, cfg: GINConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        self.convs = nn.ModuleList()
        d = cfg.node_in
        for _ in range(cfg.depth):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))))
            d = h
        self.dropout = nn.Dropout(cfg.dropout)

        ffn: list[nn.Module] = []
        d = h
        for _ in range(max(cfg.ffn_layers - 1, 0)):
            ffn += [nn.Linear(d, cfg.ffn_hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.ffn_hidden
        ffn += [nn.Linear(d, cfg.num_classes)]
        self.ffn = nn.Sequential(*ffn)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h_v = x.float()
        for conv in self.convs:
            h_v = F.relu(conv(h_v, edge_index))
            h_v = self.dropout(h_v)

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1
        graph = scatter(h_v, batch, dim=0, dim_size=num_graphs, reduce=self.cfg.pool)

        return self.ffn(graph)


ARCHS = {"dmpnn": (DMPNN, DMPNNConfig), "gin": (GIN, GINConfig)}


def make_cfg(arch: str, **kw):
    return ARCHS[arch][1](**kw)


def build_model(arch: str, cfg: dict) -> nn.Module:
    """Model from a checkpoint's ``cfg`` dict; ``arch`` defaults to the D-MPNN upstream
    (checkpoints written before the arch switch have no ``arch`` key)."""
    model_cls, cfg_cls = ARCHS[arch]
    return model_cls(cfg_cls(**cfg))
