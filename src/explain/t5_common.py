"""
RQ1 Tier 5 (runs/rq1_tier5/PLAN.md): shared pieces for the new explainers.

  load_repo     import modules of one authors' repository (third_party/<repo>) under that repository only: the repository
                directory is put first on sys.path, the modules are imported, and every module name that came from the
                repository is then removed from sys.modules again (GStarX and SAME both have top-level utils / gnnNets /
                shapley). The returned module objects keep working because their globals were bound at import time.
  IdAdapter     lets the authors' code call our model(x, edge_index, edge_attr, batch). Their code passes node subsets as
                Data(x, edge_index) and drops bond features, so augment() appends one column to x with each atom's original
                index. Their code only slices / keeps rows of x, so the column follows the atoms through removal,
                relabelling and batching. forward() strips the column, looks up every remaining bond's features by its
                (original u, original v) pair and calls the model. Accepts model(data), model(data=...) and model(x, ei).
"""
from __future__ import annotations

import importlib
import os
import random
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TP = os.path.join(ROOT, "third_party")
SHIMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t5_shims")


def load_repo(repo: str, modules: list[str], shims: bool = False) -> dict:
    d = os.path.join(TP, repo)
    before = set(sys.modules)
    paths = [d] + ([SHIMS] if shims else [])
    sys.path[:0] = paths
    try:
        out = {m: importlib.import_module(m) for m in modules}
    finally:
        del sys.path[:len(paths)]
        for name in set(sys.modules) - before:
            f = getattr(sys.modules[name], "__file__", None) or ""
            if os.path.abspath(f).startswith(d) or os.path.abspath(f).startswith(SHIMS):
                del sys.modules[name]
    return out


def seed_all(s: int):
    random.seed(s); np.random.seed(s % (2 ** 32)); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def mol_seed(tag: str, unit: str, mi: int) -> int:
    return zlib.crc32(f"{tag}|{unit}|{mi}".encode())


def edge_attr_of(d):
    ea = getattr(d, "edge_attr", None)
    return None if ea is None else ea.float()


@torch.no_grad()
def logits(model, x, edge_index, edge_attr, batch=None):
    lg = model(x, edge_index, edge_attr, batch)
    return lg.reshape(-1, lg.shape[-1])


@torch.no_grad()
def pred_class(model, d, device) -> int:
    return int(logits(model, d.x.float().to(device), d.edge_index.to(device),
                      None if edge_attr_of(d) is None else edge_attr_of(d).to(device))[0].argmax())


class IdAdapter(nn.Module):
    def __init__(self, model: nn.Module, device):
        super().__init__()
        self.model = model
        self.device = device
        self.table = None; self.valid = None

    def set_graph(self, d):
        """d: the original (un-augmented) molecule. Builds the (u, v) -> bond-feature table."""
        n = d.num_nodes; ei = d.edge_index.to(self.device); ea = edge_attr_of(d)
        self.valid = torch.zeros(n, n, dtype=torch.bool, device=self.device)
        self.valid[ei[0], ei[1]] = True
        if ea is None:
            self.table = None
        else:
            self.table = torch.zeros(n, n, ea.shape[1], device=self.device)
            self.table[ei[0], ei[1]] = ea.to(self.device)

    def augment(self, d, y=None) -> Data:
        x = d.x.float().to(self.device)
        ids = torch.arange(x.shape[0], device=self.device, dtype=x.dtype)[:, None]
        out = Data(x=torch.cat([x, ids], 1), edge_index=d.edge_index.to(self.device))
        out.y = torch.tensor([0 if y is None else int(y)], device=self.device)
        return out

    def forward(self, *args, data=None, **kwargs):
        if data is None and len(args) == 1:
            data = args[0]
        if data is not None:
            x, ei, batch = data.x, data.edge_index, getattr(data, "batch", None)
        else:
            x, ei, batch = args[0], args[1], None
        ids = x[:, -1].round().long(); feats = x[:, :-1]
        u, v = ids[ei[0]], ids[ei[1]]
        if ei.shape[1] and not bool(self.valid[u, v].all()):
            raise RuntimeError("IdAdapter: an edge that is not a bond of the original molecule")
        ea = None if self.table is None else self.table[u, v]
        return self.model(feats, ei, ea, batch)
