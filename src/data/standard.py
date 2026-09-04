"""
Standard benchmark datasets, loaded straight from PyG:

* MUTAG            -- ``TUDataset`` (188 nitro-aromatic graphs, binary mutagenicity)
* BBBP            -- ``MoleculeNet`` (blood-brain-barrier penetration, binary)
* Tox21 (SR-p53)  -- ``MoleculeNet`` Tox21, sliced to the single SR-p53 endpoint

None of these ship node/edge ground truth (see ``configs/rq1_metric_spec.md``:
GEA is not applicable to any of them).
"""

from __future__ import annotations

import os

import torch
from torch_geometric.datasets import MoleculeNet, TUDataset

from . import DatasetMeta

# Repo-root/data -- gitignored cache shared with the other loaders.
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Canonical Tox21 endpoint order in deepchem's ``tox21.csv`` (verified against the
# raw header shipped by MoleculeNet); SR-p53 is the last of the 12 columns.
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
SRP53_INDEX = TOX21_TASKS.index("SR-p53")  # == 11


def _clone(d, y):
    """Copy a Data object with a fresh label tensor, keeping only the fields
    the RQ1 pipeline uses so downstream ``Batch.from_data_list`` stays clean."""
    from torch_geometric.data import Data

    return Data(
        x=d.x.float(),
        edge_index=d.edge_index,
        edge_attr=None if d.edge_attr is None else d.edge_attr.float(),
        y=y,
        smiles=getattr(d, "smiles", None),
    )


def load_mutag(root: str | None = None) -> tuple[list, DatasetMeta]:
    root = root or os.path.join(DATA_ROOT, "TUDataset")
    ds = TUDataset(root=root, name="MUTAG")

    data_list = []
    for d in ds:
        # y: TUDataset gives shape [1] long in {0, 1}; keep [1, 1] for stacking.
        y = d.y.view(1, 1).long()
        data_list.append(_clone(d, y))

    meta = DatasetMeta(
        name="mutag",
        num_graphs=len(data_list),
        num_node_features=ds.num_node_features,   # 7-dim atom one-hot [C,N,O,F,I,Cl,Br]
        num_edge_features=ds.num_edge_features,   # 4-dim bond one-hot [aromatic,single,double,triple]
        num_classes=2,
        task_type="binary-classification",
        notes="TUDataset MUTAG; no SMILES shipped, no ground truth.",
    )
    return data_list, meta


def _load_moleculenet(name: str, task_index: int | None, pretty: str) -> tuple[list, DatasetMeta]:
    root = os.path.join(DATA_ROOT, "MoleculeNet")
    ds = MoleculeNet(root=root, name=name)

    data_list = []
    n_dropped_empty = 0
    n_dropped_nan = 0
    for d in ds:
        if d.x is None or d.num_nodes == 0 or d.edge_index.numel() == 0:
            n_dropped_empty += 1
            continue
        if task_index is None:
            y = d.y.view(1, -1).float()
        else:
            y = d.y.view(-1)[task_index].view(1, 1).float()
        if torch.isnan(y).any():
            n_dropped_nan += 1
            continue
        data_list.append(_clone(d, y))

    pos = int(sum(int(d.y.item()) for d in data_list))
    meta = DatasetMeta(
        name=pretty,
        num_graphs=len(data_list),
        num_node_features=ds.num_node_features,   # 9-dim raw atom features (PyG from_smiles)
        num_edge_features=ds.num_edge_features,   # 3-dim raw bond features
        num_classes=2,
        task_type="binary-classification",
        notes=(
            f"MoleculeNet {name}; dropped {n_dropped_empty} empty graph(s), "
            f"{n_dropped_nan} missing-label row(s); class balance neg/pos = "
            f"{len(data_list) - pos}/{pos}."
        ),
    )
    return data_list, meta


def load_bbbp(root: str | None = None) -> tuple[list, DatasetMeta]:
    return _load_moleculenet("BBBP", task_index=None, pretty="bbbp")


def load_tox21_srp53(root: str | None = None) -> tuple[list, DatasetMeta]:
    return _load_moleculenet("Tox21", task_index=SRP53_INDEX, pretty="tox21_srp53")
