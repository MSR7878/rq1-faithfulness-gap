"""
B-XAIC -- "Benchmarking Explainable AI Using Compound Data for GNNs"
(mproszewska/B-XAIC).

50k ChEMBL-derived molecules with per-atom (and, for some tasks, per-bond)
ground-truth explanation masks.  The dataset ships as two files on the HF Hub
(``data.csv`` + ``explanations.sdf``); download them once with::

    from huggingface_hub import hf_hub_download
    for f in ("data.csv", "explanations.sdf"):
        hf_hub_download("mproszewska/B-XAIC", f, repo_type="dataset",
                        local_dir="data/bxaic")

The graph featurisation and explanation-mask parsing below are ported verbatim
(bar packaging) from B-XAIC's ``dataset.py`` (``smiles_to_graph``,
``XAIMolecularDataset``), so masks line up with the atom order in the SDF.
"""

from __future__ import annotations

import os

import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from . import DatasetMeta

RDLogger.DisableLog("rdApp.*")

# B-XAIC atom vocabulary and the task -> SDF-property map (from their dataset.py).
SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
BXAIC_TASKS = {
    "B": "B", "P": "P", "X": "X",
    "indole": "indole", "PAINS": "pains",
    "rings-count": "rings", "rings-max": "largest_rings",
}
# Tasks that also carry a per-bond ("<prop>_edge") ground-truth annotation.
_EDGE_GT_TASKS = {"indole", "PAINS", "rings-count", "rings-max"}

_BOND_TYPE = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}


def _atom_label(atom) -> int:
    s = atom.GetSymbol()
    return SYMBOLS.index(s) if s in SYMBOLS else len(SYMBOLS) - 1


def _mol_to_graph(mol):
    labels = torch.tensor([_atom_label(a) for a in mol.GetAtoms()], dtype=torch.long)
    x = F.one_hot(labels, len(SYMBOLS)).float()

    row, col, etype = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = _BOND_TYPE.get(bond.GetBondType(), -1)
        row += [i, j]
        col += [j, i]
        etype += [t, t]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(etype, dtype=torch.long).view(-1, 1)
    return x, edge_index, edge_attr


def _parse_node_mask(prop: str, num_nodes: int) -> torch.Tensor:
    m = torch.zeros(num_nodes, dtype=torch.bool)
    if prop:
        m[torch.tensor([int(n) for n in prop.split(",")], dtype=torch.long)] = True
    return m


def _parse_edge_mask(prop: str, edge_index: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    if prop:
        pairs = {tuple(int(x) for x in e.split("#")) for e in prop.split(",")}
        for k in range(edge_index.shape[1]):
            a, b = int(edge_index[0, k]), int(edge_index[1, k])
            if (a, b) in pairs or (b, a) in pairs:
                m[k] = True
    return m


def _default_root() -> str:
    from .standard import DATA_ROOT

    return os.path.join(DATA_ROOT, "bxaic")


def load_bxaic(
    task: str = "indole",
    split_idx: int = 0,
    root: str | None = None,
    limit: int | None = None,
    use_cache: bool = True,
) -> tuple[list, DatasetMeta]:
    """Load B-XAIC for one binary ``task`` (see :data:`BXAIC_TASKS`).

    Each returned ``Data`` carries ``y`` ``[1, 1]``, ``node_gt_mask``, and --
    for indole / PAINS / rings-* -- ``edge_gt_mask``.  ``data.split`` is the
    ``train`` / ``valid`` / ``test`` tag from column ``split_{split_idx}``.
    """
    if task not in BXAIC_TASKS:
        raise ValueError(f"unknown B-XAIC task {task!r}; choose from {sorted(BXAIC_TASKS)}")
    root = root or _default_root()
    csv_path = os.path.join(root, "data.csv")
    sdf_path = os.path.join(root, "explanations.sdf")
    for p in (csv_path, sdf_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p!r}. Download the B-XAIC release from the HF Hub first "
                f"(see module docstring)."
            )

    prop = BXAIC_TASKS[task]
    has_edge_gt = task in _EDGE_GT_TASKS
    cache_path = os.path.join(root, f"cache_{task}_split{split_idx}.pt")
    if use_cache and limit is None and os.path.exists(cache_path):
        data_list = torch.load(cache_path, weights_only=False)
    else:
        df = pd.read_csv(csv_path)
        labels = df[task].tolist()
        split_col = df[f"split_{split_idx}"].tolist()

        data_list = []
        # sanitize=False so a handful of hypervalent records (e.g. PF6) still
        # load -- featurisation only needs atom symbols + bond types, and record
        # order must stay aligned with data.csv.  Best-effort sanitize after.
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
        n_unsanitized = 0
        for i, mol in enumerate(suppl):
            if limit is not None and i >= limit:
                break
            if mol is None:
                raise RuntimeError(f"B-XAIC SDF record {i} could not be read at all")
            if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                n_unsanitized += 1
            x, edge_index, edge_attr = _mol_to_graph(mol)
            d = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([[labels[i]]], dtype=torch.float),
                node_gt_mask=_parse_node_mask(mol.GetProp(prop), x.size(0)),
                split=split_col[i],
            )
            if has_edge_gt:
                d.edge_gt_mask = _parse_edge_mask(mol.GetProp(f"{prop}_edge"), edge_index)
            data_list.append(d)

        if n_unsanitized:
            print(f"[bxaic] note: {n_unsanitized} record(s) kept without full RDKit sanitisation")
        if use_cache and limit is None:
            torch.save(data_list, cache_path)

    pos = int(sum(int(d.y.item()) for d in data_list))
    n = len(data_list)
    split_counts = {}
    for d in data_list:
        split_counts[d.split] = split_counts.get(d.split, 0) + 1
    meta = DatasetMeta(
        name=f"bxaic:{task}",
        num_graphs=n,
        num_node_features=len(SYMBOLS),          # 11-dim atom one-hot
        num_edge_features=1,                      # scalar bond-type code {0..3}
        num_classes=2,
        task_type="binary-classification",
        has_node_gt=True,
        has_edge_gt=has_edge_gt,
        notes=(
            f"task={task} (SDF prop {prop!r}); neg/pos = {n - pos}/{pos}; "
            f"split_{split_idx} counts = {split_counts}."
        ),
    )
    return data_list, meta
