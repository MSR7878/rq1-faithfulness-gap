"""
MUTAG with GraphXAI-style substructure-matching ground truth.

GraphXAI (mims-harvard/GraphXAI) builds MUTAG explanation masks by matching the
two mutagenic motifs -- nitro (NO2) and primary amine (NH2) -- against each
molecular graph, following Debnath et al. (1991).  We vendor that logic here
rather than installing the package, because GraphXAI pins ``torch==1.10`` /
``torch-geometric==2.0.1`` and will not co-exist with this repo's environment.

Ported from, and kept faithful to:
    graphxai/datasets/real_world/MUTAG.py            (__make_explanations)
    graphxai/datasets/utils/substruct_chem_match.py  (match_NH2, match_substruct)
    graphxai/utils/misc.py                           (match_edge_presence)

One correction was required: GraphXAI's matcher hard-codes a 14-dim atom
encoding (N at index 4, O at index 1), but PyG's ``TUDataset('MUTAG')`` uses the
standard 7-dim one-hot ``[C, N, O, F, I, Cl, Br]`` (N at 1, O at 2).  MUTAG has
no explicit hydrogens, so an "NH2" is simply a terminal (degree-1) nitrogen --
exactly what GraphXAI's ``match_NH2`` checks once the encoding is fixed.

Ground truth attached to every returned ``Data``:
    data.node_gt_mask  (num_nodes,)  bool -- atom in a NO2 or NH2 motif
    data.edge_gt_mask   (num_edges,) bool -- edge incident to a motif atom
"""

from __future__ import annotations

import os

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from . import DatasetMeta

# 7-dim TUDataset MUTAG atom one-hot order (verified empirically against the
# degree distribution of each channel: C 2395 atoms deg<=4, N 345 mostly deg 3,
# O 593 mostly deg 1, then F/I/Cl/Br all terminal).
ATOM_ORDER = ["C", "N", "O", "F", "I", "Cl", "Br"]
C_IDX, N_IDX, O_IDX = 0, 1, 2

# NO2 motif as a "cherry": centre node 0 bonded to nodes 1 and 2.
_NO2 = nx.Graph()
_NO2.add_edges_from([(0, 1), (0, 2)])


def _atom_idx(G: nx.Graph, n: int) -> int:
    return int(torch.as_tensor(G.nodes[n]["x"]).argmax().item())


def _match_nh2(G: nx.Graph, n: int) -> bool:
    """GraphXAI match_NH2: a terminal nitrogen (its two H's are implicit)."""
    return G.degree[n] == 1 and _atom_idx(G, n) == N_IDX


def _match_no2(G: nx.Graph) -> list[list[int]]:
    """GraphXAI match_substruct(MUTAG_NO2): subgraph-isomorphic cherries whose
    centre is a degree-3 nitrogen and whose two leaves are degree-1 oxygens."""
    matcher = nx.algorithms.isomorphism.ISMAGS(G, _NO2)
    matches: list[list[int]] = []
    for iso in matcher.find_isomorphisms():
        # iso maps G-node -> substructure-node; find the centre (sub-node 0).
        centre = next((k for k, v in iso.items() if v == 0), None)
        if centre is None:
            continue
        if G.degree[centre] != 3 or _atom_idx(G, centre) != N_IDX:
            continue
        leaves = [k for k in iso if k != centre]
        if any(G.degree[o] != 1 or _atom_idx(G, o) != O_IDX for o in leaves):
            continue
        matches.append(sorted(iso.keys()))
    return matches


def _match_edge_presence(edge_index: torch.Tensor, nodes) -> torch.Tensor:
    """GraphXAI match_edge_presence: edge kept if *either* endpoint is in ``nodes``."""
    nodes = torch.as_tensor(list(nodes), dtype=torch.long)
    emask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for ni in nodes:
        emask |= (edge_index[0] == ni) | (edge_index[1] == ni)
    return emask


def _make_explanation(data: Data) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    G = to_networkx(data, node_attrs=["x"], to_undirected=True)
    node_gt = torch.zeros(data.num_nodes, dtype=torch.bool)
    edge_gt = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)

    no2 = _match_no2(G)
    nh2 = [[n] for n in G.nodes() if _match_nh2(G, n)]

    for motif in no2 + nh2:
        node_gt[motif] = True
        edge_gt |= _match_edge_presence(data.edge_index, motif)

    return node_gt, edge_gt, len(no2), len(nh2)


def load_mutag_graphxai(root: str | None = None) -> tuple[list, DatasetMeta]:
    from .standard import DATA_ROOT

    root = root or os.path.join(DATA_ROOT, "TUDataset")
    ds = TUDataset(root=root, name="MUTAG")

    data_list = []
    n_no2 = n_nh2 = n_with_gt = 0
    for d in ds:
        node_gt, edge_gt, k_no2, k_nh2 = _make_explanation(d)
        n_no2 += k_no2
        n_nh2 += k_nh2
        n_with_gt += int(node_gt.any())
        data_list.append(
            Data(
                x=d.x.float(),
                edge_index=d.edge_index,
                edge_attr=None if d.edge_attr is None else d.edge_attr.float(),
                y=d.y.view(1, 1).long(),
                node_gt_mask=node_gt,
                edge_gt_mask=edge_gt,
            )
        )

    meta = DatasetMeta(
        name="mutag_graphxai",
        num_graphs=len(data_list),
        num_node_features=ds.num_node_features,
        num_edge_features=ds.num_edge_features,
        num_classes=2,
        task_type="binary-classification",
        has_node_gt=True,
        has_edge_gt=True,
        notes=(
            f"GraphXAI substructure GT: {n_no2} NO2 + {n_nh2} NH2 motifs across "
            f"{n_with_gt}/{len(data_list)} graphs."
        ),
    )
    return data_list, meta
