"""
GraphXAI's Mutagenicity benchmark (class ``Mutagenicity`` in
graphxai/datasets/real_world/mutagenicity.py), vendored.

Pipeline, kept faithful to GraphXAI (quirks included -- see NOTES):

  1. ``TUDataset('Mutagenicity')`` (4,337 graphs, explicit hydrogens, 14-dim
     atom one-hot ``[C,O,Cl,H,N,F,Br,S,P,I,Na,K,Li,Ca]``, 3-dim bond one-hot).
  2. ``remove_isolated_nodes`` on every graph.
  3. Five toxicophore matchers: NH2, NO2, "aliphatic" halide, nitroso, azo.
  4. Label filter: keep a graph iff ``has_toxicophore == y``.
  5. Ground truth is a SET of masks per graph: the union of every non-empty
     subset of the matched motif instances (``make_iter_combinations`` +
     ``aggregate_explanations`` with ``torch.max``).

Ported from (and cross-checked against, see src/analysis/mutagenicity_step1.py):
    graphxai/datasets/real_world/mutagenicity.py
    graphxai/datasets/utils/substruct_chem_match.py
    graphxai/utils/nx_conversion.py   (to_networkx_conv)
    graphxai/utils/misc.py            (match_edge_presence)

NOTES (GraphXAI behaviour reproduced on purpose, not corrected):
  * ``match_aliphatic_halide`` keeps a terminal halogen only when its
    neighbour lies on a cycle-basis cycle of length > 4 -- i.e. it keeps
    RING-ATTACHED (aryl) halides and drops true aliphatic ones, the opposite
    of its name. It also accepts F although its docstring says Cl/Br/I.
  * NH2/NO2 are found with a structure-only ISMAGS cherry search followed by
    attribute checks; the direct enumeration below returns the same node sets
    (verified on all 4,337 graphs).
  * GraphXAI's own split is 70/20/10 train/test/val, sequential, no shuffle.

Attached to every returned ``Data``:
    node_gt_mask  (N,) bool -- MERGED mask (union of all motif instances)
    edge_gt_mask  (E,) bool -- edge incident to any motif atom
    orig_idx      int       -- index into the unfiltered TUDataset
The per-graph motif instances (needed for the max-over-set GEA) are in
``meta.extras['motifs'][i]`` = list of ``(type, [node, ...])``.
"""

from __future__ import annotations

import itertools
import os

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_isolated_nodes

from . import DatasetMeta

ATOM_ORDER = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
C_IDX, O_IDX, CL_IDX, H_IDX, N_IDX, F_IDX, BR_IDX, I_IDX = 0, 1, 2, 3, 4, 5, 6, 9
HALOGENS = (CL_IDX, BR_IDX, I_IDX, F_IDX)
MOTIF_TYPES = ["NH2", "NO2", "nitroso", "azo", "halide"]   # GraphXAI's all_matches order


def to_networkx_conv(edge_index: torch.Tensor) -> nx.Graph:
    """GraphXAI to_networkx_conv(to_undirected=True): nodes = sorted unique
    endpoints, one undirected edge per (u, v) column with v <= u, in column
    order (the insertion order matters for nx.cycle_basis)."""
    G = nx.Graph()
    G.add_nodes_from(sorted(torch.unique(edge_index).tolist()))
    for u, v in edge_index.t().tolist():
        if v > u:
            continue
        G.add_edge(u, v)
    return G


def _cherries(G: nx.Graph, atom: list[int], leaf_idx: int) -> list[list[int]]:
    """match_substruct_mutagenicity: centre = degree-3 N, two degree-1 leaves
    of type ``leaf_idx`` (H for NH2, O for NO2)."""
    out = []
    for k in G.nodes():
        if G.degree[k] != 3 or atom[k] != N_IDX:
            continue
        leaves = [n for n in G.neighbors(k) if G.degree[n] == 1 and atom[n] == leaf_idx]
        for a, b in itertools.combinations(sorted(leaves), 2):
            out.append(sorted([k, a, b]))
    return out


def match_aliphatic_halide(G: nx.Graph, atom: list[int]) -> list[list[int]]:
    matches = []
    all_cycles = nx.cycle_basis(G)
    for n in G.nodes():
        if G.degree[n] > 1:
            continue
        other_end = [i_n for i_n in list(G.edges(n))[0] if i_n != n][0]
        for c in all_cycles:
            if (other_end in c) and len(c) > 4:
                break
        else:
            continue
        if atom[n] in HALOGENS:
            matches.append([n])
    return matches


def match_nitroso(G: nx.Graph, atom: list[int]) -> list[list[int]]:
    matches = []
    for e1, e2 in G.edges():
        if G.degree[e1] == 2 and G.degree[e2] == 1:
            if atom[e1] == N_IDX and atom[e2] == O_IDX:
                matches.append([e1, e2])
        elif G.degree[e2] == 2 and G.degree[e1] == 1:
            if atom[e2] == N_IDX and atom[e1] == O_IDX:
                matches.append([e1, e2])
    return matches


def match_azo_type(G: nx.Graph, atom: list[int]) -> list[list[int]]:
    matches = []
    for e1, e2 in G.edges():
        if G.degree[e1] == G.degree[e2] == 2 and atom[e1] == N_IDX and atom[e2] == N_IDX:
            matches.append([e1, e2])
    return matches


def find_motifs(x: torch.Tensor, edge_index: torch.Tensor) -> list[tuple[str, list[int]]]:
    """All toxicophore instances, in GraphXAI's concatenation order
    (nh2 + no2 + nitroso + azo + halide)."""
    G = to_networkx_conv(edge_index)
    atom = x.argmax(dim=1).tolist()
    out = [("NH2", m) for m in _cherries(G, atom, H_IDX)]
    out += [("NO2", m) for m in _cherries(G, atom, O_IDX)]
    out += [("nitroso", m) for m in match_nitroso(G, atom)]
    out += [("azo", m) for m in match_azo_type(G, atom)]
    out += [("halide", m) for m in match_aliphatic_halide(G, atom)]
    return out


def _match_edge_presence(edge_index: torch.Tensor, nodes) -> torch.Tensor:
    emask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for ni in nodes:
        emask |= (edge_index[0] == ni) | (edge_index[1] == ni)
    return emask


def load_raw(root: str | None = None):
    """Unfiltered graphs after isolated-node removal, with motifs. Returns a
    list of dicts (x, edge_index, edge_attr, y, motifs, n_removed)."""
    from .standard import DATA_ROOT

    root = root or os.path.join(DATA_ROOT, "TUDataset")
    ds = TUDataset(root=root, name="Mutagenicity")
    raw = []
    for d in ds:
        ei, ea, node_mask = remove_isolated_nodes(d.edge_index, d.edge_attr, num_nodes=d.x.shape[0])
        x = d.x[node_mask]
        raw.append(dict(x=x, edge_index=ei, edge_attr=ea, y=int(d.y.item()),
                        motifs=find_motifs(x, ei), n_removed=int((~node_mask).sum())))
    return raw, ds


def load_mutagenicity_graphxai(root: str | None = None) -> tuple[list, DatasetMeta]:
    raw, ds = load_raw(root)

    data_list, motifs = [], []
    for i, r in enumerate(raw):
        has_match = int(len(r["motifs"]) > 0)
        if has_match != r["y"]:            # GraphXAI __filter_dataset
            continue
        n = r["x"].shape[0]
        node_gt = torch.zeros(n, dtype=torch.bool)
        edge_gt = torch.zeros(r["edge_index"].shape[1], dtype=torch.bool)
        for _, nodes in r["motifs"]:
            node_gt[nodes] = True
            edge_gt |= _match_edge_presence(r["edge_index"], nodes)
        data_list.append(Data(
            x=r["x"].float(),
            edge_index=r["edge_index"],
            edge_attr=None if r["edge_attr"] is None else r["edge_attr"].float(),
            y=torch.tensor([[r["y"]]], dtype=torch.long),
            node_gt_mask=node_gt,
            edge_gt_mask=edge_gt,
            orig_idx=i,
        ))
        motifs.append(r["motifs"])

    meta = DatasetMeta(
        name="mutagenicity_graphxai",
        num_graphs=len(data_list),
        num_node_features=ds.num_node_features,
        num_edge_features=ds.num_edge_features,
        num_classes=2,
        task_type="binary-classification",
        has_node_gt=True,
        has_edge_gt=True,
        notes=(f"GraphXAI Mutagenicity: {len(data_list)}/{len(raw)} graphs kept by the "
               f"has_toxicophore == y filter; GT is a set of masks (meta.extras['motifs'])."),
    )
    meta.extras = {"motifs": motifs}
    return data_list, meta
