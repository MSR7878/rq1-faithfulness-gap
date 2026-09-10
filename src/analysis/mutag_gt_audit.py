"""
Audit of the mutag_graphxai ground truth, for two advisor-meeting checks.

CHECK 1 -- dataset identity / toxicophore coverage.
  GraphXAI ships TWO real-world mutagenicity builders:
    graphxai/datasets/real_world/MUTAG.py        -> class MUTAG
        TUDataset('MUTAG')  = 188 graphs (Debnath 1991), GT = NO2 + NH2 only,
        ONE merged node mask per graph.
    graphxai/datasets/real_world/mutagenicity.py -> class Mutagenicity
        TUDataset('Mutagenicity') = 4337 graphs (Kazius 2005), filtered to
        graphs where (has_toxicophore == label) ~= 1768, GT over FIVE
        toxicophores {NH2, NO2, aliphatic halide, nitroso, azo-type}, and a
        LIST of combinatorial sub-explanations per graph (paper Eq. 2's
        "set of valid ground truths").
  Our src/data/graphxai_mutag.py ports MUTAG.py (the 188 / 2-toxicophore /
  single-mask one). This script quantifies what that costs: how many of the
  188 molecules carry a halide / nitroso / azo toxicophore that our GT leaves
  marked unimportant.

CHECK 2 -- GEA definition.
  Paper Eq. 2: GEA = max_{gt in valid-GT-set} Jaccard(gt, pred).
  Our gea.py + run_phase3 take a single Jaccard vs the merged mask (which is
  exactly what GraphXAI's own MUTAG.py does -- the max-over-set is only in
  mutagenicity.py). This recomputes mutag_graphxai GEA from the cached
  explanations under three definitions -- merged (current), max over single
  motifs, max over all motif subsets -- and re-runs F3's paired Wilcoxon.

    python -m src.analysis.mutag_gt_audit
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch_geometric.utils import to_networkx

from ..data import LOADERS
from ..data.graphxai_mutag import _match_nh2, _match_no2, C_IDX, N_IDX, O_IDX

HALO_IDX = {3, 4, 5, 6}  # F, I, Cl, Br  in our [C,N,O,F,I,Cl,Br] order
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]


def _atom(G, n):
    return int(torch.as_tensor(G.nodes[n]["x"]).argmax().item())


def _rings(G):
    return [set(c) for c in __import__("networkx").cycle_basis(G)]


def match_halide(G):
    """Terminal halogen. Split by whether its attachment atom sits in a ring
    (GraphXAI's match_aliphatic_halide literally keeps ring-attached ones;
    Kazius' 'aliphatic halide' is the non-ring case). Report both."""
    rings = _rings(G)
    ring_nodes = set().union(*rings) if rings else set()
    ring_att, aliphatic = [], []
    for n in G.nodes():
        if G.degree[n] != 1 or _atom(G, n) not in HALO_IDX:
            continue
        nb = next(iter(G.neighbors(n)))
        (ring_att if nb in ring_nodes else aliphatic).append(n)
    return ring_att, aliphatic


def match_nitroso(G):
    """R-N=O : N(deg 2) bonded to a terminal O."""
    out = []
    for a, b in G.edges():
        da, db = G.degree[a], G.degree[b]
        if da == 2 and db == 1 and _atom(G, a) == N_IDX and _atom(G, b) == O_IDX:
            out.append((a, b))
        elif db == 2 and da == 1 and _atom(G, b) == N_IDX and _atom(G, a) == O_IDX:
            out.append((b, a))
    return out


def match_azo(G):
    """R-N=N-R : an N-N bond with both N of degree 2."""
    out = []
    for a, b in G.edges():
        if G.degree[a] == 2 and G.degree[b] == 2 and _atom(G, a) == N_IDX and _atom(G, b) == N_IDX:
            out.append((a, b))
    return out


def _motifs(data):
    """Per-graph list of individual GraphXAI motif atom-sets (NO2 then NH2)."""
    G = to_networkx(data, node_attrs=["x"], to_undirected=True)
    no2 = [set(m) for m in _match_no2(G)]
    nh2 = [{n} for n in G.nodes() if _match_nh2(G, n)]
    return G, no2 + nh2


def check1_toxicophores():
    data_list, meta = LOADERS["mutag_graphxai"]()
    print("=" * 90)
    print(" CHECK 1  --  non-NO2/NH2 toxicophores in our 188-graph mutag_graphxai")
    print("=" * 90)
    print(f" loader notes: {meta.notes}")

    tally = dict(halide_ring=0, halide_aliphatic=0, nitroso=0, azo=0)
    graphs_extra = set()          # graph has >=1 halide/nitroso/azo hit
    graphs_extra_unmarked = set() # ...and >=1 of those atoms is outside our GT mask
    graphs_no_gt = 0
    n_pos_with_extra = 0
    for i, d in enumerate(data_list):
        G, motifs = _motifs(d)
        gt = d.node_gt_mask.bool()
        if not gt.any():
            graphs_no_gt += 1
        ring_h, ali_h = match_halide(G)
        nit = match_nitroso(G)
        azo = match_azo(G)
        tally["halide_ring"] += len(ring_h)
        tally["halide_aliphatic"] += len(ali_h)
        tally["nitroso"] += len(nit)
        tally["azo"] += len(azo)
        extra_atoms = set(ring_h) | set(ali_h)
        for e in nit + azo:
            extra_atoms |= set(e)
        if extra_atoms:
            graphs_extra.add(i)
            if int(d.y.view(-1)[0]) == 1:
                n_pos_with_extra += 1
            if any(not gt[a] for a in extra_atoms):
                graphs_extra_unmarked.add(i)

    n = len(data_list)
    print(f"\n total graphs: {n}   |  graphs with EMPTY GT (no NO2/NH2 match): {graphs_no_gt}")
    print("\n raw motif-instance counts across all 188 graphs:")
    print(f"   aliphatic halide (halogen on NON-ring atom) : {tally['halide_aliphatic']}")
    print(f"   ring/aromatic halide (halogen on ring atom) : {tally['halide_ring']}"
          f"   [GraphXAI match_aliphatic_halide literally keeps THESE]")
    print(f"   nitroso  R-N=O                              : {tally['nitroso']}")
    print(f"   azo-type R-N=N-R                            : {tally['azo']}")
    print(f"\n graphs containing >=1 halide/nitroso/azo instance      : {len(graphs_extra)} / {n}")
    print(f"   ...of which our NO2/NH2 GT marks some of it unimportant: {len(graphs_extra_unmarked)} / {n}")
    print(f"   (positives among the {len(graphs_extra)} graphs-with-extra: {n_pos_with_extra})")
    print("\n interpretation: in the 188-graph Debnath set the extra Kazius toxicophores are")
    print(" nearly all halogens on aromatic rings (halo-nitro-benzenes); true aliphatic")
    print(" halide / nitroso / azo are rare. The GT gap is real but small vs the 1768-graph")
    print(" Mutagenicity benchmark, where all 5 types are common.")
    return data_list


def _jaccard(gt_set: set[int], pred: torch.Tensor) -> float:
    p = set(torch.nonzero(pred).view(-1).tolist())
    if not gt_set and not p:
        return 0.0
    return len(gt_set & p) / len(gt_set | p)


def check2_gea(data_list):
    print("\n" + "=" * 90)
    print(" CHECK 2  --  mutag_graphxai GEA under merged vs max-over-motif-set (paper Eq. 2)")
    print("=" * 90)
    blob = torch.load("runs/expl_cache_mutag_graphxai.pt", weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]

    per_motifs = [_motifs(data_list[i])[1] for i in test_idx]
    n_multi = sum(len(m) >= 2 for m in per_motifs)
    n_withgt = sum(len(m) >= 1 for m in per_motifs)
    print(f" test n={len(test_idx)}  |  with >=1 motif: {n_withgt}  |  with >=2 motifs (only these can move): {n_multi}")
    tot_multi_all = sum(len(_motifs(d)[1]) >= 2 for d in data_list)
    print(f" (across all 188: {tot_multi_all} graphs have >=2 motifs)")

    defs = ("merged", "max_single_motif", "max_motif_subset")
    scores = {e: {dfn: [] for dfn in defs} for e in EXPL}
    for e in EXPL:
        for k, motifs in enumerate(per_motifs):
            if not motifs:
                continue
            imp = torch.tensor(res[e][k]["node_importance"], dtype=torch.float)
            pred = imp > imp.mean()
            merged = set().union(*motifs)
            s_merged = _jaccard(merged, pred)
            s_single = max(_jaccard(m, pred) for m in motifs)
            s_subset = s_merged
            if len(motifs) >= 2:
                for r in range(1, len(motifs) + 1):
                    for comb in itertools.combinations(range(len(motifs)), r):
                        s_subset = max(s_subset, _jaccard(set().union(*(motifs[c] for c in comb)), pred))
            else:
                s_subset = s_merged
            scores[e]["merged"].append(s_merged)
            scores[e]["max_single_motif"].append(s_single)
            scores[e]["max_motif_subset"].append(s_subset)

    for dfn in defs:
        print(f"\n --- GEA definition: {dfn} ---")
        arr = {e: np.array(scores[e][dfn], float) for e in EXPL}
        for e in EXPL:
            print(f"   {SH[e]:<4} mean={arr[e].mean():.3f}  std={arr[e].std():.3f}  n={len(arr[e])}")
        order = " > ".join(SH[e] for e in sorted(EXPL, key=lambda e: -arr[e].mean()))
        print(f"   order: {order}")
        praw = {}
        for a, b in PAIRS:
            d = arr[a] - arr[b]
            if np.allclose(d, 0):
                praw[(a, b)] = 1.0
            else:
                praw[(a, b)] = wilcoxon(arr[a], arr[b]).pvalue
        # Holm over the 3 pairs
        items = sorted(praw.items(), key=lambda kv: kv[1])
        holm = {}
        for rank, (pair, p) in enumerate(items):
            holm[pair] = min(1.0, p * (3 - rank))
        for a, b in PAIRS:
            star = "***" if holm[(a, b)] < 1e-3 else "**" if holm[(a, b)] < 1e-2 else "*" if holm[(a, b)] < 5e-2 else "ns"
            hi = SH[a] if np.median(arr[a] - arr[b]) > 0 else SH[b]
            print(f"   {SH[a]}-{SH[b]:<3} p_raw={praw[(a,b)]:.4f}  p_Holm={holm[(a,b)]:.4f}  {star}  ({hi} higher)")

    print("\n F3 as written:  GNN 0.448 ~= PG 0.300 >> SX 0.038 ;"
          " GNN-PG p_Holm=0.084 ns, GNN-SX ***, PG-SX **")


def main() -> int:
    dl = check1_toxicophores()
    check2_gea(dl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
