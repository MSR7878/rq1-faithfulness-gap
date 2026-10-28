"""
RQ1 Tier 2 (runs/rq1_tier2/PLAN.md): shared pieces.

Final protocol used by every Tier-2 number:
  - GEA at the top-25% budget k = max(1, floor(0.25 N)) on ground-truth molecules; ties at the k boundary
    broken at random, 20 seeded orders averaged per molecule (stream "<unit>|<method>|gea").
    Scorer: Jaccard (MUTAG-GT, B-XAIC), max-over-set over the GraphXAI motifs (Mutagenicity).
  - Nulls, drawn ONCE per unit (seeded by the unit name) and shared by every method:
    random-node (20 uniform k-sets) and random-connected (20 connected k-sets), mean GEA per molecule.
  - PGExplainer on stored caches: ranked by the stored mask; exact ties broken by the float64 logit-based
    node scores of the T4-addendum re-run, but only on molecules whose re-run reproduces the stored edge mask
    (max |diff| <= 1e-5); remaining ties at random. New PGExplainer runs rank by their own float64 logits.
  - Main number: hierarchical bootstrap of the mean difference (seeds, then molecules; 2,000 reps; seed 2026;
    95% percentile CI). Support: per-seed paired Wilcoxon + Holm (family stated per table). Never pooled.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from ..data import LOADERS
from ..metrics.gea_set import gea_max_over_set
from .connectivity_baseline import build_adj, random_connected_subset
from .mutagenicity_cv_report import rank_direction
from .mutagenicity_report import holm
from .revision_analysis import hboot
from .revision_pass import all_specs
from .ties_random import N_ORDERS, boundary_tie, topk_mask, tie_order

T2 = "runs/rq1_tier2"
N_DRAWS = 20
K_FRAC = 0.25
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
ARCHS = ["dmpnn", "gin", "gine", "gcn"]
BX = ["indole", "PAINS", "X", "P"]
GROUPS = {**{f"Mutagenicity {a.upper() if a != 'dmpnn' else 'D-MPNN'}": [f"mutagenicity_{a}_s{s}" for s in range(5)] for a in ARCHS},
          **{f"B-XAIC {t}": [f"bxaic_{t}_s{s}" for s in range(5)] for t in BX},
          "MUTAG-GT (10 seeds)": [f"mutag_graphxai_s{s}" for s in range(10)]}
VALID = [g for g in GROUPS if not g.startswith("MUTAG-GT")]

# atom vocabularies (x is a one-hot over these, in this order)
ATOMS = {"mutag_graphxai": ["C", "N", "O", "F", "I", "Cl", "Br"],
         "mutagenicity_graphxai": ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"],
         "bxaic": ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]}


def kbudget(n: int) -> int:
    return max(1, int(K_FRAC * n))


def unit_specs() -> dict:
    U = all_specs()
    return {u: U[u] for g in GROUPS.values() for u in g}


_DL = {}


def dataset_of(unit: str) -> str:
    if unit.startswith("mutagenicity"):
        return "mutagenicity_graphxai"
    if unit.startswith("mutag_graphxai"):
        return "mutag_graphxai"
    return "bxaic_" + unit.split("_")[1]


def load_data(key: str):
    if key not in _DL:
        _DL[key] = LOADERS[key]()
    return _DL[key]


def atom_vocab(dkey: str):
    return ATOMS["bxaic"] if dkey.startswith("bxaic") else ATOMS[dkey]


class GEAScorer:
    def __init__(self, d, motifs=None):
        self.gt = d.node_gt_mask.detach().cpu().numpy().astype(bool); self.motifs = motifs

    def __call__(self, mask):
        if self.motifs is not None:
            return gea_max_over_set(mask, self.motifs).max_over_set
        u = int((mask | self.gt).sum())
        return int((mask & self.gt).sum()) / u if u else 0.0


def scorer_for(dkey, d, mi):
    meta = load_data(dkey)[1]
    motifs = meta.extras["motifs"][mi] if dkey == "mutagenicity_graphxai" else None
    return GEAScorer(d, motifs)


def mol_rng(stream: str, mi: int) -> np.random.Generator:
    return np.random.default_rng([81000, zlib.crc32(stream.encode()), int(mi)])


def gea_random_ties(scores, k, scorer, stream, mi) -> float:
    """Mean GEA over N_ORDERS random orders within exact ties (one value if the top-k set has no boundary tie)."""
    if not boundary_tie(scores, k):
        return float(scorer(topk_mask(scores, k)))
    r = mol_rng(stream, mi)
    return float(np.mean([scorer(topk_mask(tie_order(scores, r), k)) for _ in range(N_ORDERS)]))


def lex_scores(primary, secondary) -> np.ndarray:
    """Dense ranks of (primary, secondary): secondary only separates exact ties of primary; equal pairs stay tied."""
    p = np.asarray(primary, dtype=np.float64); s = np.asarray(secondary, dtype=np.float64)
    order = np.lexsort((s, p)); out = np.empty(len(p)); r = -1; prev = None
    for i in order:
        key = (p[i], s[i])
        if key != prev:
            r += 1; prev = key
        out[i] = r
    return out


# ------------------------------------------------------------------ nulls (once per unit)
def nulls(unit: str, mol_idx, dkey: str) -> dict:
    """Random-node and random-connected null GEA per GT molecule; cached in runs/rq1_tier2/nulls/<unit>.npz."""
    f = f"{T2}/nulls/{unit}.npz"
    if os.path.exists(f):
        z = dict(np.load(f))
        if list(z["mol_idx"]) == [int(m) for m in mol_idx]:
            return z
        raise RuntimeError(f"null file {f} has a different molecule list")
    dl, _ = load_data(dkey)
    rn = np.random.default_rng([82000, zlib.crc32(unit.encode()), 0]); rc = np.random.default_rng([82000, zlib.crc32(unit.encode()), 1])
    A = {"mol_idx": [], "rn": [], "rc": [], "k": [], "n": []}
    for mi in mol_idx:
        d = dl[int(mi)]; n = d.num_nodes; k = kbudget(n); sc = scorer_for(dkey, d, int(mi)); adj = build_adj(d.edge_index, n)
        a = b = 0.0
        for _ in range(N_DRAWS):
            m = np.zeros(n, bool); m[rn.choice(n, size=k, replace=False)] = True; a += sc(m)
        for _ in range(N_DRAWS):
            m = np.zeros(n, bool); m[list(random_connected_subset(adj, n, k, rc))] = True; b += sc(m)
        A["mol_idx"].append(int(mi)); A["rn"].append(a / N_DRAWS); A["rc"].append(b / N_DRAWS); A["k"].append(k); A["n"].append(n)
    A = {k: np.array(v) for k, v in A.items()}
    os.makedirs(f"{T2}/nulls", exist_ok=True); tmp = f + ".tmp.npz"
    np.savez(tmp, **A); os.replace(tmp, f)
    return A


def gt_mol_idx(unit: str, test_idx, dkey: str) -> list[int]:
    dl, _ = load_data(dkey)
    return [int(mi) for mi in test_idx if bool(dl[int(mi)].node_gt_mask.any())]


def gea_array(unit, method, score_of, mol_idx, dkey) -> np.ndarray:
    """score_of(mi) -> node scores; GEA under the final protocol for each molecule in mol_idx."""
    dl, _ = load_data(dkey)
    out = []
    for mi in mol_idx:
        d = dl[int(mi)]; sc = score_of(int(mi))
        out.append(gea_random_ties(np.asarray(sc, dtype=np.float64), kbudget(d.num_nodes), scorer_for(dkey, d, int(mi)), f"{unit}|{method}|gea", int(mi)))
    return np.array(out)


# ------------------------------------------------------------------ heuristics
def heuristic_scores(d, dkey) -> dict:
    voc = atom_vocab(dkey); lab = d.x.detach().cpu().numpy().argmax(1); sym = [voc[i] for i in lab]
    hetero = np.array([0.0 if s in ("C", "H") else 1.0 for s in sym])
    heavy = np.array([s != "H" for s in sym]); ei = d.edge_index.detach().cpu().numpy(); n = d.num_nodes
    deg = np.zeros(n)
    for a, b in ei.T:              # directed edges: each bond appears twice, so count a's heavy neighbours once per (a, b)
        if heavy[b]:
            deg[a] += 1
    return {"hetero": hetero, "deg_high": deg, "deg_low": -deg}


# ------------------------------------------------------------------ tests
def wil(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 6 or np.allclose(a, b):
        return 1.0, 0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue), rank_direction(a, b)


def holm_signs(raw: dict) -> dict:
    adj = holm({k: v[0] for k, v in raw.items()})
    return {k: ("+" if raw[k][1] > 0 and adj[k] < .05 else "-" if raw[k][1] < 0 and adj[k] < .05 else ".") for k in raw}


def counts(signs) -> str:
    return f"{signs.count('+')}/{signs.count('-')}/{signs.count('.')}"


def ci_cat(lo, hi) -> str:
    return "above" if lo > 0 else "below" if hi < 0 else "includes 0"


def boot(real_by_seed, null_by_seed):
    p, lo, hi, _ = hboot(real_by_seed, null_by_seed)
    return p, lo, hi


def seed_sign_test(real_by_seed, null_by_seed):
    d = [float(np.mean(np.asarray(a) - np.asarray(b))) for a, b in zip(real_by_seed, null_by_seed)]
    pos = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    return pos, nz, (float(binomtest(pos, nz, 0.5).pvalue) if nz else 1.0)


def save_npz(path, **A):
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + ".tmp.npz"
    np.savez(tmp, **A); os.replace(tmp, path)


def torch_load(path):
    return torch.load(path, weights_only=False)
