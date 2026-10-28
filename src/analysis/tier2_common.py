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


# ------------------------------------------------------------------ where each method's node scores live
def m_units() -> dict:
    """Item M: seeds 5-9 of Mutagenicity D-MPNN and B-XAIC indole, with the original explain arguments and explainer seeds."""
    from .revision_logits import BXA, MUT
    U = {}
    for s in range(5, 10):
        U[f"mutagenicity_dmpnn_s{s}"] = dict(dataset="mutagenicity_graphxai", ckpt=f"{T2}/M/ckpt/mutagenicity_dmpnn_s{s}.pt",
                                             cache=f"{T2}/M/gp/mutagenicity_dmpnn_s{s}_cache.pt", seed=1000 + s, args=MUT)
        U[f"bxaic_indole_s{s}"] = dict(dataset="bxaic_indole", ckpt=f"{T2}/M/ckpt/bxaic_indole_s{s}.pt",
                                       cache=f"{T2}/M/gp/bxaic_indole_s{s}_cache.pt", seed=s, args=BXA)
    return U


M_TRAIN = {"mutagenicity_dmpnn": "--dataset mutagenicity_graphxai --depth 3 --epochs 120 --hidden 300 --val-frac 0.10 --test-frac 0.20",
           "bxaic_indole": "--dataset bxaic_indole --depth 3 --epochs 25 --batch-size 512 --hidden 256"}


def orig_spec(unit):
    from .revision_logits import units
    U = units()
    return U[unit] if unit in U else m_units()[unit]


_CACHE = {}


def stored_cache(unit):
    if unit not in _CACHE:
        _CACHE[unit] = torch_load(orig_spec(unit)["cache"])
    return _CACHE[unit]


def pg_final(unit) -> tuple[dict, dict]:
    """PGExplainer node scores under the final rule (stored mask; exact ties split by the re-run's float64 logit scores
    only where the re-run reproduces the stored edge mask). Returns (mi -> scores, stats)."""
    blob = stored_cache(unit); res = blob["results"]["pgexplainer"]; ti = [int(m) for m in blob["key"]["test_idx"]]
    lz = np.load(f"runs/rq1_revision/logits/{unit}.npz", allow_pickle=True)
    rr = torch_load(f"runs/rq1_revision/logits/rerun/{unit}_cache.pt")["results"]["pgexplainer"]
    assert [int(m) for m in lz["mol_idx"]] == ti
    out, st = {}, {"tie_boundary": 0, "resolved_by_logits": 0, "random_fallback": 0}
    for pos, mi in enumerate(ti):
        prim = np.asarray(res[pos]["node_importance"], dtype=np.float64); k = kbudget(len(prim))
        a, b = np.asarray(res[pos]["edge_importance"]), np.asarray(rr[pos]["edge_importance"])
        rep = len(a) == 0 or float(np.max(np.abs(a - b))) <= 1e-5
        sc = lex_scores(prim, lz["node_score64"][pos]) if rep else prim
        if boundary_tie(prim, k):
            st["tie_boundary"] += 1
            st["resolved_by_logits" if not boundary_tie(sc, k) else "random_fallback"] += 1
        out[mi] = sc
    return out, st


def stored_scores(unit, method) -> dict:
    blob = stored_cache(unit)
    return {int(mi): np.asarray(r["node_importance"], dtype=np.float64) for mi, r in zip(blob["key"]["test_idx"], blob["results"][method])}


def npz_scores(path, key) -> dict:
    z = np.load(path, allow_pickle=True)
    return {int(mi): np.asarray(v, dtype=np.float64) for mi, v in zip(z["mol_idx"], z[key])}


def sx_shard_scores(d) -> dict:
    import glob
    rows = [r for f in sorted(glob.glob(f"{d}/shard_*_of_*.pt")) for r in torch_load(f)]
    return {int(r["mol_idx"]): np.asarray(r["node_importance"], dtype=np.float64) for r in rows}


def sx_shard_rows(d) -> dict:
    import glob
    return {int(r["mol_idx"]): r for f in sorted(glob.glob(f"{d}/shard_*_of_*.pt")) for r in torch_load(f)}


def heuristic_dict(unit, mol_idx, which) -> dict:
    dkey = dataset_of(unit); dl, _ = load_data(dkey)
    return {int(mi): heuristic_scores(dl[int(mi)], dkey)[which] for mi in mol_idx}
