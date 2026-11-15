"""RQ1 Tier 5 (runs/rq1_tier5/PLAN.md): units, subsets, scale levels and shared helpers."""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
import torch

from . import tier2_common as C
from . import tier4_common as T

T5 = "runs/rq1_tier5"
NEW = ["eig", "gstarx", "same", "gnnshap"]
ARCH4 = ["dmpnn", "gin", "gine", "gcn"]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
WORKERS = 12
BUDGET_H = 24.0
RULE = {"bxaic_indole": "deg_high", "bxaic_PAINS": "deg_high", "mutagenicity_graphxai": "hetero", "mutag_graphxai": "hetero"}


def arch_of(unit: str) -> str:
    if unit.startswith("mutagenicity_"):
        return unit.split("_")[1]
    if unit.startswith("bxaic_"):
        p = unit.split("_")
        return p[2] if len(p) == 4 else "dmpnn"
    return "dmpnn"


def seed_of(unit: str) -> int:
    return int(unit.rsplit("_s", 1)[1])


def n1_units() -> list[str]:
    U = [f"bxaic_{t}_s{s}" for t in ("indole", "PAINS") for s in range(10)]
    U += [f"bxaic_{t}_{a}_s{s}" for t in ("indole", "PAINS") for a in ("gin", "gine", "gcn") for s in range(5)]
    U += [f"mutagenicity_dmpnn_s{s}" for s in range(10)] + [f"mutagenicity_{a}_s{s}" for a in ("gin", "gine", "gcn") for s in range(5)]
    U += [f"mutag_graphxai_s{s}" for s in range(10)]
    return U


def spec(unit: str) -> dict:
    return C.orig_spec(unit)


def test_idx(unit: str) -> list[int]:
    return [int(m) for m in C.torch_load(spec(unit)["cache"])["key"]["test_idx"]]


def gt_mols(unit: str) -> list[int]:
    return C.gt_mol_idx(unit, test_idx(unit), C.dataset_of(unit))


def subset_owner(unit: str) -> str:
    """Units that explain the same molecules share the D-MPNN unit's subset (B-XAIC GIN/GINE/GCN; Mutagenicity GIN/GINE/GCN)."""
    if unit.startswith("mutagenicity_"):
        return f"mutagenicity_dmpnn_s{seed_of(unit)}"
    if unit.startswith("bxaic_"):
        return T.dmpnn_of(unit) if arch_of(unit) != "dmpnn" else unit
    return unit


def subsets() -> dict:
    """Fixed 100-GT-molecule subsets. Existing: Tier 4 (indole, PAINS s0-4), Tier 2 (Mutagenicity s0-4, MUTAG-GT s0).
    New ones by the same subset100 rule -> runs/rq1_tier5/subsets.json (written once)."""
    f = f"{T5}/subsets.json"
    if os.path.exists(f):
        return json.load(open(f))
    t2 = json.load(open(f"{C.T2}/subsets.json")); t4 = json.load(open(f"{T.T4}/subsets.json"))
    S = {}
    owners = sorted({subset_owner(u) for u in n1_units()})
    for u in owners:
        new = T.subset100(u, test_idx(u))
        old = t4.get(u, t2.get(u))
        if old is not None and [int(m) for m in old] != new:
            raise RuntimeError(f"subset of {u} differs from the earlier tier's subsets.json")
        S[u] = new
    os.makedirs(T5, exist_ok=True); json.dump(S, open(f, "w"), indent=0)
    return S


def slice_order(unit_owner: str, sub: list[int]) -> list[int]:
    """The subset in the fixed shuffled order used for L3 / L4 (rng([83500, crc32(owner)]))."""
    r = np.random.default_rng([83500, zlib.crc32(unit_owner.encode())])
    return [int(sub[i]) for i in r.permutation(len(sub))]


def level_mols(unit: str, level: str) -> list[int] | None:
    """Molecules of a unit at a scale level; None if the unit is not part of that level (seed >= 5 at L2-L4)."""
    if level == "L0":
        return gt_mols(unit)
    if level in ("L2", "L3", "L4") and seed_of(unit) >= 5:
        return None
    o = subset_owner(unit); sub = [int(m) for m in subsets()[o]]
    if level in ("L1", "L2"):
        return sub
    m = {"L3": 50, "L4": 25}[level]
    return sorted(slice_order(o, sub)[:m])


@torch.no_grad()
def payoff_avg(model, ck, dkey, device) -> list[float]:
    """Mean predicted class probabilities over the checkpoint's test split (as GStarX's run_gstarx.py does)."""
    dl, _ = C.load_data(dkey); ps = []
    for i in ck["split"]["test"]:
        d = dl[int(i)]; ea = getattr(d, "edge_attr", None)
        lg = model(d.x.float().to(device), d.edge_index.to(device), None if ea is None else ea.float().to(device))
        ps.append(torch.softmax(lg.reshape(-1, lg.shape[-1])[0], -1).cpu().numpy())
    return [float(v) for v in np.mean(ps, 0)]


def make_explainer(name, model, ck, dkey, device):
    from ..explain import t5_explainers as X
    if name == "eig":
        return X.EiG(model, device)
    if name == "gstarx":
        return X.GStarXExplainer(model, device, payoff_avg(model, ck, dkey, device))
    if name == "same":
        return X.SAMEExplainer(model, device, dkey)
    if name == "gnnshap":
        return X.GNNShapExplainer(model, device)
    raise ValueError(name)


# ------------------------------------------------------------------ where the old methods' scores and nulls live
def _src(unit: str) -> str:
    if unit in T.m1_units():
        return "M1"
    if unit in T.m2_units():
        return "M2"
    if unit in C.m_units():
        return "T2M"
    return "T2"


def old_scores(unit: str) -> dict:
    """method -> {mi: node scores} for GNNExplainer, PG (float64 logits), SubgraphX fixed, Saliency, IG, IxG."""
    s = _src(unit)
    gdir = {"M1": f"{T.T4}/M1", "M2": f"{T.T4}/M2", "T2M": f"{C.T2}/M", "T2": f"{C.T2}/G"}[s]
    sxdir = {"M1": f"{T.T4}/M1/sx", "M2": f"{T.T4}/M2/sx", "T2M": f"{C.T2}/M/sx", "T2": f"{C.T2}/S/sx"}[s]
    if s in ("M1", "M2"):
        pg = T.pg64(unit)
    elif s == "T2M":
        pg = C.npz_scores(f"{C.T2}/M/gp/{unit}_pg_logits.npz", "node_score64")
    else:
        pg = C.npz_scores(f"runs/rq1_revision/logits/{unit}.npz", "node_score64")
    g = f"{gdir}/grad/{unit}.npz"
    return {"gnnexplainer": T.cache_scores(spec(unit)["cache"], "gnnexplainer"), "pgexplainer": pg,
            "sx_fixed": C.sx_shard_scores(f"{sxdir}/{unit}"), "sal": C.npz_scores(g, "sal"), "ig": C.npz_scores(g, "ig"),
            "ixg": C.npz_scores(g, "ixg")}


def null_arrays(unit: str, mols) -> dict:
    """Random-node / random-connected null GEA for the given molecules, read from the existing full-GT null file
    (Tier 2 for Tier-2 units and, as in Tier 4 M1, the D-MPNN unit's file for B-XAIC GIN/GINE/GCN; Tier 4 for M2)."""
    owner = T.dmpnn_of(unit) if unit in T.m1_units() else unit
    f = f"{T.T4}/nulls/{owner}.npz" if unit in T.m2_units() else f"{C.T2}/nulls/{owner}.npz"
    z = np.load(f); pos = {int(m): i for i, m in enumerate(z["mol_idx"])}
    ix = [pos[int(m)] for m in mols]
    return {"rn": z["rn"][ix], "rc": z["rc"][ix]}


def n4_units() -> list[str]:
    return [u for u in n1_units() if not u.startswith("mutag_graphxai")]


def cell_of(unit: str) -> str:
    return f"{C.dataset_of(unit)}|{arch_of(unit)}"
