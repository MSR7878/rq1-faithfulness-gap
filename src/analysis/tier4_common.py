"""
RQ1 Tier 4 (runs/rq1_tier4/PLAN.md): shared pieces. Nothing here writes under runs/rq1_tier2.

Units added in Tier 4:
  bbbp_s<S>                     L2, D-MPNN on BBBP, seeds 0-4 (explainer seed 1000 + S)
  bxaic_<task>_<arch>_s<S>      M1, GIN / GINE / GCN on B-XAIC indole and PAINS, seeds 0-4, the D-MPNN's split and explained molecules
  bxaic_PAINS_s<S>, S = 5-9     M2, D-MPNN on B-XAIC PAINS
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

from . import tier2_common as C
from .connectivity_baseline import build_adj, random_connected_subset
from .revision_logits import BXA

T4 = "runs/rq1_tier4"
HIDDEN = {"gin": 183, "gine": 182, "gcn": 258}           # matched to the B-XAIC D-MPNN's 203,522 parameters
M1_ARCHS = ["gin", "gine", "gcn"]
M1_TASKS = ["indole", "PAINS"]
BBBP_ARGS = ["--limit", "30", "--pg-train-limit", "150"]
BBBP_TRAIN = "--dataset bbbp --depth 3 --hidden 300 --epochs 100 --batch-size 64"
BX_TRAIN = "--depth 3 --epochs 25 --batch-size 512"
BAR_ACC, BAR_AUROC = 0.95, 0.98


def m1_units():
    return [f"bxaic_{t}_{a}_s{s}" for t in M1_TASKS for a in M1_ARCHS for s in range(5)]


def m2_units():
    return [f"bxaic_PAINS_s{s}" for s in range(5, 10)]


def bbbp_units():
    return [f"bbbp_s{s}" for s in range(5)]


def dmpnn_of(unit):
    """M1 unit -> the D-MPNN unit with the same task and seed."""
    t, _, s = unit.split("_")[1], unit.split("_")[2], unit.rsplit("_s", 1)[1]
    return f"bxaic_{t}_s{s}"


def t4_units() -> dict:
    U = {}
    for u in bbbp_units():
        s = int(u.rsplit("_s", 1)[1])
        U[u] = dict(dataset="bbbp", ckpt=f"{T4}/L2/ckpt/{u}.pt", cache=f"{T4}/L2/gp/{u}_cache.pt", seed=1000 + s, args=BBBP_ARGS)
    for u in m1_units():
        s = int(u.rsplit("_s", 1)[1])
        U[u] = dict(dataset=C.dataset_of(u), ckpt=f"{T4}/M1/ckpt/{u}.pt", cache=f"{T4}/M1/gp/{u}_cache.pt", seed=s, args=BXA)
    for u in m2_units():
        s = int(u.rsplit("_s", 1)[1])
        U[u] = dict(dataset="bxaic_PAINS", ckpt=f"{T4}/M2/ckpt/{u}.pt", cache=f"{T4}/M2/gp/{u}_cache.pt", seed=s, args=BXA)
    return U


def train_cmd(unit) -> str:
    sp = t4_units()[unit]; s = int(unit.rsplit("_s", 1)[1])
    if unit.startswith("bbbp"):
        a = BBBP_TRAIN
    elif unit in m2_units():
        a = f"--dataset bxaic_PAINS {BX_TRAIN} --hidden 256"
    else:
        arch = unit.split("_")[2]
        a = f"--dataset {sp['dataset']} --arch {arch} {BX_TRAIN} --hidden {HIDDEN[arch]} --split-json {T4}/M1/splits/{dmpnn_of(unit)}.json"
    return f"$PY -u -m src.train.train {a} --seed {s} --device cuda --save {sp['ckpt']}"


def export_split(dmpnn_unit):
    """The D-MPNN checkpoint's split as json for train.py --split-json (read-only on the checkpoint)."""
    import torch
    f = f"{T4}/M1/splits/{dmpnn_unit}.json"
    if os.path.exists(f):
        return f
    ck = torch.load(C.orig_spec(dmpnn_unit)["ckpt"], map_location="cpu", weights_only=False)
    sp = {k: [int(i) for i in ck["split"][k]] for k in ("train", "val", "test")}
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump({**sp, "source_ckpt": C.orig_spec(dmpnn_unit)["ckpt"]}, open(f, "w"))
    return f


# ------------------------------------------------------------------ nulls and subsets (Tier-4 directories)
def nulls4(unit, mol_idx, dkey) -> dict:
    """Same rule as tier2_common.nulls (seeded by the unit name); file under runs/rq1_tier4/nulls."""
    f = f"{T4}/nulls/{unit}.npz"
    if os.path.exists(f):
        z = dict(np.load(f))
        if list(z["mol_idx"]) == [int(m) for m in mol_idx]:
            return z
        raise RuntimeError(f"null file {f} has a different molecule list")
    if os.path.exists(f"{C.T2}/nulls/{unit}.npz"):
        raise RuntimeError(f"{unit} already has Tier-2 nulls; read those instead")
    dl, _ = C.load_data(dkey)
    rn = np.random.default_rng([82000, zlib.crc32(unit.encode()), 0]); rc = np.random.default_rng([82000, zlib.crc32(unit.encode()), 1])
    A = {"mol_idx": [], "rn": [], "rc": [], "k": [], "n": []}
    for mi in mol_idx:
        d = dl[int(mi)]; n = d.num_nodes; k = C.kbudget(n); sc = C.scorer_for(dkey, d, int(mi)); adj = build_adj(d.edge_index, n)
        a = b = 0.0
        for _ in range(C.N_DRAWS):
            m = np.zeros(n, bool); m[rn.choice(n, size=k, replace=False)] = True; a += sc(m)
        for _ in range(C.N_DRAWS):
            m = np.zeros(n, bool); m[list(random_connected_subset(adj, n, k, rc))] = True; b += sc(m)
        A["mol_idx"].append(int(mi)); A["rn"].append(a / C.N_DRAWS); A["rc"].append(b / C.N_DRAWS); A["k"].append(k); A["n"].append(n)
    A = {k: np.array(v) for k, v in A.items()}
    C.save_npz(f, **A)
    return A


def tier2_nulls(unit, mol_idx, dkey) -> dict:
    """Read-only access to an existing Tier-2 null file (never creates one)."""
    if not os.path.exists(f"{C.T2}/nulls/{unit}.npz"):
        raise RuntimeError(f"no Tier-2 null file for {unit}")
    return C.nulls(unit, mol_idx, dkey)


def subset100(unit, test_idx) -> list[int]:
    """Tier 2 subset100 rule (fixed random 100 GT molecules, seeded by the unit name) without writing any file."""
    gt = C.gt_mol_idx(unit, test_idx, C.dataset_of(unit))
    if len(gt) <= 100:
        return gt
    r = np.random.default_rng([83000, zlib.crc32(unit.encode())])
    return sorted(int(m) for m in r.choice(gt, size=100, replace=False))


def m1_subsets() -> dict:
    """Fixed 100-GT-molecule subsets of the B-XAIC D-MPNN units (indole, PAINS; seeds 0-4) -> runs/rq1_tier4/subsets.json.
    Indole must equal the Tier-2 subsets (a check)."""
    f = f"{T4}/subsets.json"
    if os.path.exists(f):
        return json.load(open(f))
    S = {}
    for t in M1_TASKS:
        for s in range(5):
            u = f"bxaic_{t}_s{s}"; S[u] = subset100(u, [int(m) for m in C.stored_cache(u)["key"]["test_idx"]])
    t2 = json.load(open(f"{C.T2}/subsets.json"))
    for u in S:
        if u in t2 and [int(m) for m in t2[u]] != S[u]:
            raise RuntimeError(f"subset of {u} differs from runs/rq1_tier2/subsets.json")
    os.makedirs(T4, exist_ok=True); json.dump(S, open(f, "w"), indent=0)
    return S


# ------------------------------------------------------------------ score sources
def pg64(unit) -> dict:
    """PGExplainer float64 logit node scores: T4 re-run file for stored units, own gp file for new ones."""
    f = f"runs/rq1_revision/logits/{unit}.npz"
    if unit in t4_units():
        d = os.path.dirname(t4_units()[unit]["cache"]); f = f"{d}/{unit}_pg_logits.npz"
    return C.npz_scores(f, "node_score64")


def cache_scores(cache_path, method) -> dict:
    blob = C.torch_load(cache_path)
    return {int(mi): np.asarray(r["node_importance"], dtype=np.float64) for mi, r in zip(blob["key"]["test_idx"], blob["results"][method])}


def test_idx_of(unit) -> list[int]:
    return [int(m) for m in C.torch_load(C.orig_spec(unit)["cache"])["key"]["test_idx"]]
