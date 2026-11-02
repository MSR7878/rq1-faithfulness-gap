"""
RQ1 Tier 3, item E (exploratory): GNNExplainer and PGExplainer on Mutagenicity under mask variants all / first / last.

    python -m src.analysis.tier3_E explain --unit U --variant V
    python -m src.analysis.tier3_E score --unit U
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from . import tier2_common as C
from .tier2_explain import run_gp
from .tier3_mask import VARIANTS, global_variant

T = "runs/rq1_tier3/E"


def explain(unit, v):
    sp = C.orig_spec(unit); ck = torch.load(sp["ckpt"], map_location="cpu", weights_only=False)
    with global_variant(v, int(ck["cfg"]["depth"])):
        run_gp(unit, sp["ckpt"], f"{T}/gp/{v}", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def rho(a, b):
    return np.nan if np.std(a) == 0 or np.std(b) == 0 else float(spearmanr(a, b).statistic)


def score(unit):
    dkey = C.dataset_of(unit); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; gt = C.gt_mol_idx(unit, ti, dkey); nl = C.nulls(unit, gt, dkey)
    H, G, S = (np.load(f"{C.T2}/{i}/gea/{unit}.npz") for i in "HGS")
    for z in (H, G, S):
        assert [int(m) for m in z["mol_idx"]] == gt, unit
    A = {"unit": unit, "mol_idx": np.array(gt), "rn": nl["rn"], "rc": nl["rc"], "stored_gnnexplainer": H["gnnexplainer"], "stored_pgexplainer": H["pgexplainer"],
         "sx_fixed": S["sx_fixed"], "ig": G["ig"], "hetero": H["hetero"]}
    pg64 = C.npz_scores(f"runs/rq1_revision/logits/{unit}.npz", "node_score64"); gnn0 = C.stored_scores(unit, "gnnexplainer")
    A["stored_pg64"] = C.gea_array(f"{unit}|E|stored", "pgexplainer", lambda x: pg64[x], gt, dkey)
    sc = {}
    for v in VARIANTS:
        c = C.torch_load(f"{T}/gp/{v}/{unit}_cache.pt"); assert [int(m) for m in c["key"]["test_idx"]] == ti, (unit, v)
        sc[v] = {"gnnexplainer": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in zip(ti, c["results"]["gnnexplainer"])},
                 "pgexplainer": C.npz_scores(f"{T}/gp/{v}/{unit}_pg_logits.npz", "node_score64")}
        for m, s in sc[v].items():
            A[f"{v}_{m}"] = C.gea_array(f"{unit}|E|{v}", m, lambda x, s=s: s[x], gt, dkey)
    for m in ("gnnexplainer", "pgexplainer"):
        for v in ("first", "last"):
            A[f"rho_{v}_{m}"] = np.array([rho(sc[v][m][x], sc["all"][m][x]) for x in gt])
        s0 = gnn0 if m == "gnnexplainer" else pg64
        A[f"rho_stored_{m}"] = np.array([rho(sc["all"][m][x], s0[x]) for x in gt])
    C.save_npz(f"{T}/gea/{unit}.npz", **A); print(f"{unit}: scored", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd"); a.add_argument("--unit", required=True); a.add_argument("--variant"); a = a.parse_args()
    explain(a.unit, a.variant) if a.cmd == "explain" else score(a.unit)
