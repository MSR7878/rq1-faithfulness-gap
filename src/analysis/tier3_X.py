"""
RQ1 Tier 3, item X: Fid under R1, R2, R3 for item-M seeds 5-9 and item-R randomised models; data for the new Fig 1.

    python -m src.analysis.tier3_X fid --unit U --kind M|full|head
    python -m src.analysis.tier3_X figdata
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import tier2_common as C
from .tier2_fid import run_fid
from .tier2_R import rand_scores

T = "runs/rq1_tier3/X"
PANELS = {"MUTAG-GT": [f"mutag_graphxai_s{s}" for s in range(10)], "Mutagenicity D-MPNN": [f"mutagenicity_dmpnn_s{s}" for s in range(10)],
          "B-XAIC indole": [f"bxaic_indole_s{s}" for s in range(10)], "B-XAIC PAINS": [f"bxaic_PAINS_s{s}" for s in range(5)]}


def fid(unit, kind):
    dkey = C.dataset_of(unit)
    if kind == "M":
        sp = C.m_units()[unit]; d = f"{C.T2}/M"; cache = C.torch_load(sp["cache"]); ti = [int(m) for m in cache["key"]["test_idx"]]
        sc = {"gnnexplainer": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in zip(ti, cache["results"]["gnnexplainer"])},
              "pgexplainer": C.npz_scores(f"{d}/gp/{unit}_pg_logits.npz", "node_score64"), "sx_fixed": C.sx_shard_scores(f"{d}/sx/{unit}"),
              "ig": C.npz_scores(f"{d}/grad/{unit}.npz", "ig"), "hetero": C.heuristic_dict(unit, ti, "hetero")}
        run_fid(unit, sp["ckpt"], f"{d}/gp/{unit}_out.json", dkey, sc, ti, f"{T}/fid/M/{unit}.npz", with_randnode=True)
        return
    r = f"{C.T2}/R"; ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; rs = rand_scores(unit, kind); ck, oj = f"{r}/ckpt/{unit}_{kind}.pt", f"{r}/gp/{kind}/{unit}_out.json"
    run_fid(f"{unit}|{kind}", ck, oj, dkey, {m: rs[m] for m in ("gnnexplainer", "pgexplainer", "ig", "sal")}, ti, f"{T}/fid/R/{unit}_{kind}.npz", with_randnode=True)
    run_fid(f"{unit}|{kind}|sx", ck, oj, dkey, {"sx_fixed": rs["subgraphx"]}, sorted(rs["subgraphx"]), f"{T}/fid/R/{unit}_{kind}_sx.npz", with_randnode=True)


def figdata():
    D = {}
    for panel, units in PANELS.items():
        P = {}
        for u in units:
            s = int(u.rsplit("_s", 1)[1]); dkey = C.dataset_of(u)
            if s >= 5 and not u.startswith("mutag_graphxai"):
                z = np.load(f"{C.T2}/M/gea/{u}.npz"); gt = [int(m) for m in z["mol_idx"]]
                arr = {k: z[k] for k in ("gnnexplainer", "pgexplainer", "sx_fixed", "ig", "rn", "rc", "hetero")}; arr["pg_final_rule"] = z["pgexplainer"]
            else:
                H, G, S = (np.load(f"{C.T2}/{i}/gea/{u}.npz") for i in "HGS"); gt = [int(m) for m in H["mol_idx"]]
                pg = C.npz_scores(f"runs/rq1_revision/logits/{u}.npz", "node_score64")
                arr = {"gnnexplainer": H["gnnexplainer"], "pgexplainer": C.gea_array(u, "fig1_v2_pg64", lambda x: pg[x], gt, dkey), "pg_final_rule": H["pgexplainer"], "sx_fixed": S["sx_fixed"],
                       "ig": G["ig"], "rn": H["rn"], "rc": H["rc"], "hetero": H["hetero"], "deg_high": H["deg_high"], "deg_low": H["deg_low"]}
            for w in ("deg_high", "deg_low"):
                if w not in arr:
                    h = C.heuristic_dict(u, gt, w); arr[w] = C.gea_array(u, w, lambda x, h=h: h[x], gt, dkey)
            for k, v in arr.items():
                P[f"{panel}|{k}|{u}"] = np.asarray(v, dtype=float)
        D.update(P)
    os.makedirs(T, exist_ok=True)
    np.savez(f"{T}/fig1_v2_data.npz", **D)
    print("figure data written", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd"); a.add_argument("--unit"); a.add_argument("--kind"); a = a.parse_args()
    fid(a.unit, a.kind) if a.cmd == "fid" else figdata()
