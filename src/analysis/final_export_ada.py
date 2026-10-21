"""
RQ1 final export, the part that needs the cached explanations (run on Ada):
  1. MUTAG-GT seeds 5-9: random-tie top-k GEA with the Phase 2b scheme
     (tag "audit_mutag_graphxai_s<seed>").
  2. Part 1 check: MUTAG-GT seed 3 (collapsed PGExplainer) re-scored under 50
     further tie streams; sign of PG vs random-node under the main-table rule.
  3. Mean-threshold GEA recomputed from the caches for every GEA unit and
     compared with the stored arrays (they have no ties; must be unchanged).
Writes only runs/rq1_final_export/.

    python -m src.analysis.final_export_ada
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data import LOADERS
from ..metrics.gea_set import gea_max_over_set
from . import exploratory_null_audit as na
from .followups_phase2b import ARCH, AUDIT_KEYS, gea_unit
from .mutagenicity_report import holm, wpair
from .ties_random import random_tie_gea

OUT = "runs/rq1_final_export"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]


def main():
    os.makedirs(f"{OUT}/gea", exist_ok=True); rep = {}
    dl, _ = LOADERS["mutag_graphxai"]()
    sc = lambda mi, d: na.Scorer(d.node_gt_mask.numpy().astype(bool))
    # 1. seeds 5-9
    for s in range(5, 10):
        z = np.load(f"runs/rq1_followups/phase1/null_audit_arrays/mutag_graphxai_s{s}.npz")
        A = gea_unit(f"audit_mutag_graphxai_s{s}", f"runs/rq1_followups/phase1/cache_mutag_graphxai_s{s}.pt", dl, sc, z["mol_idx"],
                     np.random.default_rng(40000 + s))
        np.savez(f"{OUT}/gea/audit_mutag_graphxai_s{s}.npz", **A)
        print(f"seed {s}: PG ties {int(A['pgexplainer_tie'].sum())}, SX ties {int(A['subgraphx_tie'].sum())}, GNN ties {int(A['gnnexplainer_tie'].sum())}", flush=True)
    # 2. seed-3 stream check
    blob = torch.load("runs/prio/cache_mutag_graphxai_s3.pt", weights_only=False); res = blob["results"]
    st = np.load("runs/mutagenicity/exploratory/null_audit_arrays/mutag_graphxai_s3.npz"); rn = st["rn_k"]
    signs, ps, means = [], [], []
    for j in range(50):
        real = {e: [] for e in EXPL}
        for pos, mi in enumerate(blob["key"]["test_idx"]):
            d = dl[mi]; gt = d.node_gt_mask.numpy().astype(bool)
            if not gt.any():
                continue
            k = max(1, int(0.25 * d.num_nodes)); s_ = na.Scorer(gt)
            for e in EXPL:
                real[e].append(random_tie_gea(res[e][pos]["node_importance"], k, s_, f"stability{j}_s3|{e}", pos)[0])
        raw = {e: wpair(np.array(real[e]), rn) for e in EXPL}; adj = holm(raw)
        a = np.mean(real["pgexplainer"]); b = rn.mean()
        signs.append("+" if a > b and adj["pgexplainer"] < .05 else "-" if a < b and adj["pgexplainer"] < .05 else ".")
        ps.append(adj["pgexplainer"]); means.append(a)
    rep["seed3_streams"] = {"n": 50, "plus": signs.count("+"), "minus": signs.count("-"), "ns": signs.count("."),
                            "pg_mean_min": float(min(means)), "pg_mean_max": float(max(means)), "null_mean": float(rn.mean()),
                            "holm_p_min": float(min(ps)), "holm_p_median": float(np.median(ps))}
    print("seed 3, 50 extra tie streams:", rep["seed3_streams"], flush=True)
    # 3. mean-threshold confirmation
    devs = {}
    for key, tmpl in list(AUDIT_KEYS.items()) + [("mutag_graphxai_new", "runs/rq1_followups/phase1/cache_mutag_graphxai_s{s}.pt")]:
        dk = "mutag_graphxai" if key == "mutag_graphxai_new" else key
        dlk, meta = LOADERS[dk]() if dk != "mutag_graphxai" else (dl, None)
        motifs = getattr(meta, "extras", {}).get("motifs") if meta is not None and hasattr(meta, "extras") else None
        seeds = range(5, 10) if key == "mutag_graphxai_new" else range(5)
        for s in seeds:
            arr = (f"runs/rq1_followups/phase1/null_audit_arrays/mutag_graphxai_s{s}.npz" if key == "mutag_graphxai_new"
                   else f"runs/mutagenicity/exploratory/null_audit_arrays/{key}_s{s}.npz")
            z = np.load(arr); b = torch.load(tmpl.format(s=s), weights_only=False); r = b["results"]; mx = 0.0; i = 0
            for pos, mi in enumerate(b["key"]["test_idx"]):
                d = dlk[mi]; gt = d.node_gt_mask.numpy().astype(bool)
                if not gt.any():
                    continue
                s_ = na.Scorer(gt, motifs[mi] if motifs is not None else None)
                for e in EXPL:
                    imp = np.asarray(r[e][pos]["node_importance"], dtype=np.float64)
                    mx = max(mx, abs(s_(imp > imp.mean()) - z[f"{e}_mean"][i]))
                i += 1
            devs[f"{dk}_s{s}"] = mx
    dlm, meta = LOADERS["mutagenicity_graphxai"](); motifs = meta.extras["motifs"]
    units = [(f"arch_{a}_s{s}", f"{d}/cache_s{s}.pt", f"{d}/arrays_s{s}.npz") for a, d in ARCH.items() for s in range(5)]
    units += [(f"cv_f{f}", f"runs/mutagenicity_cv/cache_s{f}.pt", f"runs/mutagenicity_cv/arrays_s{f}.npz") for f in range(5)]
    for tag, cache, arr in units:
        z = np.load(arr); b = torch.load(cache, weights_only=False); r = b["results"]; mx = 0.0; i = 0
        for pos, mi in enumerate(b["key"]["test_idx"]):
            d = dlm[mi]
            if not bool(d.node_gt_mask.any()):
                continue
            for e in EXPL:
                imp = np.asarray(r[e][pos]["node_importance"], dtype=np.float64)
                mx = max(mx, abs(gea_max_over_set(imp > imp.mean(), motifs[mi]).max_over_set - z[f"{e}_mean_set"][i]))
            i += 1
        devs[tag] = mx
    rep["mean_threshold_max_abs_dev"] = devs
    print("mean-threshold: max |recomputed - stored| over all units:", max(devs.values()), flush=True)
    json.dump(rep, open(f"{OUT}/ada_checks.json", "w"), indent=1)
    open(f"{OUT}/ADA_FINISHED", "w").write("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
