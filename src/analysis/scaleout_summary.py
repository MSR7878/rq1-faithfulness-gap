"""
Scale-out summary: 12 Tox21 endpoints + 4 B-XAIC tasks (indole from Phase 3 +
PAINS / X / P from the server sweep).

    python -m src.analysis.scaleout_summary

Reads runs/phase3_bxaic.json (Phase-3, torch 2.14) and runs/scaleout/*.json
(server, torch 2.6+cu118). Reports:
  * B-XAIC 4-task GEA (Jaccard, per explainer) + within-task Wilcoxon -- does the
    Phase-3 GNN<->SX ranking inversion generalise across task types?
  * Tox21 12-endpoint aggregate: R3/R2/R1 Fid+/Fid-/GEF means across endpoints,
    + which endpoints have a degenerate (collapsed) explainer per explainer_health.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]


def load(path):
    with open(path) as f:
        return json.load(f)


def pm(j, mk, fl, e, metric):
    return np.array([r[metric] for r in j["per_molecule"][mk][fl][e] if metric in r], float)


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def bxaic_gea():
    print("=" * 76 + "\n B-XAIC GEA (Jaccard vs substructure GT) across 4 task types\n" + "=" * 76)
    tasks = [("indole", "runs/phase3_bxaic.json"),
             ("PAINS", "runs/scaleout/phase3_bxaic_PAINS.json"),
             ("X", "runs/scaleout/phase3_bxaic_X.json"),
             ("P", "runs/scaleout/phase3_bxaic_P.json")]
    print(f"{'task':<8}{'kind':<20}{'GNN':>8}{'PG':>8}{'SX':>8}   pairwise (Wilcoxon, Holm-free)")
    kinds = {"indole": "ring substructure", "PAINS": "reactive-group alerts",
             "X": "halogen presence", "P": "phosphorus presence"}
    for t, path in tasks:
        if not os.path.exists(path):
            print(f"{t:<8}(missing {path})"); continue
        j = load(path)
        col = {e: pm(j, "R3", "mean", e, "gea") for e in EXPL}
        n = len(col["gnnexplainer"])
        means = {e: (col[e].mean() if n else float("nan")) for e in EXPL}
        sig = []
        for a, b in PAIRS:
            if n >= 6 and not np.allclose(col[a] - col[b], 0):
                _, p = wilcoxon(col[a], col[b])
                hi = SH[a] if np.median(col[a] - col[b]) > 0 else SH[b]
                sig.append(f"{SH[a]}-{SH[b]}:{star(p)}({hi}>)")
            else:
                sig.append(f"{SH[a]}-{SH[b]}:n/a")
        order = " > ".join(SH[e] for e in sorted(EXPL, key=lambda e: -means[e]))
        print(f"{t:<8}{kinds[t]:<20}{means['gnnexplainer']:>8.3f}{means['pgexplainer']:>8.3f}"
              f"{means['subgraphx']:>8.3f}   n={n}  order: {order}")
        print(f"{'':<28}{'  '.join(sig)}")


def tox21_aggregate():
    print("\n" + "=" * 76 + "\n Tox21 -- 12 endpoints, aggregate over endpoints (mean of per-endpoint means)\n" + "=" * 76)
    files = sorted(glob.glob("runs/scaleout/phase3_tox21_*.json"))
    rows = {mk: {e: {m: [] for m in ("fid_plus", "fid_minus", "gef")} for e in EXPL}
            for mk in ("R3", "R2", "R1")}
    degen = {e: [] for e in EXPL}
    aurocs = []
    for path in files:
        j = load(path)
        ep = os.path.basename(path).replace("phase3_tox21_", "").replace(".json", "")
        h = j.get("explainer_health", {})
        for e in EXPL:
            he = h.get(e, {})
            if he.get("n_degenerate", 0) / max(he.get("n", 30), 1) > 0.2:
                degen[e].append(ep)
        fl = {"R3": "mean", "R2": "zero", "R1": "hard"}
        for mk in ("R3", "R2", "R1"):
            for e in EXPL:
                for m in ("fid_plus", "fid_minus", "gef"):
                    v = pm(j, mk, fl[mk], e, m)
                    if len(v):
                        rows[mk][e][m].append(float(np.nanmean(v)))
    print(f"  endpoints: {len(files)}")
    for mk in ("R3", "R2", "R1"):
        print(f"\n  [{mk}]  (mean +/- sd across {len(files)} endpoints)")
        print(f"    {'expl':<6}{'Fid+':>16}{'Fid-':>16}{'GEF':>16}")
        for e in EXPL:
            cells = []
            for m in ("fid_plus", "fid_minus", "gef"):
                a = np.array(rows[mk][e][m], float)
                cells.append(f"{np.nanmean(a):.3f}+/-{np.nanstd(a):.3f}")
            print(f"    {SH[e]:<6}" + "".join(f"{c:>16}" for c in cells))
    print("\n  degenerate/collapsed explainer (>20% std=0 nodes) by endpoint:")
    for e in EXPL:
        print(f"    {SH[e]:<4}: {degen[e] if degen[e] else 'none'}")


def main() -> int:
    bxaic_gea()
    tox21_aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
