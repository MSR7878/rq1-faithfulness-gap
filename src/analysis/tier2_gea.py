"""
RQ1 Tier 2: per-unit GEA (final protocol, src/analysis/tier2_common.py) and Fid arrays for items H, G, S.

  H  shared nulls (random-node, random-connected); GNNExplainer, PGExplainer (final tie rule) and the stored
     (legacy-budget) SubgraphX re-scored under the final protocol; heuristics hetero / deg_high / deg_low;
     Fid for the heuristics.                       -> runs/rq1_tier2/H/{gea,fid}/<unit>.npz
  G  Saliency, IG, Input x Gradient (runs/rq1_tier2/G/grad/<unit>.npz)  -> runs/rq1_tier2/G/{gea,fid}/<unit>.npz
  S  SubgraphX with the scoring budget (runs/rq1_tier2/S/sx/<unit>/)  -> runs/rq1_tier2/S/{gea,fid}/<unit>.npz

    python -m src.analysis.tier2_gea H|G|S --units ...
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import tier2_common as C
from .revision_pass import all_specs
from .tier2_fid import run_fid


def run(item, unit):
    T = C.T2; dkey = C.dataset_of(unit); sp = C.orig_spec(unit); fs = all_specs()[unit]
    ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]
    gt = C.gt_mol_idx(unit, ti, dkey)
    if item == "H":
        nl = C.nulls(unit, gt, dkey)
        pg, st = C.pg_final(unit)
        src = {"gnnexplainer": C.stored_scores(unit, "gnnexplainer"), "pgexplainer": pg, "subgraphx": C.stored_scores(unit, "subgraphx"),
               **{h: C.heuristic_dict(unit, ti, h) for h in ("hetero", "deg_high", "deg_low")}}
        A = {"mol_idx": np.array(gt), "rn": nl["rn"], "rc": nl["rc"], "pg_tie_boundary": st["tie_boundary"],
             "pg_resolved_by_logits": st["resolved_by_logits"], "pg_random_fallback": st["random_fallback"]}
        fid_methods = ("hetero", "deg_high", "deg_low")
    elif item == "G":
        g = f"{T}/G/grad/{unit}.npz"
        src = {m: C.npz_scores(g, m) for m in ("sal", "ig", "ixg")}
        A = {"mol_idx": np.array(gt)}; fid_methods = ("sal", "ig", "ixg")
    else:
        rows = C.sx_shard_rows(f"{T}/S/sx/{unit}")
        missing = [m for m in ti if m not in rows]
        if missing:
            raise RuntimeError(f"{unit}: {len(missing)} molecules without a fixed-budget SubgraphX result")
        src = {"sx_fixed": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in rows.items()}}
        old = C.stored_cache(unit)["results"]["subgraphx"]
        A = {"mol_idx": np.array(gt), "all_mol_idx": np.array(ti),
             "k": np.array([C.kbudget(rows[m]["n"]) for m in ti]), "new_size": np.array([len(rows[m]["coalition"]) for m in ti]),
             "new_max_nodes": np.array([rows[m]["max_nodes"] for m in ti]), "old_max_nodes": np.array([r["meta"]["max_nodes"] for r in old]),
             "old_size": np.array([len(r["meta"]["coalition"]) for r in old]), "seconds": np.array([rows[m]["seconds"] for m in ti])}
        fid_methods = ("sx_fixed",)
    for m, sc in src.items():
        A[m] = C.gea_array(unit, m, lambda mi, sc=sc: sc[mi], gt, dkey)
    C.save_npz(f"{T}/{item}/gea/{unit}.npz", unit=unit, **A)
    run_fid(unit, sp["ckpt"], fs["out"], dkey, {m: src[m] for m in fid_methods}, ti, f"{T}/{item}/fid/{unit}.npz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=["H", "G", "S"])
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    for u in a.units:
        if os.path.exists(f"{C.T2}/{a.item}/fid/{u}.npz"):
            continue
        t0 = time.time(); run(a.item, u)
        print(f"{u}: item {a.item} done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
