"""
F10 multi-seed test -- is PGExplainer's "collapse" a real property of specific
Tox21 endpoints, or generic seed-noise on weak models?

For 6 endpoints (4 that collapsed at seed 0 in the scale-out + 2 non-collapsed
controls with similar/weaker AUROC) x 5 model seeds, retrain the D-MPNN (seed
drives split AND init) and rerun PGExplainer under the same n=30 R3 protocol.
Reads runs/f10ms/pg_<EP>_s<SEED>.json (+ the matching ckpt for model AUROC).

    python -m src.analysis.f10_multiseed

"collapsed" here = PGExplainer produced a degenerate (node_importance std = 0)
explanation on >= 50% of the 30 molecules -- the same signal explainer_health
records and the scale-out used to flag F10.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

COLLAPSED_AT_S0 = ["NR-AR", "NR-Aromatase", "NR-ER-LBD", "SR-ARE"]
CONTROLS = ["NR-ER", "NR-PPAR-gamma"]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/f10ms"
COLLAPSE_FRAC = 0.50


def _auroc(ep: str, s: int):
    ck = os.path.join(OUT, f"ckpt_{ep}_s{s}.pt")
    if not os.path.exists(ck):
        return None
    try:
        m = torch.load(ck, map_location="cpu", weights_only=False)["metrics"]
        return float(m["test_auroc"])
    except Exception:
        return None


def _row(ep: str, s: int):
    p = os.path.join(OUT, f"pg_{ep}_s{s}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    h = j["explainer_health"]["pgexplainer"]
    n = max(h.get("n", 30), 1)
    deg = h.get("n_degenerate", 0)
    emp = h.get("n_empty", 0)
    pg = j["summary"]["R3"]["mean"]["pgexplainer"]
    return dict(
        seed=s, auroc=_auroc(ep, s), n=n, n_degen=deg, n_empty=emp,
        frac_degen=deg / n, collapsed=deg / n >= COLLAPSE_FRAC,
        fid_plus=pg["fid_plus"][0], fid_minus=pg["fid_minus"][0], gef=pg["gef"][0],
    )


def main() -> int:
    if not glob.glob(os.path.join(OUT, "pg_*.json")):
        print(f"no {OUT}/pg_*.json yet -- run scripts/f10_multiseed.sh first")
        return 1

    groups = [("collapsed-at-seed-0", COLLAPSED_AT_S0), ("control (weak model, did NOT collapse)", CONTROLS)]
    all_rows: dict[str, list] = {}

    for label, eps in groups:
        print("=" * 92)
        print(f" {label}")
        print("=" * 92)
        for ep in eps:
            rows = [r for s in SEEDS if (r := _row(ep, s)) is not None]
            all_rows[ep] = rows
            if not rows:
                print(f"  {ep:<16} (no results yet)")
                continue
            print(f"  {ep}")
            print(f"    {'seed':>4} {'AUROC':>7} {'deg/n':>8} {'empty/n':>8} {'collapsed':>10}"
                  f" {'Fid+':>8} {'Fid-':>8} {'GEF':>8}")
            for r in rows:
                au = f"{r['auroc']:.3f}" if r["auroc"] is not None else "  ?  "
                print(f"    {r['seed']:>4} {au:>7} {r['n_degen']:>3}/{r['n']:<4}"
                      f" {r['n_empty']:>3}/{r['n']:<4} {str(r['collapsed']):>10}"
                      f" {r['fid_plus']:>8.3f} {r['fid_minus']:>8.3f} {r['gef']:>8.3f}")
            nc = sum(r["collapsed"] for r in rows)
            au_c = [r["auroc"] for r in rows if r["collapsed"] and r["auroc"] is not None]
            au_n = [r["auroc"] for r in rows if not r["collapsed"] and r["auroc"] is not None]
            msg = f"    -> collapsed in {nc}/{len(rows)} seeds"
            if au_c:
                msg += f"  | collapsed AUROC {min(au_c):.3f}-{max(au_c):.3f}"
            if au_n:
                msg += f"  | non-collapsed AUROC {min(au_n):.3f}-{max(au_n):.3f}"
            print(msg)
            print()

    # ---- verdict ---------------------------------------------------------
    print("=" * 92)
    print(" VERDICT")
    print("=" * 92)
    done = {ep: rows for ep, rows in all_rows.items() if rows}
    if len(done) < 6 or any(len(r) < len(SEEDS) for r in done.values()):
        have = sum(len(r) for r in done.values())
        print(f"  PARTIAL: {have}/{6 * len(SEEDS)} (endpoint,seed) cells present -- rerun when complete.")
    for label, eps in groups:
        print(f"\n  {label}:")
        for ep in eps:
            rows = all_rows.get(ep, [])
            if not rows:
                print(f"    {ep:<16} pending")
                continue
            nc, n = sum(r["collapsed"] for r in rows), len(rows)
            if nc == n:
                verdict = "collapses EVERY seed -> real endpoint property"
            elif nc == 0:
                verdict = "NEVER collapses -> stable"
            else:
                verdict = f"collapses {nc}/{n} seeds -> SEED-DEPENDENT (not a fixed property)"
            print(f"    {ep:<16} {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
