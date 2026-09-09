"""
Full paired-Wilcoxon / Holm-Bonferroni battery across the 16 scale-out cells
(12 Tox21 endpoints + 4 B-XAIC tasks) -- same method as F1-F8
(src/analysis/phase3_significance.py), applied at scale to test F9 / F10.

    python -m src.analysis.scaleout_significance

Blocks:
  A  B-XAIC GEA ranking, per task: GNN/PG/SX pairwise Wilcoxon + Holm  -> F9
  B  masking inflation R3->R2, R3->R1 (|Fid-|, GEF), per dataset x explainer
  C  explainer separability under R3/R2/R1 (Fid+/Fid-/GEF), per dataset  x
     masking, pairwise Wilcoxon + Holm; PGExplainer rows on collapsed
     endpoints excluded (see E)
  D  B-XAIC cross-metric ranking agreement (GEA order vs Fid+ order per masking)
  E  Tox21 cross-endpoint consistency (12 paired per-endpoint means) +
     PGExplainer collapse vs model AUROC  -> F10
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
FL = {"R3": "mean", "R2": "zero", "R1": "hard"}

# per-endpoint model test AUROC (from runs/scaleout/scaleout_logs -- "model: test_auroc=")
TOX21_AUROC = {
    "NR-AR": 0.803, "NR-AR-LBD": 0.844, "NR-AhR": 0.902, "NR-Aromatase": 0.805,
    "NR-ER": 0.736, "NR-ER-LBD": 0.725, "NR-PPAR-gamma": 0.752, "SR-ARE": 0.757,
    "SR-ATAD5": 0.879, "SR-HSE": 0.783, "SR-MMP": 0.862, "SR-p53": 0.813,
}
# the 4 endpoints whose run hit the old ">20% empty" assertion == PGExplainer collapse
COLLAPSED = {"NR-AR", "NR-Aromatase", "NR-ER-LBD", "SR-ARE"}


def load(p):
    with open(p) as f:
        return json.load(f)


def col(j, mk, fl, e, metric):
    return np.array([r[metric] for r in j["per_molecule"][mk][fl][e] if metric in r], float)


def common(j, mk, fl, metric):
    lists = {e: [(i, r[metric]) for i, r in enumerate(j["per_molecule"][mk][fl][e]) if metric in r]
             for e in EXPL}
    idx = sorted(set.intersection(*[{i for i, _ in lists[e]} for e in EXPL]))
    return idx, {e: np.array([dict(lists[e])[i] for i in idx]) for e in EXPL}


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for r, (k, p) in enumerate(items):
        run = min(max(run, (m - r) * p), 1.0)
        out[k] = run
    return out


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def W(a, b):
    if len(a) < 6 or np.allclose(a - b, 0):
        return 1.0
    return wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")[1]


def boot(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


BXAIC = [("indole", "runs/phase3_bxaic.json"),
         ("PAINS", "runs/scaleout/phase3_bxaic_PAINS.json"),
         ("X", "runs/scaleout/phase3_bxaic_X.json"),
         ("P", "runs/scaleout/phase3_bxaic_P.json")]
TOX21 = sorted(glob.glob("runs/scaleout/phase3_tox21_*.json"))


def block_A():
    print("#" * 78 + "\n# A. B-XAIC GEA ranking per task (Wilcoxon + Holm within task)  -> F9\n" + "#" * 78)
    for t, path in BXAIC:
        if not os.path.exists(path):
            print(f"[{t}] MISSING {path}"); continue
        j = load(path)
        idx, c = common(j, "R3", "mean", "gea")
        n = len(idx)
        print(f"\n[{t}]  n={n} GT-present molecules")
        for e in EXPL:
            lo, hi = boot(c[e]) if n else (float('nan'), float('nan'))
            print(f"  {SH[e]:4s} mean={c[e].mean():.3f} median={np.median(c[e]):.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
        raw = {f"{SH[a]}-{SH[b]}": W(c[a], c[b]) for a, b in PAIRS}
        adj = holm(raw)
        for a, b in PAIRS:
            k = f"{SH[a]}-{SH[b]}"
            hi = SH[a] if np.median(c[a] - c[b]) > 0 else SH[b]
            print(f"    {k:8s} medianD={np.median(c[a]-c[b]):+.3f}  p_raw={raw[k]:.4f}  p_Holm={adj[k]:.4f}  {star(adj[k]):>3}  ({hi} higher)")


def block_B():
    print("\n" + "#" * 78 + "\n# B. Masking inflation vs R3 (paired, |Fid-| and GEF)\n" + "#" * 78)
    print(f"  {'dataset':<20}{'expl':<5}  {'|Fid-| R3->R2 (p) ->R1 (p)':<34}  GEF R3->R2 (p) ->R1 (p)")
    for name, path in [(f"bxaic_{t}", p) for t, p in BXAIC] + [(os.path.basename(p)[7:-5], p) for p in TOX21]:
        j = load(path)
        for e in EXPL:
            fm = {m: np.abs(col(j, m, FL[m], e, "fid_minus")) for m in FL}
            gf = {m: col(j, m, FL[m], e, "gef") for m in FL}
            pf2, pf1 = W(fm["R3"], fm["R2"]), W(fm["R3"], fm["R1"])
            pg2, pg1 = W(gf["R3"], gf["R2"]), W(gf["R3"], gf["R1"])
            print(f"  {name:<20}{SH[e]:<5}  "
                  f"{fm['R3'].mean():.2f}->{fm['R2'].mean():.2f}({pf2:.3f})->{fm['R1'].mean():.2f}({pf1:.3f})".ljust(34)
                  + f"  {gf['R3'].mean():.2f}->{gf['R2'].mean():.2f}({pg2:.3f})->{gf['R1'].mean():.2f}({pg1:.3f})")


def block_C():
    print("\n" + "#" * 78 + "\n# C. Explainer separability under each masking (Wilcoxon + Holm)\n"
          "#    'PG*' = PGExplainer collapsed on this endpoint -> pair p's with PG are n/a\n" + "#" * 78)
    for name, path in [(f"bxaic_{t}", p) for t, p in BXAIC] + [(os.path.basename(p)[7:-5], p) for p in TOX21]:
        j = load(path)
        ep = name.replace("tox21_", "")
        pg_bad = ep in COLLAPSED
        print(f"\n[{name}]" + ("   (PGExplainer COLLAPSED -- PG pairs excluded)" if pg_bad else ""))
        for mk in ("R3", "R2", "R1"):
            for metric in ("fid_plus", "fid_minus", "gef"):
                idx, c = common(j, mk, FL[mk], metric)
                raw = {f"{SH[a]}-{SH[b]}": (float("nan") if pg_bad and "pgexplainer" in (a, b) else W(c[a], c[b]))
                       for a, b in PAIRS}
                valid = {k: v for k, v in raw.items() if v == v}
                adj = holm(valid)
                means = " ".join(f"{SH[e]}={c[e].mean():+.3f}" for e in EXPL)
                sig = " ".join(f"{k}:{'n/a' if k not in adj else star(adj[k])}" for k in raw)
                print(f"  {mk} {metric:<10} {means:<42}  {sig}")


def block_D():
    print("\n" + "#" * 78 + "\n# D. B-XAIC cross-metric ranking: GEA order vs Fid+ order per masking\n" + "#" * 78)
    for t, path in BXAIC:
        j = load(path)
        rg = {e: col(j, "R3", "mean", e, "gea").mean() for e in EXPL}
        og = sorted(EXPL, key=lambda e: -rg[e])
        print(f"\n[{t}]  GEA: {' > '.join(SH[e] for e in og)}  ({', '.join(f'{SH[e]}={rg[e]:.2f}' for e in og)})")
        for mk in ("R3", "R2", "R1"):
            rf = {e: col(j, mk, FL[mk], e, "fid_plus").mean() for e in EXPL}
            of = sorted(EXPL, key=lambda e: -rf[e])
            print(f"      {mk}-Fid+: {' > '.join(SH[e] for e in of)}"
                  f"   [{'agrees' if of == og else 'DIFFERS'} with GEA]")


def block_E():
    print("\n" + "#" * 78 + "\n# E. Tox21 cross-endpoint consistency (12 paired per-endpoint means)  -> F10\n" + "#" * 78)
    per = {mk: {e: {m: [] for m in ("fid_plus", "fid_minus", "gef")} for e in EXPL} for mk in FL}
    eps = []
    for path in TOX21:
        ep = os.path.basename(path)[13:-5]
        eps.append(ep)
        j = load(path)
        for mk in FL:
            for e in EXPL:
                for m in ("fid_plus", "fid_minus", "gef"):
                    v = col(j, mk, FL[mk], e, m)
                    per[mk][e][m].append(float(np.nanmean(v)) if len(v) else np.nan)
    print("  Across the 12 endpoints, paired Wilcoxon on per-endpoint means (n=12):")
    for mk in ("R3", "R2", "R1"):
        for m in ("fid_plus", "fid_minus", "gef"):
            raw = {}
            for a, b in PAIRS:
                x = np.array(per[mk][a][m]); y = np.array(per[mk][b][m])
                ok = ~np.isnan(x) & ~np.isnan(y)
                raw[f"{SH[a]}-{SH[b]}"] = W(x[ok], y[ok])
            adj = holm(raw)
            means = " ".join(f"{SH[e]}={np.nanmean(per[mk][e][m]):+.3f}" for e in EXPL)
            print(f"  {mk} {m:<10} {means:<42}  " + " ".join(f"{k}:{star(adj[k])}" for k in raw))
    print("\n  R3 noise-domination: #endpoints where sd(per-molecule) > mean, per explainer/metric:")
    nd = {e: {m: 0 for m in ("fid_plus", "fid_minus", "gef")} for e in EXPL}
    for path in TOX21:
        j = load(path)
        for e in EXPL:
            for m in ("fid_plus", "fid_minus", "gef"):
                v = col(j, "R3", "mean", e, m)
                if len(v) and np.nanstd(v) > abs(np.nanmean(v)):
                    nd[e][m] += 1
    for e in EXPL:
        print(f"    {SH[e]}: " + "  ".join(f"{m}={nd[e][m]}/12" for m in nd[e]))
    print("\n  PGExplainer collapse vs model AUROC (F10):")
    order = sorted(eps, key=lambda e: TOX21_AUROC[e])
    for e in order:
        print(f"    AUROC {TOX21_AUROC[e]:.3f}  {e:<16} {'COLLAPSED' if e in COLLAPSED else 'ok'}")
    ca = [TOX21_AUROC[e] for e in COLLAPSED]
    oa = [TOX21_AUROC[e] for e in eps if e not in COLLAPSED]
    print(f"    collapsed AUROC range [{min(ca):.3f}, {max(ca):.3f}]  n=4")
    print(f"    ok        AUROC range [{min(oa):.3f}, {max(oa):.3f}]  n=8")
    print(f"    -> ranges OVERLAP; 2 ok endpoints (NR-ER 0.736, NR-PPAR-gamma 0.752) have weaker")
    print(f"       models than 2 collapsed ones (NR-AR 0.803, NR-Aromatase 0.805)")


def main() -> int:
    for b in (block_A, block_B, block_C, block_D, block_E):
        b()
    print("\nWilcoxon two-sided, paired; Holm-Bonferroni over the 3 pairwise tests per block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
