"""
F11 seed-aware re-test (URGENT per user request): F11 claims R3->R2 and
R3->R1 |Fid-|/GEF inflation is significant on "~8/12 endpoints" -- computed
from the ORIGINAL 1-seed scale-out data (one paired Wilcoxon per endpoint,
n=30 molecules, that endpoint's single model). The question: does "~8/12"
reproduce independently in each of the 5 priority-sweep seeds, or was it
specific to seed 0's particular models/subsamples (the same kind of
single-replicate fragility that turned out to inflate significance for SX's
Fid-/GEF once checked per-seed)?

Unlike the SX Fid-/GEF case, F11's ORIGINAL test was already NOT pooled
across a hidden replicate dimension -- "8/12 endpoints significant" is a
COUNT of 12 independent single-seed per-molecule tests (n=30 each), not one
combined p-value. So there is no pseudo-replication bug in the ORIGINAL
1-seed claim itself. What's actually being checked here is different and
more basic: is the specific COUNT ("~8/12") a stable property of the
R2/R1-vs-R3 inflation effect, or did it depend on which single seed
happened to be used? Re-run the exact same per-endpoint, per-molecule
(n=30) paired Wilcoxon test independently for each of the 5 seeds and
compare the resulting endpoint-counts.

    python -m src.analysis.tox21_f11_seed_check
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def main() -> int:
    # sig_count[seed][explainer][metric][transition] = count of 12 endpoints significant
    results = {s: {e: {"fid_minus": {"R2": 0, "R1": 0}, "gef": {"R2": 0, "R1": 0}} for e in EXPL} for s in SEEDS}
    detail = {s: {e: {"fid_minus": {"R2": [], "R1": []}, "gef": {"R2": [], "R1": []}} for e in EXPL} for s in SEEDS}

    for s in SEEDS:
        for ep in EPS:
            path = os.path.join(OUT, f"out_tox21_{ep}_s{s}.json")
            j = json.load(open(path))
            pm = j["per_molecule"]
            for e in EXPL:
                r3 = pm["R3"]["mean"].get(e)
                r2 = pm["R2"]["zero"].get(e)
                r1 = pm["R1"]["hard"].get(e)
                if not (r3 and r2 and r1):
                    continue
                for metric in ("fid_minus", "gef"):
                    a3 = np.array([abs(row[metric]) if metric == "fid_minus" else row[metric] for row in r3])
                    a2 = np.array([abs(row[metric]) if metric == "fid_minus" else row[metric] for row in r2])
                    a1 = np.array([abs(row[metric]) if metric == "fid_minus" else row[metric] for row in r1])
                    p_r2 = _wpair(a2, a3)
                    p_r1 = _wpair(a1, a3)
                    sig_r2 = p_r2 < 0.05 and a2.mean() > a3.mean()
                    sig_r1 = p_r1 < 0.05 and a1.mean() > a3.mean()
                    results[s][e][metric]["R2"] += int(sig_r2)
                    results[s][e][metric]["R1"] += int(sig_r1)
                    detail[s][e][metric]["R2"].append((ep, p_r2, sig_r2))
                    detail[s][e][metric]["R1"].append((ep, p_r1, sig_r1))

    print("=" * 100)
    print(" F11 seed-aware re-test: count of 12 endpoints with significant R3->R2 / R3->R1")
    print(" inflation (|Fid-| and GEF), per explainer, INDEPENDENTLY per seed")
    print(" (original F11 claim, single seed 0 of the ORIGINAL scale-out: 'all three explainers ~8/12')")
    print("=" * 100)
    print(f"\n{'seed':<6}{'expl':<6}{'|Fid-| R3->R2':>16}{'|Fid-| R3->R1':>16}{'GEF R3->R2':>14}{'GEF R3->R1':>14}")
    for s in SEEDS:
        for e in EXPL:
            r = results[s][e]
            print(f"{s:<6}{SH[e]:<6}{r['fid_minus']['R2']:>13}/12{r['fid_minus']['R1']:>13}/12"
                  f"{r['gef']['R2']:>11}/12{r['gef']['R1']:>11}/12")

    print("\n" + "-" * 100)
    print(" summary: range and mean across the 5 seeds, per explainer x metric x transition")
    print("-" * 100)
    for e in EXPL:
        for metric in ("fid_minus", "gef"):
            for trans in ("R2", "R1"):
                vals = [results[s][e][metric][trans] for s in SEEDS]
                print(f"  {SH[e]:<4} {metric:<10} R3->{trans}:  {vals}  mean={np.mean(vals):.1f}/12"
                      f"  range=[{min(vals)},{max(vals)}]/12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
