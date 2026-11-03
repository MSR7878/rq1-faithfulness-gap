"""
RQ1 Tier 3, item X: new Fig 1 (figures/final/fig1_v2.pdf/.png, fig1_v2_numbers.json) from runs/rq1_tier3/X/fig1_v2_data.npz.
Same palette and hatches as figures/figures.py; IG gets its own fill and hatch. Boxes pool molecules across seeds for display only.

    python -m src.analysis.tier3_fig
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .revision_analysis import hboot

_spec = importlib.util.spec_from_file_location("figstyle", "figures/figures.py")
fs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fs)

OUT = "figures/final"
PANELS = ["MUTAG-GT", "Mutagenicity D-MPNN", "B-XAIC indole", "B-XAIC PAINS"]
METHODS = ["gnnexplainer", "pgexplainer", "sx_fixed", "ig"]
LABEL = {"gnnexplainer": "GNN-\nExplainer", "pgexplainer": "PG-\nExplainer", "sx_fixed": "SubgraphX\n(fixed)", "ig": "IG"}
STYLE = {"gnnexplainer": fs.ESTYLE["gnnexplainer"], "pgexplainer": fs.ESTYLE["pgexplainer"], "sx_fixed": fs.ESTYLE["subgraphx"],
         "ig": {"fill": "#fb9a99", "hatch": "..", "edge": "#b2182b", "dot": "#e31a1c"}}
RULES = {"hetero": "heteroatom", "deg_high": "high degree", "deg_low": "low degree"}


def main():
    z = np.load("runs/rq1_tier3/X/fig1_v2_data.npz"); D = {}
    for k in z.files:
        p, m, u = k.split("|"); D.setdefault(p, {}).setdefault(m, {})[u] = z[k]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.6), sharey=True); nums = {}; hb = None
    for ax, panel in zip(axes, PANELS):
        P = D[panel]; us = sorted(P["rn"], key=lambda u: int(u.rsplit("_s", 1)[1]))
        seed_means = lambda m: [float(P[m][u].mean()) for u in us]
        rn = seed_means("rn"); rule_means = {r: float(np.mean(seed_means(r))) for r in RULES}; best = max(rule_means, key=rule_means.get); rb = seed_means(best)
        # a range of near-identical seed means would draw as nothing: keep a minimum visible height
        def band(a, **kw):
            lo, hi = min(a), max(a); pad = max(0, 0.006 - (hi - lo) / 2)
            return ax.axhspan(lo - pad, hi + pad, **kw)
        band(rn, color="#9e9ac8", alpha=0.45, zorder=0)
        band(rb, facecolor="0.8", edgecolor="0.35", hatch="\\\\\\\\", linewidth=0, alpha=0.7, zorder=0.5)
        bp = ax.boxplot([np.concatenate([P[m][u] for u in us]) for m in METHODS], positions=[1, 2, 3, 4], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "#222222", "linewidth": 1.4}, whiskerprops={"color": "0.2", "linewidth": 0.8}, capprops={"color": "0.2", "linewidth": 0.8}, boxprops={"linewidth": 0.8})
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(STYLE[m]["fill"]); patch.set_edgecolor(STYLE[m]["edge"]); patch.set_hatch(STYLE[m]["hatch"])
        for x0, m in zip([1, 2, 3, 4], METHODS):
            hb = ax.scatter(x0 + np.linspace(-0.18, 0.18, len(us)), seed_means(m), marker="D", s=20, facecolor="white", edgecolor="black", linewidths=0.8, zorder=4)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=7.2)
        ax.set_title(f"{panel} ({len(us)} seeds)\nbest trivial rule: {RULES[best]}", fontsize=9.5)
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", zorder=-1); ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        def row(m):
            a = [P[m][u] for u in us]; d = hboot(a, [P["rn"][u] for u in us])
            return {"mean": float(np.mean(seed_means(m))), "minus_rn": list(d[:3]), "median": float(np.median(np.concatenate(a))), "per_seed": seed_means(m)}
        nums[panel] = {"n_seeds": len(us), "n_molecules": int(sum(len(P["rn"][u]) for u in us)), "rn_band": [min(rn), max(rn)], "rn_per_seed": rn, "best_rule": best, "rule_band": [min(rb), max(rb)], "rule_per_seed": rb,
                       "rule_means": rule_means, "methods": {m: row(m) for m in METHODS}, "pg_final_rule": row("pg_final_rule")}
    axes[0].set_ylabel("GEA at top-k = 0.25")
    fig.legend([Patch(color="#9e9ac8", alpha=0.45), Patch(facecolor="0.8", edgecolor="0.35", hatch="\\\\\\\\", alpha=0.7), hb],
               ["random-node null, 20 draws (range of seed means)", "best trivial rule in the panel (range of seed means)", "per-seed mean"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_v2.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    json.dump(nums, open(f"{OUT}/fig1_v2_numbers.json", "w"), indent=1)
    print("wrote figures/final/fig1_v2", flush=True)


if __name__ == "__main__":
    main()
