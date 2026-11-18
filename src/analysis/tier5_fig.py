"""
RQ1 Tier 5 follow-up F3 (runs/rq1_tier5/PLAN_followups.md): fig1_v3 = fig1_v2 (src/analysis/tier3_fig.py, same data and layout) plus
EiG and GNNShap from N1 (both at L0: the same molecules as fig1_v2; checked per seed below).
    python -m src.analysis.tier5_fig      -> runs/rq1_tier5/fig1_v3.pdf / .png / fig1_v3_numbers.json
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .revision_analysis import hboot
from .tier3_fig import LABEL as LABEL2, PANELS, RULES, STYLE as STYLE2

OUT = "runs/rq1_tier5"
METHODS = ["gnnexplainer", "pgexplainer", "sx_fixed", "ig", "eig", "gnnshap"]
LABEL = {**LABEL2, "eig": "EiG-\nSearch", "gnnshap": "GNNShap"}
STYLE = {**STYLE2, "eig": {"fill": "#ffed6f", "hatch": "--", "edge": "#7a6a00"}, "gnnshap": {"fill": "#8dd3c7", "hatch": "++", "edge": "#01665e"}}


def load():
    z = np.load("runs/rq1_tier3/X/fig1_v2_data.npz"); D = {}
    for k in z.files:
        p, m, u = k.split("|"); D.setdefault(p, {}).setdefault(m, {})[u] = z[k]
    for p in PANELS:
        for u in D[p]["rn"]:
            g = np.load(f"{OUT}/N1/gea/{u}.npz")
            # same molecules: N1's union is the unit's GT list; its random-node and IG arrays equal fig1_v2's (as multisets)
            assert len(g["union"]) == len(D[p]["rn"][u]) and np.allclose(np.sort(g["rn"]), np.sort(D[p]["rn"][u])), u
            assert np.allclose(np.sort(g["ig__gea"]), np.sort(D[p]["ig"][u])), u
            for w in ("eig", "gnnshap"):
                assert [int(m) for m in g[f"{w}__mols"]] == [int(m) for m in g["union"]], (u, w)
                D[p].setdefault(w, {})[u] = g[f"{w}__gea"]
            D[p].setdefault("rn_n1", {})[u] = g["rn"]                      # N1's random-node null in N1's (union) order
    return D


def main():
    D = load()
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9), sharey=True); nums = {}; hb = None; xs = np.arange(1, len(METHODS) + 1)
    for ax, panel in zip(axes, PANELS):
        P = D[panel]; us = sorted(P["rn"], key=lambda u: int(u.rsplit("_s", 1)[1]))
        seed_means = lambda m: [float(P[m][u].mean()) for u in us]
        rn = seed_means("rn"); rule_means = {r: float(np.mean(seed_means(r))) for r in RULES}; best = max(rule_means, key=rule_means.get); rb = seed_means(best)

        def band(a, **kw):
            lo, hi = min(a), max(a); pad = max(0, 0.006 - (hi - lo) / 2)
            return ax.axhspan(lo - pad, hi + pad, **kw)
        band(rn, color="#9e9ac8", alpha=0.45, zorder=0)
        band(rb, facecolor="0.8", edgecolor="0.35", hatch="\\\\\\\\", linewidth=0, alpha=0.7, zorder=0.5)
        bp = ax.boxplot([np.concatenate([P[m][u] for u in us]) for m in METHODS], positions=xs, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "#222222", "linewidth": 1.4}, whiskerprops={"color": "0.2", "linewidth": 0.8}, capprops={"color": "0.2", "linewidth": 0.8}, boxprops={"linewidth": 0.8})
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(STYLE[m]["fill"]); patch.set_edgecolor(STYLE[m]["edge"]); patch.set_hatch(STYLE[m]["hatch"])
        for x0, m in zip(xs, METHODS):
            hb = ax.scatter(x0 + np.linspace(-0.18, 0.18, len(us)), seed_means(m), marker="D", s=18, facecolor="white", edgecolor="black", linewidths=0.8, zorder=4)
        ax.axvline(4.5, color="0.45", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=7.0)
        ax.set_title(f"{panel} ({len(us)} seeds)\nbest trivial rule: {RULES[best]}", fontsize=9.5)
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", zorder=-1); ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        def row(m):
            a = [P[m][u] for u in us]
            ref = P["rn"] if m not in ("eig", "gnnshap") else P["rn_n1"]      # N1 arrays are in N1's molecule order
            d = hboot(a, [ref[u] for u in us])
            return {"mean": float(np.mean(seed_means(m))), "minus_rn": list(map(float, d[:3])), "median": float(np.median(np.concatenate(a))), "per_seed": seed_means(m)}
        nums[panel] = {"n_seeds": len(us), "n_molecules": int(sum(len(P["rn"][u]) for u in us)), "rn_band": [min(rn), max(rn)], "rn_per_seed": rn,
                       "best_rule": best, "rule_band": [min(rb), max(rb)], "rule_per_seed": rb, "rule_means": rule_means, "methods": {m: row(m) for m in METHODS}}
    axes[0].set_ylabel("GEA at top-k = 0.25")
    fig.legend([Patch(color="#9e9ac8", alpha=0.45), Patch(facecolor="0.8", edgecolor="0.35", hatch="\\\\\\\\", alpha=0.7), hb],
               ["random-node null, 20 draws (range of seed means)", "best trivial rule in the panel (range of seed means)", "per-seed mean"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.095, "left of the dotted line: methods of fig1_v2; right: new explainers (Tier 5, N1)", ha="center", fontsize=7.2, color="0.3")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_v3.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    json.dump(nums, open(f"{OUT}/fig1_v3_numbers.json", "w"), indent=1)
    print(f"wrote {OUT}/fig1_v3", flush=True)


if __name__ == "__main__":
    main()
