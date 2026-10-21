"""
RQ1 final figures under the random-tie rule (runs/rq1_final_export/PLAN.md).
Same design and colour style as figures/figures.py (the colour version of
src/analysis/make_figures.py); writes PDF + PNG to figures/final/ only.

  fig1_headline     per-molecule top-k GEA, seeds 0-4; PG and SX values are the mean over 20 tie orders
  fig2_sx_win_rate  SubgraphX per-molecule win rate vs the connectivity-matched null (tie-free); MUTAG-GT seeds 0-9, indole 0-4
  fig3_collapse     PGExplainer collapse rate; MUTAG-GT over 10 seeds

    python -m src.analysis.final_figures
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_spec = importlib.util.spec_from_file_location("figstyle", "figures/figures.py")
fs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fs)      # palette, rcParams, _box, _jitter_strip

OUT = "figures/final"
AUD = "runs/mutagenicity/exploratory/null_audit_arrays"
P2B = "runs/rq1_followups/phase2b/gea"
EXPL = fs.EXPL
SEEDS = [0, 1, 2, 3, 4]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)


def fig1():
    panels = [("mutag_graphxai", "MUTAG-GT"), ("bxaic_indole", "B-XAIC indole"), ("mutagenicity_graphxai", "Mutagenicity")]
    collapse = pd.read_csv("runs/prio/collapse_table.csv")
    mut_health = json.load(open("runs/mutagenicity/report.json"))["health"]
    n_coll = {"mutag_graphxai": int(collapse[(collapse.dataset == "mutag_graphxai") & (collapse.protocol == "coupled")]["collapsed"].sum()),
              "bxaic_indole": int(collapse[(collapse.dataset == "bxaic") & (collapse.task_or_endpoint == "indole")]["collapsed"].sum()),
              "mutagenicity_graphxai": int(sum(bool(mut_health[str(s)]["pgexplainer"]["collapsed"]) for s in SEEDS))}

    def conn(key):
        if key == "mutag_graphxai":
            z = np.load("runs/prio/connectivity_arrays_mutag_graphxai.npz"); return [float(z[f"rand_conn_s{s}"].mean()) for s in SEEDS]
        if key == "bxaic_indole":
            z = np.load("runs/prio/connectivity_arrays_indole.npz"); return [float(z[f"rand_conn_s{s}"].mean()) for s in SEEDS]
        return [float(np.load(f"runs/mutagenicity/arrays_s{s}.npz")["rand_conn_mean_subgraphx_set"].mean()) for s in SEEDS]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True)
    shown = {}
    for ax, (key, title) in zip(axes, panels):
        if key == "mutagenicity_graphxai":   # same arrays as table (a): the Mutagenicity report arrays + their random-tie re-scoring
            st = {s: dict(np.load(f"runs/mutagenicity/arrays_s{s}.npz")) for s in SEEDS}; rt = {s: np.load(f"{P2B}/arch_D-MPNN_s{s}.npz") for s in SEEDS}
            for s in SEEDS:
                st[s]["gnnexplainer_topk"], st[s]["rn_k"] = st[s]["gnnexplainer_topk_set"], st[s]["rand_node_topk_set"]
        else:
            st = {s: np.load(f"{AUD}/{key}_s{s}.npz") for s in SEEDS}; rt = {s: np.load(f"{P2B}/audit_{key}_s{s}.npz") for s in SEEDS}
        val = {s: {"gnnexplainer": st[s]["gnnexplainer_topk"], "pgexplainer": rt[s]["pgexplainer_b"], "subgraphx": rt[s]["subgraphx_b"]} for s in SEEDS}
        pooled = {e: np.concatenate([val[s][e] for s in SEEDS]) for e in EXPL}
        seed_mean = {e: [float(val[s][e].mean()) for s in SEEDS] for e in EXPL}
        rn = [float(st[s]["rn_k"].mean()) for s in SEEDS]; rc = conn(key)
        b1 = ax.axhspan(min(rn), max(rn), color="#9e9ac8", alpha=0.45, zorder=0)
        b2 = ax.axhspan(min(rc), max(rc), facecolor="none", edgecolor="#cb181d", hatch="////", linewidth=0, zorder=0.6)
        pos = [1, 2, 3]; fs._box(ax, pos, [pooled[e] for e in EXPL])
        for x0, e in zip(pos, EXPL):
            fs._jitter_strip(ax, x0, pooled[e], rng, color=fs.ESTYLE[e]["dot"])
            dm = ax.scatter(x0 + np.linspace(-0.16, 0.16, len(SEEDS)), seed_mean[e], marker="D", s=22, facecolor="white", edgecolor="black", linewidths=0.9, zorder=4)
        ax.set_xticks(pos); ax.set_xticklabels(["GNNExplainer", "PGExplainer", "SubgraphX"], fontsize=7.6)
        ax.set_title(title); ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", zorder=-1); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if n_coll[key] > 0:
            ax.annotate(f"{n_coll[key]} of 5 seeds\ncollapsed (included)", xy=(2, np.percentile(pooled["pgexplainer"], 75)), xytext=(2, 0.94),
                        ha="center", fontsize=6.6, color="0.15", arrowprops=dict(arrowstyle="-", color="0.3", linewidth=0.6))
        shown[title] = {"n_molecules_pooled": int(len(pooled["gnnexplainer"])), "per_seed_mean": seed_mean,
                        "median": {e: float(np.median(pooled[e])) for e in EXPL}, "random_node_band": [min(rn), max(rn)],
                        "random_connected_band": [min(rc), max(rc)], "pg_collapsed_seeds": n_coll[key]}
    axes[0].set_ylabel("GEA at top-k = 0.25 (Jaccard vs ground truth)")
    fig.legend([b1, b2, dm], ["random-node null at the top-k budget, 20 draws (range of seed means)",
                              "connectivity-matched null, sized to SubgraphX's mask (range of seed means)", "per-seed mean"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); save(fig, "fig1_headline")
    return shown


def fig2():
    m = np.load("runs/rq1_followups/phase1/connectivity_arrays_mutag_graphxai.npz"); i = np.load("runs/prio/connectivity_arrays_indole.npz")
    seeds = list(range(10))
    wr = {"MUTAG-GT": [float(m[f"win_vs_conn_s{s}"].mean()) for s in seeds],
          "B-XAIC indole": [float(i[f"win_vs_conn_s{s}"].mean()) if s < 5 else np.nan for s in seeds]}
    fig, ax = plt.subplots(figsize=(6.4, 3.1)); x = np.arange(len(seeds)); wdt = 0.36
    for j, (lab, st) in enumerate((("MUTAG-GT", fs.DSTYLE["mutag_graphxai"]), ("B-XAIC indole", fs.DSTYLE["bxaic"]))):
        ax.bar(x + (j - 0.5) * wdt, wr[lab], width=wdt, color=st["fill"], edgecolor=st["edge"], hatch=st["hatch"], linewidth=0.8, label=lab)
    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", zorder=2, label="chance (50%)")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
    ax.set_ylabel("SX win rate vs connectivity-matched null\n(fraction of molecules)"); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)); ax.grid(axis="y", zorder=-1); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    ax.set_title("SubgraphX vs connectivity-matched random null, per seed (B-XAIC indole: seeds 0-4 only)", fontsize=9, pad=32)
    fig.tight_layout(); save(fig, "fig2_sx_win_rate")
    return wr


def fig3():
    df = pd.read_csv("runs/prio/collapse_table.csv"); df = df[df.protocol == "coupled"]
    g = df.groupby("dataset")["collapsed"].agg(["sum", "count"])
    p1 = json.load(open("runs/rq1_followups/phase1/report.json"))["pg_collapsed"]
    counts = {"mutag_graphxai": (int(p1["all"]), 10), "bxaic": (int(g.loc["bxaic", "sum"]), int(g.loc["bxaic", "count"])),
              "tox21": (int(g.loc["tox21", "sum"]), int(g.loc["tox21", "count"]))}
    order = ["mutag_graphxai", "bxaic", "tox21"]
    label = {"mutag_graphxai": "MUTAG-GT\n(10 seeds)", "bxaic": "B-XAIC\n(4 tasks x 5 seeds)", "tox21": "Tox21\n(12 endpoints x 5 seeds)"}
    rate = [100 * counts[d][0] / counts[d][1] for d in order]
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); x = np.arange(3)
    bars = ax.bar(x, rate, color=[fs.DSTYLE[d]["fill"] for d in order], edgecolor=[fs.DSTYLE[d]["edge"] for d in order],
                  hatch=[fs.DSTYLE[d]["hatch"] for d in order], linewidth=0.8, width=0.6)
    for xi, d, b in zip(x, order, bars):
        ax.text(xi, b.get_height() + 1.5, f"{counts[d][0]}/{counts[d][1]}", ha="center", fontsize=8.5, color="0.1")
    ax.set_xticks(x); ax.set_xticklabels([label[d] for d in order], fontsize=7.8)
    ax.set_ylabel("PGExplainer collapse rate\n(n_degenerate/n >= 0.5)"); ax.set_ylim(0, max(rate) + 12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter()); ax.grid(axis="y", zorder=-1); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("PGExplainer collapse rate by dataset\n(coupled protocol)", fontsize=9.5)
    fig.tight_layout(); save(fig, "fig3_collapse_rate")
    return {d: {"collapsed": counts[d][0], "units": counts[d][1], "rate_pct": r} for d, r in zip(order, rate)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"fig1": fig1(), "fig2": fig2(), "fig3": fig3()}
    json.dump(rep, open(f"{OUT}/figure_numbers.json", "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
