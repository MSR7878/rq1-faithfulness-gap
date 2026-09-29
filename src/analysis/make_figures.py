"""
Draw the 3 requested figures directly from already-saved data -- no new
computation, no stats, just loading npz arrays / collapse_table.csv and
plotting. Reads from runs/prio/*.npz (pulled from Ada) and
runs/prio/collapse_table.csv. Writes vector PDFs to figures/.

  fig1_headline.pdf       mutag_graphxai + B-XAIC/indole per-molecule GEA
                          distributions (GNN/PG/SX) vs the random-null band
  fig2_sx_win_rate.pdf    SX's per-molecule win rate vs the connectivity-
                          matched null, per seed, both datasets
  fig3_collapse_rate.pdf  PGExplainer collapse rate per dataset, with the
                          per-unit counts behind each rate

Greyscale-safe by construction: explainer/dataset identity is carried by
hatch pattern + direct labels, not color alone; color is a secondary,
low-saturation grey-scale cue on top.

    python -m src.analysis.make_figures
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = "runs/prio"
FIGDIR = "figures"
SEEDS = [0, 1, 2, 3, 4]

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
ESH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
# fixed categorical assignment, used identically in every figure that shows
# explainer identity -- grey level + hatch, never color alone
ESTYLE = {
    "gnnexplainer": {"grey": "0.85", "hatch": "",   "edge": "0.15"},
    "pgexplainer":  {"grey": "0.60", "hatch": "//",  "edge": "0.15"},
    "subgraphx":    {"grey": "0.35", "hatch": "xx",  "edge": "0.05"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,  # embed as real text, not paths
})


def _box(ax, positions, data, width=0.55):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                     showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
                     whiskerprops={"color": "0.2", "linewidth": 0.8},
                     capprops={"color": "0.2", "linewidth": 0.8},
                     boxprops={"linewidth": 0.8})
    for patch, e in zip(bp["boxes"], EXPL):
        st = ESTYLE[e]
        patch.set_facecolor(st["grey"])
        patch.set_edgecolor(st["edge"])
        patch.set_hatch(st["hatch"])
    return bp


def _jitter_strip(ax, x0, vals, rng, width=0.32, max_points=250):
    vals = np.asarray(vals)
    if len(vals) > max_points:
        vals = rng.choice(vals, size=max_points, replace=False)
    jitter = rng.uniform(-width / 2, width / 2, size=len(vals))
    ax.scatter(x0 + jitter, vals, s=3.5, color="black", alpha=0.12,
               linewidths=0, zorder=3)


def fig1_headline():
    gea = {
        "mutag_graphxai": np.load(f"{OUT}/figure_gea_mutag_graphxai.npz"),
        "indole": np.load(f"{OUT}/figure_gea_indole.npz"),
    }
    baseline = {
        "mutag_graphxai": np.load(f"{OUT}/baseline_arrays_mutag_graphxai.npz"),
        "indole": np.load(f"{OUT}/baseline_arrays_bxaic_indole.npz"),
    }
    titles = {"mutag_graphxai": "mutag_graphxai", "indole": "B-XAIC / indole"}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), sharey=True)
    band_handle = None
    for ax, ds in zip(axes, ["mutag_graphxai", "indole"]):
        g, b = gea[ds], baseline[ds]
        # pooled-across-seeds per-molecule GEA, one distribution per explainer
        pooled = {e: np.concatenate([g[f"{e}_s{s}"] for s in SEEDS]) for e in EXPL}
        # random-NULL band: the range of the 5 per-seed MEANS for RandNode and
        # RandEdge (not raw per-molecule min/max -- random baselines are
        # extremely stable seed-to-seed, CV 0.03-0.07, see F15; a per-molecule
        # min/max band is dominated by single-molecule outlier variance and
        # would misrepresent "chance level" as far wider than it really is)
        seed_means = [b[f"random_node_s{s}"].mean() for s in SEEDS] + \
                     [b[f"random_edge_s{s}"].mean() for s in SEEDS]
        band_lo, band_hi = min(seed_means), max(seed_means)
        band_handle = ax.axhspan(band_lo, band_hi, color="0.5", alpha=0.35, zorder=0)

        positions = [1, 2, 3]
        _box(ax, positions, [pooled[e] for e in EXPL])
        for x0, e in zip(positions, EXPL):
            _jitter_strip(ax, x0, pooled[e], rng)

        ax.set_xticks(positions)
        ax.set_xticklabels([ESH[e] for e in EXPL])
        ax.set_title(titles[ds])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", zorder=-1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("GEA (Jaccard vs ground-truth substructure)")
    axes[0].legend([band_handle], ["random-null range\n(seed-mean RandNode/RandEdge)"],
                   loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.7", fontsize=7.2)
    fig.suptitle("Per-molecule GEA (all seeds pooled) vs the random-null range\n"
                 "SX sits BELOW the null on mutag_graphxai, FAR ABOVE it on indole",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig1_headline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig1_headline.pdf")


def fig2_sx_win_rate():
    conn = {
        "mutag_graphxai": np.load(f"{OUT}/connectivity_arrays_mutag_graphxai.npz"),
        "indole": np.load(f"{OUT}/connectivity_arrays_indole.npz"),
    }
    win_rate = {ds: [conn[ds][f"win_vs_conn_s{s}"].mean() for s in SEEDS] for ds in conn}

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    x = np.arange(len(SEEDS))
    width = 0.36
    style = {"mutag_graphxai": {"grey": "0.75", "hatch": "..", "edge": "0.1"},
             "indole": {"grey": "0.35", "hatch": "//", "edge": "0.05"}}
    for i, ds in enumerate(["mutag_graphxai", "indole"]):
        offset = (i - 0.5) * width
        st = style[ds]
        ax.bar(x + offset, win_rate[ds], width=width, color=st["grey"],
               edgecolor=st["edge"], hatch=st["hatch"], linewidth=0.8,
               label="mutag_graphxai" if ds == "mutag_graphxai" else "B-XAIC / indole")

    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(len(SEEDS) - 0.5, 0.52, "chance (50%)", fontsize=7.5, va="bottom", ha="right", color="0.2")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("SX win rate vs connectivity-matched null\n(fraction of molecules)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax.set_title("SubgraphX vs connectivity-matched random null, per seed", fontsize=9.5, pad=32)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig2_sx_win_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig2_sx_win_rate.pdf")


def fig3_collapse_rate():
    df = pd.read_csv(f"{OUT}/collapse_table.csv")
    df = df[df["protocol"] == "coupled"]
    order = ["mutag_graphxai", "bxaic", "tox21"]
    label = {"mutag_graphxai": "mutag_graphxai", "bxaic": "B-XAIC\n(4 tasks)", "tox21": "Tox21\n(12 endpoints)"}
    grouped = df.groupby("dataset")["collapsed"].agg(["sum", "count"])
    rate = (grouped["sum"] / grouped["count"]).reindex(order)
    counts = grouped.reindex(order)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(order))
    greys = ["0.75", "0.55", "0.35"]
    hatches = ["..", "//", "xx"]
    bars = ax.bar(x, rate.values * 100, color=greys, edgecolor="0.1", hatch=hatches, linewidth=0.8, width=0.6)

    for xi, ds, bar in zip(x, order, bars):
        n_c, n_t = int(counts.loc[ds, "sum"]), int(counts.loc[ds, "count"])
        ax.text(xi, bar.get_height() + 1.5, f"{n_c}/{n_t}", ha="center", fontsize=8.5, color="0.1")

    ax.set_xticks(x)
    ax.set_xticklabels([label[d] for d in order])
    ax.set_ylabel("PGExplainer collapse rate\n(n_degenerate/n >= 0.5)")
    ax.set_ylim(0, max(rate.values * 100) + 12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("PGExplainer collapse rate by dataset\n(coupled protocol, 85 units)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_collapse_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig3_collapse_rate.pdf")


def main():
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    fig1_headline()
    fig2_sx_win_rate()
    fig3_collapse_rate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
