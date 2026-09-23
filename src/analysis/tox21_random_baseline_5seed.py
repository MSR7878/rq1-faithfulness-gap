"""
Random baseline for the FULL P3 priority-sweep Tox21 data: 12 endpoints x 5
seeds = 60 (endpoint,seed) units (vs the original tox21_random_baseline.py,
which only covered the single-seed scale-out). Extends F17 to 5 seeds and
applies the seed-aware significance lesson from mutag_graphxai/B-XAIC: the
independent unit here is the (endpoint,seed) pair, so the valid test pairs
the 60 UNIT-MEANS (each already an aggregate over its own 30 molecules), not
raw per-molecule values pooled across seeds.

Real explainer Fid+/Fid-/GEF come straight from each unit's own
runs/prio/out_tox21_<ep>_s<seed>.json (already computed during the sweep --
not recomputed here). Random-node / random-edge baselines are computed fresh
per unit (uniform node scores / uniform edge scores -> node_importance_from_
edges, same construction as F15-F17), scored through the SAME masking
pipeline (R3-mean, R2-zero, R1-hard) using that unit's own ckpt + cache
(for the target class and train split).

Reports:
  A. AGGREGATE table (mean over all 60 units) -- descriptive, like F17.
  B. PRIMARY significance: paired Wilcoxon on the 60 (endpoint,seed) UNIT
     MEANS (real vs its matched random baseline), Holm over the 3 pairs per
     (masking,metric) cell. This is the valid test -- n=60 independent units,
     not pseudo-replicated.
  C. SECONDARY (mirrors mutag's per-seed view): for each of the 5 seeds
     separately, the same paired test restricted to that seed's 12
     endpoints (n=12, matching F17's original single-seed scope) -- reports
     how many of the 5 seeds show the effect, i.e. does it reproduce across
     independent model fits.
  D. PGExplainer collapse rate across all 60 units (explainer_health,
     n_degenerate/n >= 0.5), extending the cross-dataset collapse tally.

    python -m src.analysis.tox21_random_baseline_5seed
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..data import LOADERS
from ..explain.common import ExplanationResult, node_importance_from_edges
from ..explain.run_phase3 import FILLS_BY_MASKING, load_model, score_one
from ..metrics.masking import training_fill_vector

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
MATCHED = {"gnnexplainer": "random_node", "subgraphx": "random_node", "pgexplainer": "random_edge"}
RSH = {"random_node": "RandN", "random_edge": "RandE"}
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
SEEDS = [0, 1, 2, 3, 4]
MASKINGS = [("R3", "mean"), ("R2", "zero"), ("R1", "hard")]
OUT = "runs/prio"


def star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def random_means_for_unit(ep: str, seed: int, device) -> tuple[dict, bool]:
    """Returns ({masking: {'random_node'/'random_edge': {metric: mean}}}, pg_collapsed)."""
    ckpt_path = os.path.join(OUT, f"ckpt_tox21_{ep}_s{seed}.pt")
    cache_path = os.path.join(OUT, f"cache_tox21_{ep}_s{seed}.pt")
    model, ck = load_model(ckpt_path, device)
    data_list, meta = LOADERS[f"tox21_{ep}"]()
    blob = torch.load(cache_path, weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    train_idx = list(ck["split"]["train"])
    x_train = torch.cat([data_list[i].x.float() for i in train_idx], dim=0)
    fills = {s: training_fill_vector(x_train, s) for s in FILLS_BY_MASKING["R3"]}

    torch.manual_seed(400000 + seed)
    rows = {m: {"random_node": {k: [] for k in ("fid_plus", "fid_minus", "gef")},
               "random_edge": {k: [] for k in ("fid_plus", "fid_minus", "gef")}}
            for m, _ in MASKINGS}
    for k, mi in enumerate(test_idx):
        d = data_list[mi].to(device)
        n = d.num_nodes
        tgt = res["gnnexplainer"][k]["target"]
        node_score = torch.rand(n)
        edge_score = torch.rand(d.edge_index.shape[1])
        folded = node_importance_from_edges(edge_score, d.edge_index, n)
        for kind, imp in (("random_node", node_score), ("random_edge", folded)):
            result = ExplanationResult(node_importance=imp, target=tgt)
            for m, fill_name in MASKINGS:
                fv = fills.get(fill_name) if m == "R3" else None
                row = score_one(model, d, result, fv, device, False, m)
                for metric in ("fid_plus", "fid_minus", "gef"):
                    rows[m][kind][metric].append(row[metric])

    means = {m: {kind: {metric: float(np.mean(v)) for metric, v in d2.items()}
                for kind, d2 in d1.items()} for m, d1 in rows.items()}

    health = json.load(open(os.path.join(OUT, f"out_tox21_{ep}_s{seed}.json")))["explainer_health"]
    pg_h = health.get("pgexplainer", {})
    collapsed = pg_h.get("n_degenerate", 0) / max(pg_h.get("n", 30), 1) >= 0.5
    return means, collapsed


def real_means_for_unit(ep: str, seed: int) -> dict:
    j = json.load(open(os.path.join(OUT, f"out_tox21_{ep}_s{seed}.json")))
    S = j["summary"]
    out = {}
    for m, fill in MASKINGS:
        out[m] = {e: {k: S[m][fill][e][k][0] for k in ("fid_plus", "fid_minus", "gef")}
                  for e in EXPL if e in S[m][fill]}
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    units = [(ep, s) for ep in EPS for s in SEEDS]
    print(f"processing {len(units)} (endpoint,seed) units...")

    # unit_mean[masking][metric][explainer-or-random][(ep,seed)] = float
    unit_mean = {m: {k: {e: {} for e in EXPL + ["random_node", "random_edge"]}
                     for k in ("fid_plus", "fid_minus", "gef")} for m, _ in MASKINGS}
    collapsed_units = []

    for i, (ep, s) in enumerate(units):
        real = real_means_for_unit(ep, s)
        rnd, collapsed = random_means_for_unit(ep, s, device)
        if collapsed:
            collapsed_units.append((ep, s))
        for m, _ in MASKINGS:
            for k in ("fid_plus", "fid_minus", "gef"):
                for e in EXPL:
                    if e in real[m]:
                        unit_mean[m][k][e][(ep, s)] = real[m][e][k]
                for kind in ("random_node", "random_edge"):
                    unit_mean[m][k][kind][(ep, s)] = rnd[m][kind][k]
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{len(units)} units done")

    print("\n" + "=" * 100)
    print(" A. AGGREGATE across all 60 (endpoint,seed) units (mean of unit means)")
    print("=" * 100)
    for m, _ in MASKINGS:
        print(f"\n [{m}]   {'':<8}{'Fid+':>10}{'Fid-':>10}{'GEF':>10}")
        for e in EXPL + ["random_node", "random_edge"]:
            label = SH.get(e, RSH.get(e))
            cells = []
            for k in ("fid_plus", "fid_minus", "gef"):
                v = np.array(list(unit_mean[m][k][e].values()), float)
                cells.append(f"{v.mean():>10.3f}")
            print(f"    {label:<8}" + "".join(cells))

    print("\n" + "=" * 100)
    print(" B. PRIMARY significance -- paired Wilcoxon on the 60 unit-means (real vs matched random)")
    print("    Holm over the 3 real-vs-random pairs within each (masking,metric) cell")
    print("=" * 100)
    for m, _ in MASKINGS:
        for k in ("fid_plus", "fid_minus", "gef"):
            raw = {}
            for e, rnd in MATCHED.items():
                keys = sorted(set(unit_mean[m][k][e]) & set(unit_mean[m][k][rnd]))
                a = np.array([unit_mean[m][k][e][key] for key in keys])
                b = np.array([unit_mean[m][k][rnd][key] for key in keys])
                raw[e] = _wpair(a, b)
            adj = holm(raw)
            cells = []
            for e, rnd in MATCHED.items():
                keys = sorted(set(unit_mean[m][k][e]) & set(unit_mean[m][k][rnd]))
                a = np.array([unit_mean[m][k][e][key] for key in keys])
                b = np.array([unit_mean[m][k][rnd][key] for key in keys])
                hi = SH[e] if a.mean() > b.mean() else RSH[rnd]
                cells.append(f"{SH[e]}vs{RSH[rnd]}:{star(adj[e]):<3}({hi},n={len(keys)})")
            print(f"  {m:<4}{k:<10}" + "  ".join(cells))

    print("\n" + "=" * 100)
    print(" C. SECONDARY -- per-seed cross-endpoint test (n=12 endpoints, one test PER seed --")
    print("    mirrors F17's original single-seed scope; reports X/5 seeds significant, real higher)")
    print("=" * 100)
    for m, _ in MASKINGS:
        for k in ("fid_plus", "fid_minus", "gef"):
            summary_cells = []
            for e, rnd in MATCHED.items():
                sig_and_higher = 0
                for s in SEEDS:
                    keys = [(ep, s) for ep in EPS if (ep, s) in unit_mean[m][k][e] and (ep, s) in unit_mean[m][k][rnd]]
                    if len(keys) < 3:
                        continue
                    a = np.array([unit_mean[m][k][e][key] for key in keys])
                    b = np.array([unit_mean[m][k][rnd][key] for key in keys])
                    p = _wpair(a, b)
                    if p < 0.05 and a.mean() > b.mean():
                        sig_and_higher += 1
                summary_cells.append(f"{SH[e]}vs{RSH[rnd]}: {sig_and_higher}/5 seeds sig(real higher)")
            print(f"  {m:<4}{k:<10}" + "  ".join(summary_cells))

    print("\n" + "=" * 100)
    print(f" D. PGExplainer collapse across all 60 units: {len(collapsed_units)}/60"
          f" ({100*len(collapsed_units)/60:.0f}%)")
    print("=" * 100)
    for ep, s in collapsed_units:
        print(f"    {ep}  seed {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
