"""
EXPLORATORY. Scan PGExplainer's node-importance spread over EVERY completed
unit (85 coupled in runs/prio, 5 decoupled MUTAG-GT in runs/decoupled, 5
Mutagenicity in runs/mutagenicity) to compare two collapse criteria:

  strict   : fraction of molecules with std(node importance) == 0  >= 0.5
             (the rule used everywhere so far)
  revised  : median over molecules of (max - min) node importance < TAU
             ("saturated": the mask is numerically near-constant)

Needs only the explanation caches (no dataset loaders, no model).

    python -m src.analysis.exploratory_saturation_scan
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np
import torch

OUT = "runs/mutagenicity/exploratory"
TAU = 0.01


def unit_name(path):
    b = os.path.basename(path)
    d = os.path.basename(os.path.dirname(path))
    m = re.match(r"cache_(.+)_s(\d+)\.pt$", b) or re.match(r"cache_s(\d+)\.pt$", b)
    if d == "mutagenicity":
        return "mutagenicity", "", int(m.group(1)), "decoupled"
    name, seed = m.group(1), int(m.group(2))
    proto = "decoupled" if d == "decoupled" else "coupled"
    if name.startswith("bxaic_"):
        return "bxaic", name[6:], seed, proto
    if name.startswith("tox21_"):
        return "tox21", name[6:], seed, proto
    return name, "", seed, proto


def stats(results):
    rng_, std, degen = [], [], 0
    for r in results:
        imp = np.asarray(r["node_importance"], dtype=np.float64)
        rng_.append(imp.max() - imp.min()); std.append(imp.std()); degen += int(imp.std() == 0.0)
    return float(np.median(rng_)), float(np.median(std)), degen, len(results)


def main():
    paths = sorted(glob.glob("runs/prio/cache_*_s*.pt") + glob.glob("runs/decoupled/cache_*_s*.pt")
                   + glob.glob("runs/mutagenicity/cache_s*.pt"))
    rows = []
    for p in paths:
        ds, task, seed, proto = unit_name(p)
        res = torch.load(p, weights_only=False)["results"]
        row = {"dataset": ds, "task_or_endpoint": task, "seed": seed, "protocol": proto}
        for e, sh in (("pgexplainer", "pg"), ("gnnexplainer", "gnn"), ("subgraphx", "sx")):
            if e not in res:
                continue
            rm, sm, dg, n = stats(res[e])
            row.update({f"{sh}_range_median": rm, f"{sh}_std_median": sm, f"{sh}_n_degenerate": dg, "n": n})
        row["pg_strict_collapse"] = row["pg_n_degenerate"] / row["n"] >= 0.5
        row["pg_saturated"] = row["pg_range_median"] < TAU
        rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/saturation_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"units scanned: {len(rows)}   TAU = {TAU}")
    print("\nPGExplainer median (max - min) node importance, every unit, sorted:")
    for r in sorted(rows, key=lambda r: r["pg_range_median"]):
        tag = f"{r['dataset']}{'/' + r['task_or_endpoint'] if r['task_or_endpoint'] else ''}/s{r['seed']}/{r['protocol']}"
        print(f"  {tag:<34} range_med={r['pg_range_median']:.3e}  std_med={r['pg_std_median']:.3e}  degenerate={r['pg_n_degenerate']}/{r['n']}"
              f"  strict={'Y' if r['pg_strict_collapse'] else '-'}  saturated={'Y' if r['pg_saturated'] else '-'}")
    print("\nreference: smallest median range for the other explainers")
    for sh in ("gnn", "sx"):
        v = sorted(r[f"{sh}_range_median"] for r in rows if f"{sh}_range_median" in r)
        print(f"  {sh.upper()}: min {v[0]:.3e}  5th-smallest {v[4]:.3e}  median {np.median(v):.3e}")

    print("\ncollapse rate under both criteria")
    print(f"  {'group':<28}{'units':>6}{'strict':>8}{'saturated':>11}{'either':>8}")
    groups = [("mutag_graphxai coupled", lambda r: r["dataset"] == "mutag_graphxai" and r["protocol"] == "coupled"),
              ("mutag_graphxai decoupled", lambda r: r["dataset"] == "mutag_graphxai" and r["protocol"] == "decoupled"),
              ("bxaic (4 tasks)", lambda r: r["dataset"] == "bxaic"),
              ("tox21 (12 endpoints)", lambda r: r["dataset"] == "tox21"),
              ("mutagenicity", lambda r: r["dataset"] == "mutagenicity"),
              ("ALL coupled (85)", lambda r: r["protocol"] == "coupled"),
              ("ALL units", lambda r: True)]
    for name, f in groups:
        g = [r for r in rows if f(r)]
        s = sum(r["pg_strict_collapse"] for r in g); t = sum(r["pg_saturated"] for r in g)
        e = sum(r["pg_strict_collapse"] or r["pg_saturated"] for r in g)
        print(f"  {name:<28}{len(g):>6}{s:>5}/{len(g):<3}{t:>7}/{len(g):<3}{e:>5}/{len(g):<3}")
    print(f"\nsaved -> {OUT}/saturation_scan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
