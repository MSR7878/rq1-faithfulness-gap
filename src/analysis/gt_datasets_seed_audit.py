"""
Comprehensive per-seed re-test of F1/F2/F4/F5/F6/F7/F8/F13 for the datasets
that have 5-seed priority-sweep data with per_molecule detail: mutag_graphxai
and all 4 B-XAIC tasks (indole/PAINS/X/P). MUTAG-standard and BBBP are NOT
covered here -- they were never multi-seeded (no priority-sweep equivalent
exists), and are excluded per that explicit spec limitation.

Each block reproduces the ORIGINAL finding's exact method, run independently
per seed, mirroring F11/F12's audit design.

    python -m src.analysis.gt_datasets_seed_audit
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"

DATASETS = {
    "mutag_graphxai": lambda s: f"{OUT}/out_mutag_graphxai_s{s}.json",
    "indole": lambda s: f"{OUT}/out_bxaic_indole_s{s}.json",
    "PAINS": lambda s: f"{OUT}/out_bxaic_PAINS_s{s}.json",
    "X": lambda s: f"{OUT}/out_bxaic_X_s{s}.json",
    "P": lambda s: f"{OUT}/out_bxaic_P_s{s}.json",
}
GT_DATASETS = ["mutag_graphxai", "indole", "PAINS", "X", "P"]  # all have GEA


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def load(ds, s):
    return json.load(open(DATASETS[ds](s)))


def fidplus_order(j, masking, fill):
    S = j["summary"][masking][fill]
    means = {e: S[e]["fid_plus"][0] for e in EXPL if e in S}
    return tuple(sorted(means, key=lambda e: -means[e])), means


def gea_order(j, masking="R3", fill="mean"):
    pm = j["per_molecule"][masking][fill]
    means = {}
    for e in EXPL:
        vals = [row["gea"] for row in pm[e] if "gea" in row]
        if vals:
            means[e] = float(np.mean(vals))
    if len(means) < 3:
        return None, None
    return tuple(sorted(means, key=lambda e: -means[e])), means


def pw_arr(j, masking, fill, metric, absval=False):
    pm = j["per_molecule"][masking][fill]
    out = {}
    for e in EXPL:
        if e in pm:
            v = np.array([row[metric] for row in pm[e]])
            out[e] = np.abs(v) if absval else v
    return out


def section_f13_f8():
    print("\n" + "=" * 100)
    print(" F13/F8 RE-TEST: GEA order vs Fid+ order agreement, per (dataset,seed), R3-mean/R2-zero/R1-hard")
    print("=" * 100)
    maskings = [("R3", "mean"), ("R2", "zero"), ("R1", "hard")]
    agree_total, cell_total = 0, 0
    x_all3_count = 0
    for ds in GT_DATASETS:
        print(f"\n [{ds}]")
        for s in SEEDS:
            j = load(ds, s)
            go, gmeans = gea_order(j)
            if go is None:
                continue
            row = []
            agree_this_seed_all3 = True
            for m, fl in maskings:
                fo, fmeans = fidplus_order(j, m, fl)
                agree = fo == go
                agree_total += int(agree); cell_total += 1
                if not agree:
                    agree_this_seed_all3 = False
                row.append(f"{m}:{'AGREE' if agree else 'diff'}")
            print(f"   seed {s}  GEA={'>'.join(SH[e] for e in go)}  " + "  ".join(row))
            if ds == "X" and agree_this_seed_all3:
                x_all3_count += 1
    print(f"\n  TOTAL agreement across all (dataset,seed,masking) cells: {agree_total}/{cell_total}")
    print(f"  X agrees under ALL 3 maskings in {x_all3_count}/5 seeds (F13's original headline claim: 'X agrees under ALL of R3/R2/R1', from 1 seed)")


def section_f5_f6_f7():
    print("\n" + "=" * 100)
    print(" F5/F6/F7 RE-TEST: R2 separability (F5), R1 washout (F6), R1 Fid+ PG-lowest (F7)")
    print(" per (dataset,seed), pairwise Wilcoxon+Holm on Fid+/Fid-")
    print("=" * 100)
    for ds in GT_DATASETS:
        print(f"\n [{ds}]")
        for s in SEEDS:
            j = load(ds, s)
            line = {"R2": {}, "R1": {}}
            for masking, fill in (("R2", "zero"), ("R1", "hard")):
                for metric in ("fid_plus", "fid_minus"):
                    arr = pw_arr(j, masking, fill, metric)
                    if len(arr) < 3:
                        continue
                    raw = {f"{a}-{b}": _wpair(arr[a], arr[b]) for a, b in PAIRS}
                    adj = holm(raw)
                    cells = []
                    for a, b in PAIRS:
                        hi = SH[a] if arr[a].mean() > arr[b].mean() else SH[b]
                        cells.append(f"{SH[a]}-{SH[b]}:{star(adj[f'{a}-{b}']):<3}({hi})")
                    line[masking][metric] = "  ".join(cells)
            print(f"   seed {s}")
            for masking in ("R2", "R1"):
                for metric in ("fid_plus", "fid_minus"):
                    if metric in line[masking]:
                        print(f"     {masking} {metric:<10} {line[masking][metric]}")


def section_f4():
    print("\n" + "=" * 100)
    print(" F4 RE-TEST (one-hot side, mutag_graphxai): R3->R2 |Fid-|/GEF inflation, GNN & PG, per seed")
    print("=" * 100)
    for s in SEEDS:
        j = load("mutag_graphxai", s)
        r3 = {"fid_minus": pw_arr(j, "R3", "mean", "fid_minus", absval=True),
              "gef": pw_arr(j, "R3", "mean", "gef")}
        r2 = {"fid_minus": pw_arr(j, "R2", "zero", "fid_minus", absval=True),
              "gef": pw_arr(j, "R2", "zero", "gef")}
        cells = []
        for e in ("gnnexplainer", "pgexplainer"):
            for metric in ("fid_minus", "gef"):
                a3, a2 = r3[metric][e], r2[metric][e]
                p = _wpair(a2, a3)
                cells.append(f"{SH[e]} {metric}: {a3.mean():.2f}->{a2.mean():.2f} p={p:.1e} {star(p)}")
        print(f"  seed {s}: " + "  ".join(cells))


def section_f1_f2():
    print("\n" + "=" * 100)
    print(" F1 RE-TEST: R3 noise-domination (std > mean), per (dataset,seed)")
    print("=" * 100)
    for ds in GT_DATASETS:
        counts = []
        for s in SEEDS:
            j = load(ds, s)
            noisy = 0; total = 0
            for e in EXPL:
                for metric in ("fid_plus", "fid_minus", "gef"):
                    arr = pw_arr(j, "R3", "mean", metric).get(e)
                    if arr is None:
                        continue
                    total += 1
                    if arr.std() > abs(arr.mean()):
                        noisy += 1
            counts.append(f"{noisy}/{total}")
        print(f"  {ds:<16} " + "  ".join(f"s{s}:{c}" for s, c in zip(SEEDS, counts)))

    print("\n" + "=" * 100)
    print(" F2 RE-TEST: indole R3 Fid+ heavy right tail (mean >> median), per seed")
    print("=" * 100)
    for s in SEEDS:
        j = load("indole", s)
        arr = pw_arr(j, "R3", "mean", "fid_plus")["gnnexplainer"]
        print(f"  seed {s}: mean={arr.mean():.3f}  median={np.median(arr):.3f}  ratio={arr.mean()/max(np.median(arr),1e-6):.1f}x")


def main():
    section_f13_f8()
    section_f5_f6_f7()
    section_f4()
    section_f1_f2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
