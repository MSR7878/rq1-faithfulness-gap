"""
P2 (mutag_graphxai, --explain-pool all): does GEA differ between the
in-sample (train) molecules and the held-out (test) ones?

For each of the 5 priority-sweep seeds, reads runs/prio/cache_mutag_graphxai_s
<seed>.pt (cached node_importance per explained molecule, in the SAME order
as the ckpt's key.test_idx) + runs/prio/ckpt_mutag_graphxai_s<seed>.pt (the
train/val/test split those molecules were drawn from -- --explain-pool all
means every molecule in the model's own training set is ALSO in the explained
set). GEA is computed directly here (mean-threshold Jaccard vs node_gt_mask,
identical to run_phase3.score_one) so this works whether or not that unit's
JSON has the new molecule_split field (older in-flight seeds don't; new ones
do -- this script doesn't depend on it either way, only on the cache+ckpt).

Reports GEA three ways per explainer: ALL 188, TEST-only (~29, held-out),
TRAIN-only (~131, in-sample) -- plus VAL (~28) for completeness -- per seed
and pooled across seeds, with a two-sample Mann-Whitney U test (train vs
test are different molecules, not paired) on whether they differ.

    python -m src.analysis.mutag_full_split_gea
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from ..data import LOADERS

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
OUT = "runs/prio"


def jaccard(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt, pred = gt.bool(), pred.bool()
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0


def seeds_available() -> list[int]:
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, "cache_mutag_graphxai_s*.pt"))):
        m = re.search(r"_s(\d+)\.pt$", p)
        if m and os.path.exists(os.path.join(OUT, f"ckpt_mutag_graphxai_s{m.group(1)}.pt")):
            out.append(int(m.group(1)))
    return out


def per_seed(seed: int, data_list):
    ck = torch.load(os.path.join(OUT, f"ckpt_mutag_graphxai_s{seed}.pt"), map_location="cpu", weights_only=False)
    blob = torch.load(os.path.join(OUT, f"cache_mutag_graphxai_s{seed}.pt"), weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    split_of = {i: name for name, idxs in ck["split"].items() for i in idxs}
    res = blob["results"]

    rows = {e: [] for e in res}  # each: (molecule_idx, split_name, gea)
    for e in res:
        for k, mi in enumerate(test_idx):
            gt = data_list[mi].node_gt_mask.bool()
            if not bool(gt.any()):
                continue  # matches score_one: GEA only where GT exists (moot here -- all 188 have GT)
            imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
            pred = imp > imp.mean()
            rows[e].append((mi, split_of.get(mi, "unknown"), jaccard(gt, pred)))
    return rows, ck["metrics"].get("test_auroc")


def main() -> int:
    seeds = seeds_available()
    if not seeds:
        print(f"no completed/in-flight (cache+ckpt) mutag_graphxai priority-sweep seeds in {OUT}/ yet")
        return 1
    print(f"seeds with cache+ckpt present: {seeds}  (a seed's SubgraphX pass may still be running -- "
          f"its cache only has the explainers that had finished when it was last written)")

    data_list, meta = LOADERS["mutag_graphxai"]()
    per_seed_rows = {}
    for s in seeds:
        rows, auroc = per_seed(s, data_list)
        per_seed_rows[s] = rows
        n = {e: len(rows[e]) for e in rows}
        print(f"  seed {s}: test_auroc={auroc:.3f}  n explained per explainer: {n}")

    print("\n" + "=" * 92)
    print(" GEA three ways: ALL 188 / TEST-only / TRAIN-only (+ VAL) -- per seed")
    print("=" * 92)
    pooled = {e: {"all": [], "train": [], "val": [], "test": []} for e in EXPL}
    for s in seeds:
        rows = per_seed_rows[s]
        print(f"\n seed {s}")
        for e in EXPL:
            if e not in rows or not rows[e]:
                continue
            by_split = {"train": [], "val": [], "test": []}
            all_g = []
            for _, sp, g in rows[e]:
                all_g.append(g)
                if sp in by_split:
                    by_split[sp].append(g)
                pooled[e]["all"].append(g)
                if sp in pooled[e]:
                    pooled[e][sp].append(g)
            cells = " ".join(f"{k}={np.mean(v):.3f}(n={len(v)})" for k, v in
                             (("all", all_g), ("train", by_split["train"]),
                              ("val", by_split["val"]), ("test", by_split["test"])) if v)
            print(f"   {SH[e]:<4} {cells}")

    print("\n" + "=" * 92)
    print(" POOLED across all seeds -- headline comparison")
    print("=" * 92)
    for e in EXPL:
        p = pooled[e]
        if not p["all"]:
            continue
        all_a, tr_a, va_a, te_a = (np.array(p[k], float) for k in ("all", "train", "val", "test"))
        print(f"\n {SH[e]}")
        print(f"   ALL 188   mean={all_a.mean():.3f}  std={all_a.std():.3f}  n={len(all_a)}")
        print(f"   TRAIN     mean={tr_a.mean():.3f}  std={tr_a.std():.3f}  n={len(tr_a)}")
        print(f"   VAL       mean={va_a.mean():.3f}  std={va_a.std():.3f}  n={len(va_a)}")
        print(f"   TEST      mean={te_a.mean():.3f}  std={te_a.std():.3f}  n={len(te_a)}")
        if len(tr_a) and len(te_a):
            diff = tr_a.mean() - te_a.mean()
            if np.allclose(tr_a, tr_a[0]) and np.allclose(te_a, te_a[0]) and tr_a[0] == te_a[0]:
                p_mw = 1.0
            else:
                p_mw = mannwhitneyu(tr_a, te_a, alternative="two-sided").pvalue
            star = "***" if p_mw < 1e-3 else "**" if p_mw < 1e-2 else "*" if p_mw < 5e-2 else "ns"
            print(f"   TRAIN - TEST = {diff:+.3f}   Mann-Whitney U p={p_mw:.4f}  {star}"
                  f"  ({'TRAIN higher' if diff > 0 else 'TEST higher' if diff < 0 else 'tied'})")

    print("\n" + "=" * 92)
    print(" VERDICT")
    print("=" * 92)
    for e in EXPL:
        p = pooled[e]
        if not p["train"] or not p["test"]:
            continue
        tr_a, te_a = np.array(p["train"]), np.array(p["test"])
        p_mw = (1.0 if np.allclose(tr_a, tr_a[0]) and np.allclose(te_a, te_a[0]) and tr_a[0] == te_a[0]
                else mannwhitneyu(tr_a, te_a, alternative="two-sided").pvalue)
        if p_mw >= 0.05:
            print(f"  {SH[e]:<4} TRAIN ({tr_a.mean():.3f}) ~= TEST ({te_a.mean():.3f}), ns"
                  f" -> full-188 pool justified for this explainer.")
        elif tr_a.mean() > te_a.mean():
            print(f"  {SH[e]:<4} TRAIN ({tr_a.mean():.3f}) > TEST ({te_a.mean():.3f}) significantly"
                  f" -> report TEST-only ({te_a.mean():.3f}) as the headline GEA; full-188 ({np.mean(p['all']):.3f})"
                  f" is inflated by in-sample molecules.")
        else:
            print(f"  {SH[e]:<4} TEST ({te_a.mean():.3f}) > TRAIN ({tr_a.mean():.3f}) significantly"
                  f" -> full-188 not inflated (if anything conservative); report as-is, flag the direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
