"""
F9 seed-aware re-test: F9's GNN-vs-PG-vs-SX GEA significance per B-XAIC task
came from the ORIGINAL 1-seed scale-out (indole) / later individual checks --
never run through the same per-seed + seed-level method F3-REVISION v2
applied to mutag_graphxai. Now that all 4 B-XAIC tasks have 5-seed priority-
sweep data cached, apply the identical method here.

    python -m src.analysis.bxaic_f9_seed_check
"""
from __future__ import annotations
import torch
from ..data import LOADERS
from .seed_correct_significance import jaccard, per_seed_test, seed_level_test, EXPL, SH

TASKS = ["indole", "PAINS", "X", "P"]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"


def build_arrays_bxaic(task: str, seeds=SEEDS):
    data_list, meta = LOADERS[f"bxaic_{task}"]()
    by_seed = {}
    for s in seeds:
        blob = torch.load(f"{OUT}/cache_bxaic_{task}_s{s}.pt", weights_only=False)
        test_idx = list(blob["key"]["test_idx"])
        res = blob["results"]
        arr = {e: [] for e in EXPL}
        for k, mi in enumerate(test_idx):
            gt = data_list[mi].node_gt_mask.bool()
            if not bool(gt.any()):
                continue
            for e in EXPL:
                imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
                arr[e].append(jaccard(gt, imp > imp.mean()))
        by_seed[s] = {e: __import__("numpy").array(v, float) for e, v in arr.items()}
    return by_seed


def main():
    for task in TASKS:
        print("\n" + "#" * 100)
        print(f" B-XAIC task: {task}")
        print("#" * 100)
        by_seed = build_arrays_bxaic(task)
        per_seed_test(by_seed)
        seed_level_test(by_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
