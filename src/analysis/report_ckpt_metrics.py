"""
Pure read-only report: test_acc/test_auroc already stored in each trained
checkpoint's own 'metrics' dict (computed at training time, not recomputed
here). Covers every checkpoint actually explained in the priority sweep:
mutag_graphxai (5 seeds), B-XAIC (4 tasks x 5 seeds), Tox21 (12 endpoints x
5 seeds). No new computation -- just loads and prints.

    python -m src.analysis.report_ckpt_metrics
"""
from __future__ import annotations
import numpy as np
import torch

OUT = "runs/prio"
SEEDS = [0, 1, 2, 3, 4]
BXAIC_TASKS = ["indole", "PAINS", "X", "P"]
TOX21_EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
             "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


def load_metrics(path):
    ck = torch.load(path, weights_only=False)
    m = ck["metrics"]
    return m["test_acc"], m["test_auroc"]


def report_group(title, path_fn, seeds=SEEDS):
    accs, aurocs = [], []
    print(f"\n{title}")
    for s in seeds:
        acc, auroc = load_metrics(path_fn(s))
        accs.append(acc); aurocs.append(auroc)
        print(f"  seed {s}: test_acc={acc:.4f}  test_auroc={auroc:.4f}")
    accs, aurocs = np.array(accs), np.array(aurocs)
    print(f"  mean/std: test_acc={accs.mean():.4f}/{accs.std():.4f}  "
          f"test_auroc={aurocs.mean():.4f}/{aurocs.std():.4f}")
    return accs, aurocs


def main():
    print("=" * 90)
    print(" mutag_graphxai")
    print("=" * 90)
    report_group("mutag_graphxai", lambda s: f"{OUT}/ckpt_mutag_graphxai_s{s}.pt")

    print("\n" + "=" * 90)
    print(" B-XAIC (4 tasks x 5 seeds)")
    print("=" * 90)
    for t in BXAIC_TASKS:
        report_group(f"B-XAIC/{t}", lambda s, t=t: f"{OUT}/ckpt_bxaic_{t}_s{s}.pt")

    print("\n" + "=" * 90)
    print(" Tox21 (12 endpoints x 5 seeds) -- mean AUROC per endpoint")
    print("=" * 90)
    for ep in TOX21_EPS:
        aurocs = []
        for s in SEEDS:
            _, auroc = load_metrics(f"{OUT}/ckpt_tox21_{ep}_s{s}.pt")
            aurocs.append(auroc)
        aurocs = np.array(aurocs)
        print(f"  {ep:<16} mean_auroc={aurocs.mean():.4f}  std={aurocs.std():.4f}  "
              f"per-seed=[{', '.join(f'{a:.3f}' for a in aurocs)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
