"""
Export each D-MPNN checkpoint's stored split to json, so the GIN architecture
check trains on IDENTICAL train/val/test indices (train.py --split-json).
Read-only on the D-MPNN checkpoints; writes only under runs/arch_gin/.

    python -m src.analysis.arch_gin_splits
    python -m src.analysis.arch_gin_splits runs/arch_gine mutagenicity_graphxai   # other out dir / datasets
"""
from __future__ import annotations

import json
import os
import sys

import torch

SOURCES = {"mutag_graphxai": "runs/prio/ckpt_mutag_graphxai_s{s}.pt",
           "mutagenicity_graphxai": "runs/mutagenicity/ckpt_s{s}.pt"}


def main() -> int:
    R = sys.argv[1] if len(sys.argv) > 1 else "runs/arch_gin"
    for ds, tmpl in SOURCES.items():
        if len(sys.argv) > 2 and ds not in sys.argv[2:]:
            continue
        os.makedirs(f"{R}/{ds}", exist_ok=True)
        for s in range(5):
            ck = torch.load(tmpl.format(s=s), map_location="cpu", weights_only=False)
            assert ck["seed"] == s and ck.get("arch", "dmpnn") == "dmpnn"
            sp = {k: [int(i) for i in ck["split"][k]] for k in ("train", "val", "test")}
            json.dump({**sp, "source_ckpt": tmpl.format(s=s), "source_seed": s,
                       "dmpnn_test_acc": ck["metrics"]["test_acc"], "dmpnn_test_auroc": ck["metrics"]["test_auroc"]},
                      open(f"{R}/{ds}/split_s{s}.json", "w"))
            print(f"{ds} seed {s}: train/val/test {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}"
                  f"  D-MPNN test acc {ck['metrics']['test_acc']:.4f} AUROC {ck['metrics']['test_auroc']:.4f}"
                  f"  -> {R}/{ds}/split_s{s}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
