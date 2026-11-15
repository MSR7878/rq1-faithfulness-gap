"""RQ1 Tier 5: job lists for scripts/tier5_queue.sh (one shell command per line).   python -m src.analysis.tier5_jobs ITEM PHASE"""
from __future__ import annotations

import sys

from . import tier5_common as T5

PY = "$PY -u -m src.analysis"
# GNNShap's CUDA extension is built/loaded by torch.utils.cpp_extension (runs/rq1_tier5/patches.md)
GNNSHAP_ENV = "PATH=/opt/anaconda/pkgs/ninja-1.10.2-hff7bd54_1/bin:$PATH CC=gcc-9 CXX=g++-9 TORCH_CUDA_ARCH_LIST=7.0 "
SLOW_FIRST = ["same", "gstarx", "gnnshap", "eig"]


def env(name):
    return GNNSHAP_ENV if name == "gnnshap" else ""


def jobs(item, phase):
    if (item, phase) == ("N0", "time"):
        from .tier5_N0 import TIME_UNITS
        return [f"{env(e)}{PY}.tier5_N0 time --unit {u} --expl {e}" for e in SLOW_FIRST for u in TIME_UNITS]
    if (item, phase) == ("N0", "scale"):
        return [f"{PY}.tier5_N0 scale"]
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    print("\n".join(jobs(sys.argv[1], sys.argv[2])))
