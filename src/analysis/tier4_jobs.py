"""RQ1 Tier 4: job lists for scripts/tier4_queue.sh (one shell command per line).   python -m src.analysis.tier4_jobs ITEM PHASE"""
from __future__ import annotations

import json
import sys

from . import tier2_common as C
from . import tier4_common as T
from .tier2_jobs import interleave, sx_jobs

PY = "$PY -u -m src.analysis"
MUTAG = [f"mutag_graphxai_s{s}" for s in range(10)]
MUT4 = [f"mutagenicity_{a}_s{s}" for a in C.ARCHS for s in range(5)]
L6_UNITS = ["mutagenicity_dmpnn_s0", "bxaic_indole_s0", "mutag_graphxai_s0"]
L6_TAGS = ["L6a", "L6b", "L6c"]
D4 = T.T4


def explain_jobs(units, item, sx_mols):
    U = T.t4_units()
    g = [f"{PY}.tier2_explain grad --unit {u} --ckpt {U[u]['ckpt']} --mols {U[u]['cache']} --out {D4}/{item}/grad/{u}.npz" for u in units]
    sx = [sx_jobs(u, U[u]["ckpt"], sx_mols(u), f"{D4}/{item}/sx/{u}", "S", mdir=f"{D4}/{item}/mols") for u in units]
    return g + interleave(sx)


def jobs(item, phase):
    U = T.t4_units()
    if (item, phase) == ("L1", "gea"):
        return [f"{PY}.tier4_L1 gea --unit {u}" for u in MUTAG]
    if (item, phase) == ("L1", "fid"):
        return [f"{PY}.tier4_L1 fid --unit {u}" for u in MUTAG]
    if (item, phase) == ("L3", "score"):
        return [f"{PY}.tier4_L3 --unit {u}" for u in MUT4]
    if (item, phase) == ("L4", "run"):
        return [f"{PY}.tier4_L4 --unit {u}" for u in MUTAG]
    if (item, phase) == ("L5", "smoke"):
        return [f"CUBLAS_WORKSPACE_CONFIG=:4096:8 {PY}.tier4_L5 smoke"]
    if (item, phase) == ("L5", "pg"):
        dev = json.load(open(f"{D4}/L5/device.json"))["device"]
        th = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 " if dev == "cpu" else ""
        from .tier4_L5 import UNITS
        return [f"{th}CUBLAS_WORKSPACE_CONFIG=:4096:8 {PY}.tier4_L5 pg --unit {u} --rep {r} --device {dev}" for r in range(3) for u in UNITS]
    if (item, phase) == ("L7", "tune"):
        from .tier4_L7 import DRIFTED
        return [f"{PY}.tier4_L7 tune --unit {u}" for u in DRIFTED]
    if (item, phase) == ("L7", "score"):
        from .tier4_L7 import DRIFTED
        return [f"{PY}.tier4_L7 score --unit {u}" for u in DRIFTED]
    if item in ("L2", "M1", "M2"):
        units = {"L2": T.bbbp_units(), "M1": T.m1_units(), "M2": T.m2_units()}[item]
        if phase == "train":
            return [f"[ -f {U[u]['ckpt']} ] || {T.train_cmd(u)}" for u in units]
        if phase == "gp":
            return [f"{PY}.tier2_explain gp --unit {u} --ckpt {U[u]['ckpt']} --out {D4}/{item}/gp" for u in units]
        if phase == "explain":
            if item == "M1":
                sub = T.m1_subsets(); return explain_jobs(units, item, lambda u: sub[T.dmpnn_of(u)])
            return explain_jobs(units, item, T.test_idx_of)
        if (item, phase) == ("L2", "fid"):
            return [f"{PY}.tier4_L2 --unit {u}" for u in units]
        if phase == "score":
            return [f"{PY}.tier4_M score --unit {u}" for u in units]
    if (item, phase) == ("M1", "prep"):
        return [f"{PY}.tier4_M prep"]
    if (item, phase) == ("M1", "check"):
        return [f"{PY}.tier4_M check"]
    if (item, phase) == ("L6", "sx"):
        sub = json.load(open(f"{C.T2}/subsets.json"))
        return interleave([sx_jobs(u, C.orig_spec(u)["ckpt"], sub[u], f"{D4}/L6/sx/{tag}/{u}", tag, mdir=f"{D4}/L6/mols") for tag in L6_TAGS for u in L6_UNITS])
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    print("\n".join(jobs(sys.argv[1], sys.argv[2])))
