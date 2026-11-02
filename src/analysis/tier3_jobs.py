"""RQ1 Tier 3: job lists for scripts/tier3_queue.sh.   python -m src.analysis.tier3_jobs ITEM PHASE"""
from __future__ import annotations

import sys

from . import tier2_common as C

PY = "$PY -u -m src.analysis"
VARIANTS = ("all", "first", "last")
MUT = C.GROUPS["Mutagenicity D-MPNN"] + C.GROUPS["Mutagenicity GIN"] + C.GROUPS["Mutagenicity GINE"] + C.GROUPS["Mutagenicity GCN"]
ALL = [u for g in C.GROUPS.values() for u in g]
R_UNITS = [f"mutagenicity_dmpnn_s{s}" for s in range(5)] + [f"bxaic_indole_s{s}" for s in range(5)]


def jobs(item, phase):
    if (item, phase) == ("E", "explain"):
        return [f"{PY}.tier3_E explain --unit {u} --variant {v}" for u in MUT for v in VARIANTS]
    if (item, phase) == ("E", "score"):
        return [f"{PY}.tier3_E score --unit {u}" for u in MUT]
    if (item, phase) == ("F", "finetune"):
        return [f"{PY}.tier3_finetune --unit {u} --out runs/rq1_tier3/F/ckpt/{u}.pt" for u in ALL]
    if (item, phase) == ("F", "score"):
        return [f"{PY}.tier3_F --unit {u}" for u in ALL]
    if (item, phase) == ("X", "fid"):
        return [f"{PY}.tier3_X fid --unit {u} --kind M" for u in C.m_units()] + [f"{PY}.tier3_X fid --unit {u} --kind {k}" for u in R_UNITS for k in ("full", "head")]
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    print("\n".join(jobs(sys.argv[1], sys.argv[2])))
