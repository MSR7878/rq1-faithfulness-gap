"""RQ1 Tier 6: job lists for scripts/tier6_queue.sh (one shell command per line).   python -m src.analysis.tier6_jobs ITEM PHASE"""
from __future__ import annotations

import sys

from .tier5_jobs import GNNSHAP_ENV
from .tier6_explain import T1_UNITS, T2_UNITS, T3_UNITS, T4_UNITS

PY = "$PY -u -m src.analysis"
T2_SHARDS = {("R2", "sx_fixed"): 12, ("R1", "sx_fixed"): 3}          # about 30 minutes per job at the Tier 2 S time per molecule


def env(name):
    return GNNSHAP_ENV if name == "gnnshap" else ""


def jobs(item, phase):
    if (item, phase) == ("T1", "explain"):
        return [f"{PY}.tier6_explain t1 --unit {u} --expl {e} --shard {i}/{k}" for e, k in (("same", 5), ("gstarx", 6)) for u in T1_UNITS for i in range(k)]
    if (item, phase) == ("T1", "score"):
        return [f"{PY}.tier6_score t1 --unit {u}" for u in T2_UNITS]
    if (item, phase) == ("T2", "explain"):
        out = []
        for rule, w in [("R2", "sx_fixed"), ("R1", "sx_fixed")] + [(r, w) for w in ("gnnexplainer", "gnnshap", "eig", "ig") for r in ("R1", "R2")]:
            k = T2_SHARDS.get((rule, w), 1)
            out += [f"{env(w)}{PY}.tier6_explain t2 --unit {u} --expl {w} --rule {rule} --shard {i}/{k}" for u in T2_UNITS for i in range(k)]
        return out
    if (item, phase) == ("T2", "score"):
        return [f"{PY}.tier6_score t2 --unit {u}" for u in T2_UNITS]
    if (item, phase) == ("T3", "reset"):
        return [f"{PY}.tier6_explain reset --unit {u}" for u in T3_UNITS]
    if (item, phase) == ("T3", "explain"):
        return [f"{env(w)}{PY}.tier6_explain t3 --unit {u} --expl {w}" for w in ("gnnshap", "eig", "grad") for u in T3_UNITS]
    if (item, phase) == ("T3", "score"):
        return [f"{PY}.tier6_score t3 --unit {u}" for u in T3_UNITS]
    if (item, phase) == ("T4", "time"):
        return [f"{env(w)}{PY}.tier6_explain t4 --unit {u} --expl {w}" for w in ("gnnexplainer", "pgexplainer", "gnnshap", "grad", "eig") for u in T4_UNITS]
    if item == "R":
        return [f"{PY}.tier6_report {phase}"]
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    print("\n".join(jobs(sys.argv[1], sys.argv[2])))
