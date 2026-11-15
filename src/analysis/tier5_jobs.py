"""RQ1 Tier 5: job lists for scripts/tier5_queue.sh (one shell command per line).   python -m src.analysis.tier5_jobs ITEM PHASE"""
from __future__ import annotations

import json
import math
import sys

from . import tier2_common as C
from . import tier5_common as T5

PY = "$PY -u -m src.analysis"
# GNNShap's CUDA extension is built/loaded by torch.utils.cpp_extension (runs/rq1_tier5/patches.md)
GNNSHAP_ENV = "PATH=/opt/anaconda/pkgs/ninja-1.10.2-hff7bd54_1/bin:$PATH CC=gcc-9 CXX=g++-9 TORCH_CUDA_ARCH_LIST=7.0 "
SLOW_FIRST = ["same", "gstarx", "gnnshap", "eig"]


def env(name):
    return GNNSHAP_ENV if name == "gnnshap" else ""


def scale():
    return json.load(open(f"{T5.T5}/scale.json"))


def n_shards(name, unit, n_mols, sc):
    """Shards so that one job is about 30 minutes at the N0 mean seconds per molecule (at most 40)."""
    sec = sc[name]["mean_seconds"].get(f"{C.dataset_of(unit)}/{T5.arch_of(unit)}", 60.0)
    return max(1, min(40, math.ceil(n_mols * sec / 1800)))


def explain_jobs(units_of, which, out_of, tag_of, extra=lambda name, u, x: ""):
    sc = scale(); per = {}
    from .tier5_explain import mols_for
    for name in SLOW_FIRST:
        if name not in sc or sc[name].get("skipped"):
            continue
        lv = sc[name]["level"]; per[name] = []
        for u, kind_or_tag in units_of(name):
            mols = mols_for(u, lv, which)
            if mols is None:
                continue
            k = n_shards(name, u, len(mols), sc)
            for i in range(k):
                per[name].append(f"{env(name)}{PY}.tier5_explain --unit {u} --expl {name} --level {lv} --set {which} --tag {tag_of(name, kind_or_tag)} "
                                 f"--out {out_of(name, u, kind_or_tag)} --shard {i}/{k}{extra(name, u, kind_or_tag)}")
    return [j for name in SLOW_FIRST for j in per.get(name, [])]


def jobs(item, phase):
    if (item, phase) == ("N0", "time"):
        from .tier5_N0 import TIME_UNITS
        return [f"{env(e)}{PY}.tier5_N0 time --unit {u} --expl {e}" for e in SLOW_FIRST for u in TIME_UNITS]
    if (item, phase) == ("N0", "scale"):
        return [f"{PY}.tier5_N0 scale"]
    if (item, phase) == ("N4", "score"):
        return [f"{PY}.tier5_N4 --unit {u}" for u in T5.n4_units()]
    if (item, phase) == ("N1", "explain"):
        return explain_jobs(lambda name: [(u, None) for u in T5.n1_units()], "n1", lambda name, u, x: f"{T5.T5}/N1/expl/{name}/{u}", lambda name, x: "T5")
    if (item, phase) in (("N1", "score"), ("N1", "fid")):
        return [f"{PY}.tier5_N1 {phase} --unit {u}" for u in T5.n1_units()]
    if (item, phase) == ("N2", "explain"):
        from .tier5_N23 import KINDS, N2_UNITS
        return explain_jobs(lambda name: [(u, k) for k in KINDS for u in N2_UNITS], "n2", lambda name, u, k: f"{T5.T5}/N2/expl/{name}/{k}/{u}",
                            lambda name, k: "T5", extra=lambda name, u, k: f" --ckpt {C.T2}/R/ckpt/{u}_{k}.pt")
    if (item, phase) == ("N2", "score"):
        from .tier5_N23 import N2_UNITS
        return [f"{PY}.tier5_N23 n2 --unit {u}" for u in N2_UNITS]
    if (item, phase) == ("N3", "explain"):
        from .tier5_N23 import N3_TAGS, N3_UNITS
        return explain_jobs(lambda name: [(u, t) for t in N3_TAGS[name] for u in N3_UNITS], "n3", lambda name, u, t: f"{T5.T5}/N3/expl/{name}/{t}/{u}",
                            lambda name, t: t, extra=lambda name, u, t: " --budgets k" if name == "same" else "")
    if (item, phase) == ("N3", "score"):
        from .tier5_N23 import N3_UNITS
        return [f"{PY}.tier5_N23 n3 --unit {u}" for u in N3_UNITS]
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    print("\n".join(jobs(sys.argv[1], sys.argv[2])))
