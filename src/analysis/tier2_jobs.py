"""
RQ1 Tier 2: job lists for scripts/tier2_queue.sh (one shell command per line, no quotes), and the fixed
molecule subsets (100 ground-truth molecules) for R and the S sensitivity runs.

    python -m src.analysis.tier2_jobs ITEM PHASE
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np

from . import tier2_common as C

PY = "$PY -u -m src.analysis"
SHARD = 5                       # molecules per SubgraphX job
S_PRIO = [f"mutagenicity_dmpnn_s{s}" for s in range(5)] + [f"bxaic_indole_s{s}" for s in range(5)] + [f"mutag_graphxai_s{s}" for s in range(10)]
S_REST = [f"bxaic_{t}_s{s}" for t in ("PAINS", "X", "P") for s in range(5)] + [f"mutagenicity_{a}_s{s}" for a in ("gin", "gine", "gcn") for s in range(5)]
R_UNITS = [f"mutagenicity_dmpnn_s{s}" for s in range(5)] + [f"bxaic_indole_s{s}" for s in range(5)]
SENS_UNITS = ["mutagenicity_dmpnn_s0", "mutag_graphxai_s0"]
SENS = {"r50": (50, 30, "S"), "r100": (100, 30, "S"), "s100": (20, 100, "S"), "rep": (20, 30, "S-repeat")}   # (rollout, sample, seed tag)


def subset100(unit) -> list[int]:
    """Fixed random 100 ground-truth molecules of a unit's explained set (all of them if fewer)."""
    ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; gt = C.gt_mol_idx(unit, ti, C.dataset_of(unit))
    if len(gt) <= 100:
        return gt
    r = np.random.default_rng([83000, zlib.crc32(unit.encode())])
    return sorted(int(m) for m in r.choice(gt, size=100, replace=False))


def write_subsets():
    f = f"{C.T2}/subsets.json"
    S = {u: subset100(u) for u in sorted(set(R_UNITS) | set(SENS_UNITS))}
    os.makedirs(C.T2, exist_ok=True); json.dump(S, open(f, "w"), indent=0)
    return S


def mols_file(unit, mols, d):
    os.makedirs(d, exist_ok=True); f = f"{d}/{unit}.json"
    json.dump([int(m) for m in mols], open(f, "w"))
    return f


def sx_jobs(unit, ckpt, mols, out, tag, rollout=20, sample=30, mdir=None):
    mf = mols_file(unit, mols, mdir or f"{os.path.dirname(out)}/mols")
    n = max(1, -(-len(mols) // SHARD))
    return [f"{PY}.tier2_explain sx --unit {unit} --ckpt {ckpt} --mols-file {mf} --shard {i}/{n} --out {out} --tag {tag} "
            f"--rollout {rollout} --sample {sample} --budget score" for i in range(n)]


def interleave(lists):
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def jobs(item, phase):
    U = C.unit_specs(); T = C.T2
    if item == "H":
        return [f"{PY}.tier2_gea H --units {u}" for g in C.GROUPS.values() for u in g]
    if item == "G":
        return [f"{PY}.tier2_explain grad --unit {u} --ckpt {C.orig_spec(u)['ckpt']} --mols {C.orig_spec(u)['cache']} --out {T}/G/grad/{u}.npz "
                f"&& {PY}.tier2_gea G --units {u}" for g in C.GROUPS.values() for u in g]
    if item == "S" and phase in ("prio", "rest"):
        units = S_PRIO if phase == "prio" else S_REST
        per = {u: sx_jobs(u, C.orig_spec(u)["ckpt"], C.stored_cache(u)["key"]["test_idx"], f"{T}/S/sx/{u}", "S", mdir=f"{T}/S/mols") for u in units}
        slow = [per[u] for u in units if not u.startswith("mutag_graphxai")]; fast = [per[u] for u in units if u.startswith("mutag_graphxai")]
        return interleave(slow) + interleave(fast)
    if item == "S" and phase in ("prio_score", "rest_score"):
        return [f"{PY}.tier2_gea S --units {u}" for u in (S_PRIO if phase == "prio_score" else S_REST)]
    if item == "S" and phase == "sens":
        sub = json.load(open(f"{T}/subsets.json")); out = []
        for name, (r, s, tag) in SENS.items():
            out.append(interleave([sx_jobs(u, C.orig_spec(u)["ckpt"], sub[u], f"{T}/S/sens/{name}/{u}", tag, r, s, mdir=f"{T}/S/sens/mols")
                                   for u in SENS_UNITS]))
        return interleave(out)
    if item == "R" and phase == "rand":
        return [f"{PY}.tier2_explain randomise --unit {u} --kind {k} --out {T}/R/ckpt/{u}_{k}.pt" for u in R_UNITS for k in ("full", "head")]
    if item == "R" and phase == "explain":
        sub = json.load(open(f"{T}/subsets.json")); sx, other = [], []
        for u in R_UNITS:
            for k in ("full", "head"):
                ck = f"{T}/R/ckpt/{u}_{k}.pt"
                other.append(f"{PY}.tier2_explain gp --unit {u} --ckpt {ck} --out {T}/R/gp/{k}")
                other.append(f"{PY}.tier2_explain grad --unit {u} --ckpt {ck} --mols {C.orig_spec(u)['cache']} --out {T}/R/grad/{k}/{u}.npz")
                sx.append(sx_jobs(u, ck, sub[u], f"{T}/R/sx/{k}/{u}", "R", mdir=f"{T}/R/mols"))
        return other + interleave(sx)
    MU = list(C.m_units())
    if item == "M" and phase == "train":
        return [f"$PY -u -m src.train.train {C.M_TRAIN[u.rsplit('_s', 1)[0]]} --seed {u.rsplit('_s', 1)[1]} --device cuda --save {C.m_units()[u]['ckpt']}"
                for u in MU]
    if item == "M" and phase == "gp":
        return [f"{PY}.tier2_explain gp --unit {u} --ckpt {C.m_units()[u]['ckpt']} --out {T}/M/gp" for u in MU]
    if item == "M" and phase == "explain":
        g = [f"{PY}.tier2_explain grad --unit {u} --ckpt {C.m_units()[u]['ckpt']} --mols {C.m_units()[u]['cache']} --out {T}/M/grad/{u}.npz" for u in MU]
        sx = [sx_jobs(u, C.m_units()[u]["ckpt"], C.torch_load(C.m_units()[u]["cache"])["key"]["test_idx"], f"{T}/M/sx/{u}", "S", mdir=f"{T}/M/mols")
              for u in MU]
        return g + interleave(sx)
    if item == "M" and phase == "score":
        return [f"{PY}.tier2_M {u}" for u in MU]
    raise SystemExit(f"unknown item/phase {item} {phase}")


if __name__ == "__main__":
    item, phase = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "all")
    if item == "subsets":
        S = write_subsets(); print({u: len(v) for u, v in S.items()})
    else:
        print("\n".join(jobs(item, phase)))
