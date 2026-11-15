"""
RQ1 Tier 5, item N1 (runs/rq1_tier5/PLAN.md): GEA and Fid of the new explainers next to the old methods.

  score   GEA (final protocol) per molecule: new explainers at our budget (eig, same, gstarx, gnnshap) and at their own size
          (eig_own, same_own; 0/1 sets scored as sets), the old methods (gnnexplainer, pgexplainer logit, sx_fixed, sal, ig, ixg),
          the three trivial rules and the two nulls, on the union of the molecules; each method keeps its own molecule list.
          -> runs/rq1_tier5/N1/gea/<unit>.npz
  fid     Fid+/Fid- (R1-R3 computed, R2/R3 reported; probability and logit) on the unit's 100-molecule subset for every method
          that covers it, with 20 random rankings; methods at a smaller level on their own slice (no random rankings: their
          molecules are a subset of the first run's). -> runs/rq1_tier5/N1/fid/<unit>.npz (+ <unit>__<method>.npz)

    python -m src.analysis.tier5_N1 score|fid --unit U
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import tier2_common as C
from . import tier5_common as T5

OLD = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig", "ixg"]
FID_OLD = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig"]


def scale() -> dict:
    return json.load(open(f"{T5.T5}/scale.json"))


def new_rows(unit, name, root=None) -> dict:
    d = root or f"{T5.T5}/N1/expl/{name}/{unit}"
    return {int(r["mol_idx"]): r for f in sorted(glob.glob(f"{d}/shard_*_of_*.pt")) for r in C.torch_load(f)}


def new_scores(unit, sc=None, root_of=None) -> dict:
    """method -> {mi: scores}; '_own' entries are 0/1 sets. Only methods whose level includes the unit; all molecules required."""
    sc = sc or scale(); out = {}
    for name in T5.NEW:
        if name not in sc or sc[name].get("skipped"):
            continue
        want = T5.level_mols(unit, sc[name]["level"])
        if want is None:
            continue
        rows = new_rows(unit, name, None if root_of is None else root_of(name))
        miss = [m for m in want if m not in rows]
        if miss:
            raise RuntimeError(f"{unit} {name}: {len(miss)} of {len(want)} molecules not explained")
        out[name] = {m: np.asarray(rows[m]["node"]) for m in want}
        if name in ("eig", "same"):
            out[f"{name}_own"] = {m: np.asarray(rows[m]["own"]) for m in want}
    return out


def set_gea(unit, sets, mols, dkey):
    dl, _ = C.load_data(dkey)
    return np.array([C.scorer_for(dkey, dl[m], m)(np.asarray(sets[m]) > 0.5) for m in mols])


def score(unit):
    dkey = C.dataset_of(unit); new = new_scores(unit); old = T5.old_scores(unit); gt = T5.gt_mols(unit)
    meth = {**{w: old[w] for w in OLD}, **new}
    union = sorted(set().union(*[set(s) for s in new.values()]) & set(gt)) if new else []
    A = {"unit": unit, "union": np.array(union), **{k: v for k, v in T5.null_arrays(unit, union).items()}}
    for h in ("hetero", "deg_high", "deg_low"):
        hd = C.heuristic_dict(unit, union, h); A[h] = C.gea_array(unit, h, lambda x, s=hd: s[x], union, dkey)
    for w, s in meth.items():
        mols = [m for m in union if m in s]
        A[f"{w}__mols"] = np.array(mols)
        A[f"{w}__gea"] = set_gea(unit, s, mols, dkey) if w.endswith("_own") else C.gea_array(unit, w, lambda x, s=s: s[x], mols, dkey)
    C.save_npz(f"{T5.T5}/N1/gea/{unit}.npz", **A)
    print(f"{unit}: union {len(union)} molecules; " + ", ".join(f"{w} {len(A[f'{w}__mols'])}" for w in meth), flush=True)


def fill_json(unit):
    sp = T5.spec(unit); c = sp["cache"]
    if c.endswith("_cache.pt") and os.path.exists(c.replace("_cache.pt", "_out.json")):
        return c.replace("_cache.pt", "_out.json")
    from .revision_pass import all_specs
    return all_specs()[unit]["out"]


def fid(unit):
    from .tier2_fid import run_fid
    dkey = C.dataset_of(unit); new = {w: s for w, s in new_scores(unit).items() if not w.endswith("_own")}; old = T5.old_scores(unit)
    base = T5.level_mols(unit, "L1"); meth = {**{w: old[w] for w in FID_OLD}, **new}
    full = {w: s for w, s in meth.items() if all(m in s for m in base)}
    ck, fj, D = T5.spec(unit)["ckpt"], fill_json(unit), f"{T5.T5}/N1/fid"
    if not os.path.exists(f"{D}/{unit}.npz"):
        run_fid(unit, ck, fj, dkey, full, base, f"{D}/{unit}.npz", with_randnode=True)
    for w, s in meth.items():
        if w in full:
            continue
        mols = [m for m in base if m in s]
        if mols and not os.path.exists(f"{D}/{unit}__{w}.npz"):
            run_fid(unit, ck, fj, dkey, {w: s}, mols, f"{D}/{unit}__{w}.npz")
    print(f"{unit}: fid on {len(base)} molecules for {sorted(full)}; own slices for {sorted(set(meth) - set(full))}", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["score", "fid"]); a.add_argument("--unit"); a = a.parse_args()
    {"score": score, "fid": fid}[a.cmd](a.unit)
