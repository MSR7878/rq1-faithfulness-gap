"""
RQ1 Tier 5 (runs/rq1_tier5/PLAN.md): one new explainer on a list of molecules of one unit, sharded, resumable, written atomically.
Every molecule is seeded by crc32("<tag>|<unit>|<mol>") (python, numpy, torch), so a result does not depend on its shard.
SAME is run at both budgets ('k' = floor(0.25 N) and 'own' = ceil(0.5 N); N3 uses --budgets k); the others once.

    python -m src.analysis.tier5_explain --unit U --expl E --level L --set n1|n2|n3 --tag T --out DIR --shard i/n [--ckpt CK]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ..explain.run_phase3 import load_model
from ..explain.t5_common import mol_seed, seed_all
from . import tier2_common as C
from . import tier5_common as T5


def mols_for(unit, level, which):
    """n1: the unit's molecules at its method level; n2 / n3: the 100-molecule subset (or the level's slice of it)."""
    if which == "n1":
        return T5.level_mols(unit, level)
    return T5.level_mols(unit, "L1" if level == "L0" else level)


def run(unit, name, level, which, tag, out_dir, shard, ckpt=None, budgets=None):
    i, n = (int(v) for v in shard.split("/"))
    mols = mols_for(unit, level, which)
    mine = [int(m) for j, m in enumerate(mols) if j % n == i]
    f = f"{out_dir}/shard_{i}_of_{n}.pt"
    rows = torch.load(f, weights_only=False) if os.path.exists(f) else []
    done = {r["mol_idx"] for r in rows}
    if all(m in done for m in mine):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt or T5.spec(unit)["ckpt"], device); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    ex = T5.make_explainer(name, model, ck, dkey, device)
    budgets = (budgets.split(",") if budgets else ["k", "own"]) if name == "same" else [None]
    os.makedirs(out_dir, exist_ok=True)
    for mi in mine:
        if mi in done:
            continue
        d = dl[mi]; row = {"mol_idx": mi, "n": d.num_nodes, "seconds": 0.0}
        for b in budgets:
            seed_all(mol_seed(tag, unit, mi))              # the PLAN's per-molecule seed, before each run
            t0 = time.time(); e = ex.explain(d, b); row["seconds"] += time.time() - t0
            key = "node" if b in (None, "k") else "node_own"
            row[key] = np.asarray(e["node"], dtype=np.float64).tolist()
            if e.get("own") is not None:
                row["own"] = np.asarray(e["own"], dtype=np.float64).tolist()
            row[f"meta_{b or 'main'}"] = {k: v for k, v in e["meta"].items() if k not in ("bond_score",)}
        rows.append(row)
        torch.save(rows, f + ".tmp"); os.replace(f + ".tmp", f)
        print(f"{unit} {name} {tag} mol {mi}: N={d.num_nodes} {row['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    for k in ("unit", "expl", "level", "set", "tag", "out", "ckpt", "budgets"):
        a.add_argument(f"--{k}")
    a.add_argument("--shard", default="0/1"); a = a.parse_args()
    run(a.unit, a.expl, a.level, a.set, a.tag, a.out, a.shard, a.ckpt, a.budgets)
