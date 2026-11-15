"""
RQ1 Tier 5, item N0: integration check and timing (runs/rq1_tier5/PLAN.md). No GEA or Fid number is computed here.

  check   adapter and perturbation checks on every model (the checks the PLAN requires to pass) -> N0/check.json
  time    one explainer on one unit's 6 timing molecules: run time, finite / non-constant scores, perturbation
          reaches the model -> N0/time/<unit>__<expl>.json
  scale   projected N1 hours per method and level from the timings -> runs/rq1_tier5/scale.json (the PLAN's scale rule)

    python -m src.analysis.tier5_N0 check
    python -m src.analysis.tier5_N0 time --unit U --expl E
    python -m src.analysis.tier5_N0 scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

from ..explain.run_phase3 import load_model
from ..explain.t5_common import IdAdapter, edge_attr_of, load_repo, logits, mol_seed, pred_class, seed_all
from . import tier2_common as C
from . import tier5_common as T5

D0 = f"{T5.T5}/N0"
TIME_UNITS = ([f"mutagenicity_{a}_s0" for a in T5.ARCH4] + ["bxaic_indole_s0"] + [f"bxaic_indole_{a}_s0" for a in ("gin", "gine", "gcn")]
              + ["bxaic_PAINS_s0"] + [f"bxaic_PAINS_{a}_s0" for a in ("gin", "gine", "gcn")] + ["mutag_graphxai_s0"])
CHECK_UNITS = [f"mutagenicity_{a}_s0" for a in T5.ARCH4] + ["bxaic_indole_s0", "mutag_graphxai_s0"]


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timing_mols(unit) -> list[int]:
    """2 smallest, 2 median, 2 largest GT molecules of the seed-0 subset owner (same molecules for every architecture)."""
    o = T5.subset_owner(unit); dl, _ = C.load_data(C.dataset_of(o))
    g = sorted(T5.gt_mols(o), key=lambda m: (dl[m].num_nodes, m)); h = len(g) // 2
    return [g[0], g[1], g[h - 1], g[h], g[-2], g[-1]]


def p_of(model, x, ei, ea, c, device, batch=None):
    return float(torch.softmax(logits(model, x.to(device), ei.to(device), None if ea is None else ea.to(device), batch), -1)[0, c])


# ------------------------------------------------------------------------------------------------ check
def check():
    device = dev(); gx = load_repo("GStarX", ["utils"], shims=True)["utils"]; sh = load_repo("same", ["shapley"])["shapley"]
    out = {}; ok_all = True
    for u in CHECK_UNITS:
        model, ck = load_model(T5.spec(u)["ckpt"], device); dkey = C.dataset_of(u); dl, _ = C.load_data(dkey)
        d = dl[timing_mols(u)[3]]; n = d.num_nodes; ea = edge_attr_of(d); c = pred_class(model, d, device)
        A = IdAdapter(model, device).to(device); A.set_graph(d); g = A.augment(d, y=c); R = {"n": n, "bonds": int(d.edge_index.shape[1] // 2)}
        full = logits(model, d.x.float().to(device), d.edge_index.to(device), None if ea is None else ea.to(device))
        R["adapter_full_maxdiff"] = float((A(g) - full).abs().max())
        # removed-node subgraph built by hand vs the authors' graph_build_remove through the adapter
        keep = torch.ones(n); keep[torch.arange(0, n, 3)] = 0
        xr, eir = gx.graph_build_remove(g.x, g.edge_index, keep.to(device))
        ei_h, ea_h = subgraph(keep.bool(), d.edge_index, None if ea is None else ea, relabel_nodes=True, num_nodes=n)
        hand = logits(model, d.x.float()[keep.bool()].to(device), ei_h.to(device), None if ea_h is None else ea_h.to(device))
        R["adapter_removed_maxdiff"] = float((A(Data(x=xr, edge_index=eir)) - hand).abs().max())
        # zero bonds and a single atom, under remove (GStarX) and split (SAME)
        z = {}
        one = torch.zeros(n); one[0] = 1
        x1, ei1 = gx.graph_build_remove(g.x, g.edge_index, one.to(device)); z["remove_single_atom"] = A(Data(x=x1, edge_index=ei1))
        x2, ei2 = sh.graph_build_split(g.x, g.edge_index, torch.zeros(n, device=device)); z["split_no_bonds"] = A(Data(x=x2, edge_index=ei2))
        z["no_bonds_direct"] = logits(model, d.x.float().to(device), torch.zeros(2, 0, dtype=torch.long, device=device),
                                      None if ea is None else ea[:0].to(device))
        R["zero_bond_single_atom_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in z.items()}
        # batched coalition evaluation (the authors' MaskedDataset + DataLoader) vs one by one
        rng = np.random.default_rng(0); masks = torch.tensor(rng.random((8, n)) < 0.6, dtype=torch.float32); masks[:, 0] = 1
        dsm = gx.MaskedDataset(g, masks, gx.graph_build_remove)
        bt = next(iter(gx.DataLoader(dsm, batch_size=8, shuffle=False)))
        lb = A(data=bt.to(device)); lo = torch.cat([A(Data(*gx.graph_build_remove(g.x, g.edge_index, m.to(device)))) for m in masks])
        R["batched_vs_single_maxdiff"] = float((lb - lo).abs().max())
        # EiG: the authors' two-graph batch (selected atoms = graph 0) vs removing the atoms directly
        sel = torch.zeros(d.edge_index.shape[1], dtype=torch.bool); sel[: min(4, d.edge_index.shape[1])] = True
        nodes = sorted(set(d.edge_index[:, sel].flatten().tolist()))
        bvec = torch.ones(n, dtype=torch.long); bvec[nodes] = 0
        two = logits(model, d.x.float().to(device), d.edge_index[:, sel].to(device), None if ea is None else ea[sel].to(device), bvec.to(device))[0]
        from ..explain.t5_explainers import EiG
        direct = EiG(model, device)._prob_of_nodes(d, nodes, sel.numpy(), c)
        R["eig_two_graph_vs_direct"] = abs(float(torch.softmax(two, -1)[c]) - direct)
        R["pass"] = bool(R["adapter_full_maxdiff"] <= 1e-6 and R["adapter_removed_maxdiff"] <= 1e-5 and all(R["zero_bond_single_atom_finite"].values())
                         and R["batched_vs_single_maxdiff"] <= 1e-5 and R["eig_two_graph_vs_direct"] <= 1e-5)
        ok_all &= R["pass"]; out[u] = R
        print(u, json.dumps(R), flush=True)
    os.makedirs(D0, exist_ok=True); json.dump({"units": out, "all_pass": ok_all}, open(f"{D0}/check.json", "w"), indent=1)
    return ok_all


# ------------------------------------------------------------------------------------------------ time
def perturbation_reaches(name, model, d, c, device):
    """Output for the method's 'empty' input differs from the full input, under the method's own perturbation."""
    ea = edge_attr_of(d); full = p_of(model, d.x.float(), d.edge_index, ea, c, device)
    if name in ("eig", "gnnshap"):                     # all bonds removed, atoms stay
        empty = p_of(model, d.x.float(), d.edge_index[:, :0], None if ea is None else ea[:0], c, device); how = "all bonds removed"
    elif name == "gstarx":                             # node removal: the smallest admissible input is one atom
        empty = p_of(model, d.x.float()[:1], d.edge_index[:, :0], None if ea is None else ea[:0], c, device); how = "remove: one atom left"
    else:                                              # split with an empty coalition: every atom isolated
        empty = p_of(model, d.x.float(), d.edge_index[:, :0], None if ea is None else ea[:0], c, device); how = "split: no atom selected"
    return {"full": full, "empty": empty, "how": how, "differs": abs(full - empty) > 1e-6}


def time_unit(unit, name):
    device = dev(); out_f = f"{D0}/time/{unit}__{name}.json"
    if os.path.exists(out_f):
        return
    model, ck = load_model(T5.spec(unit)["ckpt"], device); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    rows = []; res = {"unit": unit, "explainer": name, "rows": rows}
    try:
        ex = T5.make_explainer(name, model, ck, dkey, device)
    except Exception as e:
        res["error"] = f"construction failed: {e!r}"; res["traceback"] = traceback.format_exc()
        os.makedirs(os.path.dirname(out_f), exist_ok=True); json.dump(res, open(out_f, "w"), indent=1); print(res["error"], flush=True); return
    budgets = ["k", "own"] if name == "same" else [None]
    for mi in timing_mols(unit):
        d = dl[mi]
        for b in budgets:
            seed_all(mol_seed("T5N0", unit, mi)); r = {"mol_idx": mi, "n": d.num_nodes, "bonds": int(d.edge_index.shape[1] // 2), "budget": b}
            try:
                t0 = time.time(); e = ex.explain(d, b); r["seconds"] = time.time() - t0
                s = np.asarray(e["node"], dtype=np.float64)
                r.update(finite=bool(np.isfinite(s).all()), nonconstant=bool(s.max() > s.min()), meta={k: v for k, v in e["meta"].items() if k not in ("bond_score",)})
                r["perturbation"] = perturbation_reaches(name, model, d, e["meta"]["target"], device)
            except Exception as ex_:
                r["error"] = repr(ex_); r["traceback"] = traceback.format_exc()
            rows.append(r); print(unit, name, mi, b, {k: r.get(k) for k in ("n", "seconds", "finite", "nonconstant", "error")}, flush=True)
    os.makedirs(os.path.dirname(out_f), exist_ok=True); json.dump(res, open(out_f + ".tmp", "w"), indent=1); os.replace(out_f + ".tmp", out_f)


# ------------------------------------------------------------------------------------------------ scale
def scale():
    """Projected N1 hours = sum over N1 molecules of the mean timing seconds for (dataset, architecture) / (12 x 3600);
    SAME counts both budgets. First level <= 24 h; otherwise L4 (flagged)."""
    sec = {}
    for name in T5.NEW:
        for u in TIME_UNITS:
            f = f"{D0}/time/{u}__{name}.json"
            if not os.path.exists(f):
                continue
            rows = [r for r in json.load(open(f))["rows"] if "seconds" in r]
            if rows:                                       # SAME: per molecule = k + own
                per = {}
                for r in rows:
                    per[r["mol_idx"]] = per.get(r["mol_idx"], 0.0) + r["seconds"]
                sec[(name, C.dataset_of(u), T5.arch_of(u))] = float(np.mean(list(per.values())))
    out = {}
    for name in T5.NEW:
        proj = {}
        for lv in T5.LEVELS:
            tot, missing = 0.0, set()
            for u in T5.n1_units():
                m = T5.level_mols(u, lv)
                if m is None:
                    continue
                key = (name, C.dataset_of(u), T5.arch_of(u))
                if key not in sec:
                    missing.add(f"{key[1]}/{key[2]}"); continue
                tot += len(m) * sec[key]
            proj[lv] = {"hours": tot / (T5.WORKERS * 3600), "cells_without_timing": sorted(missing)}
        chosen = next((lv for lv in T5.LEVELS if proj[lv]["hours"] <= T5.BUDGET_H), "L4")
        out[name] = {"level": chosen, "over_budget_even_at_L4": proj["L4"]["hours"] > T5.BUDGET_H, "projection": proj,
                     "mean_seconds": {f"{k[1]}/{k[2]}": v for k, v in sec.items() if k[0] == name}}
        print(name, chosen, {lv: round(p["hours"], 1) for lv, p in proj.items()}, flush=True)
    json.dump(out, open(f"{T5.T5}/scale.json", "w"), indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["check", "time", "scale"]); a.add_argument("--unit"); a.add_argument("--expl")
    a = a.parse_args()
    if a.cmd == "check":
        sys.exit(0 if check() else 1)
    elif a.cmd == "time":
        time_unit(a.unit, a.expl)
    else:
        scale()
