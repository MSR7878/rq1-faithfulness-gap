"""
RQ1 revision, T4 parts that need no training (runs/rq1_revision/PLAN.md):
  1. collapse recount for every unit set under the new strict rule
     (degenerate = node-importance std < 1e-6 in float64; collapsed if >= 50%),
     next to the old rule (float32 std == 0), and collapse by predicted class;
  2. DIG PGExplainer feasibility (get_emb present? does DIG's mask injection change the output?);
  3. how PGExplainer gets edge embeddings from D-MPNN, GIN, GINE, GCN (verified on one graph).
Writes runs/rq1_revision/t4_static.json and prints a report.

    python -m src.analysis.revision_t4_static
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_embeddings

from ..data import LOADERS
from ..data.standard import TOX21_TASKS
from ..explain.run_phase3 import load_model

OUT = "runs/rq1_revision"
STD_EPS = 1e-6


def unit_sets():
    S = {"Tox21 (priority sweep)": [f"runs/prio/cache_tox21_{ep}_s{s}.pt" for ep in TOX21_TASKS for s in range(5)],
         "B-XAIC (priority sweep)": [f"runs/prio/cache_bxaic_{t}_s{s}.pt" for t in ("indole", "PAINS", "X", "P") for s in range(5)],
         "MUTAG-GT seeds 0-4 (priority sweep)": [f"runs/prio/cache_mutag_graphxai_s{s}.pt" for s in range(5)],
         "MUTAG-GT seeds 5-9": [f"runs/rq1_followups/phase1/cache_mutag_graphxai_s{s}.pt" for s in range(5, 10)],
         "MUTAG-GT decoupled": [f"runs/decoupled/cache_mutag_graphxai_s{s}.pt" for s in range(5)],
         "Mutagenicity D-MPNN 5-seed": [f"runs/mutagenicity/cache_s{s}.pt" for s in range(5)],
         "Mutagenicity 5-fold CV": [f"runs/mutagenicity_cv/cache_s{s}.pt" for s in range(5)],
         "Mutagenicity GIN": [f"runs/arch_gin/mutagenicity_graphxai/cache_s{s}.pt" for s in range(5)],
         "Mutagenicity GINE": [f"runs/arch_gine/mutagenicity_graphxai/cache_s{s}.pt" for s in range(5)],
         "Mutagenicity GCN": [f"runs/arch_gcn/mutagenicity_graphxai/cache_s{s}.pt" for s in range(5)],
         "F10 mechanism (Tox21)": sorted(glob.glob("runs/f10mech/cache_*.pt")),
         "MUTAG single split": ["runs/expl_cache_mutag.pt"], "BBBP single split": ["runs/expl_cache_bbbp.pt"]}
    return S


def recount():
    P = print; rep = {}
    P("[1] PGExplainer collapse: old rule (float32 std == 0) vs new rule (float64 std < 1e-6); collapsed if >= 50% degenerate")
    P(f"  {'unit set':<38}{'units':>6}{'old collapsed':>15}{'new collapsed':>15}{'units that change':>20}")
    per_unit = {}
    for name, files in unit_sets().items():
        oc = nc = 0; changed = []
        for f in files:
            res = torch.load(f, weights_only=False)["results"]["pgexplainer"]
            s32 = np.array([float(torch.as_tensor(r["node_importance"], dtype=torch.float).std()) for r in res])
            s64 = np.array([np.std(np.asarray(r["node_importance"], dtype=np.float64), ddof=1) if len(r["node_importance"]) > 1 else 0.0 for r in res])
            tg = np.array([int(r["target"]) for r in res])
            d_old, d_new = s32 == 0, s64 < STD_EPS
            c_old, c_new = d_old.mean() >= 0.5, d_new.mean() >= 0.5
            oc += c_old; nc += c_new
            if c_old != c_new:
                changed.append(f"{os.path.basename(f)} ({int(d_old.sum())}->{int(d_new.sum())}/{len(res)})")
            per_unit[f] = {"set": name, "n": len(res), "deg_old": int(d_old.sum()), "deg_new": int(d_new.sum()), "collapsed_old": bool(c_old),
                           "collapsed_new": bool(c_new),
                           "by_class": {int(c): {"n": int((tg == c).sum()), "degenerate_new": int((d_new & (tg == c)).sum())} for c in np.unique(tg)}}
        P(f"  {name:<38}{len(files):>6}{oc:>15}{nc:>15}   {', '.join(changed) if changed else 'none'}")
        rep[name] = {"units": len(files), "collapsed_old": int(oc), "collapsed_new": int(nc), "changed": changed}
    P("\n[2] collapse by predicted (explained) class, every unit collapsed under the new rule: degenerate / molecules per class")
    for f, u in per_unit.items():
        if u["collapsed_new"]:
            P(f"  {f:<58} " + "   ".join(f"class {c}: {v['degenerate_new']}/{v['n']}" for c, v in u["by_class"].items()))
    return rep, per_unit


def dig_test():
    P = print; out = {}
    model, ck = load_model("runs/prio/ckpt_mutag_graphxai_s0.pt", torch.device("cpu"))
    dl, _ = LOADERS["mutag_graphxai"](); d = dl[0]
    out["has_get_emb"] = hasattr(model, "get_emb")
    with torch.no_grad():
        base = model(d.x, d.edge_index, d.edge_attr)
        mps = [m for m in model.modules() if isinstance(m, MessagePassing)]
        for m in mps:                                     # DIG's injection (method/pgexplainer.py __set_masks__)
            m._explain = True
            m.__edge_mask__ = torch.nn.Parameter(torch.full((d.edge_index.size(1),), -100.0))
        dig = model(d.x, d.edge_index, d.edge_attr)
        for m in mps:
            m._explain = False; m.__edge_mask__ = None
    out["dig_mask_changes_output"] = bool(not torch.allclose(base, dig))
    out["max_abs_change"] = float((base - dig).abs().max())
    P("\n[3] DIG PGExplainer feasibility")
    P(f"  model has get_emb (DIG calls model.get_emb): {out['has_get_emb']}")
    P(f"  DIG-style mask (module._explain = True, module.__edge_mask__ = -100 on every edge) changes the D-MPNN output: "
      f"{out['dig_mask_changes_output']} (max |change| {out['max_abs_change']:.2e})")
    return out


def embeddings_doc():
    P = print; out = {}
    P("\n[4] edge embeddings used by PyG's PGExplainer: z = get_embeddings(model, x, edge_index, edge_attr=...)[-1]; "
      "edge input = cat(z[src], z[dst])")
    dl, _ = LOADERS["mutagenicity_graphxai"](); d = dl[0]
    for arch, ck in (("D-MPNN", "runs/mutagenicity/ckpt_s0.pt"), ("GIN", "runs/arch_gin/mutagenicity_graphxai/ckpt_s0.pt"),
                     ("GINE", "runs/arch_gine/mutagenicity_graphxai/ckpt_s0.pt"), ("GCN", "runs/arch_gcn/mutagenicity_graphxai/ckpt_s0.pt")):
        model, _ = load_model(ck, torch.device("cpu"))
        with torch.no_grad():
            embs = get_embeddings(model, d.x, d.edge_index, edge_attr=d.edge_attr); z = embs[-1]
            if arch == "D-MPNN":                          # final edge->node aggregation, before W_o and ReLU
                captured = {}
                h = model.aggr_edges.register_forward_hook(lambda mod, inp, o: captured.__setitem__("last", o)); model(d.x, d.edge_index, d.edge_attr); h.remove()
                same = torch.allclose(z, captured["last"]); what = "output of the last call of _EdgeToNodeAggr (sum of incoming directed-edge states), before W_o and ReLU"
            else:
                captured = {}
                h = model.convs[-1].register_forward_hook(lambda mod, inp, o: captured.__setitem__("last", o)); model(d.x, d.edge_index, d.edge_attr); h.remove()
                same = torch.allclose(z, captured["last"]); what = f"output of the last {type(model.convs[-1]).__name__}, before the outer ReLU"
        n_mp = len([m for m in model.modules() if isinstance(m, MessagePassing)])
        P(f"  {arch:<7} MessagePassing modules {n_mp}; get_embeddings returns {len(embs)} tensors; [-1] shape {tuple(z.shape)}; = {what}: {same}")
        out[arch] = {"n_mp_modules": n_mp, "n_embeddings": len(embs), "shape": list(z.shape), "is": what, "verified": bool(same)}
    return out


def main():
    rep, per_unit = recount()
    out = {"recount": rep, "per_unit": per_unit, "dig": dig_test(), "embeddings": embeddings_doc()}
    json.dump(out, open(f"{OUT}/t4_static.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
