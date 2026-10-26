"""
RQ1 revision: one re-scoring pass that feeds T1, T5 and T2 (runs/rq1_revision/PLAN.md).
No training, no new explanations.

Part G (ground-truth units: MUTAG-GT 10 seeds, B-XAIC x4, Mutagenicity D-MPNN/GIN/GINE/GCN):
  per GT-bearing explained molecule and per GT mask S (Part A's mask definition),
  for M in {S, 20 random-node sets of |S|, 20 random-connected sets of |S|} and each rule
  R1/R2/R3: removed graph (M hidden) and kept graph (everything but M hidden);
  Fid+ = s(G) - s(removed), Fid- = s(G) - s(kept), GEF = 1 - exp(-KL(p(kept) || p(G))),
  s = probability of the predicted class or the logit margin. Draws scored one by one,
  then averaged; values averaged over the molecule's masks.
Part E (every unit with cached explanations, 112 units):
  per explained molecule, each explainer's top-k under random tie-breaking (20 orders,
  Phase 2b stream "<unit>|<explainer>|fid"; one order if no tie at either boundary) and
  20 random-node rankings of the same structure; Fid+, Fid-, GEF under R1/R2/R3, both spaces,
  averaged over orders / draws.
Writes runs/rq1_revision/pass/{G,E}/<unit>.npz.

    python -m src.analysis.revision_pass list
    python -m src.analysis.revision_pass run --part G|E --units ...
"""
from __future__ import annotations

import argparse
import os
import time
import zlib

import numpy as np
import torch
from torch_geometric.data import Data

from . import reliance_fidelity as rf
from .connectivity_baseline import build_adj, random_connected_subset
from .followups_phase2b import fid_units
from .ties_random import N_ORDERS, boundary_tie, mol_rng, removed_boundary_tie, tie_order

OUT = "runs/rq1_revision/pass"
RULES = ["R1", "R2", "R3"]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
N_DRAWS = 20


def unit_rng(name, stream):
    return np.random.default_rng([70000, zlib.crc32(name.encode()), stream])


def is_gt_unit(name):
    return name.startswith(("mutag_graphxai", "bxaic_", "mutagenicity_"))


def fill_nodes(G, idx, fill):
    x = G.x.clone().float(); x[torch.as_tensor(sorted(int(i) for i in idx), dtype=torch.long)] = fill.to(x.dtype)
    return Data(x=x, edge_index=G.edge_index, edge_attr=getattr(G, "edge_attr", None))


def hide(G, idx, rule, fill):
    """Graph with the nodes in idx hidden under the rule."""
    n = G.num_nodes; idx = set(int(i) for i in idx)
    if rule == "R1":
        return rf.keep_nodes(G, [v for v in range(n) if v not in idx])
    if rule == "R2":
        return rf.zero_nodes(G, idx)
    return fill_nodes(G, idx, fill)


def metrics(lg, c, i_removed, i_kept):
    """Fid+, Fid-, GEF in probability and logit space for one (removed, kept) pair; row 0 = full graph."""
    p = torch.softmax(lg, dim=-1); m = (lg[:, c] - lg[:, 1 - c])
    q, r = p[i_kept], p[0]
    kl = float((q * (q.clamp_min(1e-12).log() - r.clamp_min(1e-12).log())).sum())
    return {"fidp_prob": float(p[0, c] - p[i_removed, c]), "fidm_prob": float(p[0, c] - p[i_kept, c]),
            "fidp_logit": float(m[0] - m[i_removed]), "fidm_logit": float(m[0] - m[i_kept]), "gef": 1.0 - float(np.exp(-kl))}


KEYS = ["fidp_prob", "fidm_prob", "fidp_logit", "fidm_logit", "gef"]


# ---------------------------------------------------------------- Part G
def run_G(name, spec, device):
    model, ck, blob, out, dl, meta = rf.load_unit(name, spec, device)
    motifs_all = meta.extras.get("motifs") if hasattr(meta, "extras") else None
    fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    rng_n, rng_c = unit_rng(name + ":G", 0), unit_rng(name + ":G", 1)
    R = {"mol_idx": [], "n_masks": [], "n_skipped": [], "mask_size": [], "pred_class": []}
    for ref in ("gt", "node", "conn"):
        for rule in RULES:
            for k in KEYS:
                R[f"{ref}_{rule}_{k}"] = []
    for mi in blob["key"]["test_idx"]:
        d = dl[mi]
        if not bool(d.node_gt_mask.any()):
            continue
        G = rf.to_cpu_data(d); n = G.num_nodes; adj = build_adj(G.edge_index, n)
        masks = [S for S in rf.gt_masks(name, d, mi, motifs_all) if len(S) < n]
        skipped = len(rf.gt_masks(name, d, mi, motifs_all)) - len(masks)
        if not masks:
            R["n_skipped"].append(skipped); R["mol_idx"].append(mi); R["n_masks"].append(0); R["mask_size"].append(np.nan); R["pred_class"].append(-1)
            for k in R:
                if k not in ("mol_idx", "n_masks", "n_skipped", "mask_size", "pred_class"):
                    R[k].append(np.nan)
            continue
        sets = []                                     # (mask index, ref, node set)
        for j, S in enumerate(masks):
            sets.append((j, "gt", list(S)))
            for _ in range(N_DRAWS):
                sets.append((j, "node", rng_n.choice(n, size=len(S), replace=False).tolist()))
            for _ in range(N_DRAWS):
                sets.append((j, "conn", sorted(random_connected_subset(adj, n, len(S), rng_c))))
        graphs, idx = [G], []
        for j, ref, M in sets:
            comp = [v for v in range(n) if v not in set(M)]
            for rule in RULES:
                idx.append((j, ref, rule, len(graphs), len(graphs) + 1))
                graphs += [hide(G, M, rule, fill), hide(G, comp, rule, fill)]
        lg = rf.logits_of(model, graphs, device); c = int(lg[0].argmax())
        acc = {}
        for j, ref, rule, ir, ik in idx:
            for k, v in metrics(lg, c, ir, ik).items():
                acc.setdefault((j, ref, rule, k), []).append(v)
        R["mol_idx"].append(mi); R["n_masks"].append(len(masks)); R["n_skipped"].append(skipped)
        R["mask_size"].append(float(np.mean([len(S) for S in masks]))); R["pred_class"].append(c)
        for ref in ("gt", "node", "conn"):
            for rule in RULES:
                for k in KEYS:
                    R[f"{ref}_{rule}_{k}"].append(float(np.mean([np.mean(acc[(j, ref, rule, k)]) for j in range(len(masks))])))
    return {k: np.array(v) for k, v in R.items()}


# ---------------------------------------------------------------- Part E
def run_E(name, spec, device):
    model, ck, blob, out, dl, meta = rf.load_unit(name, spec, device)
    res = blob["results"]; fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    rng_n = unit_rng(name + ":E", 0)
    R = {"mol_idx": [], "n_orders_pg": [], "n_orders_sx": []}
    for who in EXPL + ["randnode"]:
        for rule in RULES:
            for k in KEYS:
                R[f"{who}_{rule}_{k}"] = []
    for pos, mi in enumerate(blob["key"]["test_idx"]):
        G = rf.to_cpu_data(dl[mi]); n = G.num_nodes
        k_, keep = max(1, int(rf.K_FRAC * n)), max(1, int((1 - rf.K_FRAC) * n))
        tgt = int(res[EXPL[0]][pos]["target"])
        rankings = {}
        for e in EXPL:
            imp = res[e][pos]["node_importance"]
            if boundary_tie(imp, k_) or removed_boundary_tie(imp, keep):
                r = mol_rng(f"{name}|{e}|fid", pos); rankings[e] = [tie_order(imp, r) for _ in range(N_ORDERS)]
            else:
                rankings[e] = [torch.as_tensor(imp, dtype=torch.float)]
        rankings["randnode"] = [torch.tensor(rng_n.random(n), dtype=torch.float) for _ in range(N_DRAWS)]
        graphs, idx = [G], []
        for who, rs in rankings.items():
            for r in rs:
                for rule in RULES:
                    E, GmE = rf.masked_pair(G, r, fill, rule)
                    idx.append((who, rule, len(graphs), len(graphs) + 1)); graphs += [GmE, E]
        lg = rf.logits_of(model, graphs, device)
        acc = {}
        for who, rule, ir, ik in idx:
            for k, v in metrics(lg, tgt, ir, ik).items():
                acc.setdefault((who, rule, k), []).append(v)
        R["mol_idx"].append(mi); R["n_orders_pg"].append(len(rankings["pgexplainer"])); R["n_orders_sx"].append(len(rankings["subgraphx"]))
        for who in EXPL + ["randnode"]:
            for rule in RULES:
                for k in KEYS:
                    R[f"{who}_{rule}_{k}"].append(float(np.mean(acc[(who, rule, k)])))
    return {k: np.array(v) for k, v in R.items()}


def all_specs():
    return fid_units()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--part", choices=["G", "E"])
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    U = all_specs()
    if a.cmd == "list":
        for n in U:
            print(n, "GE" if is_gt_unit(n) else "E")
        return 0
    os.makedirs(f"{OUT}/{a.part}", exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n in a.units:
        dst = f"{OUT}/{a.part}/{n}.npz"
        if os.path.exists(dst):
            continue
        t0 = time.time()
        A = (run_G if a.part == "G" else run_E)(n, U[n], device)
        np.savez(dst, unit=n, **A)
        print(f"{n}: part {a.part}, {len(A['mol_idx'])} molecules, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
