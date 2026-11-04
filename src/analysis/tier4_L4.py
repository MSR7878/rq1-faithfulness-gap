"""
RQ1 Tier 4, item L4: what the MUTAG-GT model uses (runs/rq1_tier4/PLAN.md). Scaffold = all atoms on an aromatic bond
(bond-type column 0). Per explained molecule with 0 < |scaffold| < N:
  necessity  (scaffold hidden) and sufficiency (only the scaffold kept) under R1, R2, R3 (Part-G metrics: Fid+, Fid-, GEF,
             probability and logit), vs 20 random-node and 20 random-connected sets of the same size;
  overlap    share of chosen atoms that are aromatic: SubgraphX fixed-budget coalition, GNNExplainer top-k, IG top-k
             (random ties, 20 orders), vs 20 random-connected sets of the same size.

    python -m src.analysis.tier4_L4 --unit U        -> runs/rq1_tier4/L4/<unit>.npz
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import torch

from . import reliance_fidelity as rf
from . import tier2_common as C
from . import tier4_common as T
from .connectivity_baseline import build_adj, random_connected_subset
from .revision_pass import KEYS, RULES, all_specs, hide, metrics


def aromatic(d) -> list[int]:
    ea = d.edge_attr.detach().cpu().numpy(); ei = d.edge_index.detach().cpu().numpy()
    return sorted({int(v) for v in ei[:, ea[:, 0] > 0].ravel()})


def run(unit, device):
    dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]
    model, _ = rf.load_model(C.orig_spec(unit)["ckpt"], device)
    fill = torch.tensor(json.load(open(all_specs()[unit]["out"]))["fills"]["R3"]["mean"], dtype=torch.float)
    te = {int(i) for i in torch.load(C.orig_spec(unit)["ckpt"], map_location="cpu", weights_only=False)["split"]["test"]}
    gnn = C.stored_scores(unit, "gnnexplainer"); ig = C.npz_scores(f"{C.T2}/G/grad/{unit}.npz", "ig"); sx = C.sx_shard_rows(f"{C.T2}/S/sx/{unit}")
    R = {"mol_idx": [], "is_test": [], "n": [], "scaf_size": [], "pred_class": [], "skipped": []}
    for mi in ti:
        d = dl[mi]; G = rf.to_cpu_data(d); n = G.num_nodes; S = aromatic(d)
        if not 0 < len(S) < n:
            R["skipped"].append(mi); continue
        rng = np.random.default_rng([86000, zlib.crc32(unit.encode()), mi]); adj = build_adj(G.edge_index, n)
        sets = [("scaf", S)] + [("node", rng.choice(n, size=len(S), replace=False).tolist()) for _ in range(C.N_DRAWS)] \
            + [("conn", sorted(random_connected_subset(adj, n, len(S), rng))) for _ in range(C.N_DRAWS)]
        graphs, idx = [G], []
        for ref, M in sets:
            comp = [v for v in range(n) if v not in set(M)]
            for rule in RULES:
                idx.append((ref, rule, len(graphs), len(graphs) + 1)); graphs += [hide(G, M, rule, fill), hide(G, comp, rule, fill)]
        lg = rf.logits_of(model, graphs, device); c = int(lg[0].argmax()); acc = {}
        for ref, rule, ir, ik in idx:
            for k, v in metrics(lg, c, ir, ik).items():
                acc.setdefault((ref, rule, k), []).append(v)
        R["mol_idx"].append(mi); R["is_test"].append(mi in te); R["n"].append(n); R["scaf_size"].append(len(S)); R["pred_class"].append(c)
        for (ref, rule, k), v in acc.items():
            R.setdefault(f"{ref}_{rule}_{k}", []).append(float(np.mean(v)))
        # overlap with the aromatic atoms
        arom = np.zeros(n, bool); arom[S] = True
        share = lambda mask: float((mask & arom).sum() / max(1, mask.sum()))
        k = C.kbudget(n); co = [int(v) for v in sx[mi]["coalition"]]
        conn_share = lambda size: float(np.mean([arom[list(random_connected_subset(adj, n, size, rng))].mean() for _ in range(C.N_DRAWS)])) if size else np.nan
        R.setdefault("ov_sx", []).append(arom[co].mean() if co else np.nan); R.setdefault("ov_sx_conn", []).append(conn_share(len(co))); R.setdefault("sx_size", []).append(len(co))
        R.setdefault("ov_gnn", []).append(C.gea_random_ties(gnn[mi], k, share, f"{unit}|gnnexplainer|L4", mi))
        R.setdefault("ov_ig", []).append(C.gea_random_ties(ig[mi], k, share, f"{unit}|ig|L4", mi))
        R.setdefault("ov_topk_conn", []).append(conn_share(k))
    out = {k: np.array(v) for k, v in R.items()}
    C.save_npz(f"{T.T4}/L4/{unit}.npz", unit=unit, **out)
    print(f"{unit}: scaffold test on {len(R['mol_idx'])} molecules, {len(R['skipped'])} skipped", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit", required=True)
    run(a.parse_args().unit, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
