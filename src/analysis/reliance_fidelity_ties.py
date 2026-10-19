"""
EXPLORATORY (amendment of 2026-10-19): how tie-dependent are the stored
SubgraphX fidelity values of the two single-split runs (MUTAG, BBBP)?

SubgraphX importances are 0/1, so the top-k selection is decided by tie order.
Here the selection is recomputed under 5 seeded random tie-break orders
(importance + 1e-3 * U(0,1): keeps every 1 above every 0, randomises within
ties) and Fid+ / Fid- (probability and logit margin, R1/R2/R3) are computed for
each order. GEA is added for MUTAG only if its graphs are verified identical to
MUTAG-GT's (same x and edge_index for every explained molecule); BBBP has no
ground truth. Writes runs/rq1_reliance_fidelity/ties/<unit>_subgraphx_ties.npz;
the stored values are not touched.

    python -m src.analysis.reliance_fidelity_ties
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
import torch

from ..data import LOADERS
from ..explain.common import to_cpu_data
from ..metrics.gea import compute_gea
from .reliance_fidelity import FILL_KEY, MASKINGS, OUT, all_units, load_unit, logits_of, margin, masked_pair, prob

N_ORDERS = 5
UNITS = ["mutag", "bbbp"]


def main() -> int:
    os.makedirs(f"{OUT}/ties", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = all_units()
    for name in UNITS:
        model, ck, blob, out, dl, meta = load_unit(name, U[name], device)
        test_idx = list(blob["key"]["test_idx"]); res = blob["results"]["subgraphx"]
        fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
        pm = out["per_molecule"]
        gt_src = None
        if name == "mutag":
            dg, _ = LOADERS["mutag_graphxai"]()
            same = all(torch.equal(dl[i].x.float(), dg[i].x.float()) and torch.equal(dl[i].edge_index, dg[i].edge_index) for i in test_idx)
            gt_src = dg if same else None
            print(f"{name}: graphs identical to MUTAG-GT for all {len(test_idx)} explained molecules: {same}")
        rng = np.random.default_rng([61000, zlib.crc32(name.encode())])
        A = {"mol_idx": np.array(test_idx)}
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                for sp in ("prob", "logit"):
                    A[f"{mk}_{met}_{sp}"] = np.zeros((len(test_idx), N_ORDERS))
                A[f"{mk}_{met}_prob_stored"] = np.zeros(len(test_idx))
                A[f"{mk}_{met}_prob_current"] = np.zeros(len(test_idx))
        A["gea_topk"] = np.full((len(test_idx), N_ORDERS), np.nan)
        A["gea_topk_current"] = np.full(len(test_idx), np.nan)
        A["coalition_size"] = np.zeros(len(test_idx), int); A["budget"] = np.zeros(len(test_idx), int)
        for pos, mi in enumerate(test_idx):
            G = to_cpu_data(dl[mi]); n = G.num_nodes; k = max(1, int(0.25 * n))
            imp = torch.as_tensor(res[pos]["node_importance"], dtype=torch.float); tgt = int(res[pos]["target"])
            A["coalition_size"][pos] = int((imp > 0).sum()); A["budget"][pos] = k
            orders = [imp] + [imp + 1e-3 * torch.tensor(rng.random(n), dtype=torch.float) for _ in range(N_ORDERS)]
            graphs, idx = [G], {}
            for j, o in enumerate(orders):
                for mk in MASKINGS:
                    E, GmE = masked_pair(G, o, fill, mk); idx[(j, mk)] = len(graphs); graphs += [GmE, E]
            lg = logits_of(model, graphs, device); m, p = margin(lg, tgt), prob(lg, tgt)
            for j in range(len(orders)):
                for mk in MASKINGS:
                    i = idx[(j, mk)]
                    vals = {"fid_plus": (p[0] - p[i], m[0] - m[i]), "fid_minus": (p[0] - p[i + 1], m[0] - m[i + 1])}
                    for met, (vp, vl) in vals.items():
                        if j == 0:
                            A[f"{mk}_{met}_prob_current"][pos] = vp
                            A[f"{mk}_{met}_prob_stored"][pos] = pm[mk][FILL_KEY[mk]]["subgraphx"][pos][met]
                        else:
                            A[f"{mk}_{met}_prob"][pos, j - 1] = vp; A[f"{mk}_{met}_logit"][pos, j - 1] = vl
            if gt_src is not None:
                gt = gt_src[mi].node_gt_mask.cpu().bool()
                for j, o in enumerate(orders):
                    sel = torch.zeros(n, dtype=torch.bool); sel[torch.topk(o, k).indices] = True
                    g = compute_gea(gt, sel).gea
                    if j == 0:
                        A["gea_topk_current"][pos] = g
                    else:
                        A["gea_topk"][pos, j - 1] = g
        np.savez(f"{OUT}/ties/{name}_subgraphx_ties.npz", unit=name, n_orders=N_ORDERS, gt_from_mutag_graphxai=gt_src is not None, **A)
        print(f"{name}: {len(test_idx)} molecules, coalition != budget in {int((A['coalition_size'] != A['budget']).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
