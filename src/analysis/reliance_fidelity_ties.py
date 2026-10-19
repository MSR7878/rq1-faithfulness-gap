"""
EXPLORATORY (amendment and addendum of 2026-10-19): how tie-dependent are the
stored fidelity values that do not reproduce?

Cells: SubgraphX on the two single-split runs (MUTAG, BBBP; all molecules),
plus every other (unit, explainer) with accepted boundary-tie mismatches in the
validation pass (runs/rq1_reliance_fidelity/validation/*.json; those molecules
only). For each, the top-k selection is recomputed under 5 seeded random tie
orders -- a random rank WITHIN exact ties only, so distinct importance values
keep their order -- and Fid+ / Fid- (probability and logit margin, R1/R2/R3)
are computed per order. GEA is added for MUTAG only if its graphs are verified
identical to MUTAG-GT's; BBBP has no ground truth. Writes
runs/rq1_reliance_fidelity/ties/<unit>_<explainer>_ties.npz; stored values are
not touched.

    python -m src.analysis.reliance_fidelity_ties
"""
from __future__ import annotations

import glob
import json
import os
import zlib

import numpy as np
import torch

from ..data import LOADERS
from ..explain.common import to_cpu_data
from ..metrics.gea import compute_gea
from .reliance_fidelity import FILL_KEY, K_FRAC, MASKINGS, OUT, all_units, load_unit, logits_of, margin, masked_pair, prob

N_ORDERS = 5
FULL_CELLS = [("mutag", "subgraphx"), ("bbbp", "subgraphx")]


def tie_order(imp: torch.Tensor, rng) -> torch.Tensor:
    """Scores with the same order as imp between distinct values and a random order within exact ties."""
    v = imp.double().numpy(); order = np.lexsort((rng.random(len(v)), v))
    score = np.empty(len(v)); score[order] = np.arange(len(v))
    return torch.tensor(score, dtype=torch.float)


def cells() -> dict:
    """(unit, explainer) -> list of positions (None = all molecules)."""
    c = {k: None for k in FULL_CELLS}
    for f in sorted(glob.glob(f"{OUT}/validation/*.json")):
        v = json.load(open(f))
        for r in v["mismatches"]:
            key = (r["unit"], r["explainer"])
            if key in FULL_CELLS:
                continue
            c.setdefault(key, [])
            if r["position"] not in c[key]:
                c[key].append(r["position"])
    return c


def main() -> int:
    os.makedirs(f"{OUT}/ties", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = all_units()
    for (name, expl), positions in cells().items():
        model, ck, blob, out, dl, meta = load_unit(name, U[name], device)
        test_idx = list(blob["key"]["test_idx"]); res = blob["results"][expl]
        fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float); pm = out["per_molecule"]
        positions = list(range(len(test_idx))) if positions is None else sorted(positions)
        gt_src = None
        if name == "mutag":
            dg, _ = LOADERS["mutag_graphxai"]()
            same = all(torch.equal(dl[i].x.float(), dg[i].x.float()) and torch.equal(dl[i].edge_index, dg[i].edge_index) for i in test_idx)
            gt_src = dg if same else None
            print(f"{name}: graphs identical to MUTAG-GT for all {len(test_idx)} explained molecules: {same}")
        rng = np.random.default_rng([61000, zlib.crc32(f"{name}|{expl}".encode())])
        P = len(positions)
        A = {"positions": np.array(positions), "mol_idx": np.array([test_idx[q] for q in positions])}
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                for sp in ("prob", "logit"):
                    A[f"{mk}_{met}_{sp}"] = np.zeros((P, N_ORDERS)); A[f"{mk}_{met}_{sp}_current"] = np.zeros(P)
                A[f"{mk}_{met}_prob_stored"] = np.zeros(P)
        A["gea_topk"] = np.full((P, N_ORDERS), np.nan); A["gea_topk_current"] = np.full(P, np.nan)
        A["n_distinct"] = np.zeros(P, int); A["budget"] = np.zeros(P, int)
        for j, pos in enumerate(positions):
            mi = test_idx[pos]; G = to_cpu_data(dl[mi]); n = G.num_nodes; k = max(1, int(K_FRAC * n))
            imp = torch.as_tensor(res[pos]["node_importance"], dtype=torch.float); tgt = int(res[pos]["target"])
            A["n_distinct"][j] = int(imp.unique().numel()); A["budget"][j] = k
            orders = [imp] + [tie_order(imp, rng) for _ in range(N_ORDERS)]
            graphs, idx = [G], {}
            for o_i, o in enumerate(orders):
                for mk in MASKINGS:
                    E, GmE = masked_pair(G, o, fill, mk); idx[(o_i, mk)] = len(graphs); graphs += [GmE, E]
            lg = logits_of(model, graphs, device); m, p = margin(lg, tgt), prob(lg, tgt)
            for o_i in range(len(orders)):
                for mk in MASKINGS:
                    i = idx[(o_i, mk)]
                    vals = {"fid_plus": (p[0] - p[i], m[0] - m[i]), "fid_minus": (p[0] - p[i + 1], m[0] - m[i + 1])}
                    for met, (vp, vl) in vals.items():
                        if o_i == 0:
                            A[f"{mk}_{met}_prob_current"][j] = vp; A[f"{mk}_{met}_logit_current"][j] = vl
                            A[f"{mk}_{met}_prob_stored"][j] = pm[mk][FILL_KEY[mk]][expl][pos][met]
                        else:
                            A[f"{mk}_{met}_prob"][j, o_i - 1] = vp; A[f"{mk}_{met}_logit"][j, o_i - 1] = vl
            if gt_src is not None:
                gt = gt_src[mi].node_gt_mask.cpu().bool()
                for o_i, o in enumerate(orders):
                    sel = torch.zeros(n, dtype=torch.bool); sel[torch.topk(o, k).indices] = True
                    g = compute_gea(gt, sel).gea
                    if o_i == 0:
                        A["gea_topk_current"][j] = g
                    else:
                        A["gea_topk"][j, o_i - 1] = g
        np.savez(f"{OUT}/ties/{name}_{expl}_ties.npz", unit=name, explainer=expl, n_orders=N_ORDERS,
                 all_molecules=(name, expl) in FULL_CELLS, gt_from_mutag_graphxai=gt_src is not None, **A)
        print(f"{name} {expl}: {P} molecules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
