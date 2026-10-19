"""
RQ1 follow-ups, Phase 2: deterministic tie rule (runs/rq1_followups/PLAN.md).

Rule: one total order of a molecule's atoms by (score descending, atom index
ascending), numpy stable sort. E = first k atoms; the explanation-removed graph
keeps the last max(1, int(0.75 N)) atoms of the same order. Implemented by
turning the order into strictly decreasing scores and calling the unchanged
masking code, so torch never breaks a tie.

1. Every Part B unit (107), every molecule, explainer and masking: probability
   and logit Fid+/Fid- under the rule. Outside the four tie cells the
   probability values are compared with the stored ones (P4, tolerance 1e-5).
2. The four tie cells (MUTAG/BBBP x SubgraphX/PGExplainer): stored, current
   torch (runs/rq1_reliance_fidelity/partB), deterministic rule, and the range
   over the 5 random tie orders (runs/rq1_reliance_fidelity/ties), plus MUTAG
   GEA with the MUTAG-GT mask (graphs verified identical).
Writes only runs/rq1_followups/phase2/. Stored files are read, never written.

    python -m src.analysis.followups_phase2 --units ...   # worker
    python -m src.analysis.followups_phase2 report
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from ..data import LOADERS
from ..explain.common import to_cpu_data
from ..metrics.gea import compute_gea
from .reliance_fidelity import EXPL, FILL_KEY, K_FRAC, MASKINGS, all_units, load_unit, logits_of, margin, masked_pair, prob

OUT = "runs/rq1_followups/phase2"
TOL = 1e-5
TIE_CELLS = [("mutag", "subgraphx"), ("bbbp", "subgraphx"), ("mutag", "pgexplainer"), ("bbbp", "pgexplainer")]
RF = "runs/rq1_reliance_fidelity"


def det_scores(imp) -> torch.Tensor:
    """Strictly decreasing scores following the order (score desc, atom index asc), numpy stable sort."""
    v = np.asarray(imp, dtype=np.float64); n = len(v)
    order = np.argsort(-v, kind="stable")          # stable: equal scores keep ascending atom index
    s = np.empty(n); s[order] = n - np.arange(n)
    return torch.tensor(s, dtype=torch.float)


def run_unit(name, spec, device):
    model, ck, blob, out, dl, meta = load_unit(name, spec, device)
    test_idx = list(blob["key"]["test_idx"]); res = blob["results"]; pm = out["per_molecule"]
    fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    A = {"mol_idx": np.array(test_idx)}
    for e in EXPL:
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                for sp in ("prob", "logit"):
                    A[f"{e}_{mk}_{met}_{sp}_det"] = np.zeros(len(test_idx))
                A[f"{e}_{mk}_{met}_prob_stored"] = np.zeros(len(test_idx))
    for pos, mi in enumerate(test_idx):
        G = to_cpu_data(dl[mi]); graphs, idx = [G], {}
        for mk in MASKINGS:
            for e in EXPL:
                E, GmE = masked_pair(G, det_scores(res[e][pos]["node_importance"]), fill, mk)
                idx[(e, mk)] = len(graphs); graphs += [GmE, E]
        lg = logits_of(model, graphs, device)
        for e in EXPL:
            tgt = int(res[e][pos]["target"]); m, p = margin(lg, tgt), prob(lg, tgt)
            for mk in MASKINGS:
                i = idx[(e, mk)]
                for met, j in (("fid_plus", i), ("fid_minus", i + 1)):
                    A[f"{e}_{mk}_{met}_prob_det"][pos] = p[0] - p[j]; A[f"{e}_{mk}_{met}_logit_det"][pos] = m[0] - m[j]
                    A[f"{e}_{mk}_{met}_prob_stored"][pos] = pm[mk][FILL_KEY[mk]][e][pos][met]
    dev = {}
    for e in EXPL:
        if (name, e) in TIE_CELLS:
            continue
        d = max(float(np.max(np.abs(A[f"{e}_{mk}_{met}_prob_det"] - A[f"{e}_{mk}_{met}_prob_stored"])))
                for mk in MASKINGS for met in ("fid_plus", "fid_minus"))
        dev[e] = d
    A["max_dev_nontie"] = np.array(json.dumps(dev))
    if name == "mutag":                            # GEA with the MUTAG-GT mask (graphs verified identical)
        dg, _ = LOADERS["mutag_graphxai"]()
        same = all(torch.equal(dl[i].x.float(), dg[i].x.float()) and torch.equal(dl[i].edge_index, dg[i].edge_index) for i in test_idx)
        A["graphs_identical_to_mutag_gt"] = np.array(same)
        if same:
            for e in EXPL:
                gd, gc = [], []
                for pos, mi in enumerate(test_idx):
                    n = dg[mi].num_nodes; k = max(1, int(K_FRAC * n)); gt = dg[mi].node_gt_mask.cpu().bool()
                    for sc, acc in ((det_scores(res[e][pos]["node_importance"]), gd),
                                    (torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float), gc)):
                        sel = torch.zeros(n, dtype=torch.bool); sel[torch.topk(sc, k).indices] = True
                        acc.append(compute_gea(gt, sel).gea)
                A[f"{e}_gea_topk_det"] = np.array(gd); A[f"{e}_gea_topk_current"] = np.array(gc)
    return A


def report():
    P = print
    files = sorted(glob.glob(f"{OUT}/units/*.npz"))
    U = {os.path.basename(f)[:-4]: dict(np.load(f, allow_pickle=True)) for f in files}
    P("=" * 120); P(" PHASE 2 -- deterministic tie rule (score desc, atom index asc; numpy stable sort)"); P("=" * 120)
    # ---- P4
    rows = [(u, e, d) for u, a in U.items() for e, d in json.loads(str(a["max_dev_nontie"])).items()]
    worst = max(rows, key=lambda r: r[2])
    over = [r for r in rows if r[2] > TOL]
    P(f"\n[P4] units {len(U)}/107; non-tie cells checked {len(rows)} (unit x explainer); max |det - stored| {worst[2]:.2e} ({worst[0]}, {worst[1]})")
    P(f"     cells over {TOL}: {len(over)}" + ("".join(f"\n       {u:<28}{e:<14}{d:.3e}" for u, e, d in sorted(over, key=lambda r: -r[2])[:40])))
    held = len(U) == 107 and not over
    P(f"     (P4) all non-tie cells identical to stored within {TOL}: {'HELD' if held else 'DID NOT HOLD'}")
    rep = {"P4": {"held": held, "n_units": len(U), "max_dev": worst[2], "over": over}}
    # ---- tie cells
    P("\n[T] the four tie cells: mean over ALL molecules of the unit (n = 29 MUTAG, 30 BBBP)")
    P(f"  {'cell':<20}{'metric':<14}{'stored':>9}{'torch now':>11}{'det rule':>10}   5 random orders [min, max]   | logit: torch now  det rule")
    rep["tie_cells"] = {}
    for unit, e in TIE_CELLS:
        a = U[unit]; cur = dict(np.load(f"{RF}/partB/{unit}.npz", allow_pickle=True))
        tz = dict(np.load(f"{RF}/ties/{unit}_{e}_ties.npz", allow_pickle=True)); pos = tz["positions"]
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                st = a[f"{e}_{mk}_{met}_prob_stored"]; now = cur[f"{e}_{mk}_{met}_prob"]; det = a[f"{e}_{mk}_{met}_prob_det"]
                nowl, detl = cur[f"{e}_{mk}_{met}_logit"], a[f"{e}_{mk}_{met}_logit_det"]
                orders = []
                for j in range(int(tz["n_orders"])):
                    v = now.copy(); v[pos] = tz[f"{mk}_{met}_prob"][:, j]; orders.append(v.mean())
                lab = f"{mk} {met.replace('fid_', 'Fid').replace('plus', '+').replace('minus', '-')}"
                P(f"  {unit + ' ' + e:<20}{lab:<14}{st.mean():>9.3f}{now.mean():>11.3f}{det.mean():>10.3f}   [{min(orders):.3f}, {max(orders):.3f}]"
                  f"{'':>16}| {nowl.mean():>8.3f}  {detl.mean():>8.3f}")
                rep["tie_cells"][f"{unit}|{e}|{mk}|{met}"] = {"stored": float(st.mean()), "torch_now": float(now.mean()), "det": float(det.mean()),
                                                              "orders_min": float(min(orders)), "orders_max": float(max(orders)),
                                                              "logit_torch_now": float(nowl.mean()), "logit_det": float(detl.mean())}
    m = U["mutag"]
    if bool(m.get("graphs_identical_to_mutag_gt", False)):
        P("\n  MUTAG GEA top-k (MUTAG-GT mask; graphs identical: True), mean over 29 molecules:")
        for e in EXPL:
            P(f"    {e:<13} torch now {m[f'{e}_gea_topk_current'].mean():.3f}   det rule {m[f'{e}_gea_topk_det'].mean():.3f}")
            rep["tie_cells"][f"mutag|{e}|gea"] = {"torch_now": float(m[f"{e}_gea_topk_current"].mean()), "det": float(m[f"{e}_gea_topk_det"].mean())}
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    P(f"\nsaved -> {OUT}/report.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--units", nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "report":
        return report()
    os.makedirs(f"{OUT}/units", exist_ok=True)
    U = all_units(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in args.units:
        dst = f"{OUT}/units/{name}.npz"
        if os.path.exists(dst):
            continue
        A = run_unit(name, U[name], device); np.savez(dst, unit=name, **A)
        print(f"{name}: {len(A['mol_idx'])} molecules, non-tie max dev {A['max_dev_nontie']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
