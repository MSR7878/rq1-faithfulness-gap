"""
Phase 3 -- first faithfulness-gap data point (mutag_graphxai only).

Loads the trained D-MPNN checkpoint, runs GNNExplainer / PGExplainer / SubgraphX
on the held-out test split, sanity-checks the explanations, then scores every
(explainer, molecule) pair with Fidelity+/-, GEF and GEA under the R3 masking
reference only (R2/R1 stay NotImplementedError, per the locked build order).

    python -m src.explain.run_phase3 --ckpt runs/ckpt_mutag_graphxai.pt

Outputs: a summary table to stdout + runs/phase3_mutag_graphxai.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..data import load_mutag_graphxai
from ..metrics.fidelity import compute_fidelity
from ..metrics.gef import compute_gef
from ..metrics.gea import compute_gea
from ..metrics.masking import mask_r3_distribution_aware
from ..train.dmpnn import DMPNN, DMPNNConfig
from .common import binarize_mean, to_cpu_data

K_FRAC = 0.25  # explanation sparsity: top 25% of nodes


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DMPNN(DMPNNConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def _pred_class(model, data, device) -> int:
    with torch.no_grad():
        lg = model(data.x.to(device), data.edge_index.to(device),
                   getattr(data, "edge_attr", None).to(device) if data.edge_attr is not None else None)
    return int(lg.reshape(-1, lg.shape[-1])[0].argmax())


def _masked_pair(cpu_data, node_imp, feat_means, device):
    """R3 explanation-only (E) and explanation-removed (G\\E) graphs."""
    E = mask_r3_distribution_aware(cpu_data, node_imp, feat_means, keep_top_k=K_FRAC)
    GmE = mask_r3_distribution_aware(cpu_data, -node_imp, feat_means, keep_top_k=1.0 - K_FRAC)
    return E.to(device), GmE.to(device)


def score_one(model, data, result, feat_means, device) -> dict:
    cpu_data = to_cpu_data(data)
    node_imp = result.node_importance.detach().cpu().float()
    y = result.target  # class the explanation is for (== model prediction)

    dev_data = data.to(device)
    E, GmE = _masked_pair(cpu_data, node_imp, feat_means, device)

    fid = compute_fidelity(model, dev_data, y, explanation_removed=GmE,
                           explanation_only=E, masking_reference="R3")
    gef = compute_gef(model, dev_data, masked_data=E, masking_reference="R3")

    gt = data.node_gt_mask.detach().cpu().bool()
    pred_mask = binarize_mean(node_imp)
    gea = compute_gea(gt, pred_mask)

    return dict(fid_plus=fid.fid_plus, fid_minus=fid.fid_minus, gef=gef.gef,
                gea=gea.gea, gea_tp=gea.tp, gea_fp=gea.fp, gea_fn=gea.fn)


def sanity_check(name, results, datas, n_show=5):
    print(f"\n--- sanity check: {name} ---")
    empties = wholes = 0
    for i, (r, d) in enumerate(zip(results, datas)):
        n = d.num_nodes
        sel_mean = int(binarize_mean(r.node_importance).sum())
        sel_topk = int(r.topk_node_mask(K_FRAC).sum())
        gt = d.node_gt_mask.cpu().bool()
        overlap = int((r.topk_node_mask(K_FRAC) & gt).sum())
        if sel_mean == 0:
            empties += 1
        if sel_mean >= n:
            wholes += 1
        if i < n_show:
            gt_idx = torch.nonzero(gt).view(-1).tolist()
            top_idx = torch.nonzero(r.topk_node_mask(K_FRAC)).view(-1).tolist()
            print(f"  mol {i:2d}  N={n:2d}  |expl|(mean-thr)={sel_mean:2d}  |expl|(top{int(K_FRAC*100)}%)={sel_topk:2d}"
                  f"  GT motif nodes={gt_idx}  top-k nodes={top_idx}  (overlap {overlap}/{len(gt_idx)})")
    print(f"  => {empties} empty, {wholes} whole-graph explanations out of {len(results)}")
    assert empties == 0, f"{name}: {empties} empty explanations"
    assert wholes == 0, f"{name}: {wholes} whole-graph explanations"


def summarize(rows: list[dict]) -> dict:
    keys = ["fid_plus", "fid_minus", "gef", "gea"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(v.mean()), float(v.std()))
    tp = sum(r["gea_tp"] for r in rows)
    fp = sum(r["gea_fp"] for r in rows)
    fn = sum(r["gea_fn"] for r in rows)
    out["gea_micro"] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/ckpt_mutag_graphxai.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gnn-epochs", type=int, default=200)
    ap.add_argument("--pg-epochs", type=int, default=30)
    ap.add_argument("--sx-rollout", type=int, default=15)
    ap.add_argument("--sx-sample", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="cap #test molecules (debug)")
    ap.add_argument("--out", default="runs/phase3_mutag_graphxai.json")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    feat_means = ck["feature_means"].cpu().float()

    data_list, meta = load_mutag_graphxai()
    test_idx = ck["split"]["test"]
    train_idx = ck["split"]["train"]
    if args.limit:
        test_idx = test_idx[: args.limit]
    test = [data_list[i] for i in test_idx]
    train = [data_list[i] for i in train_idx]
    print(f"model: test_acc={ck['metrics']['test_acc']:.3f} test_auroc={ck['metrics']['test_auroc']:.3f}"
          f" | explaining {len(test)} test molecules (feat_means={feat_means.tolist()})")

    from .pyg_explainers import GNNExplainerWrapper, PGExplainerWrapper
    from .dig_subgraphx import SubgraphXWrapper

    explainers = {}
    explainers["gnnexplainer"] = GNNExplainerWrapper(model, device, epochs=args.gnn_epochs)

    pg = PGExplainerWrapper(model, device, epochs=args.pg_epochs)
    t0 = time.time()
    pg_losses = pg.train(train)
    print(f"PGExplainer trained on {len(train)} graphs, {args.pg_epochs} epochs "
          f"({time.time()-t0:.0f}s), loss {pg_losses[0]:.3f} -> {pg_losses[-1]:.3f}")
    explainers["pgexplainer"] = pg

    explainers["subgraphx"] = SubgraphXWrapper(
        model, device, rollout=args.sx_rollout, sample_num=args.sx_sample, node_frac=K_FRAC)

    table = {}
    per_mol = {}
    for name, ex in explainers.items():
        t0 = time.time()
        results = [ex.explain(d) for d in test]
        dt = time.time() - t0
        sanity_check(name, results, test)
        rows = [score_one(model, d, r, feat_means, device) for d, r in zip(test, results)]
        per_mol[name] = rows
        table[name] = summarize(rows)
        table[name]["seconds"] = dt
        print(f"  {name}: explained {len(test)} mols in {dt:.0f}s")

    print("\n" + "=" * 78)
    print(f"RQ1 first data point -- mutag_graphxai, R3 masking, top-{int(K_FRAC*100)}% node explanations")
    print("=" * 78)
    hdr = f"{'explainer':<14}{'Fid+':>16}{'Fid-':>16}{'GEF':>16}{'GEA (Jaccard)':>18}"
    print(hdr)
    print("-" * 78)
    for name, s in table.items():
        print(f"{name:<14}"
              f"{s['fid_plus'][0]:>8.3f}±{s['fid_plus'][1]:<6.3f}"
              f"{s['fid_minus'][0]:>8.3f}±{s['fid_minus'][1]:<6.3f}"
              f"{s['gef'][0]:>8.3f}±{s['gef'][1]:<6.3f}"
              f"{s['gea'][0]:>9.3f}±{s['gea'][1]:<6.3f}")
    print("-" * 78)
    print(f"{'(GEA micro)':<14}" + "".join(f"{table[n]['gea_micro']:>16.3f}" if False else "" for n in table))
    for name, s in table.items():
        print(f"  {name}: GEA micro-avg (pooled TP/FP/FN) = {s['gea_micro']:.3f}")
    print("\nnotes: Fid+ higher = explanation more necessary; Fid- lower = more sufficient;")
    print("       GEF lower = more faithful (bounded [0,1)); GEA higher = better GT overlap.")

    with open(args.out, "w") as f:
        json.dump({"config": {"k_frac": K_FRAC, "ckpt": args.ckpt,
                              "n_test": len(test), "device": str(device)},
                   "summary": table, "per_molecule": per_mol}, f, indent=2)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
