"""
Phase 3 -- R3 validation per dataset variant (locked build order: run this once
per dataset before R2/R1 get built).

Loads a trained D-MPNN checkpoint, runs GNNExplainer / PGExplainer / SubgraphX
on the held-out test split, sanity-checks the explanations, then scores every
(explainer, molecule) pair with Fidelity+/-, GEF and (where ground truth exists)
GEA, under one masking reference: --masking R3 (distribution-aware, mean+mode
fill sweep) or R2 (zero-fill). R1 stays NotImplementedError per the build order.

    python -m src.explain.run_phase3 --dataset mutag_graphxai
    python -m src.explain.run_phase3 --dataset mutag --masking R2

Outputs: a summary table to stdout + runs/phase3_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..data import LOADERS
from ..metrics.fidelity import compute_fidelity
from ..metrics.gef import compute_gef
from ..metrics.gea import compute_gea
from ..metrics.masking import (
    mask_r2_zero_fill, mask_r3_distribution_aware, training_fill_vector,
)
from ..train.dmpnn import DMPNN, DMPNNConfig
from .common import binarize_mean, to_cpu_data

K_FRAC = 0.25            # explanation sparsity: top 25% of nodes
R3_FILLS = ("mean", "mode")  # both recorded -- Phase 3 R3 finding
# per-masking "fill" keys: R3 sweeps mean+mode; R2 (zero-fill) has a single mode.
FILLS_BY_MASKING = {"R3": R3_FILLS, "R2": ("zero",)}


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


def _masked_pair(cpu_data, node_imp, fill_vec, device, masking):
    """explanation-only (E) and explanation-removed (G\\E) graphs, under R3 or R2."""
    if masking == "R2":
        E = mask_r2_zero_fill(cpu_data, node_imp, keep_top_k=K_FRAC)
        GmE = mask_r2_zero_fill(cpu_data, -node_imp, keep_top_k=1.0 - K_FRAC)
    else:
        E = mask_r3_distribution_aware(cpu_data, node_imp, fill_vec, keep_top_k=K_FRAC)
        GmE = mask_r3_distribution_aware(cpu_data, -node_imp, fill_vec, keep_top_k=1.0 - K_FRAC)
    return E.to(device), GmE.to(device)


def score_one(model, data, result, fill_vec, device, has_gt: bool, masking: str) -> dict:
    cpu_data = to_cpu_data(data)
    node_imp = result.node_importance.detach().cpu().float()
    y = result.target  # class the explanation is for (== model prediction)

    dev_data = data.to(device)
    E, GmE = _masked_pair(cpu_data, node_imp, fill_vec, device, masking)

    fid = compute_fidelity(model, dev_data, y, explanation_removed=GmE,
                           explanation_only=E, masking_reference=masking)
    gef = compute_gef(model, dev_data, masked_data=E, masking_reference=masking)

    row = dict(fid_plus=fid.fid_plus, fid_minus=fid.fid_minus, gef=gef.gef)
    if has_gt:
        gt = data.node_gt_mask.detach().cpu().bool()
        # GEA only where ground truth actually exists. mutag_graphxai: every
        # molecule has a NO2/NH2 motif. B-XAIC indole: only positives contain
        # the substructure -- negatives have an all-False GT mask, so GEA there
        # would be a structural 0 that says nothing about explainer quality.
        if bool(gt.any()):
            gea = compute_gea(gt, binarize_mean(node_imp))
            row.update(gea=gea.gea, gea_tp=gea.tp, gea_fp=gea.fp, gea_fn=gea.fn)
    return row


def sanity_check(name, results, datas, has_gt: bool, n_show=5):
    print(f"\n--- sanity check: {name} ---")
    empties, wholes = [], []
    for i, (r, d) in enumerate(zip(results, datas)):
        n = d.num_nodes
        sel_mean = int(binarize_mean(r.node_importance).sum())
        sel_topk = int(r.topk_node_mask(K_FRAC).sum())
        if sel_mean == 0:
            empties.append(i)
        if sel_mean >= n:
            wholes.append(i)
        if i < n_show:
            top_idx = torch.nonzero(r.topk_node_mask(K_FRAC)).view(-1).tolist()
            line = f"  mol {i:2d}  N={n:2d}  |expl|(mean-thr)={sel_mean:2d}  |expl|(top{int(K_FRAC*100)}%)={sel_topk:2d}  top-k nodes={top_idx}"
            if has_gt:
                gt = d.node_gt_mask.cpu().bool()
                overlap = int((r.topk_node_mask(K_FRAC) & gt).sum())
                gt_idx = torch.nonzero(gt).view(-1).tolist()
                line += f"  GT motif nodes={gt_idx}  (overlap {overlap}/{len(gt_idx)})"
            print(line)
    print(f"  => {len(empties)} empty, {len(wholes)} whole-graph explanations out of {len(results)}")
    # A handful of empty/whole mean-threshold explanations happens on real, diverse
    # datasets (e.g. a molecule PGExplainer scores near-uniformly) and isn't itself a
    # bug -- print full diagnostics for each so it's inspectable, but only hard-fail
    # on near-total failure (>20%), which WOULD indicate something broken.
    frac = len(results)
    for label, idxs in (("empty", empties), ("whole-graph", wholes)):
        for i in idxs:
            r, d = results[i], datas[i]
            print(f"    [{label}] mol {i}: N={d.num_nodes}  node_importance stats:"
                  f" min={r.node_importance.min():.4g} max={r.node_importance.max():.4g}"
                  f" mean={r.node_importance.mean():.4g} std={r.node_importance.std():.4g}")
    assert len(empties) / frac <= 0.20, f"{name}: {len(empties)}/{frac} empty explanations (>20%)"
    assert len(wholes) / frac <= 0.20, f"{name}: {len(wholes)}/{frac} whole-graph explanations (>20%)"


def summarize(rows: list[dict], has_gt: bool) -> dict:
    out = {}
    for k in ("fid_plus", "fid_minus", "gef"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(v.mean()), float(v.std()))
    if has_gt:
        gea_rows = [r for r in rows if "gea" in r]   # only GT-present molecules
        out["n_gea"] = len(gea_rows)
        if gea_rows:
            v = np.array([r["gea"] for r in gea_rows], dtype=float)
            out["gea"] = (float(v.mean()), float(v.std()))
            tp = sum(r["gea_tp"] for r in gea_rows)
            fp = sum(r["gea_fp"] for r in gea_rows)
            fn = sum(r["gea_fn"] for r in gea_rows)
            out["gea_micro"] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        else:
            out["gea"] = (float("nan"), float("nan"))
            out["gea_micro"] = float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mutag_graphxai", choices=list(LOADERS))
    ap.add_argument("--masking", default="both", choices=["both", "R3", "R2"],
                    help="which masking reference(s) to score the (shared) explanations under. "
                         "both = R3 (mean+mode fill) AND R2 (zero-fill) in one pass -- explanations "
                         "computed once. R1 stays NotImplementedError per the locked build order.")
    ap.add_argument("--ckpt", default=None, help="default: runs/ckpt_<dataset>.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gnn-epochs", type=int, default=200)
    ap.add_argument("--pg-epochs", type=int, default=30)
    ap.add_argument("--sx-rollout", type=int, default=20)   # DIG default; locked (see spec)
    ap.add_argument("--sx-sample", type=int, default=30)
    ap.add_argument("--explainers", default="gnnexplainer,pgexplainer,subgraphx",
                    help="comma list; subset to re-run just one method")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #test molecules explained -- a random (seeded) subsample of the "
                         "test split, not a prefix, so bigger datasets stay comparable in n to "
                         "MUTAG/mutag_graphxai (both n=29) without hours of SubgraphX MCTS")
    ap.add_argument("--pg-train-limit", type=int, default=None,
                    help="cap #graphs PGExplainer trains on (random seeded subsample of train) -- "
                         "keeps PG training cost comparable across dataset sizes")
    ap.add_argument("--stratify-frac", type=float, default=None,
                    help="target positive-class fraction for the --limit / --pg-train-limit "
                         "subsamples (e.g. 0.5). Default None = plain random draw. Needed on "
                         "severely imbalanced datasets (Tox21 SR-p53: 6.3%% pos -> a random 30 "
                         "has ~1 positive), so positive-class explanations are actually present.")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="exclude test/pg-train molecules with more than this many nodes before "
                         "subsampling. B-XAIC has graphs up to 674 nodes -- SubgraphX MCTS at "
                         "rollout=20 on those runs for hours. Keeps the explained set comparable "
                         "in size to the other four datasets (all <~40 nodes in practice).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="recompute explanations even if a matching cache exists")
    ap.add_argument("--out", default=None, help="default: runs/phase3_<dataset>[_R2].json")
    args = ap.parse_args(argv)
    want = {s.strip() for s in args.explainers.split(",") if s.strip()}
    ckpt_path = args.ckpt or f"runs/ckpt_{args.dataset}.pt"
    suffix = "" if args.masking in ("both", "R3") else f"_{args.masking}"
    out_path = args.out or f"runs/phase3_{args.dataset}{suffix}.json"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)          # GNNExplainer / PGExplainer / SubgraphX stochasticity
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model, ck = load_model(ckpt_path, device)

    data_list, meta = LOADERS[args.dataset]()
    has_gt = meta.has_node_gt
    test_idx = list(ck["split"]["test"])
    train_idx = list(ck["split"]["train"])

    n_excluded = 0
    if args.max_nodes:
        before = len(test_idx) + len(train_idx)
        test_idx = [i for i in test_idx if data_list[i].num_nodes <= args.max_nodes]
        train_idx = [i for i in train_idx if data_list[i].num_nodes <= args.max_nodes]
        n_excluded = before - len(test_idx) - len(train_idx)

    def _subsample(idxs, size):
        if not size or size >= len(idxs):
            return list(idxs)
        if args.stratify_frac is None:
            return sorted(rng.choice(idxs, size=size, replace=False).tolist())
        lab = np.array([int(data_list[i].y.view(-1)[0]) for i in idxs])
        pos = [i for i, y in zip(idxs, lab) if y == 1]
        neg = [i for i, y in zip(idxs, lab) if y == 0]
        n_pos = min(len(pos), max(1, round(args.stratify_frac * size)))
        n_neg = min(len(neg), size - n_pos)
        pick = (rng.choice(pos, size=n_pos, replace=False).tolist()
                + rng.choice(neg, size=n_neg, replace=False).tolist())
        return sorted(pick)

    test_idx = _subsample(test_idx, args.limit)
    pg_train_idx = _subsample(train_idx, args.pg_train_limit)
    test = [data_list[i] for i in test_idx]
    train = [data_list[i] for i in train_idx]
    pg_train = [data_list[i] for i in pg_train_idx]

    x_train = torch.cat([data_list[i].x.float() for i in train_idx], dim=0)

    def _bal(idxs):
        y = np.array([int(data_list[i].y.view(-1)[0]) for i in idxs])
        return f"{int(y.sum())}pos/{int((y == 0).sum())}neg"

    test_bal = _bal(test_idx)
    pg_bal = _bal(pg_train_idx)
    print(f"dataset={args.dataset}  masking={args.masking}  has_ground_truth={has_gt}  stratify_frac={args.stratify_frac}"
          + (f"  max_nodes={args.max_nodes} (excluded {n_excluded} graphs from test+train pools)" if args.max_nodes else ""))
    print(f"model: test_acc={ck['metrics']['test_acc']:.3f} test_auroc={ck['metrics']['test_auroc']:.3f}"
          f" | explaining {len(test)}/{len(ck['split']['test'])} test molecules [{test_bal}]"
          + (f"  | PGExplainer trains on {len(pg_train)}/{len(train_idx)} [{pg_bal}]" if "pgexplainer" in want else ""))

    # ---- explanation cache -------------------------------------------------
    # Explanations don't depend on the masking reference (R2/R3), so cache them
    # keyed by everything that DOES change them: dataset, ckpt, test subsample,
    # explainer hyper-params. An R2 run then re-scores the *identical*
    # explanations an R3 run produced -- no re-running SubgraphX MCTS, and the
    # R2-vs-R3 comparison is not contaminated by explainer stochasticity.
    from .common import ExplanationResult
    cache_path = f"runs/expl_cache_{args.dataset}{suffix.replace('_R2','')}.pt"
    cache_key = dict(ckpt=ckpt_path, test_idx=test_idx, pg_train_idx=pg_train_idx,
                     gnn_epochs=args.gnn_epochs, pg_epochs=args.pg_epochs,
                     sx_rollout=args.sx_rollout, sx_sample=args.sx_sample, k_frac=K_FRAC)
    cached = None
    if not args.refresh_cache and __import__("os").path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("key") == cache_key:
            cached = blob["results"]
            print(f"  loaded cached explanations <- {cache_path}")
        else:
            print(f"  cache {cache_path} present but key mismatch -> recomputing")

    results_by = {}
    if cached is not None and want.issubset(cached):
        for name in want:
            results_by[name] = [ExplanationResult(node_importance=torch.tensor(r["node_importance"]),
                                                  edge_importance=None if r["edge_importance"] is None
                                                  else torch.tensor(r["edge_importance"]),
                                                  target=r["target"], meta=r.get("meta", {}))
                                for r in cached[name]]
            sanity_check(name, results_by[name], test, has_gt)
    else:
        from .pyg_explainers import GNNExplainerWrapper, PGExplainerWrapper
        from .dig_subgraphx import SubgraphXWrapper

        explainers = {}
        if "gnnexplainer" in want:
            explainers["gnnexplainer"] = GNNExplainerWrapper(model, device, epochs=args.gnn_epochs)
        if "pgexplainer" in want:
            pg = PGExplainerWrapper(model, device, epochs=args.pg_epochs)
            t0 = time.time()
            pg_losses = pg.train(pg_train)
            print(f"PGExplainer trained on {len(pg_train)} graphs, {args.pg_epochs} epochs "
                  f"({time.time()-t0:.0f}s), loss {pg_losses[0]:.3f} -> {pg_losses[-1]:.3f}")
            explainers["pgexplainer"] = pg
        if "subgraphx" in want:
            explainers["subgraphx"] = SubgraphXWrapper(
                model, device, rollout=args.sx_rollout, sample_num=args.sx_sample, node_frac=K_FRAC)

        for name, ex in explainers.items():
            t0 = time.time()
            results_by[name] = [ex.explain(d) for d in test]
            print(f"  {name}: explained {len(test)} mols in {time.time() - t0:.0f}s")
            sanity_check(name, results_by[name], test, has_gt)

        merged = dict(cached) if cached else {}
        for name, res in results_by.items():
            merged[name] = [dict(node_importance=r.node_importance.tolist(),
                                 edge_importance=None if r.edge_importance is None else r.edge_importance.tolist(),
                                 target=r.target, meta={k: v for k, v in r.meta.items() if k != "related_pred"})
                            for r in res]
        torch.save({"key": cache_key, "results": merged}, cache_path)
        print(f"  cached explanations -> {cache_path}")

    # score the SAME explanations under every requested masking reference
    maskings = ["R3", "R2"] if args.masking == "both" else [args.masking]
    fills_by_m = {m: ({s: training_fill_vector(x_train, s) for s in FILLS_BY_MASKING["R3"]}
                      if m == "R3" else {"zero": None}) for m in maskings}

    summary, per_mol = {}, {}
    for m in maskings:
        summary[m], per_mol[m] = {}, {}
        for fill_name, fill_vec in fills_by_m[m].items():
            summary[m][fill_name], per_mol[m][fill_name] = {}, {}
            for name, results in results_by.items():
                rows = [score_one(model, d, r, fill_vec, device, has_gt, m) for d, r in zip(test, results)]
                per_mol[m][fill_name][name] = rows
                summary[m][fill_name][name] = summarize(rows, has_gt)

    for m in maskings:
        for fill_name, table in summary[m].items():
            print("\n" + "=" * 84)
            n_gea = next(iter(table.values())).get("n_gea") if has_gt else None
            print(f"{args.dataset} | {m} masking, fill='{fill_name}' | top-{int(K_FRAC*100)}% node explanations"
                  f" | n={len(test)}" + (f" (GEA over {n_gea} GT-present mols)" if has_gt else ""))
            print("=" * 84)
            hdr = f"{'explainer':<14}{'Fid+':>15}{'Fid-':>15}{'GEF':>15}"
            if has_gt:
                hdr += f"{'GEA Jaccard':>16}{'GEA micro':>12}"
            print(hdr)
            print("-" * 80)
            for name, s in table.items():
                line = (f"{name:<14}"
                       f"{s['fid_plus'][0]:>7.3f}±{s['fid_plus'][1]:<6.3f}"
                       f"{s['fid_minus'][0]:>7.3f}±{s['fid_minus'][1]:<6.3f}"
                       f"{s['gef'][0]:>7.3f}±{s['gef'][1]:<6.3f}")
                if has_gt:
                    line += f"{s['gea'][0]:>8.3f}±{s['gea'][1]:<6.3f}{s['gea_micro']:>12.3f}"
                print(line)
    print("\nFid+ higher = explanation more necessary | Fid- lower = more sufficient")
    print("GEF lower = more faithful (bounded [0,1))" + ("" if not has_gt else
          " | GEA higher = better GT overlap (fill/masking-independent)"))
    if not has_gt:
        print(f"(no ground truth for {args.dataset} -- GEA not applicable, per spec's applicability table)")

    with open(out_path, "w") as f:
        json.dump({"config": {"dataset": args.dataset, "has_ground_truth": has_gt, "k_frac": K_FRAC,
                              "ckpt": ckpt_path, "n_test": len(test), "n_test_full_split": len(ck["split"]["test"]),
                              "n_pg_train": len(pg_train) if "pgexplainer" in want else None,
                              "masking": maskings,
                              "stratify_frac": args.stratify_frac, "max_nodes": args.max_nodes,
                              "n_excluded_by_max_nodes": n_excluded,
                              "test_balance": test_bal, "pg_train_balance": pg_bal,
                              "device": str(device), "explainers": sorted(want),
                              "sx_rollout": args.sx_rollout, "sx_sample": args.sx_sample},
                   "fills": {m: {k: (None if v is None else v.tolist()) for k, v in fb.items()}
                             for m, fb in fills_by_m.items()},
                   "summary": summary, "per_molecule": per_mol}, f, indent=2)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
