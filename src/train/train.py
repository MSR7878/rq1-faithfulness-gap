"""
D-MPNN training / smoke tests for RQ1 (Phase 2).

Protocol: 10-fold stratified CV (the TUDataset graph-classification standard),
reporting test **accuracy** and **AUROC** (mean +/- std over folds).  Class
balance varies widely across the RQ1 datasets (MUTAG 66.5%, BBBP 76.5%,
Tox21-SRp53 6.3%, B-XAIC-indole 36.8% positive), so AUROC is the cross-dataset
comparable number; the best checkpoint per fold is selected on validation AUROC.

    # MUTAG smoke test (build order: this first)
    python -m src.train.train --dataset mutag --folds 10 --epochs 120

    # dataset that ships its own split (B-XAIC): use it instead of k-fold
    python -m src.train.train --dataset bxaic --native-split --epochs 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from ..data import LOADERS
from .dmpnn import DMPNN, DMPNNConfig

# Rough published reference ranges for the base graph-classification task
# (10-fold CV, GNN literature) -- used only to sanity-check the smoke tests.
REFERENCE = {
    "mutag": "acc ~0.85-0.90 (GIN 0.894), AUROC ~0.85-0.93",
    "mutag_graphxai": "identical to mutag by construction (same graphs + labels)",
    "bbbp": "AUROC ~0.88-0.92 (scaffold split lower ~0.70); random split here",
    "tox21_srp53": "AUROC ~0.80-0.86 per-endpoint (Tox21 SR-p53)",
    "bxaic:indole": "B-XAIC report: near-perfect, AUROC > 0.95 (motif is exact)",
}


def _labels(data_list) -> list[int]:
    return [int(d.y.view(-1)[0]) for d in data_list]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _collect(model, loader, device):
    """Return (y_true, prob_pos, mean_loss) over a loader."""
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(-1).long()
        loss_sum += lossf(logits, y).item()
        n += y.numel()
        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


def _metrics(y_true, prob_pos) -> tuple[float, float]:
    acc = float(((prob_pos >= 0.5).astype(int) == y_true).mean())
    auroc = float("nan")
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, prob_pos))
    return acc, auroc


def _select_score(acc: float, auroc: float) -> float:
    """Checkpoint-selection score: val AUROC + val accuracy, so the chosen
    checkpoint has to rank *and* threshold well (a noisy early epoch with
    fluke-high AUROC but chance accuracy loses to a genuinely trained one).
    Falls back to 2*accuracy when the val fold is single-class (AUROC
    undefined).  Only consulted after a short warmup, see train_one."""
    return 2.0 * acc if np.isnan(auroc) else auroc + acc


def train_one(train_set, val_set, test_set, meta, args, device) -> dict:
    cfg = DMPNNConfig(
        node_in=meta.num_node_features,
        edge_in=meta.num_edge_features,
        hidden=args.hidden,
        depth=args.depth,
        ffn_hidden=args.hidden,
        dropout=args.dropout,
        num_classes=2,
        pool=args.pool,
    )
    model = DMPNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.7, patience=8)
    lossf = nn.CrossEntropyLoss()

    tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_set, batch_size=args.batch_size)
    testl = DataLoader(test_set, batch_size=args.batch_size) if test_set else None

    warmup = max(1, args.epochs // 10)
    best_sel, best = -1.0, {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = lossf(logits, batch.y.view(-1).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        vy, vp, _ = _collect(model, vl, device)
        val_acc, val_auroc = _metrics(vy, vp)
        sel = _select_score(val_acc, val_auroc)
        sched.step(sel)
        if epoch >= warmup and sel > best_sel:
            best_sel = sel
            best = {"val_acc": val_acc, "val_auroc": val_auroc, "epoch": epoch}
            if testl is not None:
                ty, tp, _ = _collect(model, testl, device)
                t_acc, t_auroc = _metrics(ty, tp)
                best["test_acc"], best["test_auroc"] = t_acc, t_auroc
        if args.verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  val_acc={val_acc:.3f} val_auroc={val_auroc:.3f}"
                  f"  best_sel={best_sel:.3f}")
    return best


def _summ(name, vals):
    a = np.array(vals, dtype=float)
    return f"{name} = {np.nanmean(a):.4f} +/- {np.nanstd(a):.4f}"


def run_kfold(data_list, meta, args, device) -> None:
    y = np.array(_labels(data_list))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    accs, aurocs = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=0.1, random_state=args.seed, stratify=y[tr_idx]
        )
        _set_seed(args.seed + fold)
        t0 = time.time()
        out = train_one(
            [data_list[i] for i in tr_idx], [data_list[i] for i in va_idx],
            [data_list[i] for i in te_idx], meta, args, device,
        )
        accs.append(out["test_acc"]); aurocs.append(out["test_auroc"])
        print(f"  fold {fold:2d}/{args.folds}: test_acc={out['test_acc']:.4f} "
              f"test_auroc={out['test_auroc']:.4f}  (val_auroc={out['val_auroc']:.3f}, "
              f"ep{out['epoch']})  {time.time() - t0:.0f}s")

    maj = max(y.sum(), len(y) - y.sum()) / len(y)
    print("-" * 64)
    print(f"{meta.name}: {args.folds}-fold CV")
    print(f"  {_summ('test accuracy', accs)}")
    print(f"  {_summ('test AUROC   ', aurocs)}")
    print(f"  majority-class baseline acc = {maj:.4f} (AUROC 0.5)")
    print(f"  reference: {REFERENCE.get(meta.name, 'n/a')}")


def run_native_split(data_list, meta, args, device) -> None:
    if not all(hasattr(d, "split") for d in data_list):
        raise SystemExit(f"{meta.name} has no per-graph .split field; drop --native-split")
    buckets = {"train": [], "valid": [], "test": []}
    for d in data_list:
        buckets[d.split].append(d)
    y = np.array(_labels(data_list))
    print(f"  native split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    _set_seed(args.seed)
    t0 = time.time()
    out = train_one(buckets["train"], buckets["valid"], buckets["test"], meta, args, device)
    maj = max(y.sum(), len(y) - y.sum()) / len(y)
    print("-" * 64)
    print(f"{meta.name}: native train/valid/test split")
    print(f"  test accuracy = {out['test_acc']:.4f}")
    print(f"  test AUROC    = {out['test_auroc']:.4f}  (val_auroc={out['val_auroc']:.3f}, ep{out['epoch']})")
    print(f"  majority-class baseline acc = {maj:.4f} (AUROC 0.5)   {time.time() - t0:.0f}s")
    print(f"  reference: {REFERENCE.get(meta.name, 'n/a')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--folds", type=int, default=0, help="k for stratified k-fold CV; 0 = single split")
    ap.add_argument("--native-split", action="store_true",
                    help="use the dataset's own train/valid/test split (e.g. B-XAIC) instead of CV")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--pool", default="sum", choices=["sum", "mean"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bxaic-task", default="indole")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    loader = LOADERS[args.dataset]
    data_list, meta = (
        loader(task=args.bxaic_task) if args.dataset == "bxaic" else loader()
    )
    print(f"loaded {meta.name}: {meta.num_graphs} graphs, "
          f"{meta.num_node_features} node / {meta.num_edge_features} edge feats  [device={device}]")

    if args.native_split:
        run_native_split(data_list, meta, args, device)
    elif args.folds and args.folds > 1:
        run_kfold(data_list, meta, args, device)
    else:
        raise SystemExit("pass --folds N (>=2) or --native-split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
