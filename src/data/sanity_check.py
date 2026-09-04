"""
Phase-2 dataset sanity checks.

Loads each RQ1 dataset, prints a summary, and runs structural assertions
(feature widths, index bounds, label domain, ground-truth mask shapes and
coverage).  Exits non-zero if any check fails.

    python -m src.data.sanity_check
    python -m src.data.sanity_check --datasets mutag mutag_graphxai
    python -m src.data.sanity_check --bxaic-task indole --bxaic-limit 2000
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch

from . import load_bbbp, load_bxaic, load_mutag, load_mutag_graphxai, load_tox21_srp53

CHECKS = {
    "mutag": load_mutag,
    "mutag_graphxai": load_mutag_graphxai,
    "bbbp": load_bbbp,
    "tox21_srp53": load_tox21_srp53,
    "bxaic": load_bxaic,
}

EXPECT_NODE_FEATS = {
    "mutag": 7, "mutag_graphxai": 7, "bbbp": 9, "tox21_srp53": 9, "bxaic": 11,
}


class Fail(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def _stats(values):
    t = torch.tensor(values, dtype=torch.float)
    return f"min={int(t.min())} max={int(t.max())} mean={t.mean():.1f}"


def sanity_check_one(key: str, loader, *, bxaic_task="indole", bxaic_limit=None) -> dict:
    if key == "bxaic":
        data_list, meta = loader(task=bxaic_task, limit=bxaic_limit)
    else:
        data_list, meta = loader()

    n_nodes = [d.num_nodes for d in data_list]
    n_edges = [d.edge_index.shape[1] for d in data_list]

    # ---- structural assertions -------------------------------------------------
    _check(len(data_list) == meta.num_graphs, "meta.num_graphs disagrees with len(data_list)")
    _check(len(data_list) > 0, "empty dataset")

    exp_nf = EXPECT_NODE_FEATS[key]
    _check(meta.num_node_features == exp_nf, f"expected {exp_nf} node feats, got {meta.num_node_features}")

    ys = []
    for i, d in enumerate(data_list):
        _check(d.x is not None and d.x.dim() == 2, f"graph {i}: bad x")
        _check(d.x.shape[1] == exp_nf, f"graph {i}: x width {d.x.shape[1]} != {exp_nf}")
        _check(not torch.isnan(d.x).any(), f"graph {i}: NaN in x")
        _check(d.edge_index.dtype == torch.long and d.edge_index.dim() == 2, f"graph {i}: bad edge_index")
        if d.edge_index.numel():
            _check(int(d.edge_index.max()) < d.num_nodes, f"graph {i}: edge_index out of range")
            _check(int(d.edge_index.min()) >= 0, f"graph {i}: negative edge_index")
        if d.edge_attr is not None:
            _check(d.edge_attr.shape[0] == d.edge_index.shape[1], f"graph {i}: edge_attr/edge_index length mismatch")
        _check(d.y.numel() >= 1, f"graph {i}: missing y")
        val = float(d.y.view(-1)[0])
        _check(val in (0.0, 1.0), f"graph {i}: y={val} not binary")
        ys.append(int(val))

        if meta.has_node_gt:
            _check(hasattr(d, "node_gt_mask"), f"graph {i}: node_gt_mask missing")
            _check(d.node_gt_mask.shape == (d.num_nodes,), f"graph {i}: node_gt_mask shape {tuple(d.node_gt_mask.shape)}")
            _check(d.node_gt_mask.dtype == torch.bool, f"graph {i}: node_gt_mask not bool")
        if meta.has_edge_gt:
            _check(hasattr(d, "edge_gt_mask"), f"graph {i}: edge_gt_mask missing")
            _check(d.edge_gt_mask.shape == (d.edge_index.shape[1],), f"graph {i}: edge_gt_mask shape")

    pos = sum(ys)
    balance = f"neg/pos = {len(ys) - pos}/{pos} ({pos / len(ys):.1%} pos)"

    gt_line = ""
    if meta.has_node_gt:
        cov = sum(int(d.node_gt_mask.any()) for d in data_list)
        frac_atoms = (
            torch.cat([d.node_gt_mask.float() for d in data_list]).mean().item()
        )
        _check(cov > 0, "node ground truth is empty for every graph")
        gt_line = f"\n    node GT: {cov}/{len(data_list)} graphs have >=1 flagged atom; {frac_atoms:.1%} of all atoms flagged"
        if meta.has_edge_gt:
            ecov = sum(int(d.edge_gt_mask.any()) for d in data_list)
            gt_line += f"\n    edge GT: {ecov}/{len(data_list)} graphs have >=1 flagged edge"

    print(f"[{meta.name}]  OK")
    print(f"    graphs={meta.num_graphs}  node_feats={meta.num_node_features}  edge_feats={meta.num_edge_features}  classes={meta.num_classes}")
    print(f"    nodes: {_stats(n_nodes)}   edges: {_stats(n_edges)}")
    print(f"    labels: {balance}")
    if meta.notes:
        print(f"    note: {meta.notes}")
    if gt_line:
        print(gt_line.lstrip("\n"))
    return {"name": meta.name, "graphs": meta.num_graphs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(CHECKS), choices=list(CHECKS))
    ap.add_argument("--bxaic-task", default="indole")
    ap.add_argument("--bxaic-limit", type=int, default=None,
                    help="cap B-XAIC molecules (fast smoke); default = all 50k")
    args = ap.parse_args(argv)

    failures = []
    for key in args.datasets:
        print("=" * 72)
        try:
            sanity_check_one(
                key, CHECKS[key],
                bxaic_task=args.bxaic_task, bxaic_limit=args.bxaic_limit,
            )
        except Exception as e:  # noqa: BLE001 -- report and keep going
            failures.append(key)
            print(f"[{key}]  FAIL -- {type(e).__name__}: {e}")
            traceback.print_exc()

    print("=" * 72)
    if failures:
        print(f"SANITY CHECK FAILED for: {', '.join(failures)}")
        return 1
    print(f"all {len(args.datasets)} dataset(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
