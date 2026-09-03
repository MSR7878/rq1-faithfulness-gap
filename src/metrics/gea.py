"""
GEA -- Graph Explanation Accuracy (see configs/rq1_metric_spec.md, Section 3).

Jaccard index between the predicted explanation mask and a ground-truth
importance mask. ONLY computable on datasets with ground-truth
explanations -- per the spec's applicability table, that's currently
B-XAIC and GraphXAI's ground-truth-labeled MUTAG. Do not call this on
BBBP, Tox21, or standard MUTAG; there is no ground truth to compare
against there (see fidelity.py / gef.py for the metrics that DO apply
across all five datasets).

    GEA = JAC(M_gt, M_pr) = TP / (TP + FP + FN)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GEAResult:
    gea: float
    tp: int
    fp: int
    fn: int


def compute_gea(ground_truth_mask: torch.Tensor, predicted_mask: torch.Tensor) -> GEAResult:
    """
    Args:
        ground_truth_mask: (N,) binary tensor, 1 = important per dataset's
            ground-truth annotation (e.g. B-XAIC's atom/edge labels).
        predicted_mask: (N,) binary tensor, 1 = important per the
            explainer's output. Binarize upstream (e.g. mean-threshold, as
            GraphXAI's own implementation does) before calling this.

    Returns:
        GEAResult with the Jaccard index plus raw TP/FP/FN counts, since
        the counts themselves are useful for later error analysis (e.g.
        "does explainer X systematically over- or under-select?").
    """
    gt = ground_truth_mask.bool()
    pred = predicted_mask.bool()

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())

    denom = tp + fp + fn
    gea = tp / denom if denom > 0 else 0.0

    return GEAResult(gea=gea, tp=tp, fp=fp, fn=fn)
