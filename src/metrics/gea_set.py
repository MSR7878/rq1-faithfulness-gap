"""
Max-over-set GEA (GraphXAI paper Eq. 2) for datasets whose ground truth is a
SET of valid masks per graph (GraphXAI Mutagenicity).

GraphXAI builds the set as the union of every non-empty subset of the matched
motif instances (2^L - 1 masks for L instances) and scores an explanation by
the best Jaccard over that set. We store the L motif instances and evaluate
the max without materialising the set when it would blow up:

  * L <= ENUM_CAP           exact, by enumerating all 2^L - 1 unions.
  * L >  ENUM_CAP, motifs pairwise node-disjoint
                            exact, by Dinkelbach's method: with
                            a_i = |P & m_i|, b_i = |m_i - P|,
                            J(S) = sum_S a_i / (|P| + sum_S b_i), and the
                            optimum is S = {i : a_i - lam*b_i > 0} at the fixed
                            point lam = J(S).
  * L >  ENUM_CAP, motifs overlap
                            lower bound: best of {each single motif, the full
                            union, the Dinkelbach set}; flagged ``exact=False``.

``merged`` is the plain Jaccard against the union of all instances (the
single-mask GEA the rest of this project uses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENUM_CAP = 12   # 2^12 - 1 = 4095 unions


@dataclass
class SetGEA:
    max_over_set: float
    merged: float
    n_motifs: int
    path: str      # "enum" | "dinkelbach" | "bound"
    exact: bool


def _jac(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int((pred | gt).sum())
    return int((pred & gt).sum()) / union if union else 0.0


def motif_matrix(motifs, n: int) -> np.ndarray:
    M = np.zeros((len(motifs), n), dtype=bool)
    for i, m in enumerate(motifs):
        nodes = m[1] if isinstance(m, tuple) else m
        M[i, list(nodes)] = True
    return M


def _enum(pred: np.ndarray, M: np.ndarray) -> float:
    L = M.shape[0]
    codes = np.arange(1, 2 ** L, dtype=np.int64)
    sel = ((codes[:, None] >> np.arange(L)) & 1).astype(bool)        # (2^L-1, L)
    U = (sel[:, :, None] & M[None, :, :]).any(axis=1)                # (2^L-1, N)
    inter = (U & pred).sum(axis=1)
    union = (U | pred).sum(axis=1)
    return float((inter / np.maximum(union, 1)).max())


def _dinkelbach(pred: np.ndarray, M: np.ndarray) -> float:
    a = (M & pred).sum(axis=1).astype(float)
    b = (M & ~pred).sum(axis=1).astype(float)
    if a.sum() == 0:
        return 0.0
    P = float(pred.sum())
    lam = 0.0
    for _ in range(100):
        S = (a - lam * b) > 0
        new = a[S].sum() / (P + b[S].sum())
        if new <= lam + 1e-15:
            break
        lam = new
    return float(lam)


def gea_max_over_set(pred_mask, motifs, enum_cap: int = ENUM_CAP) -> SetGEA:
    pred = np.asarray(pred_mask, dtype=bool)
    M = motif_matrix(motifs, pred.shape[0])
    L = M.shape[0]
    merged = _jac(pred, M.any(axis=0))
    if L <= enum_cap:
        return SetGEA(_enum(pred, M), merged, L, "enum", True)
    disjoint = bool((M.sum(axis=0) <= 1).all())
    dk = _dinkelbach(pred, M)
    if disjoint:
        return SetGEA(dk, merged, L, "dinkelbach", True)
    singles = max(_jac(pred, M[i]) for i in range(L))
    return SetGEA(max(dk, singles, merged), merged, L, "bound", False)
