"""
Random tie-break for top-k selections (rule (b) of runs/rq1_followups/PLAN.md).

A molecule's importance scores get a random rank WITHIN exact ties only;
distinct scores keep their order. 20 seeded orders per molecule,
numpy default_rng([62000, crc32(unit), molecule position]).
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

N_ORDERS = 20


def mol_rng(unit: str, pos: int) -> np.random.Generator:
    return np.random.default_rng([62000, zlib.crc32(unit.encode()), int(pos)])


def tie_order(imp, rng) -> torch.Tensor:
    """Scores with the same order as imp between distinct values and a random order within exact ties."""
    v = np.asarray(imp, dtype=np.float64); order = np.lexsort((rng.random(len(v)), v))
    s = np.empty(len(v)); s[order] = np.arange(len(v))
    return torch.tensor(s, dtype=torch.float)


def boundary_tie(imp, k: int) -> bool:
    """True if the k-th and (k+1)-th largest scores are equal (the top-k set depends on tie order)."""
    s = np.sort(np.asarray(imp, dtype=np.float64))[::-1]
    return k < len(s) and s[k - 1] == s[k]


def removed_boundary_tie(imp, keep: int) -> bool:
    """True if the keep-th and (keep+1)-th smallest scores are equal (the explanation-removed graph depends on tie order)."""
    s = np.sort(np.asarray(imp, dtype=np.float64))
    return keep < len(s) and s[keep - 1] == s[keep]


def topk_mask(scores, k: int) -> np.ndarray:
    m = np.zeros(len(scores), bool); m[torch.topk(torch.as_tensor(scores, dtype=torch.float), k).indices.numpy()] = True
    return m


def random_tie_gea(imp, k: int, scorer, unit: str, pos: int) -> tuple[float, bool]:
    """Mean GEA over N_ORDERS random tie orders (one value if there is no tie at the top-k boundary)."""
    if not boundary_tie(imp, k):
        return float(scorer(topk_mask(imp, k))), False
    rng = mol_rng(unit, pos)
    return float(np.mean([scorer(topk_mask(tie_order(imp, rng), k)) for _ in range(N_ORDERS)])), True
