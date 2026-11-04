"""
RQ1 Tier 3, item F: re-score cached explanations and the ground-truth masks on the fine-tuned copy under the bounded rule F
(runs/rq1_tier3/PLAN.md). Writes runs/rq1_tier3/F/score/<unit>.npz.

    python -m src.analysis.tier3_F --unit U
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import torch

from . import reliance_fidelity as rf
from . import tier2_common as C
from .connectivity_baseline import build_adj, random_connected_subset
from .revision_pass import KEYS, metrics
from .ties_random import boundary_tie, tie_order, topk_mask

T = "runs/rq1_tier3/F"
DRAWS = 20
METHODS = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig", "ixg", "hetero"]


def method_scores(unit, ti):
    g = f"{C.T2}/G/grad/{unit}.npz"
    return {"gnnexplainer": C.stored_scores(unit, "gnnexplainer"), "pgexplainer": C.pg_final(unit)[0], "sx_fixed": C.sx_shard_scores(f"{C.T2}/S/sx/{unit}"),
            "sal": C.npz_scores(g, "sal"), "ig": C.npz_scores(g, "ig"), "ixg": C.npz_scores(g, "ixg"), "hetero": C.heuristic_dict(unit, ti, "hetero")}


def idx_of(mask):
    return [int(i) for i in np.flatnonzero(np.asarray(mask))]


class Pool:
    """Graphs of one molecule, de-duplicated by hidden set; row 0 = the clean graph."""

    def __init__(self, G):
        self.G, self.graphs, self.pos = G, [G], {}

    def zero(self, M):
        key = ("Z", tuple(sorted(int(v) for v in M)))
        return self._get(key, lambda: rf.zero_nodes(self.G, key[1]))

    def r1(self, M):
        key = ("D", tuple(sorted(int(v) for v in M)))
        return self._get(key, lambda: rf.keep_nodes(self.G, [v for v in range(self.G.num_nodes) if v not in set(key[1])]))

    def _get(self, key, make):
        if key not in self.pos:
            self.pos[key] = len(self.graphs); self.graphs.append(make())
        return self.pos[key]


def pair_F(P, M, n, rng):
    """(removed row, kept row) under rule F for one set M."""
    cap = max(1, n // 2); M = [int(v) for v in M]; comp = [v for v in range(n) if v not in set(M)]
    hid = M if len(M) <= cap else rng.choice(M, size=cap, replace=False).tolist()
    out = comp if len(comp) <= cap else rng.choice(comp, size=cap, replace=False).tolist()
    return P.zero(hid), P.zero(out)


def collect(acc, name, lg, c, pairs):
    vals = [metrics(lg, c, a, b) for a, b in pairs]
    for k in KEYS:
        acc.setdefault(f"{name}_{k}", []).append(float(np.mean([v[k] for v in vals])))


def run(unit, device, ckpt=None, out_path=None):
    """ckpt / out default to the Tier-3 copy and score file (Tier 4 L7 passes its re-tuned copies)."""
    model, _ = rf.load_model(ckpt or f"{T}/ckpt/{unit}.pt", device); dkey = C.dataset_of(unit); dl, meta = C.load_data(dkey)
    motifs = meta.extras.get("motifs") if hasattr(meta, "extras") else None
    ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; S = method_scores(unit, ti)
    A, B = {"mol_idx": [], "n": [], "n_methods_tied": []}, {"mol_idx": [], "n_masks": [], "mask_size": [], "cut_fidp": [], "cut_fidm": []}
    cut_m = cut_p = 0
    for mi in ti:
        d = dl[mi]; G = rf.to_cpu_data(d); n = G.num_nodes; k = C.kbudget(n); cap = max(1, n // 2); adj = build_adj(G.edge_index, n)
        rng = np.random.default_rng([85000, zlib.crc32(unit.encode()), mi]); P = Pool(G); plan = {}; ties = 0
        for w in METHODS:
            imp = np.asarray(S[w][mi], dtype=np.float64)
            if boundary_tie(imp, k):
                ties += 1; sets = [idx_of(topk_mask(tie_order(imp, rng), k)) for _ in range(DRAWS)]
            else:
                sets = [idx_of(topk_mask(imp, k))]
            plan[w] = [pair_F(P, sets[i % len(sets)], n, rng) for i in range(DRAWS)]
        plan["rn"] = [pair_F(P, rng.choice(n, size=k, replace=False).tolist(), n, rng) for _ in range(DRAWS)]
        plan["rc"] = [pair_F(P, sorted(random_connected_subset(adj, n, k, rng)), n, rng) for _ in range(DRAWS)]
        cut_p += k > cap; cut_m += (n - k) > cap
        gtp, sizes = [], []                # ground-truth masks: Part G definition
        if bool(d.node_gt_mask.any()):
            masks = [sorted(s) for s in rf.gt_masks(unit, d, mi, motifs) if len(s) < n]; sizes = [len(M) for M in masks]
            comp = lambda X: [v for v in range(n) if v not in set(X)]
            for M in masks:
                node = [rng.choice(n, size=len(M), replace=False).tolist() for _ in range(DRAWS)]; conn = [sorted(random_connected_subset(adj, n, len(M), rng)) for _ in range(DRAWS)]
                gtp.append({"gt_F": [pair_F(P, M, n, rng) for _ in range(DRAWS)], "node_F": [pair_F(P, X, n, rng) for X in node], "conn_F": [pair_F(P, X, n, rng) for X in conn],
                            "gt_R1c": [(P.r1(M), P.r1(comp(M)))], "node_R1c": [(P.r1(X), P.r1(comp(X))) for X in node], "conn_R1c": [(P.r1(X), P.r1(comp(X))) for X in conn]})
        lg = rf.logits_of(model, P.graphs, device); c = int(lg[0].argmax())
        A["mol_idx"].append(mi); A["n"].append(n); A["n_methods_tied"].append(ties)
        for w, pairs in plan.items():
            collect(A, w, lg, c, pairs)
        if gtp:
            B["mol_idx"].append(mi); B["n_masks"].append(len(gtp)); B["mask_size"].append(float(np.mean(sizes)))
            B["cut_fidp"].append(int(any(s > cap for s in sizes))); B["cut_fidm"].append(int(any(n - s > cap for s in sizes)))
            one = {}
            for g in gtp:
                for name, pairs in g.items():
                    collect(one, name, lg, c, pairs)
            for key, v in one.items():
                B.setdefault(key, []).append(float(np.mean(v)))
    out = {k: np.array(v) for k, v in A.items()}; out.update({f"G_{k}": np.array(v) for k, v in B.items()})
    C.save_npz(out_path or f"{T}/score/{unit}.npz", unit=unit, cut_expl_fidp=cut_p, cut_expl_fidm=cut_m, **out)
    print(f"{unit}: scored {len(ti)} molecules, {len(B['mol_idx'])} with ground truth", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit", required=True); a = a.parse_args()
    run(a.unit, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
