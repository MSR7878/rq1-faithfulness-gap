"""
RQ1 Tier 6 addendum (runs/rq1_tier6/PLAN.md, "Addendum, 2026-11-24"). Existing explanations only; no explainer runs.

  t5 --unit U   what the chosen top-k atoms are: (i) GT, (ii) aromatic non-GT, (iii) other non-GT (by element C/N/O/other,
                and bonded to a GT atom or not), for EiG, GNNShap, GNNExplainer, SX fixed, IG, Saliency + random-node
                (exact expectation)                                                   -> runs/rq1_tier6/T5/<unit>.npz
  t6 --unit U   group (all GT atoms) hidden / only the group kept, vs 20 random-connected and 20 random-node sets of the same
                size, on the full graph and on the scaffold-hidden graph (R1, R2; the group is hidden with the same rule)
                                                                                      -> runs/rq1_tier6/T6/<unit>.npz

    python -m src.analysis.tier6_extra t5|t6 --unit U
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import torch

from . import reliance_fidelity as rf
from . import tier2_common as C
from . import tier5_common as T5
from .connectivity_baseline import build_adj, random_connected_subset
from .revision_pass import KEYS, hide, metrics
from .tier5_N1 import new_rows
from .tier6_explain import T6, aromatic, hidden_graph, t2_mols

T5M = ["eig", "gnnshap", "gnnexplainer", "sx_fixed", "ig", "sal"]
ELEM = {"C": [0], "N": [1], "O": [2], "other": [3, 4, 5, 6]}     # MUTAG one-hot [C, N, O, F, I, Cl, Br]
PARTS = ["gt", "arom", "other"] + [f"other_{e}" for e in ELEM] + ["other_nextgt"]


def part_masks(d):
    n = d.num_nodes; gt = d.node_gt_mask.cpu().numpy().astype(bool)
    ar = np.zeros(n, bool); ar[aromatic(d)] = True
    if (gt & ar).any():
        raise RuntimeError("a GT atom is aromatic")
    oth = ~gt & ~ar; el = d.x[:, :7].argmax(1).cpu().numpy()
    ei = d.edge_index.cpu().numpy(); nxt = np.zeros(n, bool); nxt[ei[1][gt[ei[0]]]] = True
    M = {"gt": gt, "arom": ar & ~gt, "other": oth, "other_nextgt": oth & nxt}
    for e, cols in ELEM.items():
        M[f"other_{e}"] = oth & np.isin(el, cols)
    return M


def t5(unit):
    dl, _ = C.load_data(C.dataset_of(unit)); old = T5.old_scores(unit)
    S = {w: old[w] for w in ("gnnexplainer", "sx_fixed", "ig", "sal")}
    for w in ("eig", "gnnshap"):
        S[w] = {m: np.asarray(v["node"], float) for m, v in new_rows(unit, w).items()}
    mols = T5.gt_mols(unit); A = {"unit": unit, "mols": np.array(mols)}
    for w in T5M:
        miss = [m for m in mols if m not in S[w]]
        if miss:
            raise RuntimeError(f"{unit} {w}: {len(miss)} molecules missing")
    out = {f"{w}__{p}": [] for w in T5M + ["rn"] for p in PARTS}; ks = []
    for m in mols:
        d = dl[m]; n = d.num_nodes; k = C.kbudget(n); M = part_masks(d); ks.append(k)
        for p in PARTS:
            out[f"rn__{p}"].append(float(M[p].mean()))
            for w in T5M:
                sc = lambda mask, q=M[p]: float((mask & q).sum() / mask.sum())
                out[f"{w}__{p}"].append(C.gea_random_ties(np.asarray(S[w][m], float), k, sc, f"{unit}|{w}|T6comp", m))
    for kk, v in out.items():
        A[kk] = np.array(v)
    A["k"] = np.array(ks); C.save_npz(f"{T6}/T5/{unit}.npz", **A); print(f"{unit}: T5 scored ({len(mols)} molecules)", flush=True)


def t6(unit, device):
    dl, _ = C.load_data(C.dataset_of(unit)); model, _ = rf.load_model(T5.spec(unit)["ckpt"], device); model.eval()
    R = {"mol_idx": [], "group_size": [], "n_pool": []}; skipped = {"full": [], "R1": [], "R2": []}
    for base in ("full", "R1", "R2"):
        for rule in ("R1", "R2") if base == "full" else (base,):
            for ref in ("grp", "conn", "node"):
                for kk in KEYS:
                    R[f"{base}_{rule}_{ref}_{kk}"] = []
        R[f"{base}_pred"] = []
    for m in t2_mols(unit):
        d = dl[m]; n = d.num_nodes; scaf = aromatic(d); gt = np.flatnonzero(d.node_gt_mask.cpu().numpy())
        Gf = rf.to_cpu_data(d); R["mol_idx"].append(m); R["group_size"].append(len(gt))
        for base in ("full", "R1", "R2"):
            if base == "full":
                G0, pool, loc = Gf, list(range(n)), {v: v for v in range(n)}
            else:
                G0, rem = hidden_graph(d, scaf, base); pool = rem
                loc = {v: i for i, v in enumerate(rem)} if base == "R1" else {v: v for v in rem}
            if base == "R1":
                R["n_pool"].append(len(pool))
            grp = [loc[v] for v in gt]; P = [loc[v] for v in pool]
            ok = len(grp) < len(P)
            if not ok:
                skipped[base].append(m)
            # random sets among the pool, connected in the induced graph of the pool
            sub = rf.keep_nodes(Gf, pool); adj = build_adj(sub.edge_index, len(P))
            rng = np.random.default_rng([86600, zlib.crc32(f"{unit}|{base}".encode()), m])
            conn = [[P[i] for i in sorted(random_connected_subset(adj, len(P), len(grp), rng))] for _ in range(C.N_DRAWS)]
            node = [[P[i] for i in rng.choice(len(P), size=len(grp), replace=False)] for _ in range(C.N_DRAWS)]
            sets = [("grp", grp)] + [("conn", s) for s in conn] + [("node", s) for s in node]
            rules = ("R1", "R2") if base == "full" else (base,)
            graphs, idx = [G0], []
            for ref, Mset in sets:
                comp = [v for v in P if v not in set(Mset)]
                for rule in rules:
                    idx.append((ref, rule, len(graphs), len(graphs) + 1)); graphs += [hide(G0, Mset, rule, None), hide(G0, comp, rule, None)]
            lg = rf.logits_of(model, graphs, device); c = int(lg[0].argmax()); R[f"{base}_pred"].append(c); acc = {}
            for ref, rule, ir, ik in idx:
                for kk, v in metrics(lg, c, ir, ik).items():
                    acc.setdefault((ref, rule, kk), []).append(v)
            for (ref, rule, kk), v in acc.items():
                R[f"{base}_{rule}_{ref}_{kk}"].append(float(np.mean(v)) if ok else np.nan)
    A = {k: np.array(v) for k, v in R.items()}
    for b, s in skipped.items():
        A[f"skipped_{b}"] = np.array(s, int)
    C.save_npz(f"{T6}/T6/{unit}.npz", unit=unit, **A)
    print(f"{unit}: T6 on {len(R['mol_idx'])} molecules; skipped {[len(s) for s in skipped.values()]}", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["t5", "t6"]); a.add_argument("--unit", required=True); a = a.parse_args()
    if a.cmd == "t5":
        t5(a.unit)
    else:
        t6(a.unit, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
