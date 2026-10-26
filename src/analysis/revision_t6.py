"""
RQ1 revision, T6: why does removing random atoms RAISE the margin of the GCN
model under R1 (negative null drop in the reliance test)? (runs/rq1_revision/PLAN.md)

Checks, Mutagenicity, all four architectures, 5 seeds, GT-bearing explained molecules:
  a) identity: keep_nodes(G, all atoms) reproduces G's logits exactly;
  b) sign: drop = m(G) - m(G with the set removed), recomputed directly here;
  c) drop for 20 random connected sets of the GT size, split by predicted class;
  d) node removal (R1) vs removing only the set's edges (atoms and features kept)
     vs zero-fill (R2);
  e) norm of the sum-pooled graph embedding (input to the FFN head) before/after R1 removal;
  f) GCN only: number of remaining atoms vs drop (Spearman).
Writes runs/rq1_revision/t6.json and prints a report.

    python -m src.analysis.revision_t6
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Data

from . import reliance_fidelity as rf
from .connectivity_baseline import build_adj, random_connected_subset

OUT = "runs/rq1_revision"
DIRS = {"D-MPNN": "runs/mutagenicity", "GIN": "runs/arch_gin/mutagenicity_graphxai", "GINE": "runs/arch_gine/mutagenicity_graphxai",
        "GCN": "runs/arch_gcn/mutagenicity_graphxai"}


def drop_edges(G, idx):
    idx = torch.as_tensor(sorted(idx)); ei = G.edge_index
    keep = ~(torch.isin(ei[0], idx) | torch.isin(ei[1], idx))
    ea = getattr(G, "edge_attr", None)
    return Data(x=G.x.clone().float(), edge_index=ei[:, keep], edge_attr=None if ea is None else ea[keep])


def pooled_norm(model, H, device):
    """L2 norm of the sum-pooled graph representation that enters the FFN head (hooked)."""
    got = {}
    h = model.ffn.register_forward_pre_hook(lambda mod, inp: got.__setitem__("g", inp[0]))
    with torch.no_grad():
        model(H.x.to(device), H.edge_index.to(device), None if H.edge_attr is None else H.edge_attr.to(device))
    h.remove()
    return float(got["g"].norm())


def main():
    P = print; device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rep = {}
    P("T6 -- negative null drop under R1 (drop = m(G) - m(G with the set removed); negative = margin goes UP)")
    for arch, d0 in DIRS.items():
        rows = {"pred": [], "R1": [], "edges_only": [], "R2": [], "norm_G": [], "norm_R1": [], "n_left": []}
        ident = 0.0
        for s in range(5):
            name = f"mutagenicity_{arch.lower().replace('-', '')}_s{s}"
            spec = {"ckpt": f"{d0}/ckpt_s{s}.pt", "cache": f"{d0}/cache_s{s}.pt", "out": f"{d0}/out_s{s}.json"}
            model, ck, blob, out, dl, meta = rf.load_unit(name, spec, device)
            motifs = meta.extras["motifs"]; rng = np.random.default_rng([80000, s])
            for mi in blob["key"]["test_idx"]:
                d = dl[mi]
                if not bool(d.node_gt_mask.any()):
                    continue
                G = rf.to_cpu_data(d); n = G.num_nodes; adj = build_adj(G.edge_index, n)
                size = int(np.mean([len(m[1]) for m in motifs[mi]])) or 1
                if size >= n:
                    continue
                sets = [sorted(random_connected_subset(adj, n, size, rng)) for _ in range(20)]
                graphs = [G, rf.keep_nodes(G, list(range(n)))]
                for M in sets:
                    graphs += [rf.keep_nodes(G, [v for v in range(n) if v not in set(M)]), drop_edges(G, M), rf.zero_nodes(G, M)]
                lg = rf.logits_of(model, graphs, device); c = int(lg[0].argmax()); m = (lg[:, c] - lg[:, 1 - c]).numpy()
                ident = max(ident, float((lg[0] - lg[1]).abs().max()))
                rows["pred"].append(c); rows["R1"].append(float(np.mean(m[0] - m[2::3]))); rows["edges_only"].append(float(np.mean(m[0] - m[3::3])))
                rows["R2"].append(float(np.mean(m[0] - m[4::3]))); rows["n_left"].append(n - size)
                rows["norm_G"].append(pooled_norm(model, G, device)); rows["norm_R1"].append(pooled_norm(model, graphs[2], device))
        A = {k: np.array(v) for k, v in rows.items()}
        P(f"\n  -- {arch}: {len(A['R1'])} molecules (5 seeds); identity keep_nodes(G, all) vs G: max |logit diff| {ident:.1e}")
        for lab in ("R1", "edges_only", "R2"):
            v = A[lab]
            P(f"    mean drop {lab:<11} {v.mean():+.3f}   share of molecules with a negative drop {np.mean(v < 0):.2f}   "
              + "   ".join(f"pred class {c}: {v[A['pred'] == c].mean():+.3f} (n {int((A['pred'] == c).sum())})" for c in (0, 1) if (A['pred'] == c).any()))
        P(f"    pooled-embedding norm: G {A['norm_G'].mean():.2f}  after R1 removal {A['norm_R1'].mean():.2f}  "
          f"(ratio {np.mean(A['norm_R1'] / A['norm_G']):.3f})")
        if arch == "GCN":
            r = spearmanr(A["n_left"], A["R1"]); P(f"    Spearman(remaining atoms, R1 drop) = {r.statistic:.3f} (p {r.pvalue:.1e})")
        rep[arch] = {"identity_max_abs": ident, **{f"mean_drop_{k}": float(A[k].mean()) for k in ("R1", "edges_only", "R2")},
                     **{f"neg_share_{k}": float(np.mean(A[k] < 0)) for k in ("R1", "edges_only", "R2")},
                     "drop_R1_by_class": {int(c): float(A["R1"][A["pred"] == c].mean()) for c in (0, 1) if (A["pred"] == c).any()},
                     "norm_G": float(A["norm_G"].mean()), "norm_R1": float(A["norm_R1"].mean())}
    json.dump(rep, open(f"{OUT}/t6.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
