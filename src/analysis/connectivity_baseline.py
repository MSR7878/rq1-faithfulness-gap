"""
Connectivity-matched random baseline for SubgraphX (framing-memo follow-up,
item 1). F15/F16's existing random-node baseline is UNCONSTRAINED (uniform
k-of-N node subset, fixed budget k=int(0.25*N)) -- but SubgraphX only ever
proposes CONNECTED subgraphs, so an unconstrained null may be an unfair
comparison on compact motifs. This adds two same-molecule, same-size nulls
matched to SX's own REAL per-molecule mask size (mean-threshold, the
binarization F15/F16's GEA numbers use -- see F14's mechanism note):

  rand_conn : a random CONNECTED node-induced subgraph of the SAME size as
              SX's own predicted mask on that molecule (random frontier
              growth from a random start node -- a standard way to sample a
              connected induced subgraph uniformly-ish without enumerating
              all connected subsets).
  rand_unc  : a random UNCONSTRAINED k-of-N subset of the SAME size (the
              size-matched analogue of F15/F16's RandN, but per-molecule
              size instead of a fixed 0.25*N budget).

20 random draws per molecule per condition (averaged) to control sampling
noise, especially for indole (only ~13 GT-present test molecules/seed).

Datasets: mutag_graphxai (5 seeds) + B-XAIC indole (5 seeds), per item 1's
scope (not the other 3 B-XAIC tasks).

    python -m src.analysis.connectivity_baseline
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import wilcoxon
from ..data import LOADERS

SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"
N_DRAWS = 20

DATASETS = {
    "mutag_graphxai": ("mutag_graphxai", lambda s: f"{OUT}/cache_mutag_graphxai_s{s}.pt"),
    "indole": ("bxaic_indole", lambda s: f"{OUT}/cache_bxaic_indole_s{s}.pt"),
}


def jaccard(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt, pred = gt.bool(), pred.bool()
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0


def build_adj(edge_index: torch.Tensor, n: int) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    ei = edge_index.numpy()
    for a, b in zip(ei[0], ei[1]):
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    return adj


def random_connected_subset(adj: list[set[int]], n: int, k: int, rng: np.random.Generator) -> set[int]:
    if k <= 0:
        return set()
    k = min(k, n)
    start = int(rng.integers(n))
    selected = {start}
    frontier = set(adj[start]) - selected
    while len(selected) < k:
        if not frontier:
            # disconnected component smaller than k -- restart from an
            # unvisited node (should be rare/never for these molecule graphs)
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(sorted(frontier)))
        selected.add(nxt)
        frontier |= adj[nxt]
        frontier -= selected
    return selected


def random_unconstrained_subset(n: int, k: int, rng: np.random.Generator) -> set[int]:
    if k <= 0:
        return set()
    k = min(k, n)
    return set(rng.choice(n, size=k, replace=False).tolist())


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def run_dataset(name: str, loader_key: str, path_fn):
    data_list, meta = LOADERS[loader_key]()
    print(f"\n{'='*100}\n {name}\n{'='*100}")
    for s in SEEDS:
        blob = torch.load(path_fn(s), weights_only=False)
        test_idx = list(blob["key"]["test_idx"])
        res = blob["results"]["subgraphx"]
        rng = np.random.default_rng(10_000 + s)  # separate stream, independent of any explain seed
        sx_vals, conn_vals, unc_vals, sizes = [], [], [], []
        for k, mi in enumerate(test_idx):
            gt = data_list[mi].node_gt_mask.bool()
            if not bool(gt.any()):
                continue
            n = data_list[mi].num_nodes
            adj = build_adj(data_list[mi].edge_index, n)
            imp = torch.as_tensor(res[k]["node_importance"], dtype=torch.float)
            mask = imp > imp.mean()
            ksize = int(mask.sum())
            sizes.append(ksize)
            sx_vals.append(jaccard(gt, mask))
            conn_draws = [jaccard(gt, torch.tensor([i in random_connected_subset(adj, n, ksize, rng) for i in range(n)]))
                          for _ in range(N_DRAWS)]
            unc_draws = [jaccard(gt, torch.tensor([i in random_unconstrained_subset(n, ksize, rng) for i in range(n)]))
                         for _ in range(N_DRAWS)]
            conn_vals.append(float(np.mean(conn_draws)))
            unc_vals.append(float(np.mean(unc_draws)))
        sx_a, conn_a, unc_a = np.array(sx_vals), np.array(conn_vals), np.array(unc_vals)
        p_conn = _wpair(sx_a, conn_a)
        p_unc = _wpair(sx_a, unc_a)
        sx_wins_conn = int((sx_a > conn_a).sum()); ties_conn = int((sx_a == conn_a).sum())
        sx_wins_unc = int((sx_a > unc_a).sum()); ties_unc = int((sx_a == unc_a).sum())
        n_mol = len(sx_vals)
        print(f"  seed {s}  n={n_mol}  mean|mask|={np.mean(sizes):.1f}  "
              f"SX={sx_a.mean():.3f}  RandConn={conn_a.mean():.3f}  RandUnc={unc_a.mean():.3f}  "
              f"SX-vs-Conn p={p_conn:.2e}{star(p_conn)} (SX>Conn in {sx_wins_conn}/{n_mol}, ties={ties_conn})  "
              f"SX-vs-Unc p={p_unc:.2e}{star(p_unc)} (SX>Unc in {sx_wins_unc}/{n_mol}, ties={ties_unc})")
    return 0


def main():
    for name, (loader_key, path_fn) in DATASETS.items():
        run_dataset(name, loader_key, path_fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
