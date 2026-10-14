"""
EXPLORATORY, read-only: PGExplainer top-k GEA split by degenerate (std == 0) vs valid
masks on GT-bearing Mutagenicity test molecules, per architecture and seed, and
whether top-k on a constant mask picks the first k atoms. Writes nothing.

    PYTHONPATH=. python -m src.analysis.arch_pg_tie_check
"""
import torch, numpy as np
from src.data import LOADERS
from src.metrics.gea_set import gea_max_over_set
from src.analysis.oracle_ceiling import budget
dl, meta = LOADERS["mutagenicity_graphxai"](); motifs = meta.extras["motifs"]
def g(mask, mi): return gea_max_over_set(mask, motifs[mi]).max_over_set
def mn(v): return np.mean(v) if v else float("nan")
for arch, path in [("gcn", "runs/arch_gcn"), ("gine", "runs/arch_gine"), ("gin", "runs/arch_gin"), ("dmpnn", "runs/mutagenicity")]:
    print("==", arch)
    for s in range(5):
        cp = f"{path}/mutagenicity_graphxai/cache_s{s}.pt" if arch != "dmpnn" else f"{path}/cache_s{s}.pt"
        b = torch.load(cp, weights_only=False)
        idx = b["key"]["test_idx"]; res = b["results"]["pgexplainer"]
        deg, nd, first, same = [], [], [], 0
        for p, mi in enumerate(idx):
            d = dl[mi]
            if not bool(d.node_gt_mask.any()): continue
            n = d.num_nodes; k = budget(n)
            imp = torch.tensor(res[p]["node_importance"], dtype=torch.float)
            m = np.zeros(n, bool); top = torch.topk(imp, k).indices.numpy(); m[top] = True
            fk = np.zeros(n, bool); fk[:k] = True
            if float(imp.std()) == 0:
                deg.append(g(m, mi)); first.append(g(fk, mi)); same += int(set(top.tolist()) == set(range(k)))
            else:
                nd.append(g(m, mi))
        print(f" s{s}: degenerate GT-bearing {len(deg):3d}  PG top-k GEA {mn(deg):.3f} | first-k-atoms mask GEA {mn(first):.3f}"
              f" (top-k == first k atoms in {same}/{len(deg)}) | non-degenerate {len(nd):3d}  PG GEA {mn(nd):.3f}")
