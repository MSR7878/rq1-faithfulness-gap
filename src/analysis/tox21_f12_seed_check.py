"""
F12 seed-aware re-test: F12 claims (from the ORIGINAL 1-seed scale-out)
"Under R2, SubgraphX takes a significantly higher Fid+ than GNN/PG on 5/12
Tox21 endpoints". Re-run the exact same per-endpoint 3-way pairwise test
(GNN-PG, GNN-SX, PG-SX; Fid+, Holm over the 3 pairs; n=30 molecules)
independently for each of the 5 priority-sweep seeds -- same audit as F11.

    python -m src.analysis.tox21_f12_seed_check
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def main():
    # sx_wins[seed] = count of endpoints where SX significantly beats BOTH GNN and PG on R2 Fid+
    for masking, fill in (("R2", "zero"), ("R3", "mean"), ("R1", "hard")):
        print(f"\n{'='*90}\n masking={masking} (fill={fill}) -- SX vs GNN/PG on Fid+, per seed, n=30/endpoint\n{'='*90}")
        for s in SEEDS:
            sx_beats_both = 0
            per_ep = []
            for ep in EPS:
                path = os.path.join(OUT, f"out_tox21_{ep}_s{s}.json")
                j = json.load(open(path))
                pm = j["per_molecule"][masking][fill]
                arr = {e: np.array([row["fid_plus"] for row in pm[e]]) for e in EXPL if e in pm}
                if len(arr) < 3:
                    continue
                raw = {f"{a}-{b}": _wpair(arr[a], arr[b]) for a, b in PAIRS}
                adj = holm(raw)
                sx_gt_gnn = adj["gnnexplainer-subgraphx"] < 0.05 and arr["subgraphx"].mean() > arr["gnnexplainer"].mean()
                sx_gt_pg = adj["pgexplainer-subgraphx"] < 0.05 and arr["subgraphx"].mean() > arr["pgexplainer"].mean()
                if sx_gt_gnn and sx_gt_pg:
                    sx_beats_both += 1
                    per_ep.append(ep)
            print(f"  seed {s}: SX sig-beats BOTH GNN and PG on {sx_beats_both}/12 endpoints  {per_ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
