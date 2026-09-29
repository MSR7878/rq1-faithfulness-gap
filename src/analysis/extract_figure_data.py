"""
Pure data extraction for figure-drawing -- NO new computation, no stats.
Pulls real per-molecule GEA (GNN/PG/SX, R3/mean, masking-independent) out of
each out_*.json's per_molecule block for mutag_graphxai + B-XAIC indole,
into a compact npz alongside the already-saved random-baseline arrays --
so figure code needs only npz files + collapse_table.csv, not the much
larger out_*.json / cache_*.pt files.

    python -m src.analysis.extract_figure_data
"""
from __future__ import annotations
import json
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]

DATASETS = {
    "mutag_graphxai": lambda s: f"{OUT}/out_mutag_graphxai_s{s}.json",
    "indole": lambda s: f"{OUT}/out_bxaic_indole_s{s}.json",
}


def main():
    for name, path_fn in DATASETS.items():
        flat = {}
        for s in SEEDS:
            j = json.load(open(path_fn(s)))
            pm = j["per_molecule"]["R3"]["mean"]
            for e in EXPL:
                # 'gea' is absent for GT-empty molecules (gea_tp/fp/fn undefined
                # without a GT to compare against) -- skip those rows, same
                # GT-present filter every other GEA script in this project uses.
                vals = [row["gea"] for row in pm[e] if "gea" in row]
                flat[f"{e}_s{s}"] = np.array(vals, float)
        out_path = f"{OUT}/figure_gea_{name}.npz"
        np.savez(out_path, **flat)
        print(f"saved -> {out_path}  ({', '.join(f'{k}: n={len(v)}' for k, v in flat.items() if k.endswith('_s0'))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
