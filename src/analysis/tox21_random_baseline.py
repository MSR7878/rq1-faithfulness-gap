"""
Random baseline for Tox21 Fidelity+/-/GEF (no GT -> no GEA; this is the
Fidelity-side counterpart of mutag/bxaic_random_baseline.py's GEA baseline).
Uses the ORIGINAL scale-out caches/ckpts (12 endpoints, seed 0, n=30,
runs/expl_cache_tox21_<EP>.pt + runs/ckpt_tox21_<EP>.pt) -- does not wait for
P3's 5-seed re-run; if the real explainers don't separate from random here
either, that's a second, independent confirmation of F1's noise-domination
claim (the first being std >> mean within each real explainer).

random-node: uniform random per-node score -> same top-k=0.25 masking
  pipeline (mask_r3_distribution_aware / mask_r2_zero_fill /
  mask_r1_hard_removal) as any real explainer.
random-edge: uniform random per-edge score -> node_importance_from_edges
  (PGExplainer's own fold) -> same masking pipeline.
Both scored under R3-mean/R3-mode/R2/R1 exactly like score_one, using the
SAME target class the real explainers used (pulled from the cache, not
recomputed) so it's an apples-to-apples comparison on the same molecules.

    python -m src.analysis.tox21_random_baseline
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data import LOADERS
from ..explain.common import ExplanationResult, node_importance_from_edges
from ..explain.run_phase3 import FILLS_BY_MASKING, K_FRAC, load_model, score_one
from ..metrics.masking import training_fill_vector

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
EPS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
       "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
RUNS = "runs"


def one_endpoint(ep: str, device):
    ckpt_path = os.path.join(RUNS, f"ckpt_tox21_{ep}.pt")
    cache_path = os.path.join(RUNS, f"expl_cache_tox21_{ep}.pt")
    model, ck = load_model(ckpt_path, device)
    data_list, meta = LOADERS[f"tox21_{ep}"]()
    blob = torch.load(cache_path, weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    train_idx = list(ck["split"]["train"])
    x_train = torch.cat([data_list[i].x.float() for i in train_idx], dim=0)
    fills = {s: training_fill_vector(x_train, s) for s in FILLS_BY_MASKING["R3"]}

    torch.manual_seed(300000)
    rng = np.random.default_rng(0)
    rows = {"random_node": {m: [] for m in ("R3", "R2", "R1")},
            "random_edge": {m: [] for m in ("R3", "R2", "R1")}}
    for k, mi in enumerate(test_idx):
        d = data_list[mi].to(device)
        n = d.num_nodes
        tgt = res["gnnexplainer"][k]["target"]  # same predicted class every explainer used

        node_score = torch.rand(n)
        edge_score = torch.rand(d.edge_index.shape[1])
        folded = node_importance_from_edges(edge_score, d.edge_index, n)

        for kind, imp in (("random_node", node_score), ("random_edge", folded)):
            result = ExplanationResult(node_importance=imp, target=tgt)
            for m, fill_name in (("R3", "mean"), ("R2", "zero"), ("R1", "hard")):
                fv = fills.get(fill_name) if m == "R3" else None
                row = score_one(model, d, result, fv, device, False, m)
                rows[kind][m].append(row)
    return rows, ck["metrics"].get("test_auroc")


def summarize(rows_list: list[dict], metric: str) -> tuple[float, float]:
    v = np.array([r[metric] for r in rows_list], float)
    return float(v.mean()), float(v.std())


def real_summary(ep: str) -> dict:
    j = json.load(open(f"runs/phase3_tox21_{ep}.json"))
    S = j["summary"]
    out = {}
    for m, fill in (("R3", "mean"), ("R2", "zero"), ("R1", "hard")):
        out[m] = {e: {k: S[m][fill][e][k][0] for k in ("fid_plus", "fid_minus", "gef")}
                  for e in EXPL if e in S[m][fill]}
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 106)
    print(" Tox21 Fidelity+/-/GEF -- real explainers vs random-node/random-edge baselines, per endpoint")
    print("=" * 106)
    agg = {m: {k: {"gnnexplainer": [], "pgexplainer": [], "subgraphx": [], "random_node": [], "random_edge": []}
              for k in ("fid_plus", "fid_minus", "gef")} for m in ("R3", "R2", "R1")}
    for ep in EPS:
        rows, auroc = one_endpoint(ep, device)
        real = real_summary(ep)
        print(f"\n[{ep}]  test_auroc={auroc:.3f}")
        for m in ("R3", "R2", "R1"):
            print(f"  {m:<4}{'expl':<8}{'Fid+':>10}{'Fid-':>10}{'GEF':>10}")
            for e in EXPL:
                r = real[m].get(e)
                if not r:
                    continue
                print(f"       {SH[e]:<8}{r['fid_plus']:>10.3f}{r['fid_minus']:>10.3f}{r['gef']:>10.3f}")
                for k in ("fid_plus", "fid_minus", "gef"):
                    agg[m][k][e].append(r[k])
            for kind in ("random_node", "random_edge"):
                fp, _ = summarize(rows[kind][m], "fid_plus")
                fm, _ = summarize(rows[kind][m], "fid_minus")
                gf, _ = summarize(rows[kind][m], "gef")
                tag = "RandN" if kind == "random_node" else "RandE"
                print(f"       {tag:<8}{fp:>10.3f}{fm:>10.3f}{gf:>10.3f}")
                agg[m]["fid_plus"][kind].append(fp)
                agg[m]["fid_minus"][kind].append(fm)
                agg[m]["gef"][kind].append(gf)

    print("\n" + "=" * 106)
    print(" AGGREGATE across 12 endpoints (mean of per-endpoint means) -- does ANY explainer separate from random?")
    print("=" * 106)
    for m in ("R3", "R2", "R1"):
        print(f"\n [{m}]")
        print(f"  {'':<8}{'Fid+':>10}{'Fid-':>10}{'GEF':>10}")
        for e in ("gnnexplainer", "pgexplainer", "subgraphx", "random_node", "random_edge"):
            cells = []
            for k in ("fid_plus", "fid_minus", "gef"):
                v = np.array(agg[m][k][e], float)
                cells.append(f"{v.mean():>10.3f}")
            label = SH.get(e, "RandN" if e == "random_node" else "RandE")
            print(f"  {label:<8}" + "".join(cells))

    print("\n" + "=" * 106)
    print(" real explainer vs matched random, paired by ENDPOINT (n=12 -- the seed-aware lesson: the unit")
    print(" of independence here is the endpoint, not the molecule; Wilcoxon signed-rank, Holm over the 3")
    print(" real-vs-random pairs within each (masking, metric) cell)")
    print("=" * 106)
    from itertools import product as _product
    from scipy.stats import wilcoxon as _wilcoxon

    def _p(a, b):
        d = a - b
        return 1.0 if np.allclose(d, 0) else _wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue

    def _star(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

    def _holm(pv):
        items = sorted(pv.items(), key=lambda kv: kv[1])
        out, running, mlen = {}, 0.0, len(items)
        for rank, (kk, p) in enumerate(items):
            running = min(max(running, (mlen - rank) * p), 1.0)
            out[kk] = running
        return out

    matched = {"gnnexplainer": "random_node", "subgraphx": "random_node", "pgexplainer": "random_edge"}
    for m, k in _product(("R3", "R2", "R1"), ("fid_plus", "fid_minus", "gef")):
        raw = {}
        for e, rnd in matched.items():
            a, b = np.array(agg[m][k][e]), np.array(agg[m][k][rnd])
            raw[e] = _p(a, b)
        adj = _holm(raw)
        cells = []
        for e, rnd in matched.items():
            a, b = np.array(agg[m][k][e]), np.array(agg[m][k][rnd])
            hi = SH[e] if a.mean() > b.mean() else "Rand"
            cells.append(f"{SH[e]}vs{('RandN' if rnd=='random_node' else 'RandE')}:{_star(adj[e]):<3}({hi})")
        print(f"  {m:<4}{k:<10}" + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
