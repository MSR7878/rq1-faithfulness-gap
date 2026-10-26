"""
RQ1 revision, T4 addendum (runs/rq1_revision/PLAN.md, "ADDENDUM to T4"):
pre-sigmoid PGExplainer edge logits for every re-runnable unit.

The trained explainer MLPs were never saved, so each unit's PGExplainer training is
re-run with the original command, seed and subsample (PGExplainer only), and the
trained MLP is evaluated in float64 on the explained molecules. The re-run's
subsample must equal the stored cache key (otherwise the unit is skipped and
reported); its masks are compared with the stored ones.

Per molecule saved: edge logits (float64), node scores in float64 (mean of
float64 sigmoid(logit) over incident directed edges -- same folding as
explain.common.node_importance_from_edges), and the stored float32 node mask.
Writes runs/rq1_revision/logits/<unit>.npz (+ .json summary).

    python -m src.analysis.revision_logits list
    python -m src.analysis.revision_logits run --units ...
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from ..data.standard import TOX21_TASKS
from ..explain import pyg_explainers as pe
from ..explain import run_phase3 as rp3

OUT = "runs/rq1_revision/logits"
TOX = ["--limit", "30", "--pg-train-limit", "200", "--stratify-frac", "0.5", "--max-nodes", "60"]
BXA = ["--limit", "400", "--stratify-frac", "0.9", "--max-nodes", "80", "--pg-train-limit", "200"]
MUT = ["--max-nodes", "80"]


def units() -> dict:
    U = {}
    for ep in TOX21_TASKS:
        for s in range(5):
            U[f"tox21_{ep}_s{s}"] = dict(dataset=f"tox21_{ep}", ckpt=f"runs/prio/ckpt_tox21_{ep}_s{s}.pt", cache=f"runs/prio/cache_tox21_{ep}_s{s}.pt", seed=s, args=TOX)
    for t in ("indole", "PAINS", "X", "P"):
        for s in range(5):
            U[f"bxaic_{t}_s{s}"] = dict(dataset=f"bxaic_{t}", ckpt=f"runs/prio/ckpt_bxaic_{t}_s{s}.pt", cache=f"runs/prio/cache_bxaic_{t}_s{s}.pt", seed=s, args=BXA)
    for s in range(10):
        d = "runs/prio" if s < 5 else "runs/rq1_followups/phase1"
        U[f"mutag_graphxai_s{s}"] = dict(dataset="mutag_graphxai", ckpt=f"{d}/ckpt_mutag_graphxai_s{s}.pt", cache=f"{d}/cache_mutag_graphxai_s{s}.pt",
                                         seed=s, args=["--explain-pool", "all"])
    for s in range(5):
        U[f"decoupled_mutag_graphxai_s{s}"] = dict(dataset="mutag_graphxai", ckpt=f"runs/prio/ckpt_mutag_graphxai_s{s}.pt",
                                                   cache=f"runs/decoupled/cache_mutag_graphxai_s{s}.pt", seed=1000 + s, args=["--explain-pool", "all"])
        U[f"mutagenicity_dmpnn_s{s}"] = dict(dataset="mutagenicity_graphxai", ckpt=f"runs/mutagenicity/ckpt_s{s}.pt", cache=f"runs/mutagenicity/cache_s{s}.pt",
                                             seed=1000 + s, args=MUT)
        U[f"mutagenicity_cv_f{s}"] = dict(dataset="mutagenicity_graphxai", ckpt=f"runs/mutagenicity_cv/ckpt_s{s}.pt", cache=f"runs/mutagenicity_cv/cache_s{s}.pt",
                                          seed=1000 + s, args=MUT)
        for a in ("gin", "gine", "gcn"):
            d = f"runs/arch_{a}/mutagenicity_graphxai"
            U[f"mutagenicity_{a}_s{s}"] = dict(dataset="mutagenicity_graphxai", ckpt=f"{d}/ckpt_s{s}.pt", cache=f"{d}/cache_s{s}.pt", seed=1000 + s, args=MUT)
    U["mutag"] = dict(dataset="mutag", ckpt="runs/ckpt_mutag.pt", cache="runs/expl_cache_mutag.pt", seed=0, args=[])
    U["bbbp"] = dict(dataset="bbbp", ckpt="runs/ckpt_bbbp.pt", cache="runs/expl_cache_bbbp.pt", seed=0, args=["--limit", "30", "--pg-train-limit", "150"])
    return U


def fold64(edge_vals: np.ndarray, ei: np.ndarray, n: int) -> np.ndarray:
    imp = np.zeros(n); cnt = np.zeros(n)
    for end in (0, 1):
        np.add.at(imp, ei[end], edge_vals); np.add.at(cnt, ei[end], 1.0)
    return imp / np.maximum(cnt, 1.0)


def run_unit(name, sp, device):
    holder = {}
    orig_init = pe.PGExplainerWrapper.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k); holder["w"] = self
    pe.PGExplainerWrapper.__init__ = init
    t0 = time.time()
    try:
        cache = f"{OUT}/rerun/{name}_cache.pt"
        rp3.main(["--dataset", sp["dataset"], "--ckpt", sp["ckpt"], "--seed", str(sp["seed"]), "--explainers", "pgexplainer",
                  "--masking", "R3", "--cache", cache, "--out", f"{OUT}/rerun/{name}_out.json", *sp["args"]])
    finally:
        pe.PGExplainerWrapper.__init__ = orig_init
    new = torch.load(cache, weights_only=False); old = torch.load(sp["cache"], weights_only=False)
    same = list(new["key"]["test_idx"]) == list(old["key"]["test_idx"]) and list(new["key"]["pg_train_idx"]) == list(old["key"]["pg_train_idx"])
    if not same:
        return {"unit": name, "status": "SKIPPED: re-run subsample differs from the stored cache key"}
    w = holder["w"]; alg = w.algorithm; mlp = copy.deepcopy(alg.mlp).double().eval()
    from ..data import LOADERS
    dl, _ = LOADERS[sp["dataset"]]()
    logits, nodes64, dev_mask = [], [], 0.0
    with torch.no_grad():
        for pos, mi in enumerate(new["key"]["test_idx"]):
            d = dl[mi].clone().to(device)
            z = alg._get_embeddings(w.model, d.x, d.edge_index, edge_attr=getattr(d, "edge_attr", None)).double()
            lg = mlp(alg._get_inputs(z, d.edge_index, None)).view(-1).cpu().numpy()
            logits.append(lg)
            nodes64.append(fold64(1.0 / (1.0 + np.exp(-lg)), d.edge_index.cpu().numpy(), d.num_nodes))
            dev_mask = max(dev_mask, float(np.max(np.abs(np.asarray(new["results"]["pgexplainer"][pos]["edge_importance"]) - 1.0 / (1.0 + np.exp(-lg))))) if len(lg) else 0.0)
    stored = old["results"]["pgexplainer"]; rerun = new["results"]["pgexplainer"]
    stored_dev = max((float(np.max(np.abs(np.asarray(a["edge_importance"]) - np.asarray(b["edge_importance"])))) for a, b in zip(stored, rerun)
                      if len(a["edge_importance"])), default=0.0)
    np.savez(f"{OUT}/{name}.npz", mol_idx=np.array(new["key"]["test_idx"]), logits=np.array(logits, dtype=object),
             node_score64=np.array(nodes64, dtype=object), stored_node_mask=np.array([np.asarray(r["node_importance"], dtype=np.float32) for r in stored], dtype=object),
             target=np.array([int(r["target"]) for r in stored]))
    info = {"unit": name, "status": "ok", "n": len(logits), "rerun_vs_stored_max_mask_dev": stored_dev,
            "sigmoid_logits_vs_rerun_mask_max_dev": dev_mask, "seconds": time.time() - t0}
    json.dump(info, open(f"{OUT}/{name}.json", "w"), indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    U = units()
    if a.cmd == "list":
        for n in U:
            print(n)
        return 0
    os.makedirs(f"{OUT}/rerun", exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n in a.units:
        if os.path.exists(f"{OUT}/{n}.json"):
            continue
        info = run_unit(n, U[n], device)
        if info["status"] != "ok":
            json.dump(info, open(f"{OUT}/{n}.json", "w"), indent=1)
        print(info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
