"""
RQ1 Tier 4, item L5: PGExplainer reproducibility under deterministic settings (runs/rq1_tier4/PLAN.md).
CUBLAS_WORKSPACE_CONFIG must be set in the job's environment (the job list does this). Here: use_deterministic_algorithms(True),
cudnn deterministic, seeds fixed (run_phase3 seeds with the unit's explainer seed), all before the model is loaded.

  smoke   two GPU runs of mutagenicity_dmpnn_s0; GPU is used for all repeats only if neither raises and the two float32 edge
          masks are bitwise identical; writes L5/device.json. Feasibility only, not part of P-L5.
  pg      one repeat of one unit on the chosen device -> L5/rep<r>/<unit>_cache.pt and <unit>_pg_logits.npz

    python -m src.analysis.tier4_L5 smoke
    python -m src.analysis.tier4_L5 pg --unit U --rep R --device cuda|cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import traceback

import numpy as np
import torch

from . import tier2_common as C
from . import tier4_common as T
from .tier2_explain import fold64

UNITS = [f"mutagenicity_{a}_s{s}" for a in ("dmpnn", "gine") for s in range(5)]


def deterministic(device):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set in the environment before Python starts")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if device == "cpu":
        torch.set_num_threads(1)


def run_pg(unit, out_dir, device):
    from ..explain import pyg_explainers as pe
    from ..explain import run_phase3 as rp3
    sp = C.orig_spec(unit); os.makedirs(out_dir, exist_ok=True)
    cache = f"{out_dir}/{unit}_cache.pt"; done = f"{out_dir}/{unit}_pg_logits.npz"
    if os.path.exists(done):
        return
    if os.path.exists(cache):
        os.remove(cache)
    holder = {}; orig_init = pe.PGExplainerWrapper.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k); holder["w"] = self
    pe.PGExplainerWrapper.__init__ = init
    try:
        rp3.main(["--dataset", sp["dataset"], "--ckpt", sp["ckpt"], "--seed", str(sp["seed"]), "--explainers", "pgexplainer", "--masking", "R3",
                  "--device", device, "--cache", cache, "--out", f"{out_dir}/{unit}_out.json", *sp["args"]])
    finally:
        pe.PGExplainerWrapper.__init__ = orig_init
    blob = torch.load(cache, weights_only=False); alg = holder["w"].algorithm; dev = torch.device(device)
    mlp = copy.deepcopy(alg.mlp).double().eval(); dl, _ = C.load_data(sp["dataset"]); lg_all, ns = [], []
    with torch.no_grad():
        for mi in blob["key"]["test_idx"]:
            d = dl[mi].clone().to(dev)
            z = alg._get_embeddings(holder["w"].model, d.x, d.edge_index, edge_attr=getattr(d, "edge_attr", None)).double()
            lg = mlp(alg._get_inputs(z, d.edge_index, None)).view(-1).cpu().numpy()
            lg_all.append(lg); ns.append(fold64(1.0 / (1.0 + np.exp(-lg)), d.edge_index.cpu().numpy(), d.num_nodes))
    C.save_npz(done, mol_idx=np.array(blob["key"]["test_idx"]), logits=np.array(lg_all, dtype=object), node_score64=np.array(ns, dtype=object), device=device)
    print(f"{unit}: PGExplainer on {device} done ({len(ns)} mols) -> {out_dir}", flush=True)


def edge_masks(cache):
    return [np.asarray(r["edge_importance"], dtype=np.float32) for r in C.torch_load(cache)["results"]["pgexplainer"]]


def smoke():
    f = f"{T.T4}/L5/device.json"; u = "mutagenicity_dmpnn_s0"; rep = {"unit": u}
    try:
        deterministic("cuda")
        for r in (0, 1):
            run_pg(u, f"{T.T4}/L5/smoke/run{r}", "cuda")
        a, b = (edge_masks(f"{T.T4}/L5/smoke/run{r}/{u}_cache.pt") for r in (0, 1))
        same = len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
        rep.update(error=None, identical=bool(same), n_mols=len(a), max_abs_diff=float(max(np.abs(x - y).max() if len(x) else 0.0 for x, y in zip(a, b))))
        rep["device"] = "cuda" if same else "cpu"
    except Exception as e:                      # noqa: BLE001 -- any error on GPU means CPU, by the PLAN's rule
        rep.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:], device="cpu")
    os.makedirs(os.path.dirname(f), exist_ok=True); json.dump(rep, open(f, "w"), indent=1)
    print(f"L5 smoke: device {rep['device']} ({rep.get('error') or ('identical' if rep.get('identical') else 'not identical')})", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["smoke", "pg"]); a.add_argument("--unit"); a.add_argument("--rep", type=int)
    a.add_argument("--device", choices=["cuda", "cpu"]); a = a.parse_args()
    if a.cmd == "smoke":
        smoke()
    else:
        deterministic(a.device); run_pg(a.unit, f"{T.T4}/L5/rep{a.rep}", a.device)
