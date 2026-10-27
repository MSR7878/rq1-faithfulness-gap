"""
RQ1 revision, T4 completion: re-run the T4 settings b, c, d1, d2 on the 35 Tox21
units (24 originally collapsed + 11 clean partners) with the trained explainer MLP
saved, so the pre-sigmoid edge logits are exact (the first T4 run kept only the
float32 masks, so most jobs were 'undetermined' under the logit rule).

Same command, seed, subsample and per-epoch logging as src.analysis.revision_t4
(its instrumented train is reused unchanged). After training: the MLP state dict is
saved, and the MLP is evaluated in float64 on the explained molecules (as in
src.analysis.revision_logits).

Writes runs/rq1_revision/t4_logits/: nets/<unit>_<setting>.pt (MLP state dict),
<unit>_<setting>.npz (mol_idx, logits float64, float32 node mask of the re-run,
target), <unit>_<setting>.json (training log + counts). The first T4 run's outputs
(runs/rq1_revision/t4) are not touched.

    python -m src.analysis.revision_t4_logits list
    python -m src.analysis.revision_t4_logits run --jobs unit:setting ...
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from ..explain import pyg_explainers as pe
from ..explain import run_phase3 as rp3
from .revision_t4 import SETTINGS, instrumented_train, tox21_units

OUT = "runs/rq1_revision/t4_logits"
JOB_SETTINGS = ("c", "b", "d1", "d2")  # longest first


def degenerate_logits(lg):
    lg = np.asarray(lg, dtype=np.float64)
    return len(lg) == 0 or (lg.max() - lg.min()) <= 1e-6 * max(1.0, abs(lg.mean()))


def run_job(unit, setting, spec, device):
    epochs, lr = SETTINGS[setting]
    log = {"unit": unit, "setting": setting, "epochs": epochs, "lr": lr, "loss": [], "grad_norm": [], "saturation": []}
    holder = {}
    orig_train, orig_init = pe.PGExplainerWrapper.train, pe.PGExplainerWrapper.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k); holder["w"] = self
    pe.PGExplainerWrapper.train = instrumented_train(lr, log)
    pe.PGExplainerWrapper.__init__ = init
    t0 = time.time()
    try:
        cache = f"{OUT}/rerun/{unit}_{setting}_cache.pt"
        rp3.main(["--dataset", spec["dataset"], "--ckpt", spec["ckpt"], "--seed", str(spec["explainer_seed"]), "--explainers", "pgexplainer",
                  "--masking", "R3", "--pg-epochs", str(epochs), "--cache", cache, "--out", f"{OUT}/rerun/{unit}_{setting}_out.json", *spec["args"]])
    finally:
        pe.PGExplainerWrapper.train, pe.PGExplainerWrapper.__init__ = orig_train, orig_init
    alg = holder["w"].algorithm
    torch.save(alg.mlp.state_dict(), f"{OUT}/nets/{unit}_{setting}.pt")
    mlp = copy.deepcopy(alg.mlp).double().eval()
    new = torch.load(cache, weights_only=False); res = new["results"]["pgexplainer"]
    from ..data import LOADERS
    dl, _ = LOADERS[spec["dataset"]]()
    logits, dev = [], 0.0
    with torch.no_grad():
        for pos, mi in enumerate(new["key"]["test_idx"]):
            d = dl[mi].clone().to(device)
            z = alg._get_embeddings(holder["w"].model, d.x, d.edge_index, edge_attr=getattr(d, "edge_attr", None)).double()
            lg = mlp(alg._get_inputs(z, d.edge_index, None)).view(-1).cpu().numpy()
            logits.append(lg)
            if len(lg):
                dev = max(dev, float(np.max(np.abs(np.asarray(res[pos]["edge_importance"]) - 1.0 / (1.0 + np.exp(-lg))))))
    node32 = [np.asarray(r["node_importance"], dtype=np.float32) for r in res]
    old = [float(torch.as_tensor(m, dtype=torch.float).std()) == 0.0 for m in node32]
    newd = [degenerate_logits(a) for a in logits]
    allv = np.concatenate([a for a in logits if len(a)]) if any(len(a) for a in logits) else np.zeros(0)
    sig = 1.0 / (1.0 + np.exp(-allv))
    np.savez(f"{OUT}/{unit}_{setting}.npz", mol_idx=np.array(new["key"]["test_idx"]), logits=np.array(logits, dtype=object),
             node_mask32=np.array(node32, dtype=object), target=np.array([int(r["target"]) for r in res]))
    log.update({"n": len(res), "orig_collapsed": spec["collapsed"],
                "n_deg_old_f32": int(sum(old)), "collapsed_old_f32": sum(old) / len(res) >= 0.5,
                "n_deg_logit": int(sum(newd)), "collapsed_logit": sum(newd) / len(res) >= 0.5,
                "logit_mean": float(allv.mean()) if len(allv) else None, "logit_min": float(allv.min()) if len(allv) else None,
                "logit_max": float(allv.max()) if len(allv) else None,
                "sat_1e6_explained": float(np.mean((sig < 1e-6) | (sig > 1 - 1e-6))) if len(allv) else None,
                "sigmoid_logits_vs_rerun_mask_max_dev": dev, "seconds": time.time() - t0})
    json.dump(log, open(f"{OUT}/{unit}_{setting}.json", "w"), indent=1)
    print(f"{unit} {setting}: old-rule deg {sum(old)}/{len(res)}  logit-rule deg {sum(newd)}/{len(res)}  dev {dev:.1e}  ({log['seconds']:.0f}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--jobs", nargs="*", default=[])
    a = ap.parse_args(argv)
    spec, _ = tox21_units()
    if a.cmd == "list":
        for s in JOB_SETTINGS:
            for u in spec:
                print(f"{u}:{s}")
        return 0
    os.makedirs(f"{OUT}/rerun", exist_ok=True); os.makedirs(f"{OUT}/nets", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in a.jobs:
        u, s = j.split(":")
        if not os.path.exists(f"{OUT}/{u}_{s}.json"):
            run_job(u, s, spec[u], device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
