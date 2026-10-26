"""
RQ1 revision, T4 (runs/rq1_revision/PLAN.md): PGExplainer re-training on the
24 collapsed priority-sweep Tox21 units and their clean partners, under five
settings, with per-epoch logging. Each re-run uses the unit's original
command, seed and subsample (PGExplainer only; in the original runs PGExplainer
is trained before any explanation is made).

Settings: a = 30 epochs, lr 0.003 (current = PyG defaults); b = 100 epochs;
c = 300 epochs; d1 = lr 0.0009; d2 = lr 0.009 (30 epochs each).

Logged per epoch: mean loss; gradient norm of the explainer MLP (L2 over all
parameters, read after each train call, averaged over the epoch's graphs);
every 5 epochs and at the end: fraction of sigmoid edge outputs < 1e-3 or
> 1 - 1e-3 on the first 50 training graphs (no random numbers drawn, so the
training path is unchanged). After training: collapse status on the explained
molecules (degenerate = node-importance std < 1e-6 in float64; collapsed if
>= 50% degenerate) and mean edge-mask std.

    python -m src.analysis.revision_t4 list
    python -m src.analysis.revision_t4 run --jobs unit:setting ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from ..explain import pyg_explainers as pe
from ..explain import run_phase3 as rp3
from .followups_phase3 import units as p3_units

OUT = "runs/rq1_revision/t4"
SETTINGS = {"a": (30, 0.003), "b": (100, 0.003), "c": (300, 0.003), "d1": (30, 0.0009), "d2": (30, 0.009)}
STD_EPS = 1e-6


def tox21_units():
    spec, pairs = p3_units()
    keep = {u for p in pairs for u in p if u is not None and u.startswith("coupled_tox21")}
    return {u: spec[u] for u in sorted(keep)}, [p for p in pairs if p[0].startswith("coupled_tox21")]


def instrumented_train(lr, log):
    """Replacement for PGExplainerWrapper.train: same calls in the same order, plus logging."""
    def train(self, train_data):
        alg = self.algorithm
        for g in alg.optimizer.param_groups:
            g["lr"] = lr
        losses = []
        bar = tqdm(range(self.epochs), desc="PGExplainer train", unit="epoch", file=sys.stderr, mininterval=10.0)
        for epoch in bar:
            ep_loss, gn = 0.0, []
            for data in train_data:
                data = data.to(self.device)
                tgt = pe._predicted_class(self.model, data)
                loss = alg.train(epoch, self.model, data.x, data.edge_index, target=torch.tensor([tgt], device=self.device),
                                 edge_attr=getattr(data, "edge_attr", None))
                ep_loss += float(loss)
                gn.append(float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in alg.parameters() if p.grad is not None))))
            losses.append(ep_loss / max(len(train_data), 1))
            log["loss"].append(losses[-1]); log["grad_norm"].append(float(np.mean(gn)))
            if epoch % 5 == 4 or epoch == self.epochs - 1:
                log["saturation"].append((epoch + 1, saturation(self, train_data[:50])))
        bar.close()
        self._trained = True
        return losses
    return train


@torch.no_grad()
def saturation(wrapper, graphs):
    alg, sat, tot = wrapper.algorithm, 0, 0
    for data in graphs:
        data = data.to(wrapper.device)
        z = alg._get_embeddings(wrapper.model, data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None))
        s = torch.sigmoid(alg.mlp(alg._get_inputs(z, data.edge_index, None)).view(-1))
        sat += int(((s < 1e-3) | (s > 1 - 1e-3)).sum()); tot += s.numel()
    return sat / max(tot, 1)


def run_job(unit, setting, spec):
    epochs, lr = SETTINGS[setting]
    log = {"unit": unit, "setting": setting, "epochs": epochs, "lr": lr, "loss": [], "grad_norm": [], "saturation": []}
    orig = pe.PGExplainerWrapper.train
    pe.PGExplainerWrapper.train = instrumented_train(lr, log)
    t0 = time.time()
    try:
        cache = f"{OUT}/rerun/{unit}_{setting}_cache.pt"
        rp3.main(["--dataset", spec["dataset"], "--ckpt", spec["ckpt"], "--seed", str(spec["explainer_seed"]), "--explainers", "pgexplainer",
                  "--masking", "R3", "--pg-epochs", str(epochs), "--cache", cache, "--out", f"{OUT}/rerun/{unit}_{setting}_out.json", *spec["args"]])
    finally:
        pe.PGExplainerWrapper.train = orig
    res = torch.load(cache, weights_only=False)["results"]["pgexplainer"]
    stds = np.array([np.std(np.asarray(r["node_importance"], dtype=np.float64)) for r in res])
    deg = int((stds < STD_EPS).sum())
    log.update({"n": len(res), "n_degenerate": deg, "collapsed": deg / len(res) >= 0.5, "orig_collapsed": spec["collapsed"],
                "mask_std_edge": float(np.mean([np.std(np.asarray(r["edge_importance"], dtype=np.float64)) for r in res])),
                "seconds": time.time() - t0})
    json.dump(log, open(f"{OUT}/jobs/{unit}_{setting}.json", "w"), indent=1)
    print(f"{unit} {setting}: degenerate {deg}/{len(res)} collapsed={log['collapsed']} ({log['seconds']:.0f}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--jobs", nargs="*", default=[])
    a = ap.parse_args(argv)
    spec, pairs = tox21_units()
    if a.cmd == "list":
        for u in spec:
            for s in SETTINGS:
                print(f"{u}:{s}")
        return 0
    os.makedirs(f"{OUT}/rerun", exist_ok=True); os.makedirs(f"{OUT}/jobs", exist_ok=True)
    for j in a.jobs:
        u, s = j.split(":")
        if not os.path.exists(f"{OUT}/jobs/{u}_{s}.json"):
            run_job(u, s, spec[u])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
