"""
RQ1 Tier 2, item G check failure diagnostic (not part of any registered test): the IG completeness check
(median |sum attr - (f(x) - f(0))| / |f(x) - f(0)| <= 0.05) failed on one unit. For the given units, recompute IG
with 50, 100, 200 and 500 steps (Captum, Gauss-Legendre, zero baseline) and report the completeness error and the
IG top-25% GEA (final protocol) at each step count. Writes runs/rq1_tier2/G/ig_check.txt.

    python -m src.analysis.tier2_ig_check UNIT ...
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from captum.attr import IntegratedGradients

from . import tier2_common as C
from ..explain.run_phase3 import load_model
from .tier2_explain import batched_forward


def main(units):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); lines = []
    for u in units:
        sp = C.orig_spec(u); model, _ = load_model(sp["ckpt"], dev); model.eval()
        dkey = C.dataset_of(u); dl, _ = C.load_data(dkey)
        ti = [int(m) for m in C.stored_cache(u)["key"]["test_idx"]]; gt = C.gt_mol_idx(u, ti, dkey); nl = C.nulls(u, gt, dkey)
        for steps in (50, 100, 200, 500):
            rel, sc = [], {}
            for mi in ti:
                d = dl[mi]; x = d.x.float().to(dev)[None]; ei = d.edge_index.to(dev)
                ea = None if getattr(d, "edge_attr", None) is None else d.edge_attr.float().to(dev)
                f = batched_forward(model, ei, ea)
                with torch.no_grad():
                    c = int(f(x)[0].argmax()); fd = float(f(x)[0, c] - f(torch.zeros_like(x))[0, c])
                a, delta = IntegratedGradients(f).attribute(x, baselines=torch.zeros_like(x), target=c, n_steps=steps, return_convergence_delta=True)
                rel.append(float(delta.abs().sum()) / max(abs(fd), 1e-12)); sc[mi] = a[0].detach().abs().sum(-1).double().cpu().numpy()
            g = C.gea_array(u, "ig", lambda m: sc[m], gt, dkey)
            s = (f"{u} steps {steps:>3}: completeness rel. error median {np.median(rel):.4f}, 90th pct {np.percentile(rel, 90):.3f}, max {max(rel):.2f}; "
                 f"IG GEA {g.mean():.4f} (random-node {nl['rn'].mean():.4f})")
            print(s, flush=True); lines.append(s)
    open(f"{C.T2}/G/ig_check.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
