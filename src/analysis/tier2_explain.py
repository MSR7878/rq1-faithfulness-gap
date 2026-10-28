"""
RQ1 Tier 2 (runs/rq1_tier2/PLAN.md): explanation jobs. Every job is resumable and writes atomically.

  grad      Saliency, Integrated Gradients (50 steps, zero baseline, Captum's default Gauss-Legendre rule) and
            Input x Gradient (Captum 0.7.0) on every explained molecule of a unit; target = the predicted-class
            logit; node attribution = sum over atom features of |attribution|. The graph is passed to Captum as
            one example x[None] (1, N, F); the interpolation steps are run as one disjoint batch of N-atom graphs
            with the same bonds. Checks per molecule: finite, non-zero, IG completeness |sum attr - (f(x) - f(0))|.
  sx        SubgraphX (DIG; zero filling; mc_l_shapley) on a list of molecules, sharded; every molecule seeded by
            crc32("<tag>|<unit>|<mol>") (python, numpy, torch), so a molecule's result does not depend on its shard.
  gp        GNNExplainer + PGExplainer through src.explain.run_phase3 with the unit's original arguments and
            explainer seed; the trained PGExplainer MLP is saved and evaluated in float64 (edge logits, node
            scores = mean float64 sigmoid over incident directed edges).
  randomise re-initialise a trained D-MPNN: "full" = reset_parameters() on every module, "head" = only the
            output FFN; seed 5000 + model seed; saved as a new checkpoint (same cfg and split) with test metrics.

    python -m src.analysis.tier2_explain grad --unit U --ckpt CK --mols CACHE --out F
    python -m src.analysis.tier2_explain sx --unit U --ckpt CK --mols-file F --shard i/n --out DIR [--rollout 20 --sample 30 --budget score --tag T]
    python -m src.analysis.tier2_explain gp --unit U --ckpt CK --out DIR
    python -m src.analysis.tier2_explain randomise --unit U --kind full|head --out CK
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
import zlib

import numpy as np
import torch

from ..explain import pyg_explainers as pe
from ..explain import run_phase3 as rp3
from ..explain.run_phase3 import load_model
from .revision_logits import units as orig_units
from .tier2_common import dataset_of, load_data, orig_spec, save_npz


def seed_all(s: int):
    random.seed(s); np.random.seed(s % (2 ** 32)); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ gradients
def batched_forward(model, edge_index, edge_attr):
    def f(xb):                                   # xb: (B, N, F) -> (B, C)
        B, N, Fd = xb.shape
        x = xb.reshape(B * N, Fd)
        ei = torch.cat([edge_index + i * N for i in range(B)], 1)
        ea = None if edge_attr is None else edge_attr.repeat(B, 1)
        batch = torch.arange(B, device=xb.device).repeat_interleave(N)
        return model(x, ei, ea, batch)
    return f


def run_grad(unit, ckpt, mols, out, device):
    from captum.attr import InputXGradient, IntegratedGradients, Saliency
    model, ck = load_model(ckpt, device); model.eval()
    dl, _ = load_data(dataset_of(unit))
    R = {"mol_idx": [], "target": [], "sal": [], "ig": [], "ixg": [], "ig_delta": [], "ig_fx_f0": [], "finite": [], "nonzero": []}
    for mi in mols:
        d = dl[int(mi)]
        x = d.x.float().to(device)[None]; ei = d.edge_index.to(device)
        ea = None if getattr(d, "edge_attr", None) is None else d.edge_attr.float().to(device)
        f = batched_forward(model, ei, ea)
        with torch.no_grad():
            c = int(f(x)[0].argmax()); fx = float(f(x)[0, c]); f0 = float(f(torch.zeros_like(x))[0, c])
        sal = Saliency(f).attribute(x, target=c, abs=True)
        ig, delta = IntegratedGradients(f).attribute(x, baselines=torch.zeros_like(x), target=c, n_steps=50, return_convergence_delta=True)
        ixg = InputXGradient(f).attribute(x, target=c)
        node = {k: v[0].detach().abs().sum(-1).double().cpu().numpy() for k, v in (("sal", sal), ("ig", ig), ("ixg", ixg))}
        R["mol_idx"].append(int(mi)); R["target"].append(c); R["ig_delta"].append(float(delta.abs().sum())); R["ig_fx_f0"].append(fx - f0)
        R["finite"].append(all(np.isfinite(v).all() for v in node.values())); R["nonzero"].append(all(bool((v != 0).any()) for v in node.values()))
        for k, v in node.items():
            R[k].append(v)
    A = {k: (np.array(v, dtype=object) if k in ("sal", "ig", "ixg") else np.array(v)) for k, v in R.items()}
    save_npz(out, unit=unit, ckpt=ckpt, **A)
    print(f"{unit}: grad {len(mols)} mols, finite {int(A['finite'].sum())}, nonzero {int(A['nonzero'].sum())}, "
          f"max |IG delta| {A['ig_delta'].max():.2e}, median |f(x)-f(0)| {np.median(np.abs(A['ig_fx_f0'])):.2f}", flush=True)


# ------------------------------------------------------------------ SubgraphX
def run_sx(unit, ckpt, mols, out_dir, shard, rollout, sample, budget, tag, device):
    from ..explain.dig_subgraphx import SubgraphXWrapper
    i, n = (int(v) for v in shard.split("/"))
    mine = [int(m) for j, m in enumerate(mols) if j % n == i]
    f = f"{out_dir}/shard_{i}_of_{n}.pt"
    rows = torch.load(f, weights_only=False) if os.path.exists(f) else []
    done = {r["mol_idx"] for r in rows}
    model, _ = load_model(ckpt, device)
    dl, _ = load_data(dataset_of(unit))
    sx = SubgraphXWrapper(model, device, rollout=rollout, sample_num=sample, node_frac=0.25, budget=budget)
    os.makedirs(out_dir, exist_ok=True)
    for mi in mine:
        if mi in done:
            continue
        seed_all(zlib.crc32(f"{tag}|{unit}|{mi}".encode()))
        t0 = time.time(); r = sx.explain(dl[mi]); dt = time.time() - t0
        rows.append({"mol_idx": mi, "node_importance": r.node_importance.tolist(), "target": r.target, "coalition": r.meta["coalition"],
                     "max_nodes": r.meta["max_nodes"], "min_atoms": r.meta["min_atoms"], "n": dl[mi].num_nodes, "seconds": dt})
        torch.save(rows, f + ".tmp"); os.replace(f + ".tmp", f)
        print(f"{unit} {tag} mol {mi}: N={dl[mi].num_nodes} k={r.meta['max_nodes']} |S|={len(r.meta['coalition'])} {dt:.1f}s", flush=True)


# ------------------------------------------------------------------ GNNExplainer + PGExplainer via run_phase3
def fold64(edge_vals, ei, n):
    imp = np.zeros(n); cnt = np.zeros(n)
    for end in (0, 1):
        np.add.at(imp, ei[end], edge_vals); np.add.at(cnt, ei[end], 1.0)
    return imp / np.maximum(cnt, 1.0)


def run_gp(unit, ckpt, out_dir, device, orig=None):
    sp = (orig or {unit: orig_spec(unit)})[unit]
    os.makedirs(out_dir, exist_ok=True)
    cache = f"{out_dir}/{unit}_cache.pt"; done = f"{out_dir}/{unit}_pg_logits.npz"
    if os.path.exists(done):
        return
    if os.path.exists(cache):                      # a cache without the logits file = an interrupted job
        os.remove(cache)
    holder = {}; orig_init = pe.PGExplainerWrapper.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k); holder["w"] = self
    pe.PGExplainerWrapper.__init__ = init
    try:
        rp3.main(["--dataset", sp["dataset"], "--ckpt", ckpt, "--seed", str(sp["seed"]), "--explainers", "gnnexplainer,pgexplainer",
                  "--masking", "R3", "--cache", cache, "--out", f"{out_dir}/{unit}_out.json", *sp["args"]])
    finally:
        pe.PGExplainerWrapper.__init__ = orig_init
    blob = torch.load(cache, weights_only=False); alg = holder["w"].algorithm
    torch.save(alg.mlp.state_dict(), f"{out_dir}/{unit}_pg_mlp.pt")
    mlp = copy.deepcopy(alg.mlp).double().eval(); dl, _ = load_data(sp["dataset"])
    lg_all, ns = [], []
    with torch.no_grad():
        for mi in blob["key"]["test_idx"]:
            d = dl[mi].clone().to(device)
            z = alg._get_embeddings(holder["w"].model, d.x, d.edge_index, edge_attr=getattr(d, "edge_attr", None)).double()
            lg = mlp(alg._get_inputs(z, d.edge_index, None)).view(-1).cpu().numpy()
            lg_all.append(lg); ns.append(fold64(1.0 / (1.0 + np.exp(-lg)), d.edge_index.cpu().numpy(), d.num_nodes))
    save_npz(done, mol_idx=np.array(blob["key"]["test_idx"]), logits=np.array(lg_all, dtype=object), node_score64=np.array(ns, dtype=object))
    print(f"{unit}: gp done ({len(ns)} mols) -> {out_dir}", flush=True)


# ------------------------------------------------------------------ randomised models
@torch.no_grad()
def test_metrics(model, ck, device):
    from sklearn.metrics import roc_auc_score
    dl, _ = load_data(ck["dataset"]); ys, ps = [], []
    for i in ck["split"]["test"]:
        d = dl[i]; ea = None if getattr(d, "edge_attr", None) is None else d.edge_attr.float().to(device)
        lg = model(d.x.float().to(device), d.edge_index.to(device), ea)
        ps.append(float(torch.softmax(lg.reshape(-1, lg.shape[-1])[0], -1)[1])); ys.append(int(d.y.view(-1)[0]))
    ys, ps = np.array(ys), np.array(ps)
    return {"test_acc": float(((ps >= 0.5) == ys).mean()), "test_auroc": float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else float("nan"),
            "frac_pred_pos": float((ps >= 0.5).mean())}


def run_randomise(unit, kind, out, device):
    sp = orig_units()[unit]; model, ck = load_model(sp["ckpt"], device)
    seed = 5000 + sp_seed(unit); torch.manual_seed(seed)
    mods = list(model.ffn.modules()) if kind == "head" else list(model.modules())
    n_reset = 0
    for m in mods:
        if hasattr(m, "reset_parameters") and m is not model:
            m.reset_parameters(); n_reset += 1
    model.eval(); met = test_metrics(model, ck, device)
    new = dict(ck); new["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new["metrics"] = {**ck["metrics"], **met}; new["randomised"] = {"kind": kind, "seed": seed, "modules_reset": n_reset, "from": sp["ckpt"]}
    os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(new, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"{unit} {kind}: reset {n_reset} modules; test acc {met['test_acc']:.3f} auroc {met['test_auroc']:.3f} "
          f"(trained {ck['metrics']['test_acc']:.3f} / {ck['metrics']['test_auroc']:.3f}); predicted positive {met['frac_pred_pos']:.2f}", flush=True)


def sp_seed(unit):
    return int(unit.rsplit("_s", 1)[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grad", "sx", "gp", "randomise"])
    ap.add_argument("--unit"); ap.add_argument("--ckpt"); ap.add_argument("--out")
    ap.add_argument("--mols", help="explanation cache whose key.test_idx lists the molecules")
    ap.add_argument("--mols-file", help="json list of molecule indices")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--rollout", type=int, default=20); ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--budget", default="score"); ap.add_argument("--tag", default="sx")
    ap.add_argument("--kind", choices=["full", "head"])
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.cmd == "grad":
        if not os.path.exists(a.out):
            run_grad(a.unit, a.ckpt, list(torch.load(a.mols, weights_only=False)["key"]["test_idx"]), a.out, device)
    elif a.cmd == "sx":
        run_sx(a.unit, a.ckpt, json.load(open(a.mols_file)), a.out, a.shard, a.rollout, a.sample, a.budget, a.tag, device)
    elif a.cmd == "gp":
        run_gp(a.unit, a.ckpt, a.out, device)
    else:
        if not os.path.exists(a.out):
            run_randomise(a.unit, a.kind, a.out, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
