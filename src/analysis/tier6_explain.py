"""
RQ1 Tier 6 (runs/rq1_tier6/PLAN.md): explanation and timing jobs. Sharded, resumable, written atomically.

  t1      GStarX / SAME on MUTAG-GT seeds 5-9 (the Tier 5 subsets and slice rule; tag T5; SAME at budget k only)
  t2      the aromatic scaffold hidden (R1 or R2), one method (eig, gnnshap, gnnexplainer, sx_fixed, ig) on the hidden graph;
          per-molecule seed crc32("T6|<rule>|<unit>|<mol>"); a molecule that raises an error is recorded, not dropped silently
  reset   T3 full reset of a trained model (reset_parameters() on every module, torch seed 5000 + model seed) with a check that
          every parameter tensor changed, plus the reset model's test metrics
  t3      EiG / GNNShap (tag T5) or the gradient methods (Saliency, IG, IxG) on a reset model, the unit's 100-molecule subset
  t4      seconds per molecule of one method on the unit's 100-molecule subset (gnnexplainer, pgexplainer = training + inference,
          grad = Saliency / IG / IxG each on its own, and eig / gnnshap as the load check)

    python -m src.analysis.tier6_explain t1 --unit U --expl E --shard i/n
    python -m src.analysis.tier6_explain t2 --unit U --expl E --rule R1|R2 --shard i/n
    python -m src.analysis.tier6_explain reset --unit U
    python -m src.analysis.tier6_explain t3 --unit U --expl eig|gnnshap|grad [--shard i/n]
    python -m src.analysis.tier6_explain t4 --unit U --expl gnnexplainer|pgexplainer|grad|eig|gnnshap
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
from torch_geometric.data import Data

from ..explain.run_phase3 import load_model
from ..explain.t5_common import mol_seed, seed_all
from . import tier2_common as C
from . import tier5_common as T5

T6 = "runs/rq1_tier6"
T1_UNITS = [f"mutag_graphxai_s{s}" for s in range(5, 10)]
T2_UNITS = [f"mutag_graphxai_s{s}" for s in range(10)]
T2_METHODS = ["eig", "gnnshap", "gnnexplainer", "sx_fixed", "ig"]
T3_UNITS = [f"{b}_s{s}" for b in ("bxaic_PAINS", "bxaic_indole_gcn", "bxaic_PAINS_gcn", "mutagenicity_gcn") for s in range(5)]
T4_UNITS = [f"bxaic_{t}_s0" for t in ("indole", "PAINS")] + [f"bxaic_{t}_{a}_s0" for t in ("indole", "PAINS") for a in ("gin", "gine", "gcn")] \
    + [f"mutagenicity_{a}_s0" for a in ("dmpnn", "gin", "gine", "gcn")] + ["mutag_graphxai_s0"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def aromatic(d) -> list[int]:
    """Atoms on an aromatic bond (bond-type column 0), as Tier 4 L4."""
    ea = d.edge_attr.detach().cpu().numpy(); ei = d.edge_index.detach().cpu().numpy()
    return sorted({int(v) for v in ei[:, ea[:, 0] > 0].ravel()})


def t1_mols(unit, name):
    sub = [int(m) for m in T5.subsets()[unit]]
    return sub if name == "gstarx" else sorted(T5.slice_order(unit, sub)[:25])


def subset100(unit):
    return [int(m) for m in T5.subsets()[T5.subset_owner(unit)]]


def _shard(mols, shard):
    i, n = (int(v) for v in shard.split("/"))
    return i, n, [int(m) for j, m in enumerate(mols) if j % n == i]


def _load_rows(f):
    rows = torch.load(f, weights_only=False) if os.path.exists(f) else []
    return rows, {r["mol_idx"] for r in rows}


def _save_rows(rows, f):
    os.makedirs(os.path.dirname(f), exist_ok=True); torch.save(rows, f + ".tmp"); os.replace(f + ".tmp", f)


# ------------------------------------------------------------------ gradients on one graph (as tier2_explain.run_grad)
def grad_scores(model, G, which=("sal", "ig", "ixg"), timer=None):
    from captum.attr import InputXGradient, IntegratedGradients, Saliency
    from .tier2_explain import batched_forward
    x = G.x.float().to(DEV)[None]; ei = G.edge_index.to(DEV)
    ea = None if getattr(G, "edge_attr", None) is None else G.edge_attr.float().to(DEV)
    f = batched_forward(model, ei, ea); out = {}
    for w in which:
        sync(); t0 = time.time()
        with torch.no_grad():
            c = int(f(x)[0].argmax())
        if w == "sal":
            a = Saliency(f).attribute(x, target=c, abs=True)
        elif w == "ig":
            a = IntegratedGradients(f).attribute(x, baselines=torch.zeros_like(x), target=c, n_steps=50)
        else:
            a = InputXGradient(f).attribute(x, target=c)
        out[w] = a[0].detach().abs().sum(-1).double().cpu().numpy(); sync()
        if timer is not None:
            timer[w] = time.time() - t0
    return out


# ------------------------------------------------------------------ T1
def run_t1(unit, name, shard):
    out_dir = f"{T6}/T1/expl/{name}/{unit}"; i, n, mine = _shard(t1_mols(unit, name), shard)
    f = f"{out_dir}/shard_{i}_of_{n}.pt"; rows, done = _load_rows(f)
    if all(m in done for m in mine):
        return
    model, ck = load_model(T5.spec(unit)["ckpt"], DEV); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    ex = T5.make_explainer(name, model, ck, dkey, DEV)
    for mi in mine:
        if mi in done:
            continue
        d = dl[mi]; seed_all(mol_seed("T5", unit, mi)); t0 = time.time()
        e = ex.explain(d, "k" if name == "same" else None)
        rows.append({"mol_idx": mi, "n": d.num_nodes, "seconds": time.time() - t0, "node": np.asarray(e["node"], np.float64).tolist(),
                     "meta_main": {k: v for k, v in e["meta"].items() if k != "bond_score"}})
        _save_rows(rows, f); print(f"{unit} {name} T5 mol {mi}: N={d.num_nodes} {rows[-1]['seconds']:.1f}s", flush=True)


# ------------------------------------------------------------------ T2
def hidden_graph(d, scaf, rule):
    """R1: induced graph of the remaining atoms (reindexed in increasing original order); R2: scaffold rows of x set to 0."""
    from . import reliance_fidelity as rf
    G = Data(x=d.x.float().cpu(), edge_index=d.edge_index.cpu(), edge_attr=d.edge_attr.float().cpu())
    n = d.num_nodes; rem = [v for v in range(n) if v not in set(scaf)]
    return (rf.keep_nodes(G, rem) if rule == "R1" else rf.zero_nodes(G, scaf)), rem


def t2_mols(unit):
    dl, _ = C.load_data(C.dataset_of(unit))
    return [m for m in T5.gt_mols(unit) if 0 < len(aromatic(dl[m])) < dl[m].num_nodes]


def make_t2(name, model, ck, dkey):
    if name in ("eig", "gnnshap"):
        ex = T5.make_explainer(name, model, ck, dkey, DEV); return lambda G: np.asarray(ex.explain(G)["node"], np.float64)
    if name == "gnnexplainer":
        from ..explain.pyg_explainers import GNNExplainerWrapper
        ex = GNNExplainerWrapper(model, DEV, epochs=200); return lambda G: ex.explain(G).node_importance.double().numpy()
    if name == "sx_fixed":
        from ..explain.dig_subgraphx import SubgraphXWrapper
        ex = SubgraphXWrapper(model, DEV, rollout=20, sample_num=30, node_frac=0.25, budget="score")
        return lambda G: ex.explain(G).node_importance.double().numpy()
    if name == "ig":
        return lambda G: grad_scores(model, G, ("ig",))["ig"]
    raise ValueError(name)


@torch.no_grad()
def pred(model, G):
    lg = model(G.x.float().to(DEV), G.edge_index.to(DEV), G.edge_attr.float().to(DEV))
    return int(lg.reshape(-1, lg.shape[-1])[0].argmax())


def run_t2(unit, name, rule, shard):
    out_dir = f"{T6}/T2/expl/{rule}/{name}/{unit}"; i, n, mine = _shard(t2_mols(unit), shard)
    f = f"{out_dir}/shard_{i}_of_{n}.pt"; rows, done = _load_rows(f)
    if all(m in done for m in mine):
        return
    model, ck = load_model(T5.spec(unit)["ckpt"], DEV); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    fn = make_t2(name, model, ck, dkey)
    for mi in mine:
        if mi in done:
            continue
        d = dl[mi]; scaf = aromatic(d); G, rem = hidden_graph(d, scaf, rule)
        row = {"mol_idx": mi, "n": d.num_nodes, "rem": rem, "pred_full": pred(model, d), "pred_hidden": pred(model, G)}
        seed_all(mol_seed(f"T6|{rule}", unit, mi)); t0 = time.time()
        try:
            s = fn(G); row["node"] = (s if rule == "R1" else s[rem]).tolist()       # scores of the remaining atoms, in `rem` order
        except Exception:
            row["error"] = traceback.format_exc(limit=3)
        row["seconds"] = time.time() - t0; rows.append(row); _save_rows(rows, f)
        print(f"{unit} {rule} {name} mol {mi}: N={d.num_nodes} rem={len(rem)} {row['seconds']:.1f}s" + (" ERROR" if "error" in row else ""), flush=True)


# ------------------------------------------------------------------ T3
def reset_ckpt(unit):
    return f"{T6}/T3/ckpt/{unit}_full.pt"


@torch.no_grad()
def test_metrics(model, ck, dkey):
    """As tier2_explain.test_metrics, with the dataset given (not every checkpoint stores it)."""
    from sklearn.metrics import roc_auc_score
    dl, _ = C.load_data(dkey); ys, ps = [], []
    for i in ck["split"]["test"]:
        d = dl[int(i)]; ea = None if getattr(d, "edge_attr", None) is None else d.edge_attr.float().to(DEV)
        lg = model(d.x.float().to(DEV), d.edge_index.to(DEV), ea)
        ps.append(float(torch.softmax(lg.reshape(-1, lg.shape[-1])[0], -1)[1])); ys.append(int(d.y.view(-1)[0]))
    ys, ps = np.array(ys), np.array(ps)
    return {"test_acc": float(((ps >= 0.5) == ys).mean()), "test_auroc": float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else float("nan"),
            "frac_pred_pos": float((ps >= 0.5).mean())}


def run_reset(unit):
    out = reset_ckpt(unit)
    if os.path.exists(out):
        return
    model, ck = load_model(T5.spec(unit)["ckpt"], DEV); before = {k: v.detach().clone() for k, v in model.named_parameters()}
    seed = 5000 + T5.seed_of(unit); torch.manual_seed(seed); n_reset = 0
    for m in model.modules():
        if hasattr(m, "reset_parameters") and m is not model:
            m.reset_parameters(); n_reset += 1
    same = [k for k, v in model.named_parameters() if torch.equal(v.detach(), before[k])]
    if same:
        raise RuntimeError(f"{unit}: parameters not changed by the reset: {same}")
    model.eval(); met = test_metrics(model, ck, C.dataset_of(unit))
    new = dict(ck); new["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new["metrics"] = {**ck["metrics"], **met}; new["randomised"] = {"kind": "full", "seed": seed, "modules_reset": n_reset, "from": T5.spec(unit)["ckpt"],
                                                                    "params_changed": len(before)}
    os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(new, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"{unit} full reset: {n_reset} modules, {len(before)} parameter tensors all changed; test acc {met['test_acc']:.3f} auroc {met['test_auroc']:.3f} "
          f"(trained {ck['metrics']['test_acc']:.3f} / {ck['metrics']['test_auroc']:.3f})", flush=True)


def run_t3(unit, name, shard):
    mols = subset100(unit)
    if name == "grad":
        from .tier2_explain import run_grad
        out = f"{T6}/T3/grad/{unit}.npz"
        if not os.path.exists(out):
            model_ck = reset_ckpt(unit); os.makedirs(os.path.dirname(out), exist_ok=True); run_grad(unit, model_ck, mols, out, DEV)
        return
    out_dir = f"{T6}/T3/expl/{name}/{unit}"; i, n, mine = _shard(mols, shard)
    f = f"{out_dir}/shard_{i}_of_{n}.pt"; rows, done = _load_rows(f)
    if all(m in done for m in mine):
        return
    model, ck = load_model(reset_ckpt(unit), DEV); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    ex = T5.make_explainer(name, model, ck, dkey, DEV)
    for mi in mine:
        if mi in done:
            continue
        d = dl[mi]; seed_all(mol_seed("T5", unit, mi)); t0 = time.time(); e = ex.explain(d)
        rows.append({"mol_idx": mi, "n": d.num_nodes, "seconds": time.time() - t0, "node": np.asarray(e["node"], np.float64).tolist()})
        _save_rows(rows, f); print(f"{unit} reset {name} mol {mi}: {rows[-1]['seconds']:.1f}s", flush=True)


# ------------------------------------------------------------------ T4
def run_t4(unit, name):
    out = f"{T6}/T4/time/{unit}__{name}.json"
    if os.path.exists(out):
        return
    sp = T5.spec(unit); model, ck = load_model(sp["ckpt"], DEV); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey)
    key = C.torch_load(sp["cache"])["key"]; mols = subset100(unit); res = {"unit": unit, "method": name, "molecules": len(mols)}
    seed_all(mol_seed("T6t", unit, 0))

    def per_mol(fn):
        fn(dl[mols[0]])                                            # untimed warm-up
        t = []
        for mi in mols:
            sync(); t0 = time.time(); fn(dl[mi]); sync(); t.append(time.time() - t0)
        return t
    if name == "gnnexplainer":
        from ..explain.pyg_explainers import GNNExplainerWrapper
        ex = GNNExplainerWrapper(model, DEV, epochs=int(key["gnn_epochs"])); res["epochs"] = int(key["gnn_epochs"])
        res["seconds"] = per_mol(lambda d: ex.explain(d.clone()))
    elif name == "pgexplainer":
        from ..explain.pyg_explainers import PGExplainerWrapper
        tr = [dl[int(i)] for i in key["pg_train_idx"]]; ex = PGExplainerWrapper(model, DEV, epochs=int(key["pg_epochs"]))
        sync(); t0 = time.time(); ex.train(tr); sync(); res["train_seconds"] = time.time() - t0
        res["train_molecules"] = len(tr); res["epochs"] = int(key["pg_epochs"]); res["seconds"] = per_mol(lambda d: ex.explain(d.clone()))
    elif name == "grad":
        model.eval(); grad_scores(model, dl[mols[0]])               # untimed warm-up
        for w in ("sal", "ig", "ixg"):
            t = []
            for mi in mols:
                tm = {}; grad_scores(model, dl[mi], (w,), tm); t.append(tm[w])
            res[f"seconds_{w}"] = t
    elif name in ("eig", "gnnshap"):
        ex = T5.make_explainer(name, model, ck, dkey, DEV); res["seconds"] = per_mol(lambda d: ex.explain(d))
    else:
        raise ValueError(name)
    res["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES"); res["finished"] = time.strftime("%F %T")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    print(f"{unit} {name}: timed", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["t1", "t2", "reset", "t3", "t4"])
    for k in ("unit", "expl", "rule"):
        a.add_argument(f"--{k}")
    a.add_argument("--shard", default="0/1"); a = a.parse_args()
    if a.cmd == "t1":
        run_t1(a.unit, a.expl, a.shard)
    elif a.cmd == "t2":
        run_t2(a.unit, a.expl, a.rule, a.shard)
    elif a.cmd == "reset":
        run_reset(a.unit)
    elif a.cmd == "t3":
        run_t3(a.unit, a.expl, a.shard)
    else:
        run_t4(a.unit, a.expl)
