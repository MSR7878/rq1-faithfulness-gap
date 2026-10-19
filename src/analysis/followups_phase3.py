"""
RQ1 follow-ups, Phase 3: PGExplainer collapse diagnosis (descriptive; PLAN.md).

Units: priority-sweep Tox21 / B-XAIC / MUTAG-GT and the decoupled MUTAG-GT runs
listed in runs/prio/collapse_table.csv. Each collapsed unit is paired with the
clean unit of the same endpoint / task / dataset and protocol with the lowest seed.

Per unit:
  - PGExplainer loss per epoch: the unit's explanation step is re-run with the
    same command, seed and subsample (PGExplainer only -- in the original runs
    PGExplainer is trained before any explanation is computed, and the
    GNNExplainer wrapper draws no random numbers at construction), with every
    epoch's loss saved; the re-run's masks are compared with the stored ones.
  - mask std and normalised entropy of the STORED PG edge masks (mean over molecules);
  - node-embedding norm (mean, std) and class separability (5-fold linear probe,
    balanced accuracy; silhouette) of node and graph embeddings on the
    checkpoint's test split. Embeddings = the last message-passing layer
    (what PGExplainer reads); graph embedding = sum pool; label = predicted class.
Writes only runs/rq1_followups/phase3/.

    python -m src.analysis.followups_phase3 run --units ...
    python -m src.analysis.followups_phase3 report
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import get_embeddings

from ..data import LOADERS
from ..explain import pyg_explainers
from ..explain import run_phase3 as rp3

OUT = "runs/rq1_followups/phase3"
MAX_NODES_PROBE = 20000


def units() -> dict:
    """name -> spec, for every unit of the collapse table that is collapsed or a pair partner."""
    df = pd.read_csv("runs/prio/collapse_table.csv")
    df = df[df["dataset"].isin(["tox21", "bxaic", "mutag_graphxai"])]
    df["group"] = df["dataset"] + "|" + df["task_or_endpoint"].fillna("") + "|" + df["protocol"]
    spec, pairs = {}, []
    for g, d in df.groupby("group"):
        clean = d[~d["collapsed"]].sort_values("seed")
        for _, r in d[d["collapsed"]].iterrows():
            if clean.empty:
                pairs.append((uname(r), None)); continue
            pairs.append((uname(r), uname(clean.iloc[0])))
    for _, r in df.iterrows():
        n = uname(r)
        if any(n in p for p in pairs):
            spec[n] = make_spec(r)
    return spec, pairs


def uname(r):
    t = "" if pd.isna(r["task_or_endpoint"]) else f"_{r['task_or_endpoint']}"
    return f"{r['protocol']}_{r['dataset']}{t}_s{int(r['seed'])}"


def make_spec(r):
    s, ds, t, proto = int(r["seed"]), r["dataset"], r["task_or_endpoint"], r["protocol"]
    if ds == "tox21":
        key = f"tox21_{t}"; args = ["--limit", "30", "--pg-train-limit", "200", "--stratify-frac", "0.5", "--max-nodes", "60"]
    elif ds == "bxaic":
        key = f"bxaic_{t}"; args = ["--limit", "400", "--stratify-frac", "0.9", "--max-nodes", "80", "--pg-train-limit", "200"]
    else:
        key = "mutag_graphxai"; args = ["--explain-pool", "all"]
    d = "runs/prio" if proto == "coupled" else "runs/decoupled"
    eseed = s if proto == "coupled" else 1000 + s
    return {"dataset": key, "seed": s, "explainer_seed": eseed, "protocol": proto, "collapsed": bool(r["collapsed"]),
            "n_degenerate": int(r["n_degenerate"]), "n": int(r["n"]), "group": f"{ds}|{'' if pd.isna(t) else t}|{proto}",
            "ckpt": f"runs/prio/ckpt_{key}_s{s}.pt", "cache": f"{d}/cache_{key}_s{s}.pt", "out": f"{d}/out_{key}_s{s}.json", "args": args}


def mask_stats(cache):
    res = torch.load(cache, weights_only=False)["results"]["pgexplainer"]
    std, ent = [], []
    for r in res:
        m = np.asarray(r["edge_importance"], dtype=np.float64); E = len(m)
        std.append(m.std())
        if E < 2:
            continue
        p = m / m.sum() if m.sum() > 0 else np.full(E, 1.0 / E)       # all-zero mask -> uniform
        ent.append(float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(E)))
    return float(np.mean(std)), float(np.mean(ent))


def rerun_pg(name, sp):
    """Re-run the unit's explanation step (PGExplainer only) with per-epoch losses saved; compare masks with the stored ones."""
    losses = {}
    orig = pyg_explainers.PGExplainerWrapper.train

    def train(self, data):
        out = orig(self, data); losses["curve"] = [float(x) for x in out]; return out
    pyg_explainers.PGExplainerWrapper.train = train
    try:
        cache = f"{OUT}/rerun/{name}_cache.pt"
        rp3.main(["--dataset", sp["dataset"], "--ckpt", sp["ckpt"], "--seed", str(sp["explainer_seed"]), "--explainers", "pgexplainer",
                  "--masking", "R3", "--cache", cache, "--out", f"{OUT}/rerun/{name}_out.json", *sp["args"]])
    finally:
        pyg_explainers.PGExplainerWrapper.train = orig
    new = torch.load(cache, weights_only=False)
    old = torch.load(sp["cache"], weights_only=False)
    same_idx = list(new["key"]["test_idx"]) == list(old["key"]["test_idx"]) and new["key"]["pg_train_idx"] == old["key"]["pg_train_idx"]
    dev = max(float(np.max(np.abs(np.asarray(a["edge_importance"]) - np.asarray(b["edge_importance"]))))
              for a, b in zip(new["results"]["pgexplainer"], old["results"]["pgexplainer"])) if same_idx else float("nan")
    deg_new = sum(float(np.std(r["node_importance"])) == 0.0 for r in new["results"]["pgexplainer"])
    return losses["curve"], same_idx, dev, deg_new


@torch.no_grad()
def embedding_stats(sp, device):
    model, ck = rp3.load_model(sp["ckpt"], device)
    dl, _ = LOADERS[sp["dataset"]]()
    test = list(ck["split"]["test"])
    node_e, node_y, graph_e, graph_y = [], [], [], []
    for i in test:
        d = dl[i].to(device)
        ea = getattr(d, "edge_attr", None)
        emb = get_embeddings(model, d.x, d.edge_index, edge_attr=ea)[-1]
        pred = int(model(d.x, d.edge_index, ea).argmax())
        node_e.append(emb.cpu().numpy()); node_y.append(np.full(emb.shape[0], pred))
        graph_e.append(emb.sum(0).cpu().numpy()); graph_y.append(pred)
    N, Ny = np.concatenate(node_e), np.concatenate(node_y)
    Gm, Gy = np.stack(graph_e), np.array(graph_y)
    norms = np.linalg.norm(N, axis=1)
    rng = np.random.default_rng(0)
    if len(N) > MAX_NODES_PROBE:
        sel = rng.choice(len(N), MAX_NODES_PROBE, replace=False); N, Ny = N[sel], Ny[sel]

    def sep(X, y):
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 5:
            return float("nan"), float("nan")
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        acc = cross_val_score(probe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy").mean()
        Xs = StandardScaler().fit_transform(X)
        sil = silhouette_score(Xs, y, sample_size=min(5000, len(y)), random_state=0)
        return float(acc), float(sil)
    na, ns = sep(N, Ny); ga, gs = sep(Gm, Gy)
    return {"emb_norm_mean": float(norms.mean()), "emb_norm_std": float(norms.std()), "node_probe_bacc": na, "node_silhouette": ns,
            "graph_probe_bacc": ga, "graph_silhouette": gs, "n_test_graphs": len(test), "pred_pos_frac": float(Gy.mean())}


def run(names):
    spec, _ = units()
    os.makedirs(f"{OUT}/rerun", exist_ok=True); os.makedirs(f"{OUT}/units", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n in names:
        dst = f"{OUT}/units/{n}.json"
        if os.path.exists(dst):
            continue
        sp = spec[n]
        curve, same_idx, dev, deg_new = rerun_pg(n, sp)
        ms, me = mask_stats(sp["cache"])
        rec = {"unit": n, **{k: v for k, v in sp.items() if k != "args"}, "loss_curve": curve, "rerun_same_subsample": same_idx,
               "rerun_max_abs_mask_dev": dev, "rerun_n_degenerate": deg_new, "mask_std": ms, "mask_entropy": me, **embedding_stats(sp, device)}
        json.dump(rec, open(dst, "w"), indent=1)
        print(f"{n}: collapsed={sp['collapsed']} rerun mask dev {dev:.2e} (degenerate {deg_new} vs stored {sp['n_degenerate']})", flush=True)


QUANT = [("final loss", lambda r: r["loss_curve"][-1]), ("loss drop (first - last)", lambda r: r["loss_curve"][0] - r["loss_curve"][-1]),
         ("mask std", lambda r: r["mask_std"]), ("mask entropy", lambda r: r["mask_entropy"]),
         ("emb norm mean", lambda r: r["emb_norm_mean"]), ("emb norm std", lambda r: r["emb_norm_std"]),
         ("node probe bal. acc", lambda r: r["node_probe_bacc"]), ("node silhouette", lambda r: r["node_silhouette"]),
         ("graph probe bal. acc", lambda r: r["graph_probe_bacc"]), ("graph silhouette", lambda r: r["graph_silhouette"]),
         ("predicted-positive fraction", lambda r: r["pred_pos_frac"])]


def report():
    P = print
    spec, pairs = units()
    U = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{OUT}/units/*.json")}
    P("=" * 120); P(" PHASE 3 -- PGExplainer collapse diagnosis (descriptive)"); P("=" * 120)
    P(f"\n units with results {len(U)}/{len(spec)}; pairs {len(pairs)} (unpaired collapsed units: {[c for c, k in pairs if k is None]})")
    rr = [(n, r) for n, r in sorted(U.items())]
    P(f"\n [R] reproduction of the stored PG masks by the re-run: same subsample in {sum(r['rerun_same_subsample'] for _, r in rr)}/{len(rr)};"
      f" max |mask dev| median {np.nanmedian([r['rerun_max_abs_mask_dev'] for _, r in rr]):.2e}, max {np.nanmax([r['rerun_max_abs_mask_dev'] for _, r in rr]):.2e};"
      f" degenerate count equal to stored in {sum(r['rerun_n_degenerate'] == r['n_degenerate'] for _, r in rr)}/{len(rr)}")
    P(f"\n [U] per unit")
    P(f"  {'unit':<34}{'coll':>5}{'deg/n':>9}{'loss 1st':>9}{'last':>8}{'mask sd':>9}{'entropy':>8}{'|emb|':>8}{'node acc':>9}{'node sil':>9}{'graph acc':>10}{'graph sil':>10}{'pos frac':>9}")
    for n, r in rr:
        P(f"  {n:<34}{'yes' if r['collapsed'] else 'no':>5}{str(r['n_degenerate']) + '/' + str(r['n']):>9}{r['loss_curve'][0]:>9.3f}{r['loss_curve'][-1]:>8.3f}"
          f"{r['mask_std']:>9.2e}{r['mask_entropy']:>8.3f}{r['emb_norm_mean']:>8.2f}{r['node_probe_bacc']:>9.3f}{r['node_silhouette']:>9.3f}"
          f"{r['graph_probe_bacc']:>10.3f}{r['graph_silhouette']:>10.3f}{r['pred_pos_frac']:>9.3f}")
    rep = {"pairs": pairs, "counts": {}}
    for scope, sel in (("Tox21 pairs", lambda c: c.startswith("coupled_tox21")), ("all pairs", lambda c: True)):
        pp = [(c, k) for c, k in pairs if k is not None and sel(c) and c in U and k in U]
        P(f"\n [D] {scope} (n = {len(pp)}): collapsed vs its clean partner -- higher / lower / equal-or-NaN")
        rep["counts"][scope] = {}
        for lab, f in QUANT:
            h = sum(f(U[c]) > f(U[k]) for c, k in pp); l = sum(f(U[c]) < f(U[k]) for c, k in pp); e = len(pp) - h - l
            P(f"    {lab:<30}{h:>4} /{l:>4} /{e:>4}")
            rep["counts"][scope][lab] = [h, l, e]
    pp = [(c, k) for c, k in pairs if k is not None and c.startswith("coupled_tox21") and c in U and k in U]
    lower = sum(U[c]["node_probe_bacc"] < U[k]["node_probe_bacc"] for c, k in pp)
    held = lower > len(pp) / 2
    P(f"\n [P5] Tox21: node-embedding probe balanced accuracy lower for the collapsed unit in {lower}/{len(pp)} pairs -> {'HELD' if held else 'DID NOT HOLD'}")
    rep["P5"] = {"lower": lower, "pairs": len(pp), "held": held}
    rep["units"] = U
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    P(f"\nsaved -> {OUT}/report.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "report"])
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "list":
        spec, pairs = units()
        for n in spec:
            print(n)
        return 0
    return report() if a.cmd == "report" else (run(a.units) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
