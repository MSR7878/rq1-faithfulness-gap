"""
RQ1 Tier 2, item R: score the randomised-model explanations (runs/rq1_tier2/R/).

Per (unit, kind in full/head): GEA under the final protocol on the trained unit's GT molecules and shared nulls --
GNNExplainer (run_phase3 cache), PGExplainer (own float64 logit node scores), IG and Saliency (all GT molecules),
SubgraphX (scoring budget; the fixed 100-molecule subset). Trained vs randomised atom scores per GT molecule:
Spearman correlation (molecules with a constant score vector on either side are skipped and counted) and the
Jaccard of the two top-k sets (one fixed random order within ties on each side). Trained scores: GNNExplainer and
SubgraphX-legacy from the stored cache, PGExplainer final rule, IG / Saliency from item G, SubgraphX fixed budget
from item S (filled when it exists).
Also stores the trained model's GEA on the same molecules (items H / G / S, final protocol) for GEA(trained) - GEA(reset).
PGExplainer diagnostic (pg_diag): edge-mask saturation (share of edges with sigmoid > 0.99 / < 0.01) for the trained and
both randomised explainers, and atoms ranked by the mean incident edge LOGIT instead of the mean sigmoid (the float64
sigmoid rounds to exactly 1.0 above logit ~37, so a mask saturated high ties its top atoms; one saturated low does not).
Writes <out>/<unit>_<kind>.npz and <out>/<unit>_pgdiag.npz (out default runs/rq1_tier2/R/score; the re-score after
S priority writes runs/rq1_tier2/R/rescore so the first scoring is kept).

    python -m src.analysis.tier2_R [--out DIR] UNIT ...
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import spearmanr

from . import tier2_common as C
from .ties_random import tie_order, topk_mask

T = f"{C.T2}/R"
METHODS = ["gnnexplainer", "pgexplainer", "subgraphx", "ig", "sal"]


def trained_scores(unit):
    pg, _ = C.pg_final(unit); g = f"{C.T2}/G/grad/{unit}.npz"
    out = {"gnnexplainer": C.stored_scores(unit, "gnnexplainer"), "pgexplainer": pg, "ig": C.npz_scores(g, "ig"), "sal": C.npz_scores(g, "sal")}
    d = f"{C.T2}/S/sx/{unit}"
    out["subgraphx"] = C.sx_shard_scores(d) if os.path.isdir(d) else {}
    return out


def rand_scores(unit, kind):
    cache = C.torch_load(f"{T}/gp/{kind}/{unit}_cache.pt")
    ti = [int(m) for m in cache["key"]["test_idx"]]
    assert ti == [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]], f"{unit} {kind}: explained set differs from the trained unit"
    g = f"{T}/grad/{kind}/{unit}.npz"
    return {"gnnexplainer": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in zip(ti, cache["results"]["gnnexplainer"])},
            "pgexplainer": C.npz_scores(f"{T}/gp/{kind}/{unit}_pg_logits.npz", "node_score64"),
            "ig": C.npz_scores(g, "ig"), "sal": C.npz_scores(g, "sal"), "subgraphx": C.sx_shard_scores(f"{T}/sx/{kind}/{unit}")}


def trained_gea(unit) -> dict:
    """method -> {mol -> GEA of the trained model's explanation} (final protocol, items H / G / S)."""
    src = {"gnnexplainer": ("H", "gnnexplainer"), "pgexplainer": ("H", "pgexplainer"), "ig": ("G", "ig"), "sal": ("G", "sal"),
           "subgraphx": ("S", "sx_fixed")}
    out = {}
    for m, (item, key) in src.items():
        f = f"{C.T2}/{item}/gea/{unit}.npz"
        z = np.load(f) if os.path.exists(f) else None
        out[m] = {int(x): float(v) for x, v in zip(z["mol_idx"], z[key])} if z is not None and key in z.files else {}
    return out


def score(unit, kind, out=f"{T}/score"):
    dkey = C.dataset_of(unit); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]
    gt = C.gt_mol_idx(unit, ti, dkey); nl = C.nulls(unit, gt, dkey); pos = {m: i for i, m in enumerate(gt)}
    rs, tr, tg = rand_scores(unit, kind), trained_scores(unit), trained_gea(unit)
    A = {"unit": unit, "kind": kind}
    for m in METHODS:
        mols = [x for x in gt if x in rs[m]]
        if m == "subgraphx":
            assert len(mols) == 100 or len(mols) == len(gt), (unit, kind, len(mols))
        A[f"{m}_mol_idx"] = np.array(mols)
        A[f"{m}_gea"] = C.gea_array(f"{unit}|{kind}", m, lambda x, s=rs[m]: s[x], mols, dkey)
        A[f"{m}_trained_gea"] = np.array([tg[m].get(x, np.nan) for x in mols])
        A[f"{m}_rn"] = nl["rn"][[pos[x] for x in mols]]; A[f"{m}_rc"] = nl["rc"][[pos[x] for x in mols]]
        rho, jac = [], []
        for x in mols:
            if x not in tr[m]:
                rho.append(np.nan); jac.append(np.nan); continue
            a, b = np.asarray(tr[m][x], float), np.asarray(rs[m][x], float); k = C.kbudget(len(a))
            rho.append(np.nan if a.std() == 0 or b.std() == 0 else float(spearmanr(a, b).statistic))
            ta = topk_mask(tie_order(a, C.mol_rng(f"{unit}|{m}|trained-jac", x)), k); tb = topk_mask(tie_order(b, C.mol_rng(f"{unit}|{kind}|{m}|jac", x)), k)
            jac.append(float((ta & tb).sum() / max((ta | tb).sum(), 1)))
        A[f"{m}_spearman"] = np.array(rho); A[f"{m}_jaccard"] = np.array(jac)
    C.save_npz(f"{out}/{unit}_{kind}.npz", **A)


def fold64(edge_vals, ei, n):
    imp = np.zeros(n); cnt = np.zeros(n)
    for end in (0, 1):
        np.add.at(imp, ei[end], edge_vals); np.add.at(cnt, ei[end], 1.0)
    return imp / np.maximum(cnt, 1.0)


def pg_diag(unit, out=f"{T}/score"):
    """PGExplainer: saturation of the edge masks and logit-ranked atoms, trained vs full / head reset, on the GT molecules."""
    dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]
    gt = C.gt_mol_idx(unit, ti, dkey); nl = C.nulls(unit, gt, dkey)
    files = {"trained": f"runs/rq1_revision/logits/{unit}.npz", "full": f"{T}/gp/full/{unit}_pg_logits.npz", "head": f"{T}/gp/head/{unit}_pg_logits.npz"}
    A = {"unit": unit, "mol_idx": np.array(gt), "rn": nl["rn"]}; lg, sg = {}, {}
    for w, f in files.items():
        z = np.load(f, allow_pickle=True)
        assert [int(m) for m in z["mol_idx"]] == ti, (unit, w)
        L = {int(m): np.asarray(v, dtype=np.float64) for m, v in zip(z["mol_idx"], z["logits"])}
        allL = np.concatenate([L[m] for m in gt]); s = 1.0 / (1.0 + np.exp(-np.clip(allL, -700, 700)))
        A[f"{w}_frac_hi"] = float(np.mean(s > 0.99)); A[f"{w}_frac_lo"] = float(np.mean(s < 0.01)); A[f"{w}_median_logit"] = float(np.median(allL))
        lg[w] = {m: fold64(L[m], dl[m].edge_index.cpu().numpy(), dl[m].num_nodes) for m in gt}
        sg[w] = {int(m): np.asarray(v, dtype=np.float64) for m, v in zip(z["mol_idx"], z["node_score64"])}
        A[f"{w}_atoms_at_one"] = float(np.mean(np.concatenate([sg[w][m] == 1.0 for m in gt])))   # sigmoid score exactly 1.0 in float64
        A[f"{w}_gea_logit"] = C.gea_array(f"{unit}|{w}", "pg_logit", lambda x, s=lg[w]: s[x], gt, dkey)
    for k in ("full", "head"):
        A[f"{k}_gea_neglogit"] = C.gea_array(f"{unit}|{k}", "pg_neglogit", lambda x, s=lg[k]: -s[x], gt, dkey)
        for tag, S in (("sigmoid", sg), ("logit", lg)):
            A[f"{k}_rho_{tag}"] = np.array([np.nan if S["trained"][m].std() == 0 or S[k][m].std() == 0
                                            else float(spearmanr(S["trained"][m], S[k][m]).statistic) for m in gt])
    C.save_npz(f"{out}/{unit}_pgdiag.npz", **A)


if __name__ == "__main__":
    a = sys.argv[1:]; out = f"{T}/score"
    if a and a[0] == "--out":
        out, a = a[1], a[2:]
    for u in a:
        for k in ("full", "head"):
            score(u, k, out); print(f"{u} {k}: scored", flush=True)
        pg_diag(u, out); print(f"{u}: PGExplainer diagnostic", flush=True)
