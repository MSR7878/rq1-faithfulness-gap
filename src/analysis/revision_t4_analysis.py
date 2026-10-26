"""
RQ1 revision, T4 analysis (runs/rq1_revision/PLAN.md, T4 and "ADDENDUM to T4").

  [A] saturation statistics of the pre-sigmoid edge logits per unit (re-runs, float64);
  [B] collapse recount: new rule (edge logits have no ordering: max - min <= 1e-6 * max(1, |mean|),
      float64; run collapsed if >= half of its molecules) next to the old rule (stored float32
      node mask std == 0); the prediction of the addendum;
  [C] PGExplainer top-k GEA re-scored with float64 logit-based node scores (random ties only for exact
      float64 ties) -> direction counts old vs new: main table, architecture table, MUTAG-GT 10 seeds;
  [D] T4 re-training: per setting collapse (old rule on the re-run masks, new rule on logits
      reconstructed from the float32 masks), loss, gradient norm, saturation, mask std; P-T4.

    python -m src.analysis.revision_t4_analysis
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from ..data import LOADERS
from ..metrics.gea_set import gea_max_over_set
from . import exploratory_null_audit as na
from .mutagenicity_cv_report import FAMILY, family_tests
from .ties_random import random_tie_gea

R = "runs/rq1_revision"
L = f"{R}/logits"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
LOG = []


def P(s=""):
    print(s); LOG.append(s)


def degenerate_logits(lg):
    lg = np.asarray(lg, dtype=np.float64)
    return len(lg) == 0 or (lg.max() - lg.min()) <= 1e-6 * max(1.0, abs(lg.mean()))


def group_of(u):
    if u.startswith("tox21"):
        return "Tox21"
    if u.startswith("bxaic"):
        return "B-XAIC " + u.split("_")[1]
    if u.startswith("decoupled"):
        return "MUTAG-GT decoupled"
    if u.startswith("mutag_graphxai"):
        return "MUTAG-GT seeds 0-9"
    if u.startswith("mutagenicity_cv"):
        return "Mutagenicity 5-fold CV"
    if u.startswith("mutagenicity_"):
        return "Mutagenicity " + {"dmpnn": "D-MPNN", "gin": "GIN", "gine": "GINE", "gcn": "GCN"}[u.split("_")[1]]
    return u.upper()


def part_ab(rep):
    infos = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{L}/*.json")}
    skipped = {u: i["status"] for u, i in infos.items() if i["status"] != "ok"}
    P("[A] PGExplainer pre-sigmoid edge logits (re-run, trained MLP evaluated in float64), per unit")
    P("    sat = fraction of sigmoid outputs < 1e-6 or > 1 - 1e-6; distinct = median over molecules of distinct values per molecule")
    P(f"  {'unit':<32}{'mean':>9}{'std':>9}{'min':>10}{'max':>9}{'sat':>7}{'mask f32':>9}{'mask f64':>9}{'logits':>8}"
      f"{'old coll':>9}{'new coll':>9}{'rerun dev':>11}")
    rows = {}
    for u in sorted(infos):
        if infos[u]["status"] != "ok":
            continue
        z = np.load(f"{L}/{u}.npz", allow_pickle=True)
        lgs = [np.asarray(a, dtype=np.float64) for a in z["logits"]]; allg = np.concatenate([a for a in lgs if len(a)])
        sig = 1.0 / (1.0 + np.exp(-allg))
        d32 = np.median([len(np.unique((1.0 / (1.0 + np.exp(-a))).astype(np.float32))) for a in lgs if len(a)])
        d64 = np.median([len(np.unique(1.0 / (1.0 + np.exp(-a)))) for a in lgs if len(a)])
        dl_ = np.median([len(np.unique(a)) for a in lgs if len(a)])
        old_deg = np.array([float(torch.as_tensor(m, dtype=torch.float).std()) == 0.0 for m in z["stored_node_mask"]])
        new_deg = np.array([degenerate_logits(a) for a in lgs])
        r = {"group": group_of(u), "mean": float(allg.mean()), "std": float(allg.std()), "min": float(allg.min()), "max": float(allg.max()),
             "sat": float(np.mean((sig < 1e-6) | (sig > 1 - 1e-6))), "distinct_mask_f32": float(d32), "distinct_mask_f64": float(d64),
             "distinct_logits": float(dl_), "old_deg": int(old_deg.sum()), "new_deg": int(new_deg.sum()), "n": len(lgs),
             "old_collapsed": bool(old_deg.mean() >= 0.5), "new_collapsed": bool(new_deg.mean() >= 0.5), "rerun_dev": infos[u]["rerun_vs_stored_max_mask_dev"]}
        rows[u] = r
        P(f"  {u:<32}{r['mean']:>9.2f}{r['std']:>9.2f}{r['min']:>10.1f}{r['max']:>9.1f}{r['sat']:>7.2f}{d32:>9.0f}{d64:>9.0f}{dl_:>8.0f}"
          f"{('yes' if r['old_collapsed'] else 'no'):>9}{('yes' if r['new_collapsed'] else 'no'):>9}{r['rerun_dev']:>11.1e}")
    if skipped:
        P(f"  skipped: {skipped}")
    P("\n[B] collapse recount per dataset / architecture")
    P("    new rule: a molecule is degenerate if its edge logits have no ordering, (max - min) <= 1e-6 * max(1, |mean|) in float64;")
    P("    a run is collapsed if >= half of its explained molecules are degenerate. Old rule: stored float32 node mask std == 0.")
    P(f"  {'dataset / architecture':<28}{'units':>6}{'old collapsed':>15}{'new collapsed':>15}   units collapsed under the old rule but ordered under the new")
    groups = {}
    for u, r in rows.items():
        groups.setdefault(r["group"], []).append(u)
    old_c = new_from_old = 0
    for g in sorted(groups):
        us = groups[g]; oc = [u for u in us if rows[u]["old_collapsed"]]; nc = [u for u in us if rows[u]["new_collapsed"]]
        freed = [u for u in oc if not rows[u]["new_collapsed"]]
        old_c += len(oc); new_from_old += len(freed)
        P(f"  {g:<28}{len(us):>6}{len(oc):>15}{len(nc):>15}   {', '.join(freed) if freed else '-'}")
        rep.setdefault("recount", {})[g] = {"units": len(us), "old": len(oc), "new": len(nc), "freed": freed}
    frac = new_from_old / old_c if old_c else float("nan")
    P(f"\n  ADDENDUM PREDICTION: of {old_c} units collapsed under the old rule, {new_from_old} have ordered logits ({frac:.2f}) -> "
      f"{'HELD' if old_c and frac >= 1 / 3 else 'DID NOT HOLD'} (held if >= 1/3)")
    rep["addendum_prediction"] = {"old_collapsed": old_c, "ordered": new_from_old, "fraction": frac, "held": bool(old_c and frac >= 1 / 3)}
    rep["units"] = rows
    return rows


def part_c(rep):
    P("\n[C] PGExplainer top-k GEA re-scored with float64 logit-based node scores (random ties only for exact float64 ties)")
    dl_mg, _ = LOADERS["mutag_graphxai"](); dl_ix, _ = LOADERS["bxaic_indole"](); dl_mu, meta = LOADERS["mutagenicity_graphxai"](); motifs = meta.extras["motifs"]

    def pg_new(unit, dl, scorer, mol_order):
        z = np.load(f"{L}/{unit}.npz", allow_pickle=True); pos = {int(m): i for i, m in enumerate(z["mol_idx"])}
        out = []
        for mi in mol_order:
            sc = z["node_score64"][pos[int(mi)]]; n = len(sc); k = max(1, int(0.25 * n))
            out.append(random_tie_gea(sc, k, scorer(int(mi), dl[int(mi)]), f"{unit}|pgexplainer|logit", pos[int(mi)])[0])
        return np.array(out)
    jac = lambda mi, d: na.Scorer(d.node_gt_mask.numpy().astype(bool))
    mos = lambda mi, d: (lambda m: gea_max_over_set(m, motifs[mi]).max_over_set)
    na.SEEDS = list(range(10))
    changes = []
    # main table rule (Holm over 3 explainers vs random-node, direction = higher mean)
    for title, key, seeds, dl, scorer in (("MUTAG-GT seeds 0-4", "mutag_graphxai", range(5), dl_mg, jac),
                                          ("MUTAG-GT seeds 0-9", "mutag_graphxai", range(10), dl_mg, jac),
                                          ("B-XAIC indole", "bxaic_indole", range(5), dl_ix, jac),
                                          ("Mutagenicity D-MPNN (main table)", "mutagenicity_graphxai", range(5), dl_mu, mos)):
        po, pn = {}, {}
        for s in seeds:
            st = dict(np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/{key}_s{s}.npz") if s < 5
                      else np.load(f"runs/rq1_followups/phase1/null_audit_arrays/{key}_s{s}.npz"))
            rt = np.load(f"runs/rq1_followups/phase2b/gea/audit_{key}_s{s}.npz") if s < 5 else np.load(f"runs/rq1_final_export/gea/audit_{key}_s{s}.npz")
            base = dict(st)
            for e in EXPL:
                base[f"{e}_topk"] = rt[f"{e}_b"]
            unit = {"mutag_graphxai": f"mutag_graphxai_s{s}", "bxaic_indole": f"bxaic_indole_s{s}", "mutagenicity_graphxai": f"mutagenicity_dmpnn_s{s}"}[key]
            new = dict(base); new["pgexplainer_topk"] = pg_new(unit, dl, scorer, st["mol_idx"])
            po[s], pn[s] = (base,), (new,)
        fam = [(f"{e}_topk", "rn_k") for e in EXPL]
        for e in EXPL:
            a1, b1, _, c1 = na.sig_counts(po, f"{e}_topk", "rn_k", fam); a2, b2, _, c2 = na.sig_counts(pn, f"{e}_topk", "rn_k", fam)
            ch = (a1, b1) != (a2, b2)
            P(f"  {title:<34}{e:<13} random-tie {a1}/{b1}  ->  logit-ranked {a2}/{b2}{'   CHANGE' if ch else ''}")
            if ch:
                changes.append((title, e, f"{a1}/{b1}", f"{a2}/{b2}"))
    # architecture table (registered 4-test family)
    dirs = {"D-MPNN": ("runs/mutagenicity", "dmpnn"), "GIN": ("runs/arch_gin/mutagenicity_graphxai", "gin"),
            "GINE": ("runs/arch_gine/mutagenicity_graphxai", "gine"), "GCN": ("runs/arch_gcn/mutagenicity_graphxai", "gcn")}
    for A, (d, tag) in dirs.items():
        so, sn = [], []
        for s in range(5):
            z = dict(np.load(f"{d}/arrays_s{s}.npz")); rt = np.load(f"runs/rq1_followups/phase2b/gea/arch_{A}_s{s}.npz")
            for e in EXPL:
                z[f"{e}_topk_set"] = rt[f"{e}_b"]
            zn = dict(z); zn["pgexplainer_topk_set"] = pg_new(f"mutagenicity_{tag}_s{s}", dl_mu, mos, z["mol_idx"])
            to, tn = family_tests(z, FAMILY)["PG vs RandNode"], family_tests(zn, FAMILY)["PG vs RandNode"]
            so.append("+" if to["above"] else "-" if to["below"] else "."); sn.append("+" if tn["above"] else "-" if tn["below"] else ".")
        ch = so != sn
        P(f"  architecture {A:<20}PG vs RandNode   random-tie {''.join(so)} ({so.count('+')}/{so.count('-')})  ->  logit-ranked {''.join(sn)} "
          f"({sn.count('+')}/{sn.count('-')}){'   CHANGE' if ch else ''}")
        if ch:
            changes.append((f"architecture {A}", "pgexplainer", "".join(so), "".join(sn)))
    rep["gea_changes"] = changes


def part_d(rep):
    P("\n[D] T4 re-training of PGExplainer (Tox21: 24 originally collapsed units + 11 clean partners), per setting")
    P("    settings: a = 30 epochs lr 0.003 (current); b = 100 epochs; c = 300 epochs; d1 = lr 0.0009; d2 = lr 0.009")
    P("    logits here are reconstructed from the saved float32 masks (log m - log(1 - m)); m == 0 or 1 gives only a bound;")
    P("    a molecule whose edge masks are all exactly 0 (or all exactly 1) is 'undetermined'.")
    jobs = [json.load(open(f)) for f in glob.glob(f"{R}/t4/jobs/*.json")]
    by = {}
    for j in jobs:
        by.setdefault(j["setting"], []).append(j)
    P(f"  {'setting':<8}{'jobs':>5}{'old coll (orig coll)':>22}{'old coll (orig clean)':>23}{'new coll':>9}{'undet. units':>13}"
      f"{'final loss':>11}{'grad norm':>11}{'sat 1e-3':>9}{'sat 1e-6':>9}{'distinct f32':>13}")
    rates = {}
    for s in ("a", "b", "c", "d1", "d2"):
        js = by.get(s, [])
        if not js:
            continue
        oc_c = oc_k = nc = und = 0; fl, gn, s3, s6, dm = [], [], [], [], []
        for j in js:
            cache = torch.load(f"{R}/t4/rerun/{j['unit']}_{s}_cache.pt", weights_only=False)["results"]["pgexplainer"]
            old = np.mean([float(torch.as_tensor(r["node_importance"], dtype=torch.float).std()) == 0.0 for r in cache]) >= 0.5
            states = []
            for r in cache:
                m = np.asarray(r["edge_importance"], dtype=np.float32).astype(np.float64)
                if len(m) and (np.all(m == 0) or np.all(m == 1)):
                    states.append("und")
                    continue
                fin = (m > 0) & (m < 1)
                lg = np.log(m[fin]) - np.log1p(-m[fin])
                if fin.all():
                    states.append("deg" if degenerate_logits(lg) else "ord")
                else:
                    states.append("ord")                 # some edges at 0/1 and others not: an ordering exists
                s6.append(np.mean((m < 1e-6) | (m > 1 - 1e-6))); dm.append(len(np.unique(np.asarray(r["edge_importance"], dtype=np.float32))))
            new = np.mean([x == "deg" for x in states]) >= 0.5; un = np.mean([x == "und" for x in states]) >= 0.5
            if j["orig_collapsed"]:
                oc_c += old
            else:
                oc_k += old
            nc += new; und += un
            fl.append(j["loss"][-1]); gn.append(j["grad_norm"][-1]); s3.append(j["saturation"][-1][1])
        n_c = sum(j["orig_collapsed"] for j in js); n_k = len(js) - n_c
        rates[s] = oc_c / n_c if n_c else float("nan")
        P(f"  {s:<8}{len(js):>5}{f'{oc_c}/{n_c}':>22}{f'{oc_k}/{n_k}':>23}{nc:>9}{und:>13}{np.median(fl):>11.3f}{np.median(gn):>11.2e}"
          f"{np.median(s3):>9.2f}{np.median(s6):>9.2f}{np.median(dm):>13.0f}")
        rep.setdefault("t4", {})[s] = {"jobs": len(js), "old_collapsed_orig_collapsed": oc_c, "n_orig_collapsed": n_c, "old_collapsed_orig_clean": oc_k,
                                       "new_collapsed": int(nc), "undetermined_units": int(und), "final_loss_median": float(np.median(fl)),
                                       "grad_norm_median": float(np.median(gn)), "sat_1e3_median": float(np.median(s3))}
    if "a" in rates and "b" in rates and "c" in rates:
        if rates["a"] == 0:
            P("  P-T4: not testable (no collapsed units under setting a)")
        else:
            held = rates["b"] <= rates["a"] / 2 and rates["c"] <= rates["a"] / 2
            P(f"  P-T4 (old float32 rule on the re-run masks -- the registered absolute rule was withdrawn by the addendum): collapse rate of the "
              f"24 units a {rates['a']:.2f}, b {rates['b']:.2f}, c {rates['c']:.2f} -> {'HELD' if held else 'DID NOT HOLD'}")
            rep["P-T4"] = {"rates": rates, "held": bool(held)}


def main():
    rep = {}
    part_ab(rep)
    part_c(rep)
    part_d(rep)
    open(f"{R}/t4_analysis.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{R}/t4_analysis.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
