"""
RQ1 revision, T4 completion: summary of the b, c, d1, d2 re-runs with exact logits
(runs/rq1_revision/t4_logits), next to setting a.

Setting a: old-rule counts, gradient norm and training saturation from the first T4
run (runs/rq1_revision/t4); logit-rule counts from the addendum re-runs of the same
Tox21 units (runs/rq1_revision/logits), used only where that unit's command matches.
Per setting: collapsed under the old float32 rule (node std == 0 in float32) and under
the logit rule ((max - min) of the edge logits <= 1e-6 * max(1, |mean|) for >= half of
the molecules), split by the unit's original status (24 collapsed / 11 clean); median
final-epoch gradient norm; median saturation (sigmoid < 1e-3 or > 1 - 1e-3 on the first
50 training graphs at the last epoch; sigmoid < 1e-6 or > 1 - 1e-6 on the explained
molecules); median logit min / max; agreement with the first T4 run (same old-rule
status; fraction of molecules whose edge mask matches within 1e-5).
Tox21 has no ground-truth masks, so no GEA is computed.
Writes runs/rq1_revision/t4_logits.txt / .json.

    python -m src.analysis.revision_t4_logits_analysis
"""
from __future__ import annotations

import glob
import json

import numpy as np
import torch

from .revision_logits import units as logit_units
from .revision_t4 import tox21_units
from .revision_t4_logits import degenerate_logits

R = "runs/rq1_revision"; N = f"{R}/t4_logits"
LOG = []


def P(s=""):
    print(s); LOG.append(s)


def old_rule(node_masks):
    return np.mean([float(torch.as_tensor(m, dtype=torch.float).std()) == 0.0 for m in node_masks]) >= 0.5


def main():
    spec, _ = tox21_units(); LU = logit_units(); rep = {"settings": {}, "a_match": {}}
    P("T4 completion: PGExplainer re-training with exact float64 logits (24 originally collapsed Tox21 units + 11 clean partners)")
    P("  settings: a = 30 epochs lr 0.003 (current); b = 100 epochs; c = 300 epochs; d1 = lr 0.0009; d2 = lr 0.009")
    P("  old rule = node mask std == 0 in float32; logit rule = (max - min) of edge logits <= 1e-6 * max(1, |mean|); run collapsed if >= half its molecules are")
    P("  Tox21 has no ground-truth masks: no GEA for these units.\n")
    # setting a: logit-rule status from the addendum re-runs when the command matches
    a_logit = {}
    for u, sp in spec.items():
        lu = u.replace("coupled_", "")
        m = lu in LU and LU[lu]["ckpt"] == sp["ckpt"] and LU[lu]["seed"] == sp["explainer_seed"] and list(LU[lu]["args"]) == list(sp["args"])
        rep["a_match"][u] = bool(m)
        if m:
            z = np.load(f"{R}/logits/{lu}.npz", allow_pickle=True)
            a_logit[u] = np.mean([degenerate_logits(x) for x in z["logits"]]) >= 0.5
    P(f"  setting a logit-rule status taken from the addendum re-runs: {len(a_logit)}/{len(spec)} units have a matching command\n")
    hdr = (f"  {'set':<4}{'jobs':>5}{'old coll':>10}{'old coll':>10}{'logit coll':>12}{'logit coll':>12}{'deg mols':>10}{'grad norm':>11}"
           f"{'sat 1e-3':>9}{'sat 1e-6':>9}{'logit min':>10}{'logit max':>10}{'same old':>9}{'mask repro':>11}")
    P(hdr)
    P(f"  {'':<4}{'':>5}{'(of 24)':>10}{'(of 11)':>10}{'(of 24)':>12}{'(of 11)':>12}{'(logit)':>10}{'(median)':>11}{'(train)':>9}{'(expl.)':>9}"
      f"{'(median)':>10}{'(median)':>10}{'status':>9}{'(median)':>11}")
    first = {json.load(open(f))["unit"] + ":" + json.load(open(f))["setting"]: json.load(open(f)) for f in glob.glob(f"{R}/t4/jobs/*.json")}
    rates_old, rates_logit = {}, {}
    for s in ("a", "b", "c", "d1", "d2"):
        rows = []
        for u in spec:
            f1 = first.get(f"{u}:{s}")
            c1 = torch.load(f"{R}/t4/rerun/{u}_{s}_cache.pt", weights_only=False)["results"]["pgexplainer"] if f1 else None
            if s == "a":
                if f1 is None:
                    continue
                rows.append({"unit": u, "orig": f1["orig_collapsed"], "old": old_rule([r["node_importance"] for r in c1]), "logit": a_logit.get(u),
                             "deg": None, "gn": f1["grad_norm"][-1], "s3": f1["saturation"][-1][1], "s6": None, "lmin": None, "lmax": None,
                             "same": None, "repro": None})
                continue
            try:
                j = json.load(open(f"{N}/{u}_{s}.json"))
            except FileNotFoundError:
                continue
            z = np.load(f"{N}/{u}_{s}.npz", allow_pickle=True)
            lg = [np.asarray(x, dtype=np.float64) for x in z["logits"]]
            c2 = torch.load(f"{N}/rerun/{u}_{s}_cache.pt", weights_only=False)["results"]["pgexplainer"]
            same = repro = None
            if c1 is not None:
                same = bool(old_rule([r["node_importance"] for r in c1]) == j["collapsed_old_f32"])
                repro = float(np.mean([len(a["edge_importance"]) == 0 or float(np.max(np.abs(np.asarray(a["edge_importance"]) - np.asarray(b["edge_importance"])))) <= 1e-5
                                       for a, b in zip(c1, c2)]))
            rows.append({"unit": u, "orig": j["orig_collapsed"], "old": j["collapsed_old_f32"], "logit": j["collapsed_logit"], "deg": j["n_deg_logit"],
                         "gn": j["grad_norm"][-1], "s3": j["saturation"][-1][1], "s6": j["sat_1e6_explained"],
                         "lmin": float(np.median([x.min() for x in lg if len(x)])), "lmax": float(np.median([x.max() for x in lg if len(x)])),
                         "same": same, "repro": repro})
        if not rows:
            continue
        oc = [r for r in rows if r["orig"]]; ok = [r for r in rows if not r["orig"]]

        def cnt(rs, k):
            v = [r[k] for r in rs if r[k] is not None]
            return f"{sum(v)}/{len(v)}" if v else "-"

        def med(k, fmt):
            v = [r[k] for r in rows if r[k] is not None]
            return format(float(np.median(v)), fmt) if v else "-"
        degs = [r["deg"] for r in rows if r["deg"] is not None]
        P(f"  {s:<4}{len(rows):>5}{cnt(oc, 'old'):>10}{cnt(ok, 'old'):>10}{cnt(oc, 'logit'):>12}{cnt(ok, 'logit'):>12}"
          f"{(str(sum(degs)) if degs else '-'):>10}{med('gn', '.2e'):>11}{med('s3', '.2f'):>9}{med('s6', '.2f'):>9}{med('lmin', '.1f'):>10}{med('lmax', '.1f'):>10}"
          f"{cnt(rows, 'same'):>9}{med('repro', '.2f'):>11}")
        rates_old[s] = sum(r["old"] for r in oc) / len(oc)
        lv = [r["logit"] for r in oc if r["logit"] is not None]
        rates_logit[s] = (sum(lv), len(lv))
        rep["settings"][s] = {"jobs": len(rows), "old_coll_orig_coll": cnt(oc, "old"), "old_coll_orig_clean": cnt(ok, "old"),
                              "logit_coll_orig_coll": cnt(oc, "logit"), "logit_coll_orig_clean": cnt(ok, "logit"), "deg_molecules_logit": sum(degs) if degs else None,
                              "grad_norm_median": med("gn", ".3e"), "sat_1e3_median": med("s3", ".4f"), "sat_1e6_explained_median": med("s6", ".4f"),
                              "logit_min_median": med("lmin", ".3f"), "logit_max_median": med("lmax", ".3f"),
                              "same_old_status_as_first_run": cnt(rows, "same"), "mask_reproduction_median": med("repro", ".3f"),
                              "per_unit": {r["unit"]: r for r in rows}}
    P("\n  'deg mols' = molecules with no logit ordering, summed over the jobs; 'same old status' / 'mask repro' compare with the first T4 run")
    P("  (setting a row: the first T4 run; its logit-rule column comes from the addendum re-runs).")
    if all(k in rates_old for k in "abc"):
        P(f"\n  P-T4 (registered: collapse rate halves with 100+ epochs): old float32 rule a {rates_old['a']:.2f}, b {rates_old['b']:.2f}, c {rates_old['c']:.2f}"
          f" -> {'held' if rates_old['b'] <= rates_old['a'] / 2 and rates_old['c'] <= rates_old['a'] / 2 else 'failed'} on this re-run")
        P("  logit rule, collapsed of the 24: " + ", ".join(f"{s} {v[0]}/{v[1]}" for s, v in rates_logit.items()))
    rep["rates_old"] = rates_old; rep["rates_logit"] = {k: list(v) for k, v in rates_logit.items()}
    open(f"{R}/t4_logits.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{R}/t4_logits.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
