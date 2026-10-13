"""
Report for one architecture check on GraphXAI Mutagenicity (runs/arch_<arch>/;
arch = gine | gcn): 5 seeds, the D-MPNN runs' splits, decoupled seeds.

Per-molecule arrays come from the same code as the D-MPNN and GIN reports
(mutagenicity_report.analyse_seed, max-over-set GEA).  The D-MPNN and GIN
columns are the same tests on runs/mutagenicity/arrays_s<S>.npz and
runs/arch_gin/mutagenicity_graphxai/arrays_s<S>.npz, both read-only.

Registered family, Holm within seed (four tests), direction = rank test:
    GNN / PG / SX top-k 0.25 vs RandNode at the budget (20 draws)
    SX mean-threshold mask vs RandConn sized to that mask (20 draws)

Writes only under runs/arch_<arch>/:
    mutagenicity_graphxai/arrays_s<S>.npz, report.json

    python -m src.analysis.arch_report gine | tee runs/arch_gine/report.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from ..data import LOADERS
from . import mutagenicity_report as mr
from .arch_gin_report import fidelity, gea_table, health, sign
from .mutagenicity_cv_report import FAMILY, family_tests, table
from .mutagenicity_report import EXPL

MU = "mutagenicity_graphxai"
SEEDS = [0, 1, 2, 3, 4]
DESC = {"gine": "plain PyG GINE (3 x GINEConv, edge_dim 3, hidden 214, sum pool; bond features used)",
        "gcn": "plain PyG GCN (3 x GCNConv, hidden 300, sum pool; no bond features)"}
# registered predictions per architecture: (key, label, comparison, counted field, threshold)
PRED = {"gine": [("a", "GNNExplainer significantly above RandNode in >= 4/5 seeds", "GNN vs RandNode", "above", 4),
                 ("b", "SubgraphX significantly above its RandConn null in >= 4/5 seeds", "SX vs RandConn", "above", 4),
                 ("d", "PGExplainer NOT significantly above RandNode in >= 3/5 seeds", "PG vs RandNode", "not_above", 3)],
        "gcn": [("b", "SubgraphX significantly above its RandConn null in >= 4/5 seeds", "SX vs RandConn", "above", 4),
                ("d", "PGExplainer NOT significantly above RandNode in >= 3/5 seeds", "PG vs RandNode", "not_above", 3)]}
REF = {"D-MPNN": "runs/mutagenicity", "GIN": f"runs/arch_gin/{MU}"}


def hit(t, field):
    return (not t["above"]) if field == "not_above" else bool(t[field])


def main():
    arch = sys.argv[1]
    R = f"runs/arch_{arch}"
    P = print
    rep = {"arch": arch, "n_draws": mr.N_DRAWS}
    P("=" * 124); P(f" {arch.upper()} architecture check on Mutagenicity -- {DESC[arch]}; D-MPNN splits, seeds (S, 1000+S)"); P("=" * 124)

    dl, meta = LOADERS[MU](); motifs_all = meta.extras["motifs"]
    seeds = [s for s in SEEDS if os.path.exists(f"{R}/{MU}/out_s{s}.json")]
    mr.R = f"{R}/{MU}"                              # analyse_seed reads AND writes arrays under this dir only
    per = {s: mr.analyse_seed(s, dl, motifs_all) for s in seeds}
    ref = {name: {s: dict(np.load(f"{d}/arrays_s{s}.npz")) for s in seeds} for name, d in REF.items()}
    dm_ck = {s: torch.load(f"runs/mutagenicity/ckpt_s{s}.pt", map_location="cpu", weights_only=False)["metrics"] for s in seeds}
    gin_ck = {s: torch.load(f"runs/arch_gin/{MU}/ckpt_s{s}.pt", map_location="cpu", weights_only=False)["metrics"] for s in seeds}

    P(f"\n[1] MODEL (test split)   seed  {arch.upper()} acc   AUROC | D-MPNN acc  AUROC | GIN acc   AUROC | explained (<=80 atoms)  GT-bearing | same molecules as D-MPNN / GIN")
    rep["model"] = {}
    for s in seeds:
        A, ck, out = per[s]; m = ck["metrics"]
        same = [bool(np.array_equal(A["mol_idx"], ref[n][s]["mol_idx"])) for n in REF]
        P(f"{'':>23}{s:>4}  {m['test_acc']:.4f}   {m['test_auroc']:.4f} |  {dm_ck[s]['test_acc']:.4f}    {dm_ck[s]['test_auroc']:.4f} |"
          f" {gin_ck[s]['test_acc']:.4f}   {gin_ck[s]['test_auroc']:.4f} | {out['config']['n_test']:>12}{len(A['mol_idx']):>18}    | {same[0]} / {same[1]}")
        rep["model"][s] = {"test_acc": m["test_acc"], "test_auroc": m["test_auroc"], "n_explained": out["config"]["n_test"],
                           "n_gt_bearing": int(len(A["mol_idx"])), "same_molecules_as_dmpnn": same[0], "same_molecules_as_gin": same[1],
                           "params": int(sum(v.numel() for v in ck["state_dict"].values()))}
    P(f"     parameters: {rep['model'][seeds[0]]['params']:,}")

    P("\n[2] top-k 0.25 GEA (max-over-set), per-seed means over GT-bearing test molecules")
    cols = [(f"{e}_topk_set", mr.SH[e]) for e in EXPL] + [("rand_node_topk_set", "RandNode"), ("rand_edge_topk_set", "RandEdge"),
                                                         ("rand_conn_topk_set", "RandConn@k"), ("ceiling_set", "ceiling")]
    rep["gea_topk"] = gea_table(P, per, cols)
    for name in REF:
        P(f"     {name}, same molecules:")
        gea_table(P, {s: (ref[name][s],) for s in seeds}, cols)
    P("\n     mean-threshold: SubgraphX mask vs RandConn sized to that mask")
    rep["gea_mean"] = gea_table(P, per, [("subgraphx_mean_set", "SX mean-thr"), ("rand_conn_mean_subgraphx_set", "RandConn|SX"),
                                         ("subgraphx_mean_size", "|M| SX"), ("budget", "budget k")])

    P("\n[3] per-seed paired Wilcoxon, Holm within seed over the four registered comparisons (cells: explainer/null)")
    T = {s: family_tests(per[s][0], FAMILY) for s in seeds}
    table(P, seeds, T, FAMILY, "seed")
    TR = {name: {s: family_tests(ref[name][s], FAMILY) for s in seeds} for name in REF}
    rep["tests"] = {str(s): T[s] for s in seeds}
    dis = [(s, lab) for s in seeds for lab, _, _ in FAMILY if not T[s][lab]["mean_and_rank_agree"]]
    P(f"     cells where the mean and the rank test disagree in direction: {dis if dis else 'none'}")

    P("\n[4] direction per seed (+ sig. above, - sig. below, . n.s.) and the cells that differ from the D-MPNN and from GIN")
    P(f"  {'comparison':<18}{arch.upper() + ' s0-4':>14}{'ab/be':>7} |{'D-MPNN s0-4':>14}{'ab/be':>7} |{'GIN s0-4':>14}{'ab/be':>7}")
    rep["vs"] = {}
    for lab, _, _ in FAMILY:
        g = [sign(T[s][lab]) for s in seeds]
        row = f"  {lab:<18}{' '.join(g):>14}{str(g.count('+')) + '/' + str(g.count('-')):>7} |"
        rep["vs"][lab] = {arch: g}
        diffs = []
        for name in REF:
            d = [sign(TR[name][s][lab]) for s in seeds]
            row += f"{' '.join(d):>14}{str(d.count('+')) + '/' + str(d.count('-')):>7} |"
            dd = [f"s{s} {b}->{a}" for s, a, b in zip(seeds, g, d) if a != b]
            diffs.append(f"vs {name}: {', '.join(dd) if dd else 'none'}")
            rep["vs"][lab][name] = {"signs": d, "differs": dd}
        P(row); P(f"  {'':<18}   differs {';  '.join(diffs)}")

    P("\n[5] explainer health")
    npg = health(P, per, "health", rep)
    collapsed = [s for s in seeds if rep["health"][s]["pgexplainer"]["collapsed"]]

    P("\n[6] REGISTERED PREDICTIONS (committed before training; scoring rules fixed in the spec)")
    rep["predictions"] = {"pg_collapsed_seeds": collapsed}
    for key, text, lab, field, thr in PRED[arch]:
        n = sum(hit(T[s][lab], field) for s in seeds)
        P(f"  ({key}) {text}: observed {n}/{len(seeds)} -> {'HELD' if n >= thr else 'DID NOT HOLD'}")
        rep["predictions"][key] = {"observed": n, "of": len(seeds), "held": bool(n >= thr)}

    P(f"\n[6'] SAME COUNTS WITH COLLAPSED PGExplainer SEEDS NOTED (strict rule): collapsed seeds = {collapsed if collapsed else 'none'}")
    keep = [s for s in seeds if s not in collapsed]
    P(f"  {'comparison':<18}{'all seeds above/below/n.s.':>30}{'non-collapsed seeds above/below/n.s.':>40}   (of {len(seeds)} and {len(keep)})")
    rep["excluding_collapsed"] = {"seeds_kept": keep, "counts": {}}
    for lab, _, _ in FAMILY:
        g = [sign(T[s][lab]) for s in seeds]; k = [sign(T[s][lab]) for s in keep]
        P(f"  {lab:<18}{'/'.join(str(g.count(c)) for c in '+-.'):>30}{'/'.join(str(k.count(c)) for c in '+-.'):>40}")
        rep["excluding_collapsed"]["counts"][lab] = {"all": {c: g.count(c) for c in "+-."}, "kept": {c: k.count(c) for c in "+-."}}
    if collapsed:
        P("  (on a constant PGExplainer mask the top-k selection is decided by tie-breaking, so the PG cells of collapsed seeds do not measure a ranking)")
        for key, text, lab, field, thr in PRED[arch]:
            n = sum(hit(T[s][lab], field) for s in keep)
            P(f"  ({key}) on the {len(keep)} non-collapsed seeds (not registered in this form): {n}/{len(keep)}")
            rep["excluding_collapsed"][key] = n

    P("\n[7] Fid+ / Fid- / GEF, R3 (mean fill), per-seed means over all explained molecules")
    rep["fidelity_R3"] = fidelity(P, per)

    P("\n[8] BUDGET ROUNDING (not changed): SubgraphX is asked for max(3, round(0.25 N)) atoms; top-k scoring and R3 masking keep max(1, int(0.25 N)).")
    coal, k = [], []
    for s in seeds:
        blob = torch.load(f"{R}/{MU}/cache_s{s}.pt", weights_only=False); pos = {mi: i for i, mi in enumerate(blob["key"]["test_idx"])}
        for mi, kk in zip(per[s][0]["mol_idx"], per[s][0]["budget"]):
            coal.append(int(np.sum(np.array(blob["results"]["subgraphx"][pos[int(mi)]]["node_importance"]) > 0))); k.append(int(kk))
    coal, k = np.array(coal), np.array(k)
    P(f"  GT-bearing molecule-units {len(k)}: coalition > budget in {int((coal > k).sum())} ({(coal > k).mean():.1%}), equal in {int((coal == k).sum())},"
      f" smaller in {int((coal < k).sum())}; mean coalition {coal.mean():.2f} vs budget {k.mean():.2f}")
    rep["budget_rounding"] = {"n": int(len(k)), "coalition_gt_budget": int((coal > k).sum()), "equal": int((coal == k).sum()),
                              "smaller": int((coal < k).sum()), "mean_coalition": float(coal.mean()), "mean_budget": float(k.mean())}

    json.dump(rep, open(f"{R}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {R}/report.json, {R}/{MU}/arrays_s<S>.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
