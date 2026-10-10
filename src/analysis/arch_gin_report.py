"""
Report for the GIN architecture check (runs/arch_gin/): plain PyG GIN on
MUTAG-GT and GraphXAI Mutagenicity, 5 seeds each, on the D-MPNN runs' splits.

Mutagenicity per-molecule arrays come from the same code as the D-MPNN report
(mutagenicity_report.analyse_seed, max-over-set GEA).  MUTAG-GT uses the
single merged GT mask (Jaccard), as every MUTAG-GT D-MPNN result does; the
D-MPNN reference there is recomputed by THIS code from the decoupled-protocol
caches (runs/decoupled/, read-only) so both architectures share one test.

Registered family, Holm within seed (four tests), direction = rank test:
    GNN / PG / SX top-k 0.25 vs RandNode at the budget (20 draws)
    SX mean-threshold mask vs RandConn sized to that mask (20 draws)

Writes only under runs/arch_gin/:
    <dataset>/arrays_s<S>.npz, dmpnn_ref/mutag_graphxai_s<S>.npz, report.json

    python -m src.analysis.arch_gin_report | tee runs/arch_gin/report.txt
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data import LOADERS
from . import mutagenicity_report as mr
from .connectivity_baseline import build_adj, random_connected_subset
from .mutagenicity_cv_report import FAMILY, family_tests, table
from .mutagenicity_report import EXPL, SH
from .oracle_ceiling import budget

R = "runs/arch_gin"
SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = 20
MG, MU = "mutag_graphxai", "mutagenicity_graphxai"
# MUTAG-GT: same four comparisons on the single merged GT mask
FAMILY_M = [(lab, r.replace("_set", ""), n.replace("_set", "")) for lab, r, n in FAMILY]
DMPNN_MG = {"ckpt": "runs/prio/ckpt_mutag_graphxai_s{s}.pt", "cache": "runs/decoupled/cache_mutag_graphxai_s{s}.pt",
            "out": "runs/decoupled/out_mutag_graphxai_s{s}.json"}
GIN_MG = {k: f"{R}/{MG}/{k}_s{{s}}.{'json' if k == 'out' else 'pt'}" for k in ("ckpt", "cache", "out")}


def jac(gt: np.ndarray, m: np.ndarray) -> float:
    u = int((gt | m).sum())
    return int((gt & m).sum()) / u if u else 0.0


def analyse_mutag_gt(paths: dict, s: int, dl, save_to: str):
    """Per-molecule arrays for one MUTAG-GT unit. RandNode and RandConn use separate
    RNG streams, so RandNode is identical for both architectures (same molecules, same k)."""
    ck = torch.load(paths["ckpt"].format(s=s), map_location="cpu", weights_only=False)
    blob = torch.load(paths["cache"].format(s=s), weights_only=False)
    out = json.load(open(paths["out"].format(s=s)))
    idx = list(blob["key"]["test_idx"]); res = blob["results"]
    rng_node, rng_conn = np.random.default_rng([40_000, s, 0]), np.random.default_rng([40_000, s, 1])
    A = {k: [] for k in ["mol_idx", "n_nodes", "gt_size", "budget", "rand_node_topk", "rand_conn_topk",
                         "rand_conn_mean_subgraphx", "sx_coalition"]
         + [f"{e}_{b}" for e in EXPL for b in ("topk", "mean", "mean_size")]}
    for pos, mi in enumerate(idx):
        d = dl[mi]; gt = d.node_gt_mask.cpu().numpy().astype(bool)
        if not gt.any():
            continue
        n = d.num_nodes; k = budget(n); adj = build_adj(d.edge_index.cpu(), n)
        A["mol_idx"].append(mi); A["n_nodes"].append(n); A["gt_size"].append(int(gt.sum())); A["budget"].append(k)
        A["rand_node_topk"].append(np.mean([jac(gt, mr.mask_from_nodes(rng_node.choice(n, size=k, replace=False), n))
                                            for _ in range(N_DRAWS)]))
        for e in EXPL:
            imp = torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float)
            A[f"{e}_topk"].append(jac(gt, mr.topk_mask(imp, k)))
            mm = (imp > imp.mean()).numpy()
            A[f"{e}_mean"].append(jac(gt, mm)); A[f"{e}_mean_size"].append(int(mm.sum()))
        size = A["subgraphx_mean_size"][-1]; A["sx_coalition"].append(int(np.sum(np.array(res["subgraphx"][pos]["node_importance"]) > 0)))
        A["rand_conn_topk"].append(np.mean([jac(gt, mr.mask_from_nodes(random_connected_subset(adj, n, k, rng_conn), n))
                                            for _ in range(N_DRAWS)]))
        A["rand_conn_mean_subgraphx"].append(np.mean([jac(gt, mr.mask_from_nodes(random_connected_subset(adj, n, size, rng_conn), n))
                                                      for _ in range(N_DRAWS)]) if size > 0 else 0.0)
    A = {k: np.array(v) for k, v in A.items()}
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    np.savez(save_to, model_seed=s, explainer_seed=1000 + s, explained_idx=np.array(idx),
             split_train=np.array(ck["split"]["train"]), split_val=np.array(ck["split"]["val"]),
             split_test=np.array(ck["split"]["test"]), **A)
    return A, ck, out


def sign(t):
    return "+" if t["above"] else "-" if t["below"] else "."


def gea_table(P, per, cols, unit="seed"):
    P(f"  {unit:>6}" + "".join(f"{lab:>13}" for _, lab in cols))
    arr = np.array([[float(per[s][0][k].mean()) for k, _ in cols] for s in per])
    for s, row in zip(per, arr):
        P(f"  {s:>6}" + "".join(f"{x:>13.3f}" for x in row))
    P(f"  {'mean':>6}" + "".join(f"{x:>13.3f}" for x in arr.mean(0)))
    P(f"  {'std':>6}" + "".join(f"{x:>13.3f}" for x in arr.std(0)))
    return {str(s): dict(zip([l for _, l in cols], row.tolist())) for s, row in zip(per, arr)}


def health(P, per, rep_key, rep):
    rep[rep_key] = {}
    for s in per:
        h = per[s][2]["explainer_health"]
        P(f"  seed {s}: " + "   ".join(f"{SH[e]} degenerate {h[e]['n_degenerate']}/{h[e]['n']} empty {h[e]['n_empty']} whole {h[e]['n_whole']}"
                                      f"{'  COLLAPSED' if h[e]['n_degenerate'] / max(h[e]['n'], 1) >= 0.5 else ''}" for e in EXPL))
        rep[rep_key][s] = {e: {**h[e], "collapsed": h[e]["n_degenerate"] / max(h[e]["n"], 1) >= 0.5} for e in EXPL}
    npg = sum(rep[rep_key][s]["pgexplainer"]["collapsed"] for s in per)
    P(f"  PGExplainer collapsed (strict rule, n_degenerate/n >= 0.5) in {npg}/{len(per)} seeds")
    return npg


def fidelity(P, per):
    out = {}
    for e in EXPL:
        a = np.array([[per[s][2]["summary"]["R3"]["mean"][e][k][0] for k in ("fid_plus", "fid_minus", "gef")] for s in per])
        P(f"     {SH[e]:<4} Fid+ " + " ".join(f"{x:.3f}" for x in a[:, 0]) + f"  (mean {a[:, 0].mean():.3f})"
          + "   Fid- " + " ".join(f"{x:.3f}" for x in a[:, 1]) + f"  (mean {a[:, 1].mean():.3f})"
          + "   GEF " + " ".join(f"{x:.3f}" for x in a[:, 2]) + f"  (mean {a[:, 2].mean():.3f})")
        out[e] = {"fid_plus": a[:, 0].tolist(), "fid_minus": a[:, 1].tolist(), "gef": a[:, 2].tolist()}
    return out


def compare(P, family, T_gin, T_dm, seeds, name):
    """Per-seed direction strings for both architectures and the cells that differ."""
    P(f"  {'comparison':<18}{'GIN  seeds 0-4':>18}{'above/below':>13}   |{'D-MPNN seeds 0-4':>18}{'above/below':>13}   cells that differ")
    diff = {}
    for lab, _, _ in family:
        g = [sign(T_gin[s][lab]) for s in seeds]; d = [sign(T_dm[s][lab]) for s in seeds]
        dd = [f"s{s}: D-MPNN {b} -> GIN {a}" for s, a, b in zip(seeds, g, d) if a != b]
        P(f"  {lab:<18}{' '.join(g):>18}{str(g.count('+')) + '/' + str(g.count('-')):>13}   |{' '.join(d):>18}"
          f"{str(d.count('+')) + '/' + str(d.count('-')):>13}   {'; '.join(dd) if dd else 'none'}")
        diff[lab] = {"gin": g, "dmpnn": d, "differs": dd}
    return diff


def main():
    P = print
    rep = {"arch": "gin", "n_draws": N_DRAWS}
    P("=" * 124); P(" GIN architecture check -- plain PyG GIN (3 x GINConv, hidden 212, sum pool), D-MPNN splits, decoupled seeds (S, 1000+S)"); P("=" * 124)

    # ------------------------------------------------------------------ Mutagenicity
    dl, meta = LOADERS[MU](); motifs_all = meta.extras["motifs"]
    seeds_u = [s for s in SEEDS if os.path.exists(f"{R}/{MU}/out_s{s}.json")]
    mr.R = f"{R}/{MU}"                              # analyse_seed reads AND writes arrays under this dir only
    per_u = {s: mr.analyse_seed(s, dl, motifs_all) for s in seeds_u}
    dm_u = {s: dict(np.load(f"runs/mutagenicity/arrays_s{s}.npz")) for s in seeds_u}      # D-MPNN, read-only
    dm_ck = {s: torch.load(f"runs/mutagenicity/ckpt_s{s}.pt", map_location="cpu", weights_only=False)["metrics"] for s in seeds_u}

    P(f"\n[U1] MUTAGENICITY -- model (test split)   seed  GIN acc  GIN AUROC | D-MPNN acc  AUROC | explained (<=80 atoms)  GT-bearing | same molecules as D-MPNN")
    rep["mutagenicity_model"] = {}
    for s in seeds_u:
        A, ck, out = per_u[s]; m = ck["metrics"]; same = bool(np.array_equal(A["mol_idx"], dm_u[s]["mol_idx"]))
        P(f"{'':>42}{s:>4}  {m['test_acc']:.4f}   {m['test_auroc']:.4f}   |  {dm_ck[s]['test_acc']:.4f}    {dm_ck[s]['test_auroc']:.4f} |"
          f" {out['config']['n_test']:>12}{len(A['mol_idx']):>18}    | {same}")
        rep["mutagenicity_model"][s] = {"gin_test_acc": m["test_acc"], "gin_test_auroc": m["test_auroc"], "dmpnn_test_acc": dm_ck[s]["test_acc"],
                                        "dmpnn_test_auroc": dm_ck[s]["test_auroc"], "n_explained": out["config"]["n_test"],
                                        "n_gt_bearing": int(len(A["mol_idx"])), "same_molecules_as_dmpnn": same}

    P("\n[U2] MUTAGENICITY -- top-k 0.25 GEA (max-over-set), per-seed means over GT-bearing test molecules")
    cols = [(f"{e}_topk_set", SH[e]) for e in EXPL] + [("rand_node_topk_set", "RandNode"), ("rand_edge_topk_set", "RandEdge"),
                                                      ("rand_conn_topk_set", "RandConn@k"), ("ceiling_set", "ceiling")]
    rep["mutagenicity_gea_topk"] = gea_table(P, per_u, cols)
    P("     D-MPNN, same molecules (runs/mutagenicity/arrays_s<S>.npz):")
    gea_table(P, {s: (dm_u[s],) for s in seeds_u}, cols)
    P("\n     mean-threshold: SubgraphX mask vs RandConn sized to that mask")
    rep["mutagenicity_gea_mean"] = gea_table(P, per_u, [("subgraphx_mean_set", "SX mean-thr"), ("rand_conn_mean_subgraphx_set", "RandConn|SX"),
                                                       ("subgraphx_mean_size", "|M| SX"), ("budget", "budget k")])

    P("\n[U3] MUTAGENICITY -- per-seed paired Wilcoxon, Holm within seed over the four registered comparisons (cells: explainer/null)")
    Tu = {s: family_tests(per_u[s][0], FAMILY) for s in seeds_u}
    table(P, seeds_u, Tu, FAMILY, "seed")
    Tu_dm = {s: family_tests(dm_u[s], FAMILY) for s in seeds_u}
    P("\n     direction per seed (+ sig. above, - sig. below, . n.s.), GIN against the D-MPNN 5-seed run")
    rep["mutagenicity_vs_dmpnn"] = compare(P, FAMILY, Tu, Tu_dm, seeds_u, "Mutagenicity")
    rep["mutagenicity_tests"] = {str(s): Tu[s] for s in seeds_u}
    dis = [(s, lab) for s in seeds_u for lab, _, _ in FAMILY if not Tu[s][lab]["mean_and_rank_agree"]]
    P(f"     cells where the mean and the rank test disagree in direction: {dis if dis else 'none'}")

    P("\n[U4] MUTAGENICITY -- explainer health")
    npg_u = health(P, per_u, "mutagenicity_health", rep)
    P("\n[U5] MUTAGENICITY -- Fid+ / Fid- / GEF, R3 (mean fill), per-seed means over all explained molecules")
    rep["mutagenicity_fidelity_R3"] = fidelity(P, per_u)

    # ------------------------------------------------------------------ MUTAG-GT
    dlg, _ = LOADERS[MG]()
    seeds_g = [s for s in SEEDS if os.path.exists(GIN_MG["out"].format(s=s))]
    per_g = {s: analyse_mutag_gt(GIN_MG, s, dlg, f"{R}/{MG}/arrays_s{s}.npz") for s in seeds_g}
    per_d = {s: analyse_mutag_gt(DMPNN_MG, s, dlg, f"{R}/dmpnn_ref/mutag_graphxai_s{s}.npz") for s in seeds_g}

    P(f"\n[G1] MUTAG-GT -- model (29-graph test split; all 188 graphs explained, 131 of them training graphs)")
    P(f"  {'seed':>6}{'GIN acc':>10}{'GIN AUROC':>11} |{'D-MPNN acc':>12}{'AUROC':>8} |{'explained':>11}{'GT-bearing':>12}")
    rep["mutag_gt_model"] = {}
    for s in seeds_g:
        A, ck, out = per_g[s]; m = ck["metrics"]; md = per_d[s][1]["metrics"]
        P(f"  {s:>6}{m['test_acc']:>10.4f}{m['test_auroc']:>11.4f} |{md['test_acc']:>12.4f}{md['test_auroc']:>8.4f} |{out['config']['n_test']:>11}{len(A['mol_idx']):>12}")
        rep["mutag_gt_model"][s] = {"gin_test_acc": m["test_acc"], "gin_test_auroc": m["test_auroc"], "dmpnn_test_acc": md["test_acc"],
                                    "dmpnn_test_auroc": md["test_auroc"], "n_explained": out["config"]["n_test"], "n_gt_bearing": int(len(A["mol_idx"]))}

    P("\n[G2] MUTAG-GT -- top-k 0.25 GEA (Jaccard vs the merged GT mask), per-seed means over the 188 graphs")
    colg = [(f"{e}_topk", SH[e]) for e in EXPL] + [("rand_node_topk", "RandNode"), ("rand_conn_topk", "RandConn@k")]
    rep["mutag_gt_gea_topk"] = gea_table(P, per_g, colg)
    P("     D-MPNN, decoupled-protocol caches (runs/decoupled/), same code, same RandNode draws:")
    rep["mutag_gt_gea_topk_dmpnn"] = gea_table(P, per_d, colg)
    P("\n     mean-threshold: SubgraphX mask vs RandConn sized to that mask")
    rep["mutag_gt_gea_mean"] = gea_table(P, per_g, [("subgraphx_mean", "SX mean-thr"), ("rand_conn_mean_subgraphx", "RandConn|SX"),
                                                   ("subgraphx_mean_size", "|M| SX"), ("budget", "budget k")])

    P("\n[G3] MUTAG-GT -- per-seed paired Wilcoxon, Holm within seed over the four registered comparisons (cells: explainer/null)")
    Tg = {s: family_tests(per_g[s][0], FAMILY_M) for s in seeds_g}
    table(P, seeds_g, Tg, FAMILY_M, "seed")
    Td = {s: family_tests(per_d[s][0], FAMILY_M) for s in seeds_g}
    P("\n     D-MPNN (decoupled caches), same tests:")
    table(P, seeds_g, Td, FAMILY_M, "seed")
    P("\n     direction per seed (+ sig. above, - sig. below, . n.s.)")
    rep["mutag_gt_vs_dmpnn"] = compare(P, FAMILY_M, Tg, Td, seeds_g, "MUTAG-GT")
    rep["mutag_gt_tests"] = {str(s): Tg[s] for s in seeds_g}; rep["mutag_gt_tests_dmpnn"] = {str(s): Td[s] for s in seeds_g}
    dis = [(s, lab) for s in seeds_g for lab, _, _ in FAMILY_M if not Tg[s][lab]["mean_and_rank_agree"]]
    P(f"     cells where the mean and the rank test disagree in direction: {dis if dis else 'none'}")

    P("\n[G4] MUTAG-GT -- explainer health")
    npg_g = health(P, per_g, "mutag_gt_health", rep)
    P("\n[G5] MUTAG-GT -- Fid+ / Fid- / GEF, R3 (mean fill), per-seed means over all explained molecules")
    rep["mutag_gt_fidelity_R3"] = fidelity(P, per_g)

    # ------------------------------------------------------------------ predictions
    P("\n[P] REGISTERED PREDICTIONS (committed before any GIN explanation was scored)")
    a = sum(Tu[s]["GNN vs RandNode"]["above"] for s in seeds_u); b = sum(Tu[s]["SX vs RandConn"]["above"] for s in seeds_u)
    c = sum(Tg[s]["SX vs RandNode"]["below"] for s in seeds_g); d_ = sum(not Tu[s]["PG vs RandNode"]["above"] for s in seeds_u)
    P(f"  (a) Mutagenicity: GNNExplainer significantly above RandNode in >= 4/5 seeds:        observed {a}/{len(seeds_u)} -> {'HELD' if a >= 4 else 'DID NOT HOLD'}")
    P(f"  (b) Mutagenicity: SubgraphX significantly above its RandConn null in >= 4/5 seeds: observed {b}/{len(seeds_u)} -> {'HELD' if b >= 4 else 'DID NOT HOLD'}")
    P(f"  (c) MUTAG-GT: SubgraphX significantly below RandNode in >= 4/5 seeds:              observed {c}/{len(seeds_g)} -> {'HELD' if c >= 4 else 'DID NOT HOLD'}")
    P(f"  (d) Mutagenicity: PGExplainer NOT significantly above RandNode in >= 3/5 seeds:    observed {d_}/{len(seeds_u)} -> {'HELD' if d_ >= 3 else 'DID NOT HOLD'}")
    rep["predictions"] = {"a": a, "b": b, "c": c, "d": d_, "a_held": a >= 4, "b_held": b >= 4, "c_held": c >= 4, "d_held": d_ >= 3,
                          "pg_collapse_mutagenicity": npg_u, "pg_collapse_mutag_gt": npg_g}

    # ------------------------------------------------------------------ budget rounding
    P("\n[Q] BUDGET ROUNDING (not changed): SubgraphX is asked for max(3, round(0.25 N)) atoms (dig_subgraphx.py), while top-k scoring"
      "\n    and R3 masking keep max(1, int(0.25 N)) (masking.py). Where the coalition is larger, top-k drops atoms from a 0/1 vector.")
    rep["budget_rounding"] = {}
    for name, per in (("Mutagenicity", per_u), ("MUTAG-GT", per_g)):
        if name == "MUTAG-GT":
            coal = np.concatenate([per[s][0]["sx_coalition"] for s in per]); k = np.concatenate([per[s][0]["budget"] for s in per])
        else:
            coal, k = [], []
            for s in per:
                blob = torch.load(f"{R}/{MU}/cache_s{s}.pt", weights_only=False); pos = {mi: i for i, mi in enumerate(blob["key"]["test_idx"])}
                for mi, kk in zip(per[s][0]["mol_idx"], per[s][0]["budget"]):
                    coal.append(int(np.sum(np.array(blob["results"]["subgraphx"][pos[int(mi)]]["node_importance"]) > 0))); k.append(int(kk))
            coal, k = np.array(coal), np.array(k)
        P(f"  {name:<13} GT-bearing molecule-units {len(k)}: coalition > budget in {int((coal > k).sum())} ({(coal > k).mean():.1%}),"
          f" equal in {int((coal == k).sum())}, smaller in {int((coal < k).sum())}; mean coalition {coal.mean():.2f} vs budget {k.mean():.2f}")
        rep["budget_rounding"][name] = {"n": int(len(k)), "coalition_gt_budget": int((coal > k).sum()), "equal": int((coal == k).sum()),
                                        "smaller": int((coal < k).sum()), "mean_coalition": float(coal.mean()), "mean_budget": float(k.mean())}

    json.dump(rep, open(f"{R}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {R}/report.json, {R}/<dataset>/arrays_s<S>.npz, {R}/dmpnn_ref/mutag_graphxai_s<S>.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
