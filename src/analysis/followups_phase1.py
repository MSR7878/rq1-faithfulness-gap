"""
RQ1 follow-ups, Phase 1: MUTAG-GT with 10 seeds (runs/rq1_followups/PLAN.md).

Seeds 0-4: the stored priority-sweep runs (runs/prio). Seeds 5-9: new runs with
the same protocol (runs/rq1_followups/phase1). For every seed, the SAME code
as the paper computes:
  a) top-k GEA of GNN/PG/SX vs RandNode (exploratory_null_audit.collect; Holm
     over the 3 explainers) and SX mean-threshold GEA vs size-matched RandConn
     (connectivity_baseline.run_dataset; one test per seed);
  b) Part A reliance, R1/R2, random-connected null (reliance_fidelity.run_part_a;
     Holm over the 4 Part A tests);
  c) random-corrected Fid+/Fid- under R3 vs the random-node null
     (reliance_fidelity.run_part_b; Holm over its 18 tests);
  d) PGExplainer degenerate masks (strict rule) from each run's output json.
Seeds 0-4 are recomputed and must equal the stored arrays exactly; otherwise
the script stops. Writes only runs/rq1_followups/phase1/.

    python -m src.analysis.followups_phase1 | tee runs/rq1_followups/phase1/report.txt
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest

from . import connectivity_baseline as cb
from . import exploratory_null_audit as na
from . import reliance_fidelity as rf
from .mutagenicity_cv_report import rank_direction
from .mutagenicity_report import holm, wpair
from .reliance_fidelity_report import b_signs

R = "runs/rq1_followups/phase1"
SEEDS = list(range(10))
OLD, NEW = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
SHA = {  # first 16 hex of SHA-256 before the run (PLAN.md)
    "ckpt": "56d85aaecf94a053 07a3c1231cfddfce b7cf7ca769a9de8c d01510675acc81fa 4691f6c6b8ba0e62".split(),
    "cache": "57fee6ec883e4597 d9deed6b8c50cdc1 277c5b7d4d0ac026 7bdc0a1b1c232f30 9b5cabf13367961a".split(),
    "out": "64ceb411f1b9ef23 5d143aef57b7ee31 26f6f93b5c176931 860c2bcacc44cd72 0f3d0262e7039acd".split()}
P = print


def paths(s):
    d, ext = ("runs/prio", "") if s in OLD else (R, "")
    return {"ckpt": f"{d}/ckpt_mutag_graphxai_s{s}.pt", "cache": f"{d}/cache_mutag_graphxai_s{s}.pt", "out": f"{d}/out_mutag_graphxai_s{s}.json"}


class _Tmpl:
    """format(s=) -> the right cache for that seed (stored for 0-4, new for 5-9)."""
    def format(self, s):
        return paths(s)["cache"]


def stop(msg):
    P(f"\nSTOP: {msg}"); open(f"{R}/STOPPED", "w").write(msg + "\n"); sys.exit(1)


def sign(p_holm, d):
    return "+" if (d > 0 and p_holm < 0.05) else "-" if (d < 0 and p_holm < 0.05) else "."


def counts(sg):
    return f"{sg.count('+')}/{sg.count('-')}/{sg.count('.')}"


def sign_test(dirs):
    """exact two-sided sign test over seeds on the per-seed rank-test direction (+1 / -1; 0 dropped)."""
    pos, neg = sum(d > 0 for d in dirs), sum(d < 0 for d in dirs)
    return pos, neg, (binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0)


def main():
    os.makedirs(R, exist_ok=True)
    rep = {}
    # ---------------------------------------------------------------- 0. integrity of seeds 0-4
    P("=" * 120); P(" PHASE 1 -- MUTAG-GT, 10 seeds (0-4 stored priority-sweep runs; 5-9 new, same protocol)"); P("=" * 120)
    bad = []
    for kind in ("ckpt", "cache", "out"):
        for s in OLD:
            h = hashlib.sha256(open(paths(s)[kind], "rb").read()).hexdigest()[:16]
            if h != SHA[kind][s]:
                bad.append(f"{kind} s{s}")
    P(f"\n[0] seeds 0-4 stored files unchanged (SHA-256 vs PLAN.md): {'yes' if not bad else 'NO: ' + str(bad)}")
    if bad:
        stop(f"stored seed 0-4 files changed: {bad}")
    # ---------------------------------------------------------------- model accuracy
    P("\n[M] model, 29-graph test split          seed   test acc   test AUROC")
    rep["model"] = {}
    for s in SEEDS:
        m = torch.load(paths(s)["ckpt"], map_location="cpu", weights_only=False)["metrics"]
        P(f"{'':>40}{s:>4}   {m['test_acc']:.4f}     {m['test_auroc']:.4f}" + ("   (new)" if s in NEW else ""))
        rep["model"][s] = {"test_acc": m["test_acc"], "test_auroc": m["test_auroc"]}

    # ---------------------------------------------------------------- a) GEA
    na.OUT, na.SEEDS = R, SEEDS
    gea = na.collect("mutag_graphxai", _Tmpl())                      # writes R/null_audit_arrays/mutag_graphxai_s<S>.npz
    for s in OLD:
        old = np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/mutag_graphxai_s{s}.npz")
        new = np.load(f"{R}/null_audit_arrays/mutag_graphxai_s{s}.npz")
        diff = [k for k in old.files if not np.array_equal(old[k], new[k], equal_nan=True)]
        if diff:
            stop(f"GEA arrays for stored seed {s} do not reproduce: {diff}")
    cb.SEEDS = SEEDS
    conn = cb.run_dataset("mutag_graphxai", "mutag_graphxai", lambda s: paths(s)["cache"], save_dir=R)
    old = np.load("runs/prio/connectivity_arrays_mutag_graphxai.npz")
    diff = [k for k in old.files if not np.array_equal(old[k], np.load(f"{R}/connectivity_arrays_mutag_graphxai.npz")[k])]
    if diff:
        stop(f"connectivity arrays for stored seeds do not reproduce: {diff}")
    P("\n[a] seeds 0-4 GEA and connectivity arrays recomputed: identical to the stored ones")

    # ---------------------------------------------------------------- b) Part A, c) Part B
    os.makedirs(f"{R}/partA", exist_ok=True); os.makedirs(f"{R}/partB", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A, B = {}, {}
    for s in SEEDS:
        name, spec = f"mutag_graphxai_s{s}", {**paths(s), "seed": s}
        fa, fb = f"{R}/partA/{name}.npz", f"{R}/partB/{name}.npz"
        if not os.path.exists(fa):
            a, _ = rf.run_part_a(name, spec, device); np.savez(fa, unit=name, seed=s, **a)
        if not os.path.exists(fb):
            if s in OLD:
                b = dict(np.load(f"runs/rq1_reliance_fidelity/partB/{name}.npz", allow_pickle=True))   # stored, validated earlier
            else:
                b, _ = rf.run_part_b(name, spec, device)
            np.savez(fb, unit=name, seed=s, **{k: v for k, v in b.items() if k not in ("unit", "seed")})
        A[s] = dict(np.load(fa, allow_pickle=True)); B[s] = dict(np.load(fb, allow_pickle=True))
    for s in OLD:
        st = np.load(f"runs/rq1_reliance_fidelity/partA/mutag_graphxai_s{s}.npz", allow_pickle=True)
        diff = [k for k in ("mol_idx", "delta_gt_R1", "delta_gt_R2", "null_conn_R1", "null_conn_R2", "null_node_R1", "null_node_R2")
                if not np.allclose(st[k], A[s][k], atol=1e-5, equal_nan=True)]
        if diff:
            stop(f"Part A for stored seed {s} does not reproduce: {diff}")
    P("[b] Part A for seeds 0-4 recomputed: identical to the stored arrays (atol 1e-5)")

    # ---------------------------------------------------------------- per-seed tests
    T = {}
    for s in SEEDS:
        g, _, _ = gea[s]
        raw = {e: wpair(g[f"{e}_topk"], g["rn_k"]) for e in EXPL}; adj = holm(raw)
        row = {f"GEA {SH[e]} vs RandNode": (sign(adj[e], rank_direction(g[f"{e}_topk"], g["rn_k"])), rank_direction(g[f"{e}_topk"], g["rn_k"]),
                                            float(g[f"{e}_topk"].mean()), float(g["rn_k"].mean())) for e in EXPL}
        c = conn[s]; p = wpair(c["sx"], c["rand_conn"]); d = rank_direction(c["sx"], c["rand_conn"])
        row["GEA SX vs RandConn (SX size)"] = (sign(p, d), d, float(c["sx"].mean()), float(c["rand_conn"].mean()))
        a = A[s]; ok = ~np.isnan(a["delta_gt_R1"])
        fam = {(mk, ref): a[f"delta_gt_{mk}"][ok] - a[f"null_{ref}_{mk}"][ok] for mk in ("R1", "R2") for ref in ("conn", "node")}
        rawA = {k: wpair(v, np.zeros_like(v)) for k, v in fam.items()}; adjA = holm(rawA)
        for mk in ("R1", "R2"):
            v = fam[(mk, "conn")]; d = rank_direction(v, np.zeros_like(v))
            row[f"reliance {mk} vs RandConn"] = (sign(adjA[(mk, "conn")], d), d, float(a[f"delta_gt_{mk}"][ok].mean()), float(a[f"null_conn_{mk}"][ok].mean()))
        sb = b_signs(B[s])["node"]
        for e in EXPL:
            for met in ("fid_plus", "fid_minus"):
                corr = B[s][f"{e}_R3_{met}_logit"] - B[s][f"rand_node_R3_{met}_logit"]
                lab = f"R3 {'Fid+' if met == 'fid_plus' else 'Fid-'} {SH[e]} - RandNode"
                row[lab] = (sb[(e, "R3", met)], rank_direction(corr, np.zeros_like(corr)), float(corr.mean()), 0.0)
        h = json.load(open(paths(s)["out"]))["explainer_health"]["pgexplainer"]
        row["PG degenerate"] = (h["n_degenerate"], h["n"], h["n_degenerate"] / h["n"] >= 0.5)
        T[s] = row
    keys = [k for k in T[0] if k != "PG degenerate"]

    P("\n[T] per-seed results: mean real / mean null and direction (+ above, - below, . n.s.)")
    P(f"  {'comparison':<34}" + "".join(f"{'s' + str(s):>15}" for s in SEEDS))
    for k in keys:
        cell = lambda s: (f"{T[s][k][2]:.3f}/{T[s][k][3]:.3f}{T[s][k][0]}" if not k.startswith("R3") else f"{T[s][k][2]:+.3f}{T[s][k][0]}")
        P(f"  {k:<34}" + "".join(f"{cell(s):>15}" for s in SEEDS))
    P(f"  {'PG degenerate / n (collapsed)':<34}" + "".join(f"{str(T[s]['PG degenerate'][0]) + '/' + str(T[s]['PG degenerate'][1]) + ('*' if T[s]['PG degenerate'][2] else ''):>15}" for s in SEEDS))

    P("\n[C] direction counts above/below/n.s.           seeds 0-4     seeds 5-9      all 10    | exploratory sign test over 10 seeds (rank direction): +/-, p")
    rep["counts"] = {}
    for k in keys:
        c = [counts([T[s][k][0] for s in grp]) for grp in (OLD, NEW, SEEDS)]
        line = f"  {k:<42}{c[0]:>12}{c[1]:>12}{c[2]:>12}"
        if k.startswith("GEA") or k.startswith("reliance"):
            pos, neg, p = sign_test([T[s][k][1] for s in SEEDS]); line += f"    | {pos}/{neg}, p = {p:.4f}"
            rep["counts"][k] = {"old": c[0], "new": c[1], "all": c[2], "sign_test": [pos, neg, p]}
        else:
            rep["counts"][k] = {"old": c[0], "new": c[1], "all": c[2]}
        P(line)
    ncol = [sum(T[s]["PG degenerate"][2] for s in grp) for grp in (OLD, NEW, SEEDS)]
    P(f"  {'PGExplainer collapsed (strict rule)':<42}{ncol[0]:>12}{ncol[1]:>12}{ncol[2]:>12}")
    rep["pg_collapsed"] = {"old": ncol[0], "new": ncol[1], "all": ncol[2], "per_seed": {s: T[s]["PG degenerate"] for s in SEEDS}}

    P("\n[P] PREDICTIONS (PLAN.md)")
    p1 = sum(T[s]["GEA SX vs RandConn (SX size)"][0] == "-" for s in SEEDS)
    p2 = [sum(T[s][f"reliance {mk} vs RandConn"][0] == "-" for s in SEEDS) for mk in ("R1", "R2")]
    p3 = sum(T[s]["GEA GNN vs RandNode"][0] == "+" for s in SEEDS)
    P(f"  P1 SX significantly below its size-matched RandConn on GEA in >= 8/10 seeds: {p1}/10 -> {'HELD' if p1 >= 8 else 'DID NOT HOLD'}")
    P(f"  P2 reliance significantly below RandConn in >= 8/10 seeds, R1 and R2: R1 {p2[0]}/10, R2 {p2[1]}/10 -> {'HELD' if min(p2) >= 8 else 'DID NOT HOLD'}")
    P(f"  P3 GNN significantly above RandNode on top-k GEA in >= 7/10 seeds: {p3}/10 -> {'HELD' if p3 >= 7 else 'DID NOT HOLD'}")
    rep["P1"], rep["P2"], rep["P3"] = {"n": p1, "held": p1 >= 8}, {"R1": p2[0], "R2": p2[1], "held": min(p2) >= 8}, {"n": p3, "held": p3 >= 7}
    # ---------------------------------------------------------------- addendum: scoring ways (a) / (b) / (c)
    from ..data import LOADERS
    from .connectivity_baseline import build_adj, random_connected_subset
    from .ties_random import N_ORDERS, boundary_tie, random_tie_gea
    dl, _ = LOADERS["mutag_graphxai"]()
    Bt, ties = {}, {}
    for s in SEEDS:
        blob = torch.load(paths(s)["cache"], weights_only=False); res = blob["results"]
        g = gea[s][0]; rngc = np.random.default_rng(40000 + s)
        arr = {f"{e}_topk_b": [] for e in EXPL}; arr["rand_conn_k"] = []; mol = []; tc = {e: 0 for e in EXPL}
        for pos, mi in enumerate(blob["key"]["test_idx"]):
            d = dl[mi]; gt = d.node_gt_mask.numpy().astype(bool)
            if not gt.any():
                continue
            n = d.num_nodes; k = max(1, int(0.25 * n)); sc = na.Scorer(gt); mol.append(mi)
            for e in EXPL:
                v, tied = random_tie_gea(res[e][pos]["node_importance"], k, sc, f"mutag_graphxai_s{s}|{e}", pos)
                arr[f"{e}_topk_b"].append(v); tc[e] += int(tied)
            adj = build_adj(d.edge_index, n)
            arr["rand_conn_k"].append(np.mean([sc(np.isin(np.arange(n), list(random_connected_subset(adj, n, k, rngc)))) for _ in range(N_ORDERS)]))
        if not np.array_equal(np.array(mol), g["mol_idx"].astype(int)):
            stop(f"seed {s}: molecule order of the tie scoring differs from the GEA arrays")
        Bt[s] = {k_: np.array(v) for k_, v in arr.items()}; ties[s] = tc
        np.savez(f"{R}/null_audit_arrays/mutag_graphxai_s{s}_random_ties.npz", mol_idx=np.array(mol), **Bt[s])

    def fam3(real, null):
        raw = {e: wpair(real[e], null[e]) for e in EXPL}; adj = holm(raw)
        return {e: sign(adj[e], rank_direction(real[e], null[e])) for e in EXPL}
    W = {}
    for s in SEEDS:
        g = gea[s][0]
        W[s] = {"a": fam3({e: g[f"{e}_topk"] for e in EXPL}, {e: g["rn_k"] for e in EXPL}),
                "b": fam3({e: Bt[s][f"{e}_topk_b"] for e in EXPL}, {e: g["rn_k"] for e in EXPL}),
                "c": fam3({e: g[f"{e}_mean"] for e in EXPL}, {e: g[f"{e}_unc"] for e in EXPL})}
        p = wpair(Bt[s]["subgraphx_topk_b"], Bt[s]["rand_conn_k"]); W[s]["b_conn"] = sign(p, rank_direction(Bt[s]["subgraphx_topk_b"], Bt[s]["rand_conn_k"]))
        W[s]["ac_conn"] = T[s]["GEA SX vs RandConn (SX size)"][0]
    P("\n[W] ADDENDUM -- GEA scored three ways: (a) current torch order, (b) PRIMARY random ties (20 orders, averaged per molecule),"
      "\n    (c) mean-threshold vs size-matched nulls. Molecules with a tie at the top-k boundary, per seed (of GT-bearing):")
    P("     " + "   ".join(f"s{s}: GNN {ties[s]['gnnexplainer']} PG {ties[s]['pgexplainer']} SX {ties[s]['subgraphx']}" for s in SEEDS))
    P(f"  {'comparison':<40}{'way':>5}{'seeds 0-4':>12}{'seeds 5-9':>12}{'all 10':>10}   per seed")
    rep["three_ways"] = {}
    rows = [(f"{SH[e]} vs RandNode (top-k)" if w != "c" else f"{SH[e]} mean-thr vs RandUnc (size-matched)", w, lambda s, e=e, w=w: W[s][w][e])
            for e in EXPL for w in ("a", "b", "c")]
    rows += [("SX mean-thr vs RandConn (SX size) [P1 as registered]", "a=c", lambda s: W[s]["ac_conn"]),
             ("SX top-k vs RandConn at the budget", "b", lambda s: W[s]["b_conn"])]
    for lab, w, f in rows:
        sg = [f(s) for s in SEEDS]
        P(f"  {lab:<40}{w:>5}{counts(sg[:5]):>12}{counts(sg[5:]):>12}{counts(sg):>10}   {' '.join(sg)}")
        rep["three_ways"][f"{lab}|{w}"] = {"old": counts(sg[:5]), "new": counts(sg[5:]), "all": counts(sg), "signs": sg}
    p1 = {"a": sum(W[s]["ac_conn"] == "-" for s in SEEDS), "b": sum(W[s]["b_conn"] == "-" for s in SEEDS)}; p1["c"] = p1["a"]
    p3 = {w: sum(W[s][w]["gnnexplainer"] == "+" for s in SEEDS) for w in ("a", "b", "c")}
    P("\n[P'] PREDICTIONS under each way ((b) primary)")
    for w in ("b", "a", "c"):
        P(f"  ({w})  P1 {p1[w]}/10 -> {'HELD' if p1[w] >= 8 else 'DID NOT HOLD'}   P2 R1 {p2[0]}/10, R2 {p2[1]}/10 -> "
          f"{'HELD' if min(p2) >= 8 else 'DID NOT HOLD'} (no explainer; same under every way)   P3 {p3[w]}/10 -> {'HELD' if p3[w] >= 7 else 'DID NOT HOLD'}")
    rep["predictions_three_ways"] = {w: {"P1": p1[w], "P1_held": p1[w] >= 8, "P2": p2, "P2_held": min(p2) >= 8, "P3": p3[w], "P3_held": p3[w] >= 7}
                                     for w in ("a", "b", "c")}
    rep["boundary_ties"] = ties
    rep["per_seed"] = {s: {k: list(v) for k, v in T[s].items()} for s in SEEDS}
    json.dump(rep, open(f"{R}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {R}/report.json, {R}/null_audit_arrays/, {R}/connectivity_arrays_mutag_graphxai.npz, {R}/partA/, {R}/partB/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
