"""
RQ1 follow-ups, Phase 2b: tie robustness of all stored top-k SubgraphX and
PGExplainer results under rule (b) -- random tie-break, 20 seeded orders per
molecule, averaged per molecule (runs/rq1_followups/PLAN.md, addendum).

    python -m src.analysis.followups_phase2b gea            # all GEA units (no model needed)
    python -m src.analysis.followups_phase2b fid --units ...  # fidelity, Part B units (+ MUTAG-GT seeds 5-9)
    python -m src.analysis.followups_phase2b report

Nulls are the stored ones. Each count table is redone with its ORIGINAL test
code. Writes only runs/rq1_followups/phase2b/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from ..data import LOADERS
from ..explain.common import to_cpu_data
from ..metrics.gea_set import gea_max_over_set
from . import exploratory_null_audit as na
from . import reliance_fidelity as rf
from .connectivity_baseline import build_adj, random_connected_subset
from .mutagenicity_cv_report import FAMILY, family_tests
from .reliance_fidelity_report import b_signs, family as holm_family, test as rtest
from .ties_random import N_ORDERS, boundary_tie, mol_rng, random_tie_gea, removed_boundary_tie, tie_order

OUT = "runs/rq1_followups/phase2b"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
TIED = ["pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
SEEDS = [0, 1, 2, 3, 4]
AUD = "runs/mutagenicity/exploratory/null_audit_arrays"
ARCH = {"D-MPNN": "runs/mutagenicity", "GIN": "runs/arch_gin/mutagenicity_graphxai", "GINE": "runs/arch_gine/mutagenicity_graphxai",
        "GCN": "runs/arch_gcn/mutagenicity_graphxai"}
AUDIT_KEYS = {"mutag_graphxai": "runs/prio/cache_mutag_graphxai_s{s}.pt", "bxaic_indole": "runs/prio/cache_bxaic_indole_s{s}.pt",
              "bxaic_PAINS": "runs/prio/cache_bxaic_PAINS_s{s}.pt", "bxaic_X": "runs/prio/cache_bxaic_X_s{s}.pt",
              "bxaic_P": "runs/prio/cache_bxaic_P_s{s}.pt", "mutagenicity_graphxai": "runs/mutagenicity/cache_s{s}.pt"}


# ============================================================================ GEA
def gea_unit(tag, cache, dl, scorer_of, expect_mol, conn_k_rng=None):
    """Random-tie top-k GEA (and tie flags) for every explainer, molecules in the stored arrays' order."""
    blob = torch.load(cache, weights_only=False); res = blob["results"]
    A = {f"{e}_b": [] for e in EXPL}; A.update({f"{e}_tie": [] for e in EXPL}); A["mol_idx"] = []; A["rand_conn_k"] = []
    for pos, mi in enumerate(blob["key"]["test_idx"]):
        d = dl[mi]
        if not bool(d.node_gt_mask.any()):
            continue
        n = d.num_nodes; k = max(1, int(0.25 * n)); sc = scorer_of(mi, d)
        A["mol_idx"].append(mi)
        for e in EXPL:
            v, tied = random_tie_gea(res[e][pos]["node_importance"], k, sc, f"{tag}|{e}", pos)
            A[f"{e}_b"].append(v); A[f"{e}_tie"].append(tied)
        if conn_k_rng is not None:
            adj = build_adj(d.edge_index, n)
            A["rand_conn_k"].append(np.mean([sc(np.isin(np.arange(n), list(random_connected_subset(adj, n, k, conn_k_rng)))) for _ in range(N_ORDERS)]))
    A = {k_: np.array(v) for k_, v in A.items()}
    if not np.array_equal(A["mol_idx"], np.asarray(expect_mol).astype(int)):
        raise SystemExit(f"STOP: {tag}: molecule order differs from the stored arrays")
    return A


def run_gea():
    os.makedirs(f"{OUT}/gea", exist_ok=True)
    loaded = {}

    def data(key):
        if key not in loaded:
            loaded[key] = LOADERS[key]()
        return loaded[key]
    # null-audit units (main count table, B-XAIC tasks)
    for key, tmpl in AUDIT_KEYS.items():
        dl, meta = data(key); motifs = getattr(meta, "extras", {}).get("motifs") if hasattr(meta, "extras") else None
        for s in SEEDS:
            dst = f"{OUT}/gea/audit_{key}_s{s}.npz"
            if os.path.exists(dst):
                continue
            z = np.load(f"{AUD}/{key}_s{s}.npz")
            scorer = (lambda mi, d: na.Scorer(d.node_gt_mask.numpy().astype(bool), motifs[mi] if motifs is not None else None))
            rng = np.random.default_rng(40000 + s) if key != "mutagenicity_graphxai" else None
            A = gea_unit(f"audit_{key}_s{s}", tmpl.format(s=s), dl, scorer, z["mol_idx"], rng)
            np.savez(dst, **A); print(f"audit {key} s{s}: ties PG {A['pgexplainer_tie'].mean():.2f} SX {A['subgraphx_tie'].mean():.2f} GNN {A['gnnexplainer_tie'].mean():.2f}", flush=True)
    # Mutagenicity arrays (architecture table, max-over-set) and 5-fold CV
    dl, meta = data("mutagenicity_graphxai"); motifs = meta.extras["motifs"]
    scorer = lambda mi, d: (lambda m: gea_max_over_set(m, motifs[mi]).max_over_set)
    units = [(f"arch_{a}_s{s}", f"{d}/cache_s{s}.pt", f"{d}/arrays_s{s}.npz") for a, d in ARCH.items() for s in SEEDS]
    units += [(f"cv_f{f}", f"runs/mutagenicity_cv/cache_s{f}.pt", f"runs/mutagenicity_cv/arrays_s{f}.npz") for f in SEEDS]
    for tag, cache, arr in units:
        dst = f"{OUT}/gea/{tag}.npz"
        if os.path.exists(dst):
            continue
        A = gea_unit(tag, cache, dl, scorer, np.load(arr)["mol_idx"])
        np.savez(dst, **A); print(f"{tag}: ties PG {A['pgexplainer_tie'].mean():.2f} SX {A['subgraphx_tie'].mean():.2f} GNN {A['gnnexplainer_tie'].mean():.2f}", flush=True)


# ============================================================================ fidelity
def fid_unit(name, spec, device):
    model, ck, blob, out, dl, meta = rf.load_unit(name, spec, device)
    test_idx = list(blob["key"]["test_idx"]); res = blob["results"]
    fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    A = {"mol_idx": np.array(test_idx)}
    for e in TIED:
        A[f"{e}_tie_E"] = np.zeros(len(test_idx), bool); A[f"{e}_tie_GmE"] = np.zeros(len(test_idx), bool)
        for mk in rf.MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                for sp in ("prob", "logit"):
                    A[f"{e}_{mk}_{met}_{sp}_b"] = np.full(len(test_idx), np.nan)
    for pos, mi in enumerate(test_idx):
        G = None
        for e in TIED:
            imp = res[e][pos]["node_importance"]; n = len(imp)
            k, keep = max(1, int(rf.K_FRAC * n)), max(1, int((1 - rf.K_FRAC) * n))
            tE, tR = boundary_tie(imp, k), removed_boundary_tie(imp, keep)
            A[f"{e}_tie_E"][pos], A[f"{e}_tie_GmE"][pos] = tE, tR
            if not (tE or tR):
                continue                                         # identical under every order: keep the current value
            G = G if G is not None else to_cpu_data(dl[mi])
            rng = mol_rng(f"{name}|{e}|fid", pos); graphs, idx = [G], []
            for _ in range(N_ORDERS):
                o = tie_order(imp, rng)
                for mk in rf.MASKINGS:
                    E, GmE = rf.masked_pair(G, o, fill, mk); idx.append((mk, len(graphs))); graphs += [GmE, E]
            lg = rf.logits_of(model, graphs, device); tgt = int(res[e][pos]["target"])
            m, p = rf.margin(lg, tgt), rf.prob(lg, tgt)
            for mk in rf.MASKINGS:
                ii = np.array([i for mk_, i in idx if mk_ == mk])
                A[f"{e}_{mk}_fid_plus_prob_b"][pos] = np.mean(p[0] - p[ii]); A[f"{e}_{mk}_fid_minus_prob_b"][pos] = np.mean(p[0] - p[ii + 1])
                A[f"{e}_{mk}_fid_plus_logit_b"][pos] = np.mean(m[0] - m[ii]); A[f"{e}_{mk}_fid_minus_logit_b"][pos] = np.mean(m[0] - m[ii + 1])
    return A


def fid_units():
    U = {n: v for n, v in rf.all_units().items() if "B" in v["parts"]}
    for s in range(5, 10):
        d = "runs/rq1_followups/phase1"
        U[f"mutag_graphxai_s{s}"] = {"ckpt": f"{d}/ckpt_mutag_graphxai_s{s}.pt", "cache": f"{d}/cache_mutag_graphxai_s{s}.pt",
                                     "out": f"{d}/out_mutag_graphxai_s{s}.json", "parts": "B", "seed": s}
    return U


def stored_partB(name):
    f = f"runs/rq1_reliance_fidelity/partB/{name}.npz"
    return f if os.path.exists(f) else f"runs/rq1_followups/phase1/partB/{name}.npz"


def merged_B(name):
    """Part B arrays with the PG/SX fidelity of tied molecules replaced by the random-tie (b) average."""
    B = dict(np.load(stored_partB(name), allow_pickle=True)); F = dict(np.load(f"{OUT}/fid/{name}.npz"))
    assert np.array_equal(B["mol_idx"], F["mol_idx"])
    Bb = dict(B)
    for e in TIED:
        for mk in rf.MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                for sp in ("prob", "logit"):
                    v = F[f"{e}_{mk}_{met}_{sp}_b"]; cur = B[f"{e}_{mk}_{met}_{sp}"]
                    Bb[f"{e}_{mk}_{met}_{sp}"] = np.where(np.isnan(v), cur, v)
    return B, Bb, F


# ============================================================================ report
def mark(a, b):
    return "" if a == b else "  <-- changes"


def report():
    P = print; rep = {}
    P("=" * 124); P(" PHASE 2b -- tie robustness of stored top-k SubgraphX / PGExplainer results, rule (b): random ties, 20 orders, averaged per molecule"); P("=" * 124)
    G = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in glob.glob(f"{OUT}/gea/*.npz")}
    # ---- tie fractions
    P("\n[1] fraction of GT-bearing molecules with a tie at the top-k boundary (mean over seeds / folds)")
    P(f"  {'unit group':<34}{'GNN':>8}{'PG':>8}{'SX':>8}")
    groups = {f"audit_{k}": [f"audit_{k}_s{s}" for s in SEEDS] for k in AUDIT_KEYS}
    groups.update({f"arch_{a}": [f"arch_{a}_s{s}" for s in SEEDS] for a in ARCH}); groups["cv"] = [f"cv_f{f}" for f in SEEDS]
    rep["tie_fraction_gea"] = {}
    for g, us in groups.items():
        fr = [np.mean([G[u][f"{e}_tie"].mean() for u in us]) for e in EXPL]
        P(f"  {g:<34}" + "".join(f"{x:>8.2f}" for x in fr)); rep["tie_fraction_gea"][g] = dict(zip(EXPL, fr))
    # ---- [2] main count table + all B-XAIC tasks: original rule (exploratory_null_audit.sig_counts: Holm over 3, mean direction)
    P("\n[2] matched-budget counts, top-k GEA vs RandNode (original rule: Holm over the 3 explainers; above/below of 5 seeds)")
    P(f"  {'dataset':<24}{'explainer':<6}{'stored':>10}{'random ties':>14}")
    rep["main_counts"] = {}
    na.SEEDS = SEEDS
    for key in AUDIT_KEYS:
        per_s = {s: (dict(np.load(f"{AUD}/{key}_s{s}.npz")),) for s in SEEDS}
        per_b = {}
        for s in SEEDS:
            A = dict(per_s[s][0]); g = G[f"audit_{key}_s{s}"]
            for e in EXPL:
                A[f"{e}_topk"] = g[f"{e}_b"]
            per_b[s] = (A,)
        fam = [(f"{e}_topk", "rn_k") for e in EXPL]
        for e in EXPL:
            a1, b1, _, _ = na.sig_counts(per_s, f"{e}_topk", "rn_k", fam); a2, b2, _, _ = na.sig_counts(per_b, f"{e}_topk", "rn_k", fam)
            P(f"  {key:<24}{SH[e]:<6}{f'{a1}/{b1}':>10}{f'{a2}/{b2}':>14}{mark((a1, b1), (a2, b2))}")
            rep["main_counts"][f"{key}|{e}"] = {"stored": [a1, b1], "random_ties": [a2, b2]}
        if key != "mutagenicity_graphxai":
            sg = []
            for s in SEEDS:
                g = G[f"audit_{key}_s{s}"]; p, d = rtest(g["subgraphx_b"], g["rand_conn_k"])
                sg.append("+" if d > 0 and p < .05 else "-" if d < 0 and p < .05 else ".")
            P(f"  {key:<24}{'SX vs RandConn at the budget (top-k), random ties, one test per seed: ' + str(sg.count('+')) + '/' + str(sg.count('-'))}  (no stored top-k version)")
            rep["main_counts"][f"{key}|SX_topk_vs_conn_k"] = {"random_ties": [sg.count("+"), sg.count("-")]}
    P("  (SX mean-threshold vs size-matched RandConn -- the paper's connectivity check -- has no ties and is unchanged)")
    # ---- [3] architecture table + [4] 5-fold CV: registered 4-test family (rank direction, Holm within unit)
    for title, items in (("[3] architecture table, Mutagenicity (registered 4-test family; above/below of 5 seeds)",
                          [(a, [(f"arch_{a}_s{s}", f"{d}/arrays_s{s}.npz") for s in SEEDS]) for a, d in ARCH.items()]),
                         ("[4] Mutagenicity 5-fold CV (same family; above/below of 5 folds)",
                          [("CV", [(f"cv_f{f}", f"runs/mutagenicity_cv/arrays_s{f}.npz") for f in SEEDS])])):
        P(f"\n{title}")
        P(f"  {'model':<8}{'comparison':<18}{'stored':>10}{'random ties':>14}   per unit stored -> random ties")
        for lab_m, us in items:
            Ts, Tb = [], []
            for tag, arr in us:
                A = dict(np.load(arr)); Ab = dict(A); g = G[tag]
                for e in EXPL:
                    Ab[f"{e}_topk_set"] = g[f"{e}_b"]
                Ts.append(family_tests(A, FAMILY)); Tb.append(family_tests(Ab, FAMILY))
            for lab, _, _ in FAMILY:
                s1 = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in Ts]
                s2 = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in Tb]
                c1, c2 = f"{s1.count('+')}/{s1.count('-')}", f"{s2.count('+')}/{s2.count('-')}"
                P(f"  {lab_m:<8}{lab:<18}{c1:>10}{c2:>14}   {''.join(s1)} -> {''.join(s2)}{mark(s1, s2)}")
                rep.setdefault("family_counts", {})[f"{lab_m}|{lab}"] = {"stored": s1, "random_ties": s2}
    # ---- [5] fidelity: tie fractions, changed cells, GEA-vs-fidelity comparison
    fids = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{OUT}/fid/*.npz"))
    P(f"\n[5] fidelity under random ties: {len(fids)} units")
    tieE = {e: np.mean([dict(np.load(f'{OUT}/fid/{u}.npz'))[f'{e}_tie_E'].mean() for u in fids]) for e in TIED}
    tieR = {e: np.mean([dict(np.load(f'{OUT}/fid/{u}.npz'))[f'{e}_tie_GmE'].mean() for u in fids]) for e in TIED}
    P("  mean fraction of explained molecules with a tie at the E boundary / the removed-set boundary: "
      + "   ".join(f"{SH[e]} {tieE[e]:.2f} / {tieR[e]:.2f}" for e in TIED))
    changed, total = [], 0
    for u in fids:
        B, Bb, _ = merged_B(u); s1, s2 = b_signs(B)["node"], b_signs(Bb)["node"]
        for k in s1:
            if k[0] in TIED:
                total += 1
                if s1[k] != s2[k]:
                    changed.append((u, SH[k[0]], k[1], k[2], s1[k], s2[k]))
    P(f"  Part B random-corrected (vs random-node) direction cells for PG and SX: {total}; changed under random ties: {len(changed)}")
    for c in changed[:80]:
        P(f"    {c[0]:<28}{c[1]:<4}{c[2]} {c[3]:<10} {c[4]} -> {c[5]}")
    rep["partB_changed"] = changed
    # GEA-vs-fidelity comparison (reliance_fidelity_report.gea_compare rule), stored vs random ties
    P("\n[6] GEA vs random-node-corrected logit Fid+ under R3 (original comparison rule): cells whose direction differs")
    src = {f"mutag_graphxai_s{s}": f"audit_mutag_graphxai_s{s}" for s in SEEDS}
    src.update({f"bxaic_{t}_s{s}": f"audit_bxaic_{t}_s{s}" for t in rf.BXAIC for s in SEEDS})
    src.update({f"mutagenicity_{a.lower().replace('-', '')}_s{s}": f"arch_{a}_s{s}" for a in ARCH for s in SEEDS})
    out = {}
    for which in ("stored", "random_ties"):
        diffs = []
        for u, tag in src.items():
            if u not in fids:
                continue
            if tag.startswith("audit"):
                z = dict(np.load(f"{AUD}/{tag[6:]}.npz")); real = {e: z[f"{e}_topk"] for e in EXPL}; null = z["rn_k"]
            else:
                a = tag.split("_")[1]; s = int(tag.rsplit("_s", 1)[1]); z = dict(np.load(f"{ARCH[a]}/arrays_s{s}.npz"))
                real = {e: z[f"{e}_topk_set"] for e in EXPL}; null = z["rand_node_topk_set"]
            if which == "random_ties":
                real = {e: G[tag][f"{e}_b"] for e in EXPL}
            Fm = holm_family({e: rtest(real[e], null) for e in EXPL})
            B, Bb, _ = merged_B(u); sb = b_signs(B if which == "stored" else Bb)["node"]
            for e in EXPL:
                if Fm[e][0] != sb[(e, "R3", "fid_plus")]:
                    diffs.append((u, SH[e], Fm[e][0], sb[(e, "R3", "fid_plus")]))
        out[which] = diffs
        P(f"  {which}: {len(diffs)} cells differ")
    gone = [d for d in out["stored"] if d not in out["random_ties"]]; new = [d for d in out["random_ties"] if d not in out["stored"]]
    P(f"  disagreements that disappear under random ties: {len(gone)}; new ones: {len(new)}")
    for d in gone:
        P(f"    gone: {d}")
    for d in new:
        P(f"    new:  {d}")
    rep["gea_vs_fid"] = out
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    P(f"\nsaved -> {OUT}/report.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gea", "fid", "report", "list"])
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n in fid_units():
            print(n)
        return 0
    if a.cmd == "gea":
        run_gea(); return 0
    if a.cmd == "report":
        return report()
    os.makedirs(f"{OUT}/fid", exist_ok=True); U = fid_units(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n in a.units:
        dst = f"{OUT}/fid/{n}.npz"
        if not os.path.exists(dst):
            A = fid_unit(n, U[n], device); np.savez(dst, **A)
            print(f"{n}: ties PG {A['pgexplainer_tie_E'].mean():.2f}/{A['pgexplainer_tie_GmE'].mean():.2f} SX {A['subgraphx_tie_E'].mean():.2f}/{A['subgraphx_tie_GmE'].mean():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
