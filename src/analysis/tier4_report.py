"""
RQ1 Tier 4 reports (runs/rq1_tier4/PLAN.md): writes runs/rq1_tier4/<item>.txt/.json.

    python -m src.analysis.tier4_report L1|L3|L4|L5|L7|L2|L6|M1|M2
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import revision_analysis as RA
from . import tier2_common as C
from .tier2_report import SH, fmt_ci

T4 = "runs/rq1_tier4"
LOG = []
MUTAG = [f"mutag_graphxai_s{s}" for s in range(10)]
RULES = ["R1", "R2", "R3"]
SH = {**SH, "pgexplainer": "PG (logit)", "sx_fixed": "SX fixed", "deg_high": "high deg", "deg_low": "low deg"}


def P(s=""):
    print(s); LOG.append(s)


def save(name, rep):
    open(f"{T4}/{name}.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{T4}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load(path):
    return dict(np.load(path, allow_pickle=True))


def ok_pairs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); k = ~(np.isnan(a) | np.isnan(b))
    return a[k], b[k]


def boot(real, null):
    """Hierarchical bootstrap after dropping NaN pairs; seeds with no molecule left are dropped (number of seeds returned)."""
    pr = [ok_pairs(a, b) for a, b in zip(real, null)]; pr = [(a, b) for a, b in pr if len(a)]
    if not pr:
        return (np.nan, np.nan, np.nan), 0
    return C.boot([a for a, _ in pr], [b for _, b in pr]), len(pr)


def cat(ci):
    return "n/a" if np.isnan(ci[1]) else C.ci_cat(ci[1], ci[2])


def holm_counts(per_seed_raw):
    """per_seed_raw: list over seeds of {key: (p, direction)} -> key -> list of signs (+/-/.) after Holm within each seed."""
    sig = {}
    for raw in per_seed_raw:
        for k, s in RA.fam_signs(raw).items():
            sig.setdefault(k, []).append(s)
    return sig


def flip(sign, metric):
    """rank-test sign of (x - null) -> better (+) / worse (-) / n.s., where lower is better for Fid- and GEF."""
    return RA.beat_sign(sign, metric)


# ------------------------------------------------------------------ L1
def L1():
    from .tier4_L1 import METHODS
    rep = {}; G = {u: load(f"{T4}/L1/gea/{u}.npz") for u in MUTAG}; Fd = {u: load(f"{T4}/L1/fid/{u}.npz") for u in MUTAG}
    O = {u: load(f"runs/rq1_revision/pass/G/{u}.npz") for u in MUTAG}
    te = {u: {int(m) for m, t in zip(G[u]["mol_idx"], G[u]["is_test"]) if t} for u in MUTAG}
    P("L1 -- MUTAG-GT, TEST MOLECULES ONLY (each seed's own 29 test molecules) vs ALL 188 EXPLAINED MOLECULES; 10 seeds; no new explanations")
    P(f"  test molecules with ground truth per seed: {[len(te[u]) for u in MUTAG]}")
    sel = lambda u, Z, part: np.ones(len(Z["mol_idx"]), bool) if part == "all" else np.array([int(m) in te[u] for m in Z["mol_idx"]])
    # [1] GEA
    P("\n  [1] GEA minus null, 95% CI and counts better/worse/n.s.; Holm per seed = 9 methods x 2 nulls = 18 (computed within each molecule set)")
    P(f"  {'method':<12}{'set':<6}{'minus random-node':>30}{'counts':>9}{'minus random-conn.':>30}{'counts':>9}{'mean GEA':>10}")
    rep["gea"] = {}
    for part in ("test", "all"):
        sig = holm_counts([{(m, r): C.wil(G[u][m][sel(u, G[u], part)], G[u][r][sel(u, G[u], part)]) for m in METHODS for r in ("rn", "rc")} for u in MUTAG])
        for m in METHODS:
            a = [G[u][m][sel(u, G[u], part)] for u in MUTAG]
            n, _ = boot(a, [G[u]["rn"][sel(u, G[u], part)] for u in MUTAG]); c, _ = boot(a, [G[u]["rc"][sel(u, G[u], part)] for u in MUTAG])
            rep["gea"][f"{m}|{part}"] = {"rn": list(n), "rc": list(c), "cat_rn": cat(n), "signs_rn": sig[(m, "rn")], "signs_rc": sig[(m, "rc")], "mean": float(np.mean([x.mean() for x in a]))}
    for m in METHODS:
        for part in ("test", "all"):
            r = rep["gea"][f"{m}|{part}"]
            P(f"  {SH[m]:<12}{part:<6}{fmt_ci(*r['rn']):>30}{C.counts(r['signs_rn']):>9}{fmt_ci(*r['rc']):>30}{C.counts(r['signs_rc']):>9}{r['mean']:>10.3f}")
    ch = [m for m in METHODS if rep["gea"][f"{m}|test"]["cat_rn"] != rep["gea"][f"{m}|all"]["cat_rn"]]
    P(f"  CI category vs random-node changes (test vs all): {', '.join(SH[m] for m in ch) if ch else 'none'}"); rep["gea_category_changes"] = ch
    # [2] oracle
    P("\n  [2] reliance / sufficiency (T1 oracle, Part-G arrays): ground-truth mask minus size-matched null; counts GT better/worse/n.s. (T1 families: prob 18, logit 12 per seed) and bootstrap CI")
    rep["oracle"] = {}
    for space, mets in (("prob", ["fidp", "fidm", "gef"]), ("logit", ["fidp", "fidm"])):
        ko = lambda mt: f"{mt}_{space}" if mt != "gef" else "gef"
        P(f"  [{space}] {'metric':<6}{'null':<6}" + "".join(f"{r + ' ' + part:>36}" for r in RULES for part in ("test", "all")))
        for part in ("test", "all"):
            sig = holm_counts([{(mt, r, nl): RA.wp(O[u][f"gt_{r}_{ko(mt)}"][sel(u, O[u], part)], O[u][f"{nl}_{r}_{ko(mt)}"][sel(u, O[u], part)])
                                for mt in mets for r in RULES for nl in ("node", "conn")} for u in MUTAG])
            for mt in mets:
                for r in RULES:
                    for nl in ("node", "conn"):
                        d, ns = boot([O[u][f"gt_{r}_{ko(mt)}"][sel(u, O[u], part)] for u in MUTAG], [O[u][f"{nl}_{r}_{ko(mt)}"][sel(u, O[u], part)] for u in MUTAG])
                        rep["oracle"][f"{space}|{mt}|{r}|{nl}|{part}"] = {"ci": list(d), "signs": [flip(s, mt) for s in sig[(mt, r, nl)]], "seeds": ns}
        for mt in mets:
            for nl in ("node", "conn"):
                cells = ""
                for r in RULES:
                    for part in ("test", "all"):
                        x = rep["oracle"][f"{space}|{mt}|{r}|{nl}|{part}"]; cells += f"{C.counts(x['signs']):>9} {fmt_ci(*x['ci']):>26}"
                P(f"  [{space}] {mt:<6}{nl:<6}{cells}")
    # [3] Fid vs random rankings
    P("\n  [3] Fid+ and Fid- under R3 (probability and logit) minus 20 random rankings; counts better/worse/n.s.; Holm per (seed, space) = 9 methods x 2 metrics = 18")
    rep["fid"] = {}
    for space in ("prob", "logit"):
        for part in ("test", "all"):
            for rule in RULES:
                sig = holm_counts([{(m, mt): C.wil(Fd[u][f"{m}_{rule}_{mt}_{space}"][sel(u, Fd[u], part)], Fd[u][f"randnode_{rule}_{mt}_{space}"][sel(u, Fd[u], part)])
                                    for m in METHODS for mt in ("fidp", "fidm")} for u in MUTAG])
                for m in METHODS:
                    for mt in ("fidp", "fidm"):
                        d, _ = boot([Fd[u][f"{m}_{rule}_{mt}_{space}"][sel(u, Fd[u], part)] for u in MUTAG], [Fd[u][f"randnode_{rule}_{mt}_{space}"][sel(u, Fd[u], part)] for u in MUTAG])
                        rep["fid"][f"{space}|{part}|{rule}|{m}|{mt}"] = {"ci": list(d), "cat": cat(d), "signs": [flip(s, mt) for s in sig[(m, mt)]]}
        P(f"  [{space}] {'method':<12}" + "".join(f"{lab + ' ' + part:>40}" for lab in ("Fid+", "Fid-") for part in ("test", "all")))
        for m in METHODS:
            cells = "".join(f"{C.counts(rep['fid'][f'{space}|{part}|R3|{m}|{mt}']['signs']):>10} {fmt_ci(*rep['fid'][f'{space}|{part}|R3|{m}|{mt}']['ci']):>29}"
                            for mt in ("fidp", "fidm") for part in ("test", "all"))
            P(f"  [{space}] {SH[m]:<12}{cells}")
    chf = [k for k in rep["fid"] if k.split("|")[1] == "test" and rep["fid"][k]["cat"] != rep["fid"][k.replace("|test|", "|all|")]["cat"]]
    P(f"  Fid CI categories that change between test-only and all-188 (all rules, both spaces): {len(chf)} of {sum(1 for k in rep['fid'] if '|test|' in k)}")
    for k in chf:
        P(f"    {k}: test {rep['fid'][k]['cat']} {fmt_ci(*rep['fid'][k]['ci'])}; all {rep['fid'][k.replace('|test|', '|all|')]['cat']} {fmt_ci(*rep['fid'][k.replace('|test|', '|all|')]['ci'])}")
    rep["fid_category_changes"] = chf
    # [4] fresh all-188 vs the currently reported pass-E computation
    P("\n  [4] fresh all-188 Fid vs the currently reported pass-E arrays (stored PG, legacy SubgraphX): mean Fid+ R3 (prob) per method, mean over seeds")
    E = {u: load(f"runs/rq1_revision/pass/E/{u}.npz") for u in MUTAG}
    for m, em in (("gnnexplainer", "gnnexplainer"), ("pgexplainer", "pgexplainer"), ("sx_fixed", "subgraphx")):
        a = np.mean([Fd[u][f"{m}_R3_fidp_prob"].mean() for u in MUTAG]); b = np.mean([E[u][f"{em}_R3_fidp_prob"].mean() for u in MUTAG])
        P(f"    {SH[m]:<12} fresh {a:+.4f}   pass E ({em}) {b:+.4f}   difference {a - b:+.4f}")
        rep.setdefault("vs_passE", {})[m] = {"fresh": float(a), "passE": float(b)}
    # P-L1
    sx = rep["gea"]["sx_fixed|test"]["rn"]; he = rep["gea"]["hetero|test"]["rn"]; oc = rep["oracle"]["logit|fidp|R1|conn|test"]
    a_, b_, nw = sx[2] < 0, he[1] > 0, oc["signs"].count("-"); c_ = nw >= 8
    P("\n  P-L1 (test molecules only, 10 seeds):")
    P(f"    (a) SubgraphX fixed minus random-node {fmt_ci(*sx)} -> {'CI entirely below 0' if a_ else 'NOT entirely below 0'}")
    P(f"    (b) hetero minus random-node {fmt_ci(*he)} -> {'CI entirely above 0' if b_ else 'NOT entirely above 0'}")
    P(f"    (c) R1 logit Fid+, GT minus random-connected {fmt_ci(*oc['ci'])}; GT significantly worse in {nw}/10 seeds (counts {C.counts(oc['signs'])}) -> {'>= 8' if c_ else '< 8'}")
    held = a_ and b_ and c_; rep["P-L1"] = {"a": a_, "b": b_, "c": c_, "c_worse_seeds": nw, "held": held}
    P(f"  P-L1 {'HELD' if held else 'FAILED'}")
    save("L1", rep)


# ------------------------------------------------------------------ L3
def L3():
    rep = {}; MODELS = {"D-MPNN": "dmpnn", "GIN": "gin", "GINE": "gine", "GCN": "gcn"}
    P("L3 -- PGEXPLAINER SATURATION AND THE MODEL ORDERING (Mutagenicity, 5 seeds, GT-bearing explained molecules)")
    P("  main split: the T4 re-run's float32 edge mask constant (max == min) or not; GEA ranked by the same run's float64 logits.")
    P("  Holm per (model, seed) = 2 parts. Seeds with no molecule in a part are dropped from that part (number of seeds shown).")
    for split in ("const_rerun", "const_stored"):
        P(f"\n  [{'main' if split == 'const_rerun' else 'secondary'}] split on {'the T4 re-run' if split == 'const_rerun' else 'the stored'} float32 mask")
        P(f"  {'model':<8}{'part':<14}{'molecules (per seed)':<30}{'PG logit minus random-node':>32}{'counts':>9}{'seeds':>6}{'PG final rule minus random-node':>34}")
        for lab, a in MODELS.items():
            us = [f"mutagenicity_{a}_s{s}" for s in range(5)]; Z = {u: load(f"{T4}/L3/{u}.npz") for u in us}
            parts = {"not constant": lambda z: ~z[split].astype(bool), "constant": lambda z: z[split].astype(bool)}
            sig = holm_counts([{p: C.wil(Z[u]["pg64"][f(Z[u])], Z[u]["rn"][f(Z[u])]) for p, f in parts.items()} for u in us])
            for p, f in parts.items():
                d, ns = boot([Z[u]["pg64"][f(Z[u])] for u in us], [Z[u]["rn"][f(Z[u])] for u in us]); df, _ = boot([Z[u]["pg_final"][f(Z[u])] for u in us], [Z[u]["rn"][f(Z[u])] for u in us])
                sizes = [int(f(Z[u]).sum()) for u in us]
                rep[f"{split}|{lab}|{p}"] = {"sizes": sizes, "pg64": list(d), "pg_final": list(df), "signs": sig[p], "seeds": ns}
                P(f"  {lab:<8}{p:<14}{str(sum(sizes)) + ' ' + str(sizes):<30}{fmt_ci(*d):>32}{C.counts(sig[p]):>9}{ns:>6}{fmt_ci(*df):>34}")
    pts = {lab: rep[f"const_rerun|{lab}|not constant"]["pg64"][0] for lab in MODELS}; rng = max(pts.values()) - min(pts.values())
    P("\n  P-L3: PG (logit) minus random-node on non-constant molecules (re-run split): " + ", ".join(f"{k} {v:+.3f}" for k, v in pts.items()))
    P(f"    range (max - min) {rng:.3f} -> {'< 0.10' if rng < 0.10 else '>= 0.10'}")
    rep["P-L3"] = {"points": pts, "range": rng, "held": rng < 0.10}
    P(f"  P-L3 {'HELD' if rng < 0.10 else 'FAILED'}")
    save("L3", rep)


# ------------------------------------------------------------------ L4
def L4():
    rep = {}; Z = {u: load(f"{T4}/L4/{u}.npz") for u in MUTAG}
    P("L4 -- WHAT THE MUTAG-GT MODEL USES: the aromatic scaffold (all atoms on an aromatic bond) vs random sets of the same size; 10 seeds")
    P(f"  molecules per seed (all / test): {[(len(Z[u]['mol_idx']), int(Z[u]['is_test'].sum())) for u in MUTAG]}; skipped (no aromatic atom or all atoms aromatic): {[len(Z[u]['skipped']) for u in MUTAG]}")
    P(f"  mean scaffold size {np.mean([Z[u]['scaf_size'].mean() for u in MUTAG]):.1f} atoms of {np.mean([Z[u]['n'].mean() for u in MUTAG]):.1f}")
    P("  Fid+ = scaffold hidden (necessity: higher = more needed); Fid- = only the scaffold kept (sufficiency: lower = more sufficient); GEF lower = better.")
    P("  counts = scaffold better/worse/n.s.; Holm per seed over 3 rules x {Fid+, Fid-} = 6, per space and per null; GEF in its own family of 3.")
    sel = lambda u, part: np.ones(len(Z[u]["mol_idx"]), bool) if part == "all" else Z[u]["is_test"].astype(bool)
    rep["scaffold"] = {}
    for space in ("logit", "prob"):
        fams = [("fidp", "fidm")] + ([("gef",)] if space == "prob" else [])
        ko = lambda mt: "gef" if mt == "gef" else f"{mt}_{space}"
        for nl in ("conn", "node"):
            for part in ("all", "test"):
                for fam in fams:
                    sig = holm_counts([{(mt, r): RA.wp(Z[u][f"scaf_{r}_{ko(mt)}"][sel(u, part)], Z[u][f"{nl}_{r}_{ko(mt)}"][sel(u, part)]) for mt in fam for r in RULES} for u in MUTAG])
                    for mt in fam:
                        for r in RULES:
                            d, _ = boot([Z[u][f"scaf_{r}_{ko(mt)}"][sel(u, part)] for u in MUTAG], [Z[u][f"{nl}_{r}_{ko(mt)}"][sel(u, part)] for u in MUTAG])
                            rep["scaffold"][f"{space}|{nl}|{part}|{mt}|{r}"] = {"ci": list(d), "signs": [flip(s, mt) for s in sig[(mt, r)]]}
        P(f"\n  [{space}] scaffold minus null")
        P(f"  {'null':<6}{'set':<6}{'metric':<7}" + "".join(f"{r:>37}" for r in RULES))
        for nl in ("conn", "node"):
            for part in ("all", "test"):
                for mt in (("fidp", "fidm", "gef") if space == "prob" else ("fidp", "fidm")):
                    cells = "".join(f"{C.counts(rep['scaffold'][f'{space}|{nl}|{part}|{mt}|{r}']['signs']):>9} {fmt_ci(*rep['scaffold'][f'{space}|{nl}|{part}|{mt}|{r}']['ci']):>27}" for r in RULES)
                    P(f"  {nl:<6}{part:<6}{mt:<7}{cells}")
    P("\n  [overlap] share of chosen atoms that are aromatic, minus random-connected sets of the same size; Holm per seed = 3")
    P(f"  {'method':<14}{'set':<6}{'mean share':>11}{'null share':>11}{'difference [95% CI]':>30}{'counts':>9}")
    rep["overlap"] = {}
    OV = {"sx_fixed": ("ov_sx", "ov_sx_conn"), "gnnexplainer": ("ov_gnn", "ov_topk_conn"), "ig": ("ov_ig", "ov_topk_conn")}
    for part in ("all", "test"):
        sig = holm_counts([{m: RA.wp(Z[u][a][sel(u, part)], Z[u][b][sel(u, part)]) for m, (a, b) in OV.items()} for u in MUTAG])
        for m, (a, b) in OV.items():
            d, _ = boot([Z[u][a][sel(u, part)] for u in MUTAG], [Z[u][b][sel(u, part)] for u in MUTAG])
            ma = np.mean([np.nanmean(Z[u][a][sel(u, part)]) for u in MUTAG]); mb = np.mean([np.nanmean(Z[u][b][sel(u, part)]) for u in MUTAG])
            rep["overlap"][f"{m}|{part}"] = {"ci": list(d), "signs": sig[m], "share": float(ma), "null": float(mb)}
            P(f"  {SH[m]:<14}{part:<6}{ma:>11.3f}{mb:>11.3f}{fmt_ci(*d):>30}{C.counts(sig[m]):>9}")
    P(f"  SubgraphX coalition size: mean {np.mean([Z[u]['sx_size'].mean() for u in MUTAG]):.1f}; molecules with an empty coalition: {sum(int((Z[u]['sx_size'] == 0).sum()) for u in MUTAG)}")
    a = rep["scaffold"]["logit|conn|all|fidp|R1"]; b = rep["overlap"]["sx_fixed|all"]
    na, nb = a["signs"].count("+"), b["signs"].count("+")
    P("\n  P-L4 (all explained molecules with an aromatic scaffold):")
    P(f"    (a) R1 necessity, logit Fid+ scaffold minus random-connected {fmt_ci(*a['ci'])}; scaffold significantly higher in {na}/10 seeds -> {'>= 8' if na >= 8 else '< 8'}")
    P(f"    (b) SubgraphX aromatic share minus random-connected {fmt_ci(*b['ci'])}; significantly higher in {nb}/10 seeds -> {'>= 8' if nb >= 8 else '< 8'}")
    held = na >= 8 and nb >= 8; rep["P-L4"] = {"a_seeds": na, "b_seeds": nb, "held": held}
    P(f"  P-L4 {'HELD' if held else 'FAILED'}")
    save("L4", rep)


# ------------------------------------------------------------------ L5
def _pg_cache(path):
    r = C.torch_load(path)["results"]["pgexplainer"]
    return [np.asarray(x["edge_importance"], dtype=np.float32) for x in r], [np.asarray(x["node_importance"], dtype=np.float64) for x in r]


def _spear(a, b):
    from scipy.stats import spearmanr
    v = [np.nan if np.std(x) == 0 or np.std(y) == 0 else spearmanr(x, y).statistic for x, y in zip(a, b)]
    return float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan


def L5():
    from .tier4_L5 import UNITS
    rep = {}; dev = json.load(open(f"{T4}/L5/device.json")); rep["device"] = dev
    P("L5 -- PGEXPLAINER REPRODUCIBILITY (Mutagenicity D-MPNN and GINE, seeds 0-4; 3 repeats; original arguments and explainer seed)")
    P(f"  device rule (PLAN): GPU smoke test -> {dev.get('error') or ('identical' if dev.get('identical') else 'not identical')} "
      f"(max |edge mask difference| {dev.get('max_abs_diff', float('nan'))}); all repeats ran on: {dev['device'].upper()}")
    P("  settings: use_deterministic_algorithms(True), cudnn deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8, fixed seeds" + (", 1 CPU thread" if dev["device"] == "cpu" else ""))
    P("\n  [1] mask equality across the 3 repeats (all explained molecules)")
    P(f"  {'unit':<26}{'molecules':>10}{'identical masks':>17}{'mols differing':>16}{'max |edge diff|':>17}{'max |logit diff|':>18}")
    allsame = True; rep["units"] = {}
    for u in UNITS:
        E_ = [_pg_cache(f"{T4}/L5/rep{r}/{u}_cache.pt")[0] for r in range(3)]
        L_ = [list(np.load(f"{T4}/L5/rep{r}/{u}_pg_logits.npz", allow_pickle=True)["logits"]) for r in range(3)]
        diff_mols = sum(any(not np.array_equal(E_[0][i], E_[r][i]) for r in (1, 2)) for i in range(len(E_[0])))
        me = max(float(np.abs(E_[0][i] - E_[r][i]).max()) if len(E_[0][i]) else 0.0 for r in (1, 2) for i in range(len(E_[0])))
        ml = max(float(np.abs(np.asarray(L_[0][i], float) - np.asarray(L_[r][i], float)).max()) if len(L_[0][i]) else 0.0 for r in (1, 2) for i in range(len(L_[0])))
        same = diff_mols == 0; allsame &= same
        rep["units"][u] = {"n": len(E_[0]), "identical": same, "mols_differing": diff_mols, "max_edge_diff": me, "max_logit_diff": ml}
        P(f"  {u:<26}{len(E_[0]):>10}{'yes' if same else 'NO':>17}{diff_mols:>16}{me:>17.2e}{ml:>18.2e}")
    P("\n  [2] PGExplainer GEA minus random-node per run (logit-ranked unless stated), 95% CI, counts (Holm per seed = 1)")
    runs = {f"repeat {r}": (lambda u, r=r: C.npz_scores(f"{T4}/L5/rep{r}/{u}_pg_logits.npz", "node_score64")) for r in range(3)}
    runs.update({"earlier: stored (final rule)": None, "earlier: T4 re-run": (lambda u: C.npz_scores(f"runs/rq1_revision/logits/{u}.npz", "node_score64")),
                 "earlier: Tier-3 E 'all' re-run": (lambda u: C.npz_scores(f"runs/rq1_tier3/E/gp/all/{u}_pg_logits.npz", "node_score64"))})
    rep["gea"] = {}
    for arch in ("dmpnn", "gine"):
        us = [u for u in UNITS if f"_{arch}_" in u]; H = {u: load(f"{C.T2}/H/gea/{u}.npz") for u in us}
        for name, f in runs.items():
            if f is None:
                a = [H[u]["pgexplainer"] for u in us]
            else:
                a = []
                for u in us:
                    sc = f(u); a.append(C.gea_array(u, "pgexplainer", lambda x, sc=sc: sc[x], [int(m) for m in H[u]["mol_idx"]], C.dataset_of(u)))
            d, _ = boot(a, [H[u]["rn"] for u in us]); sg = [RA.fam_signs({0: RA.wp(x, H[u]["rn"])})[0] for x, u in zip(a, us)]
            rep["gea"][f"{arch}|{name}"] = {"ci": list(d), "cat": cat(d), "signs": sg}
            P(f"  {'D-MPNN' if arch == 'dmpnn' else 'GINE':<8}{name:<34}{fmt_ci(*d):>28}{C.counts(sg):>9}   {cat(d)}")
    P("\n  [3] spread of the earlier non-deterministic runs: pairwise max |edge mask difference| (float32) and mean Spearman of node scores")
    pairs = [("stored", "T4 re-run"), ("stored", "E all"), ("T4 re-run", "E all")]; rep["earlier"] = {}
    P(f"  {'unit':<26}" + "".join(f"{a + ' vs ' + b:>34}" for a, b in pairs))
    for u in UNITS:
        src = {"stored": _pg_cache(C.orig_spec(u)["cache"]), "T4 re-run": _pg_cache(f"runs/rq1_revision/logits/rerun/{u}_cache.pt"), "E all": _pg_cache(f"runs/rq1_tier3/E/gp/all/{u}_cache.pt")}
        cells = ""
        for a, b in pairs:
            me = max(float(np.abs(x - y).max()) if len(x) else 0.0 for x, y in zip(src[a][0], src[b][0])); sp = _spear(src[a][1], src[b][1])
            rep["earlier"][f"{u}|{a}|{b}"] = {"max_edge_diff": me, "spearman": sp}; cells += f"{me:>20.2e}  rho {sp:>6.3f}  "
        P(f"  {u:<26}{cells}")
    rep["P-L5"] = {"held": allsame}
    P(f"\n  P-L5: three deterministic repeats give bitwise-identical float32 edge masks on all 10 units -> {'yes' if allsame else 'NO'}")
    P(f"  P-L5 {'HELD' if allsame else 'FAILED'}")
    save("L5", rep)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
