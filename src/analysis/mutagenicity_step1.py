"""
STEP 1 (CPU only) for the GraphXAI Mutagenicity experiment: dataset, encoding
check, matcher cross-check against the vendored GraphXAI code, ground-truth
set statistics. Writes runs/mutagenicity/step1_report.json; touches nothing
else.

    python -m src.analysis.mutagenicity_step1
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import Counter

import networkx as nx
import numpy as np
import torch

from ..data.graphxai_mutagenicity import (
    ATOM_ORDER, MOTIF_TYPES, find_motifs, load_mutagenicity_graphxai, load_raw,
)
from ..data.standard import DATA_ROOT
from ..metrics.gea_set import ENUM_CAP, _dinkelbach, _enum, gea_max_over_set, motif_matrix

OUTDIR = "runs/mutagenicity"
GX = "third_party/GraphXAI/graphxai"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vendored_matches(raw):
    """Run GraphXAI's OWN matcher code (loaded by file path) on every graph."""
    sub = _load_by_path("gx_sub", f"{GX}/datasets/utils/substruct_chem_match.py")
    conv = _load_by_path("gx_conv", f"{GX}/utils/nx_conversion.py")
    from torch_geometric.data import Data
    out = []
    for r in raw:
        g = Data(x=r["x"], edge_index=r["edge_index"])
        G = conv.to_networkx_conv(g, node_attrs=["x"], to_undirected=True)
        nh2 = sub.match_substruct_mutagenicity(G, sub.MUTAG_NH2, nh2_no2=0)
        no2 = sub.match_substruct_mutagenicity(G, sub.MUTAG_NO2, nh2_no2=1)
        hal = sub.match_aliphatic_halide(G)
        nso = sub.match_nitroso(G)
        azo = sub.match_azo_type(G)
        d = {"NH2": [sorted(m.tolist()) for m in nh2], "NO2": [sorted(m.tolist()) for m in no2],
             "nitroso": [sorted(m.tolist()) for m in nso], "azo": [sorted(m.tolist()) for m in azo],
             "halide": [[int(m)] for m in hal]}
        out.append(d)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rep = {"networkx": nx.__version__, "torch": torch.__version__}
    t0 = time.time()
    raw, ds = load_raw()
    rep["load_seconds"] = round(time.time() - t0, 1)

    # ---------------- (a) dataset -----------------
    readme = None
    for fn in os.listdir(os.path.join(DATA_ROOT, "TUDataset", "Mutagenicity", "raw")):
        if "readme" in fn.lower() or "label" in fn.lower():
            p = os.path.join(DATA_ROOT, "TUDataset", "Mutagenicity", "raw", fn)
            if os.path.getsize(p) < 20000:
                readme = (fn, open(p, errors="replace").read())
    rep["tu_readme_file"] = readme[0] if readme else None
    print("=" * 100, "\n (a) DATASET\n", "=" * 100, sep="")
    if readme:
        print(f"--- raw/{readme[0]} ---\n{readme[1]}\n--- end ---")

    y = np.array([r["y"] for r in raw])
    has = np.array([len(r["motifs"]) > 0 for r in raw])
    ct = {f"y={a},match={int(b)}": int(((y == a) & (has == b)).sum()) for a in (0, 1) for b in (False, True)}
    rep["raw_graphs"] = len(raw)
    rep["raw_class_counts"] = {int(k): int(v) for k, v in Counter(y.tolist()).items()}
    rep["isolated_nodes_removed"] = int(sum(r["n_removed"] for r in raw))
    rep["graphs_with_isolated_nodes"] = int(sum(r["n_removed"] > 0 for r in raw))
    rep["label_x_match"] = ct
    print(f"raw graphs: {len(raw)}   raw class counts: {rep['raw_class_counts']}")
    print(f"isolated nodes removed: {rep['isolated_nodes_removed']} across {rep['graphs_with_isolated_nodes']} graphs")
    print(f"label x has_toxicophore crosstab (unfiltered): {ct}")
    print(f"  toxicophore prevalence: y=0 -> {has[y == 0].mean():.3f}   y=1 -> {has[y == 1].mean():.3f}")
    rep["toxicophore_prevalence_by_label"] = {"y0": float(has[y == 0].mean()), "y1": float(has[y == 1].mean())}
    kept_gx = int((has.astype(int) == y).sum())
    kept_inv = int((has.astype(int) == (1 - y)).sum())
    print(f"GraphXAI filter (has_toxicophore == y): keeps {kept_gx}")
    print(f"  [for reference only] opposite polarity (has_toxicophore == 1-y) would keep {kept_inv}")
    rep["kept_graphxai_filter"] = kept_gx
    rep["kept_if_polarity_inverted"] = kept_inv

    data_list, meta = load_mutagenicity_graphxai()
    motifs = meta.extras["motifs"]
    n_nodes = np.array([d.num_nodes for d in data_list])
    n_edges = np.array([d.edge_index.shape[1] // 2 for d in data_list])
    yk = np.array([int(d.y.item()) for d in data_list])
    rep.update(
        graphs_after_filter=len(data_list),
        class_balance={"y0_no_toxicophore": int((yk == 0).sum()), "y1_toxicophore": int((yk == 1).sum())},
        nodes_mean=float(n_nodes.mean()), nodes_median=float(np.median(n_nodes)), nodes_max=int(n_nodes.max()),
        nodes_min=int(n_nodes.min()), edges_mean=float(n_edges.mean()),
        node_feature_dim=int(meta.num_node_features), edge_feature_dim=int(meta.num_edge_features),
    )
    print(f"\nafter filter: {len(data_list)} graphs   class balance: {rep['class_balance']}")
    print(f"nodes: mean {n_nodes.mean():.1f}  median {np.median(n_nodes):.0f}  min {n_nodes.min()}  max {n_nodes.max()}"
          f"   undirected edges mean {n_edges.mean():.1f}")
    print(f"node feature dim {meta.num_node_features}   edge feature dim {meta.num_edge_features}")
    for thr in (60, 80, 100):
        print(f"  graphs with > {thr} nodes: {(n_nodes > thr).sum()} ({100 * (n_nodes > thr).mean():.2f}%)"
              f"   among GT-present: {(n_nodes[yk == 1] > thr).sum()} ({100 * (n_nodes[yk == 1] > thr).mean():.2f}%)")
    rep["exclusion_max_nodes_80"] = {"all": float((n_nodes > 80).mean()), "gt_present": float((n_nodes[yk == 1] > 80).mean()),
                                     "n_all": int((n_nodes > 80).sum()), "n_gt_present": int((n_nodes[yk == 1] > 80).sum())}

    # ---------------- (b) encoding + matchers -----------------
    print("\n" + "=" * 100, "\n (b) ATOM ENCODING + MATCHER CHECK\n", "=" * 100, sep="")
    chan_count = Counter(); chan_deg = {i: Counter() for i in range(14)}
    onehot_ok = True
    for r in raw:
        x = r["x"]
        onehot_ok &= bool((x.sum(1) == 1).all()) and bool(((x == 0) | (x == 1)).all())
        deg = torch.bincount(r["edge_index"][0], minlength=x.shape[0]).tolist()
        for a, dg in zip(x.argmax(1).tolist(), deg):
            chan_count[a] += 1; chan_deg[a][dg] += 1
    print(f"x is strictly one-hot, 14-dim: {onehot_ok}  (dim={raw[0]['x'].shape[1]})")
    print(f"{'ch':>3} {'GraphXAI atom':>13} {'count':>8}  degree histogram")
    enc = {}
    for i in range(14):
        print(f"{i:>3} {ATOM_ORDER[i]:>13} {chan_count[i]:>8}  {dict(sorted(chan_deg[i].items()))}")
        enc[ATOM_ORDER[i]] = {"count": chan_count[i], "max_degree": max(chan_deg[i]) if chan_deg[i] else None}
    rep["encoding_check"] = enc

    if os.path.isdir(GX):
        t0 = time.time()
        ven = vendored_matches(raw)
        n_exact = n_setequal = 0; dup = Counter(); mism = []
        for i, (r, v) in enumerate(zip(raw, ven)):
            mine = {t: sorted(m for tt, m in r["motifs"] if tt == t) for t in MOTIF_TYPES}
            exact = all(mine[t] == sorted(v[t]) for t in MOTIF_TYPES)
            seteq = all({tuple(m) for m in mine[t]} == {tuple(m) for m in v[t]} for t in MOTIF_TYPES)
            n_exact += exact; n_setequal += seteq
            for t in MOTIF_TYPES:
                dup[t] += len(v[t]) - len({tuple(m) for m in v[t]})
            if not seteq and len(mism) < 5:
                mism.append((i, mine, v))
        print(f"\ncross-check vs GraphXAI's own matcher code (ISMAGS, networkx {nx.__version__}), all {len(raw)} graphs "
              f"[{time.time() - t0:.0f}s]:")
        print(f"  identical match lists: {n_exact}/{len(raw)}   identical match SETS: {n_setequal}/{len(raw)}")
        print(f"  duplicate node-sets emitted by GraphXAI's matchers, per type: {dict(dup)}")
        for i, mine, v in mism:
            print(f"  MISMATCH graph {i}: mine={mine}  graphxai={v}")
        rep["crosscheck"] = {"identical_lists": n_exact, "identical_sets": n_setequal, "n": len(raw),
                             "graphxai_duplicates": dict(dup)}
    else:
        print("vendored GraphXAI not found -- cross-check skipped")

    comp = {t: Counter() for t in MOTIF_TYPES}
    for d, ms in zip(data_list, motifs):
        atom = d.x.argmax(1).tolist()
        for t, nodes in ms:
            comp[t][tuple(sorted(ATOM_ORDER[atom[n]] for n in nodes))] += 1
    print("\nmotif atom composition (kept graphs):")
    for t in MOTIF_TYPES:
        print(f"  {t:<8} {dict(comp[t])}")
    rep["motif_composition"] = {t: {"+".join(k): v for k, v in comp[t].items()} for t in MOTIF_TYPES}

    # halide: what is the halogen attached to? ring size of the neighbour, and how many terminal
    # halogens GraphXAI's rule DROPS (true aliphatic / small-ring / chain-attached)
    from ..data.graphxai_mutagenicity import HALOGENS, to_networkx_conv
    kept_h = dropped_h = 0
    for r in raw:
        G = to_networkx_conv(r["edge_index"]); atom = r["x"].argmax(1).tolist()
        matched = {m[0] for t, m in r["motifs"] if t == "halide"}
        for n in G.nodes():
            if G.degree[n] == 1 and atom[n] in HALOGENS:
                kept_h += n in matched; dropped_h += n not in matched
    print(f"\nhalide rule (GraphXAI, followed exactly): terminal halogens KEPT (neighbour on a cycle-basis ring of "
          f"length > 4) = {kept_h};  terminal halogens DROPPED (chain / small-ring attached) = {dropped_h}  [all 4337 graphs]")
    rep["halide_rule"] = {"kept_ring_attached": kept_h, "dropped_non_ring": dropped_h}

    # ---------------- (c) GT set -----------------
    print("\n" + "=" * 100, "\n (c) GROUND-TRUTH SET (max-over-set GEA)\n", "=" * 100, sep="")
    pos = [i for i in range(len(data_list)) if yk[i] == 1]
    L = np.array([len(motifs[i]) for i in pos])
    print(f"motif instances per GT-present graph: mean {L.mean():.2f}  median {np.median(L):.0f}  max {L.max()}")
    hist = Counter(L.tolist())
    print(f"  histogram L -> graphs: {dict(sorted(hist.items()))}")
    print(f"  GraphXAI set size 2^L-1: median {int(2 ** np.median(L) - 1)}  max 2^{L.max()}-1 = {2 ** int(L.max()) - 1}")
    over = [i for i in pos if len(motifs[i]) > ENUM_CAP]
    disjoint = [bool((motif_matrix(motifs[i], data_list[i].num_nodes).sum(0) <= 1).all()) for i in pos]
    over_disj = [i for i in over if (motif_matrix(motifs[i], data_list[i].num_nodes).sum(0) <= 1).all()]
    print(f"enumeration cap: L <= {ENUM_CAP} ({2 ** ENUM_CAP - 1} masks) enumerated exactly")
    print(f"  graphs over the cap: {len(over)}/{len(pos)}  -> exact via Dinkelbach (disjoint motifs): {len(over_disj)}"
          f"   lower-bound only (overlapping motifs): {len(over) - len(over_disj)}")
    print(f"  graphs whose motif instances overlap at all (any L): {len(pos) - sum(disjoint)}/{len(pos)}")
    rep["gt_set"] = {"L_mean": float(L.mean()), "L_max": int(L.max()), "L_hist": {int(k): v for k, v in sorted(hist.items())},
                     "enum_cap": ENUM_CAP, "over_cap": len(over), "over_cap_exact_dinkelbach": len(over_disj),
                     "over_cap_bound_only": len(over) - len(over_disj),
                     "graphs_with_overlapping_motifs": len(pos) - sum(disjoint)}

    # validate Dinkelbach == enumeration wherever both apply (random predicted masks)
    rng = np.random.default_rng(0); n_cmp = 0; max_diff = 0.0
    for i in pos:
        M = motif_matrix(motifs[i], data_list[i].num_nodes)
        if M.shape[0] > ENUM_CAP or not (M.sum(0) <= 1).all():
            continue
        for frac in (0.1, 0.25, 0.5):
            n = M.shape[1]; pred = np.zeros(n, bool); pred[rng.choice(n, max(1, int(frac * n)), replace=False)] = True
            max_diff = max(max_diff, abs(_enum(pred, M) - _dinkelbach(pred, M))); n_cmp += 1
    print(f"  Dinkelbach vs exhaustive enumeration on {n_cmp} (graph, random mask) pairs: max |diff| = {max_diff:.2e}")
    rep["gt_set"]["dinkelbach_vs_enum_max_diff"] = max_diff; rep["gt_set"]["dinkelbach_vs_enum_n"] = n_cmp

    # how much does max-over-set raise GEA over merged, for a random 25% mask?
    mo, mg = [], []
    for i in pos:
        n = data_list[i].num_nodes; pred = np.zeros(n, bool); pred[rng.choice(n, max(1, int(0.25 * n)), replace=False)] = True
        g = gea_max_over_set(pred, motifs[i]); mo.append(g.max_over_set); mg.append(g.merged)
    print(f"  sanity (uniform random top-25% mask, one draw/graph): max-over-set GEA {np.mean(mo):.3f}  merged GEA {np.mean(mg):.3f}")
    rep["random25_sanity"] = {"max_over_set": float(np.mean(mo)), "merged": float(np.mean(mg))}

    # ---------------- (d) toxicophore statistics -----------------
    print("\n" + "=" * 100, "\n (d) TOXICOPHORE STATISTICS (kept graphs)\n", "=" * 100, sep="")
    g_with = Counter(); inst = Counter(); ntypes = Counter()
    for i in pos:
        types = [t for t, _ in motifs[i]]
        for t in set(types): g_with[t] += 1
        for t in types: inst[t] += 1
        ntypes[len(set(types))] += 1
    print(f"{'type':<9}{'graphs':>8}{'instances':>11}")
    for t in MOTIF_TYPES:
        print(f"{t:<9}{g_with[t]:>8}{inst[t]:>11}")
    print(f"graphs by number of distinct toxicophore TYPES: {dict(sorted(ntypes.items()))}")
    print(f"graphs with 2+ types: {sum(v for k, v in ntypes.items() if k >= 2)}/{len(pos)}")
    gt_size = np.array([int(data_list[i].node_gt_mask.sum()) for i in pos])
    gt_frac = gt_size / n_nodes[pos]
    print(f"GT-present (positives): {len(pos)}/{len(data_list)}")
    print(f"merged GT size: mean {gt_size.mean():.2f} atoms  median {np.median(gt_size):.0f}  max {gt_size.max()}"
          f"   as fraction of molecule: mean {gt_frac.mean():.3f}")
    print(f"top-k=0.25 budget on GT-present graphs: mean {np.mean([max(1, int(0.25 * n)) for n in n_nodes[pos]]):.2f} atoms")
    rep["toxicophores"] = {"graphs_with_type": dict(g_with), "instances": dict(inst),
                           "n_types_hist": {int(k): v for k, v in sorted(ntypes.items())},
                           "graphs_2plus_types": int(sum(v for k, v in ntypes.items() if k >= 2)),
                           "gt_present": len(pos), "gt_size_mean": float(gt_size.mean()),
                           "gt_size_median": float(np.median(gt_size)), "gt_frac_mean": float(gt_frac.mean())}

    json.dump(rep, open(f"{OUTDIR}/step1_report.json", "w"), indent=1)
    print(f"\nsaved -> {OUTDIR}/step1_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
