"""
RQ1 Part A (ground-truth reliance) and Part B (logit-margin / random-corrected
fidelity) on the EXISTING models and cached explanations. No training, no new
explanations; reads checkpoints, explanation caches and run outputs, writes
only under runs/rq1_reliance_fidelity/. Plan, predictions and scoring rules:
configs/predictions_reliance_fidelity.md (committed before any value existed).

Margin m(H) = logit_c(H) - logit_other(H), c = predicted class on the full G.

    python -m src.analysis.reliance_fidelity list
    python -m src.analysis.reliance_fidelity run --part B --units mutag mutag_graphxai_s0 [--limit 5] [--out DIR]
    python -m src.analysis.reliance_fidelity run --part A --all
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from ..data import LOADERS
from ..data.standard import TOX21_TASKS
from ..explain.common import to_cpu_data
from ..explain.run_phase3 import K_FRAC, load_model
from ..metrics.masking import mask_r1_hard_removal, mask_r2_zero_fill, mask_r3_distribution_aware
from .connectivity_baseline import build_adj, random_connected_subset

OUT = "runs/rq1_reliance_fidelity"
SEEDS = [0, 1, 2, 3, 4]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
MASKINGS = ["R1", "R2", "R3"]
FILL_KEY = {"R1": "hard", "R2": "zero", "R3": "mean"}
N_DRAWS = 20
BXAIC = ["indole", "PAINS", "X", "P"]
ARCH_DIRS = {"dmpnn": "runs/mutagenicity", **{a: f"runs/arch_{a}/mutagenicity_graphxai" for a in ("gin", "gine", "gcn")}}
# Addendum of 2026-10-19 (spec; supersedes the SubgraphX-only amendment): a recomputed probability value may differ
# from the stored one by more than VALIDATE_TOL only if a tie in importance is verified at the relevant selection
# boundary; such mismatches are logged and accepted, anything else stops the run. Stored values are never modified.
VALIDATE_TOL = 1e-5


# ----------------------------------------------------------------------------- units
def all_units() -> dict:
    """name -> {ckpt, cache, out, parts} for every existing run used here."""
    u = {}
    u["mutag"] = dict(ckpt="runs/ckpt_mutag.pt", cache="runs/expl_cache_mutag.pt", out="runs/phase3_mutag.json", parts="B")
    u["bbbp"] = dict(ckpt="runs/ckpt_bbbp.pt", cache="runs/expl_cache_bbbp.pt", out="runs/phase3_bbbp.json", parts="B")
    for s in SEEDS:
        p = "runs/prio"
        u[f"mutag_graphxai_s{s}"] = dict(ckpt=f"{p}/ckpt_mutag_graphxai_s{s}.pt", cache=f"{p}/cache_mutag_graphxai_s{s}.pt",
                                         out=f"{p}/out_mutag_graphxai_s{s}.json", parts="AB")
        for t in BXAIC:
            u[f"bxaic_{t}_s{s}"] = dict(ckpt=f"{p}/ckpt_bxaic_{t}_s{s}.pt", cache=f"{p}/cache_bxaic_{t}_s{s}.pt",
                                        out=f"{p}/out_bxaic_{t}_s{s}.json", parts="AB")
        for ep in TOX21_TASKS:
            u[f"tox21_{ep}_s{s}"] = dict(ckpt=f"{p}/ckpt_tox21_{ep}_s{s}.pt", cache=f"{p}/cache_tox21_{ep}_s{s}.pt",
                                         out=f"{p}/out_tox21_{ep}_s{s}.json", parts="B")
        for a, d in ARCH_DIRS.items():
            u[f"mutagenicity_{a}_s{s}"] = dict(ckpt=f"{d}/ckpt_s{s}.pt", cache=f"{d}/cache_s{s}.pt", out=f"{d}/out_s{s}.json", parts="AB")
    for name, v in u.items():
        v["seed"] = int(name.rsplit("_s", 1)[1]) if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else 0
    return u


def unit_rng(name: str, stream: int) -> np.random.Generator:
    return np.random.default_rng([60000, zlib.crc32(name.encode()), stream])


# ----------------------------------------------------------------------------- helpers
def keep_nodes(data: Data, keep_idx) -> Data:
    """Induced subgraph on keep_idx (R1 semantics: other nodes and their edges deleted, survivors reindexed)
    -- the same construction as masking.mask_r1_hard_removal, for an explicit node set."""
    n = data.x.size(0)
    keep_idx = torch.as_tensor(sorted(int(i) for i in keep_idx), dtype=torch.long)
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(keep_idx.numel())
    ei = data.edge_index
    ek = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    ea = getattr(data, "edge_attr", None)
    return Data(x=data.x[keep_idx].clone().float(), edge_index=remap[ei[:, ek]].contiguous(), edge_attr=None if ea is None else ea[ek])


def zero_nodes(data: Data, idx) -> Data:
    """R2 semantics for an explicit node set: those rows of x set to 0, topology kept."""
    x = data.x.clone().float()
    x[torch.as_tensor(sorted(int(i) for i in idx), dtype=torch.long)] = 0.0
    return Data(x=x, edge_index=data.edge_index, edge_attr=getattr(data, "edge_attr", None))


def masked_pair(cpu_data, node_imp, fill_vec, masking):
    """Explanation-only E and explanation-removed G\\E -- identical calls to run_phase3._masked_pair."""
    if masking == "R1":
        return (mask_r1_hard_removal(cpu_data, node_imp, keep_top_k=K_FRAC),
                mask_r1_hard_removal(cpu_data, -node_imp, keep_top_k=1.0 - K_FRAC))
    if masking == "R2":
        return (mask_r2_zero_fill(cpu_data, node_imp, keep_top_k=K_FRAC),
                mask_r2_zero_fill(cpu_data, -node_imp, keep_top_k=1.0 - K_FRAC))
    return (mask_r3_distribution_aware(cpu_data, node_imp, fill_vec, keep_top_k=K_FRAC),
            mask_r3_distribution_aware(cpu_data, -node_imp, fill_vec, keep_top_k=1.0 - K_FRAC))


@torch.no_grad()
def logits_of(model, graphs: list[Data], device, chunk: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, len(graphs), chunk):
        b = Batch.from_data_list(graphs[i:i + chunk]).to(device)
        out.append(model(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch).float().cpu())
    return torch.cat(out, 0)


def margin(logits: torch.Tensor, c: int) -> np.ndarray:
    return (logits[:, c] - logits[:, 1 - c]).numpy()


def prob(logits: torch.Tensor, c: int) -> np.ndarray:
    return torch.softmax(logits, dim=-1)[:, c].numpy()


def gt_masks(name, data, mi, motifs_all):
    """Valid GT masks: Mutagenicity -> the GraphXAI motif set; MUTAG-GT / B-XAIC -> connected components of the merged mask."""
    if name.startswith("mutagenicity"):
        return [sorted(set(int(v) for v in nodes)) for _, nodes in motifs_all[mi]]
    gt = data.node_gt_mask.detach().cpu().bool().numpy()
    adj = build_adj(data.edge_index.cpu(), data.num_nodes)
    seen, comps = set(), []
    for v in np.flatnonzero(gt):
        if v in seen:
            continue
        stack, comp = [int(v)], []
        seen.add(int(v))
        while stack:
            a = stack.pop(); comp.append(a)
            for b in adj[a]:
                if gt[b] and b not in seen:
                    seen.add(b); stack.append(b)
        comps.append(sorted(comp))
    return comps


def load_unit(name, spec, device):
    model, ck = load_model(spec["ckpt"], device)
    blob = torch.load(spec["cache"], weights_only=False)
    out = json.load(open(spec["out"]))
    assert blob["key"]["ckpt"] == spec["ckpt"], (name, blob["key"]["ckpt"], spec["ckpt"])
    loader = LOADERS[ck["dataset"]]
    dl, meta = loader()
    return model, ck, blob, out, dl, meta


# ----------------------------------------------------------------------------- Part A
def run_part_a(name, spec, device, limit=None):
    model, ck, blob, out, dl, meta = load_unit(name, spec, device)
    motifs_all = getattr(meta, "extras", {}).get("motifs") if hasattr(meta, "extras") else None
    rng_node, rng_conn = unit_rng(name + ":A", 0), unit_rng(name + ":A", 1)
    test_idx = list(blob["key"]["test_idx"])
    R = {k: [] for k in ["mol_idx", "n_nodes", "pred_class", "m_full", "n_masks", "n_skipped", "mask_size",
                         "delta_gt_R1", "delta_gt_R2", "null_node_R1", "null_node_R2", "null_conn_R1", "null_conn_R2",
                         "suff_gt", "suff_null_conn"]}
    done = 0
    for mi in test_idx:
        d = dl[mi]
        if not bool(d.node_gt_mask.any()):
            continue
        if limit is not None and done >= limit:
            break
        G = to_cpu_data(d); n = G.num_nodes; adj = build_adj(G.edge_index, n)
        masks = gt_masks(name, d, mi, motifs_all)
        graphs, plan, skipped = [G], [], 0
        for S in masks:
            if len(S) >= n:
                skipped += 1
                continue
            comp = [v for v in range(n) if v not in set(S)]
            base = len(graphs)
            graphs += [keep_nodes(G, comp), zero_nodes(G, S), keep_nodes(G, S)]
            for _ in range(N_DRAWS):
                r = rng_node.choice(n, size=len(S), replace=False)
                graphs += [keep_nodes(G, [v for v in range(n) if v not in set(r.tolist())]), zero_nodes(G, r)]
            for _ in range(N_DRAWS):
                r = random_connected_subset(adj, n, len(S), rng_conn)
                graphs += [keep_nodes(G, [v for v in range(n) if v not in r]), zero_nodes(G, r), keep_nodes(G, r)]
            plan.append((base, len(S)))
        if not plan:
            R["n_skipped"].append(skipped)   # keep the record aligned: molecule had no removable mask
            for k in R:
                if k != "n_skipped":
                    R[k].append(np.nan if k not in ("mol_idx", "n_nodes", "n_masks") else (mi if k == "mol_idx" else n if k == "n_nodes" else 0))
            continue
        lg = logits_of(model, graphs, device)
        c = int(lg[0].argmax()); m = margin(lg, c); m0 = m[0]
        per = {k: [] for k in ["dg1", "dg2", "nn1", "nn2", "nc1", "nc2", "sg", "sc"]}
        for base, size in plan:
            per["dg1"].append(m0 - m[base]); per["dg2"].append(m0 - m[base + 1]); per["sg"].append(m[base + 2])
            nb = base + 3
            node = m[nb: nb + 2 * N_DRAWS].reshape(N_DRAWS, 2)
            cb = nb + 2 * N_DRAWS
            conn = m[cb: cb + 3 * N_DRAWS].reshape(N_DRAWS, 3)
            per["nn1"].append(m0 - node[:, 0].mean()); per["nn2"].append(m0 - node[:, 1].mean())
            per["nc1"].append(m0 - conn[:, 0].mean()); per["nc2"].append(m0 - conn[:, 1].mean()); per["sc"].append(conn[:, 2].mean())
        R["mol_idx"].append(mi); R["n_nodes"].append(n); R["pred_class"].append(c); R["m_full"].append(m0)
        R["n_masks"].append(len(plan)); R["n_skipped"].append(skipped); R["mask_size"].append(np.mean([s for _, s in plan]))
        for k, kk in [("delta_gt_R1", "dg1"), ("delta_gt_R2", "dg2"), ("null_node_R1", "nn1"), ("null_node_R2", "nn2"),
                      ("null_conn_R1", "nc1"), ("null_conn_R2", "nc2"), ("suff_gt", "sg"), ("suff_null_conn", "sc")]:
            R[k].append(float(np.mean(per[kk])))
        done += 1
    return {k: np.array(v) for k, v in R.items()}, ck


# ----------------------------------------------------------------------------- validation (addendum of 2026-10-19)
def boundary_tie(imp: torch.Tensor, metric: str) -> dict:
    """Tie at the selection boundary that decides the graph behind `metric`:
    Fid- uses E (top-k by importance, k = max(1, int(K_FRAC N)));
    Fid+ uses G minus E (keeps the max(1, int((1-K_FRAC) N)) LOWEST importances)."""
    v = imp.double(); n = v.numel()
    if metric == "fid_minus":
        k = max(1, int(K_FRAC * n)); s = torch.sort(v, descending=True).values
    else:
        k = max(1, int((1.0 - K_FRAC) * n)); s = torch.sort(v).values
    if k >= n:
        return {"tie": False, "value": float(s[-1]), "n_inside": 0, "n_outside": 0}
    tie = bool(s[k - 1] == s[k]); val = float(s[k - 1])
    return {"tie": tie, "value": val, "n_inside": int((s[:k] == s[k - 1]).sum()), "n_outside": int((s[k:] == s[k - 1]).sum())}


def check_molecule(name, e, pos, mk, imp, recomputed: dict, stored: dict, log: list):
    """Addendum rule: |recomputed - stored| <= VALIDATE_TOL, or a verified tie at the metric's boundary
    (then logged and accepted). Anything else raises."""
    for met in ("fid_plus", "fid_minus"):
        dev = abs(recomputed[met] - stored[met])
        if dev <= VALIDATE_TOL:
            continue
        bt = boundary_tie(imp, met)
        row = {"unit": name, "explainer": e, "position": pos, "masking": mk, "metric": met, "stored": float(stored[met]),
               "recomputed": float(recomputed[met]), "abs_dev": float(dev), "tie_at_boundary": bt["tie"], "tied_value": bt["value"],
               "n_tied_inside": bt["n_inside"], "n_tied_outside": bt["n_outside"]}
        log.append(row)
        if not bt["tie"]:
            raise RuntimeError(f"UNVERIFIED MISMATCH (no boundary tie): {row}")


def validate_unit(name, spec, device):
    """Validation-only pass: recompute probability Fid+/Fid- for every molecule, explainer and masking and compare
    with the stored values. Returns (n_checked, mismatch rows). Raises only via check_molecule's rule."""
    model, ck, blob, out, dl, meta = load_unit(name, spec, device)
    test_idx = list(blob["key"]["test_idx"]); res = blob["results"]; pm = out["per_molecule"]
    fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    if "molecule_index" in out:
        assert list(out["molecule_index"]) == test_idx, f"{name}: out json rows are not the cache's test_idx"
    log, checked = [], 0
    for pos, mi in enumerate(test_idx):
        G = to_cpu_data(dl[mi]); graphs, idx = [G], {}
        for mk in MASKINGS:
            for e in EXPL:
                imp = torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float)
                E, GmE = masked_pair(G, imp, fill, mk); idx[(e, mk)] = len(graphs); graphs += [GmE, E]
        lg = logits_of(model, graphs, device)
        for mk in MASKINGS:
            for e in EXPL:
                tgt = int(res[e][pos]["target"]); p = prob(lg, tgt); i = idx[(e, mk)]
                imp = torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float)
                check_molecule(name, e, pos, mk, imp, {"fid_plus": p[0] - p[i], "fid_minus": p[0] - p[i + 1]},
                               pm[mk][FILL_KEY[mk]][e][pos], log)
                checked += 2
    return checked, log


# ----------------------------------------------------------------------------- Part B
def run_part_b(name, spec, device, limit=None):
    model, ck, blob, out, dl, meta = load_unit(name, spec, device)
    rng_node, rng_conn = unit_rng(name + ":B", 0), unit_rng(name + ":B", 1)
    test_idx = list(blob["key"]["test_idx"]); res = blob["results"]
    fill = torch.tensor(out["fills"]["R3"]["mean"], dtype=torch.float)
    pm = out["per_molecule"]
    if "molecule_index" in out:
        assert list(out["molecule_index"]) == test_idx, f"{name}: out json rows are not the cache's test_idx"
    keys = ["mol_idx", "n_nodes", "target", "pred_matches_target", "m_full", "p_full"]
    for e in EXPL:
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                keys += [f"{e}_{mk}_{met}_logit", f"{e}_{mk}_{met}_prob", f"{e}_{mk}_{met}_prob_stored", f"{e}_{mk}_{met}_mismatch_accepted"]
    for mk in MASKINGS:
        for met in ("fid_plus", "fid_minus"):
            for ref in ("node", "conn"):
                keys += [f"rand_{ref}_{mk}_{met}_logit", f"rand_{ref}_{mk}_{met}_prob"]
    R = {k: [] for k in keys}
    log, max_dev_ok = [], 0.0
    n_use = len(test_idx) if limit is None else min(limit, len(test_idx))
    for pos in range(n_use):
        mi = test_idx[pos]; d = dl[mi]
        G = to_cpu_data(d); n = G.num_nodes; adj = build_adj(G.edge_index, n)
        k = max(1, int(K_FRAC * n))
        tgt = int(res[EXPL[0]][pos]["target"])
        assert all(int(res[e][pos]["target"]) == tgt for e in EXPL), f"{name} pos {pos}: explainers explained different classes"
        node_rank = [torch.tensor(rng_node.random(n), dtype=torch.float) for _ in range(N_DRAWS)]
        conn_rank = []
        for _ in range(N_DRAWS):
            sc = rng_conn.random(n); sc[list(random_connected_subset(adj, n, k, rng_conn))] += 2.0
            conn_rank.append(torch.tensor(sc, dtype=torch.float))
        graphs, idx = [G], {}
        for mk in MASKINGS:
            for e in EXPL:
                imp = torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float)
                E, GmE = masked_pair(G, imp, fill, mk)
                idx[(e, mk)] = len(graphs); graphs += [GmE, E]
            for ref, ranks in (("node", node_rank), ("conn", conn_rank)):
                idx[(ref, mk)] = len(graphs)
                for r in ranks:
                    E, GmE = masked_pair(G, r, fill, mk); graphs += [GmE, E]
        lg = logits_of(model, graphs, device)
        m, p = margin(lg, tgt), prob(lg, tgt)
        R["mol_idx"].append(mi); R["n_nodes"].append(n); R["target"].append(tgt)
        R["pred_matches_target"].append(int(lg[0].argmax()) == tgt); R["m_full"].append(m[0]); R["p_full"].append(p[0])
        for mk in MASKINGS:
            for e in EXPL:
                i = idx[(e, mk)]
                fp_l, fm_l, fp_p, fm_p = m[0] - m[i], m[0] - m[i + 1], p[0] - p[i], p[0] - p[i + 1]
                R[f"{e}_{mk}_fid_plus_logit"].append(fp_l); R[f"{e}_{mk}_fid_minus_logit"].append(fm_l)
                R[f"{e}_{mk}_fid_plus_prob"].append(fp_p); R[f"{e}_{mk}_fid_minus_prob"].append(fm_p)
                st = pm[mk][FILL_KEY[mk]][e][pos]       # stored probability-space values of the original run (never modified)
                n_before = len(log)
                check_molecule(name, e, pos, mk, torch.as_tensor(res[e][pos]["node_importance"], dtype=torch.float),
                               {"fid_plus": fp_p, "fid_minus": fm_p}, st, log)
                acc = {r["metric"] for r in log[n_before:]}
                for met, val in (("fid_plus", fp_p), ("fid_minus", fm_p)):
                    R[f"{e}_{mk}_{met}_prob_stored"].append(st[met]); R[f"{e}_{mk}_{met}_mismatch_accepted"].append(met in acc)
                    if met not in acc:
                        max_dev_ok = max(max_dev_ok, abs(val - st[met]))
            for ref in ("node", "conn"):
                i = idx[(ref, mk)]
                gm, ee = np.arange(i, i + 2 * N_DRAWS, 2), np.arange(i + 1, i + 2 * N_DRAWS, 2)
                R[f"rand_{ref}_{mk}_fid_plus_logit"].append(float(np.mean(m[0] - m[gm])))
                R[f"rand_{ref}_{mk}_fid_minus_logit"].append(float(np.mean(m[0] - m[ee])))
                R[f"rand_{ref}_{mk}_fid_plus_prob"].append(float(np.mean(p[0] - p[gm])))
                R[f"rand_{ref}_{mk}_fid_minus_prob"].append(float(np.mean(p[0] - p[ee])))
    A = {k: np.array(v) for k, v in R.items()}
    A["validation_max_abs_dev"] = np.array(max_dev_ok)          # over all non-accepted cells (all within VALIDATE_TOL)
    A["validation_accepted_mismatches"] = np.array(json.dumps(log))
    return A, ck


# ----------------------------------------------------------------------------- driver
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "validate"])
    ap.add_argument("--part", choices=["A", "B"])
    ap.add_argument("--units", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="molecules per unit (smoke test)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    U = all_units()
    if args.cmd == "list":
        for n, v in U.items():
            print(n, v["parts"], all(os.path.exists(v[f]) for f in ("ckpt", "cache", "out")))
        return 0
    device = torch.device(args.device)
    if args.cmd == "validate":                  # validation-only pass; one json per unit (rows = every mismatch found)
        names = [n for n in U if "B" in U[n]["parts"]] if args.all else args.units
        os.makedirs(f"{args.out}/validation", exist_ok=True)
        for name in names:
            t0 = time.time()
            try:
                checked, log = validate_unit(name, U[name], device); status = "ok"
            except RuntimeError as ex:
                checked, log, status = -1, [], f"STOP: {ex}"
            json.dump({"unit": name, "checked_values": checked, "mismatches": log, "status": status},
                      open(f"{args.out}/validation/{name}.json", "w"), indent=1)
            print(f"{name}: validation {status}, {checked} values, {len(log)} accepted mismatches, {time.time() - t0:.1f}s", flush=True)
        return 0
    names = [n for n in U if args.part in U[n]["parts"]] if args.all else args.units
    os.makedirs(f"{args.out}/part{args.part}", exist_ok=True)
    for name in names:
        spec = U[name]
        dst = f"{args.out}/part{args.part}/{name}.npz"
        if os.path.exists(dst) and args.limit is None:
            print(f"{name}: exists, skip", flush=True); continue
        t0 = time.time()
        A, ck = (run_part_a if args.part == "A" else run_part_b)(name, spec, device, args.limit)
        np.savez(dst, unit=name, seed=spec["seed"], ckpt=spec["ckpt"], cache=spec["cache"], dataset=ck["dataset"],
                 arch=ck.get("arch", "dmpnn"), n_draws=N_DRAWS, **A)
        extra = (f"  validation max|dev| (non-accepted) {float(A['validation_max_abs_dev']):.1e},"
                 f" accepted tie mismatches {len(json.loads(str(A['validation_accepted_mismatches'])))}") if args.part == "B" else ""
        print(f"{name}: part {args.part}, {len(A['mol_idx'])} molecules, {time.time() - t0:.1f}s{extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
