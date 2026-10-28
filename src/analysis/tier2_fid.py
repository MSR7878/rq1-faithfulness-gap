"""
RQ1 Tier 2: Fid+, Fid- and GEF under R1, R2, R3 (probability and logit space) for new methods, with the
Part-E protocol of src.analysis.revision_pass: per explained molecule, the method's top-k (k = 25%) with random
tie-breaking (20 orders when the top-k or the removed-set boundary is tied; stream "<unit>|<method>|fid"),
explanation-removed and explanation-only graphs, averaged over orders. The random-node reference is the Part-E
array of the same unit and molecules (runs/rq1_revision/pass/E) or, for new units, 20 random rankings drawn here.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import torch

from . import reliance_fidelity as rf
from .revision_pass import KEYS, RULES, metrics
from .tier2_common import load_data, mol_rng, save_npz
from .ties_random import N_ORDERS, boundary_tie, removed_boundary_tie, tie_order


def run_fid(unit, ckpt, out_json, dkey, scores: dict, mol_idx, out, with_randnode=False):
    """scores: method -> {mi: node scores}. Writes per-molecule arrays <method>_<rule>_<key>."""
    model, _ = rf.load_model(ckpt, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    fill = torch.tensor(json.load(open(out_json))["fills"]["R3"]["mean"], dtype=torch.float)
    dl, _ = load_data(dkey); rng_n = np.random.default_rng([84000, zlib.crc32(unit.encode()), 0])
    R = {"mol_idx": []}
    names = list(scores) + (["randnode"] if with_randnode else [])
    for w in names:
        R[f"{w}_n_orders"] = []
        for rule in RULES:
            for k in KEYS:
                R[f"{w}_{rule}_{k}"] = []
    for mi in mol_idx:
        G = rf.to_cpu_data(dl[int(mi)]); n = G.num_nodes
        k_, keep = max(1, int(rf.K_FRAC * n)), max(1, int((1 - rf.K_FRAC) * n))
        rankings = {}
        for w in scores:
            imp = np.asarray(scores[w][int(mi)], dtype=np.float64)
            if boundary_tie(imp, k_) or removed_boundary_tie(imp, keep):
                r = mol_rng(f"{unit}|{w}|fid", int(mi)); rankings[w] = [tie_order(imp, r) for _ in range(N_ORDERS)]
            else:
                rankings[w] = [torch.as_tensor(imp, dtype=torch.float)]
        if with_randnode:
            rankings["randnode"] = [torch.tensor(rng_n.random(n), dtype=torch.float) for _ in range(20)]
        graphs, idx = [G], []
        for w, rs in rankings.items():
            for r in rs:
                for rule in RULES:
                    E, GmE = rf.masked_pair(G, r, fill, rule)
                    idx.append((w, rule, len(graphs), len(graphs) + 1)); graphs += [GmE, E]
        lg = rf.logits_of(model, graphs, device); c = int(lg[0].argmax())
        acc = {}
        for w, rule, ir, ik in idx:
            for kk, v in metrics(lg, c, ir, ik).items():
                acc.setdefault((w, rule, kk), []).append(v)
        R["mol_idx"].append(int(mi))
        for w in names:
            R[f"{w}_n_orders"].append(len(rankings[w]))
            for rule in RULES:
                for kk in KEYS:
                    R[f"{w}_{rule}_{kk}"].append(float(np.mean(acc[(w, rule, kk)])))
    save_npz(out, unit=unit, **{k: np.array(v) for k, v in R.items()})
