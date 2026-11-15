"""
RQ1 Tier 5 (runs/rq1_tier5/PLAN.md): the four new explainers on our models. Settings and deviations are fixed in the PLAN;
every replaced or added function is listed in runs/rq1_tier5/patches.md.

Each class has explain(d, budget) -> dict with
  node      per-atom scores used for top-25% GEA (ties broken at random by the scorer)
  own       0/1 per-atom set at the method's own size (EiG, SAME) or None
  meta      method details (target class, sizes, ...)
d is a PyG Data on CPU; the target is the model's predicted class.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from .t5_common import IdAdapter, edge_attr_of, load_repo, logits, pred_class


def bonds_of(edge_index: torch.Tensor):
    """Undirected bonds of a symmetric edge_index: list of (u, v) with u < v, and for every directed edge its bond id."""
    ei = edge_index.cpu().numpy(); key = {}
    bid = np.empty(ei.shape[1], dtype=np.int64)
    for j, (a, b) in enumerate(zip(ei[0], ei[1])):
        p = (min(a, b), max(a, b))
        if p not in key:
            key[p] = len(key)
        bid[j] = key[p]
    pairs = [None] * len(key)
    for p, i in key.items():
        pairs[i] = p
    return pairs, bid


def keep_edges_batch(d, masks, device):
    """One Batch with a copy of d per row of masks (bool over directed edges): the edges in the row are kept, atoms stay."""
    x = d.x.float(); ea = edge_attr_of(d); out = []
    for m in masks:
        g = Data(x=x, edge_index=d.edge_index[:, m])
        if ea is not None:
            g.edge_attr = ea[m]
        out.append(g)
    return Batch.from_data_list(out).to(device)


def batch_logits(model, b):
    return logits(model, b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch)


# ----------------------------------------------------------------------------------------------------- EiG-Search
class EiG:
    """Port of the graph-classification branch of EiG-Search main.py (commit b1e975d) to our models; see the PLAN."""
    name = "eig"

    def __init__(self, model, device, sparsity=0.7):
        self.model = model.eval(); self.device = device; self.sparsity = sparsity

    def _prob_of_nodes(self, d, keep_nodes, edge_keep, c):
        """Probability of class c for the graph made of the atoms keep_nodes and the directed edges edge_keep (bool) among
        them; an empty graph gets 1/2 (PLAN)."""
        keep_nodes = np.asarray(sorted(keep_nodes), dtype=np.int64)
        if len(keep_nodes) == 0:
            return 0.5
        remap = -np.ones(d.num_nodes, dtype=np.int64); remap[keep_nodes] = np.arange(len(keep_nodes))
        ei = d.edge_index[:, torch.as_tensor(edge_keep)]
        ei = torch.as_tensor(remap)[ei]
        ea = edge_attr_of(d); ea = None if ea is None else ea[torch.as_tensor(edge_keep)]
        x = d.x.float()[torch.as_tensor(keep_nodes)]
        lg = logits(self.model, x.to(self.device), ei.to(self.device), None if ea is None else ea.to(self.device))
        return float(torch.softmax(lg[0], -1)[c])

    def explain(self, d, budget=None):
        dev = self.device; n = d.num_nodes
        c = pred_class(self.model, d, dev)
        lg0 = logits(self.model, d.x.float().to(dev), d.edge_index.to(dev), None if edge_attr_of(d) is None else edge_attr_of(d).to(dev))[0]
        lp0 = float(torch.log_softmax(lg0, -1)[c]); p0 = float(torch.softmax(lg0, -1)[c])
        pairs, bid = bonds_of(d.edge_index); B = len(pairs)
        if B == 0:
            return {"node": np.zeros(n), "own": np.zeros(n), "meta": {"target": c, "bonds": 0, "chosen_bonds": 0}}
        # 1. bond scores: drop in log p(c) when the bond (both directions) is removed; one batched forward
        masks = [bid != b for b in range(B)]
        lp = torch.log_softmax(batch_logits(self.model, keep_edges_batch(d, masks, dev)), -1)[:, c].cpu().numpy()
        score = lp0 - lp
        order = np.argsort(-score, kind="stable")                  # ties: lower bond index first (deterministic)
        # 2. size search over undirected bonds, Fid+ - Fid- in probability
        M = max(1, int(math.floor((1.0 - self.sparsity) * B)))
        sizes = list(range(2, M + 1)) if M >= 2 else [M]
        ei = d.edge_index.numpy(); best, best_m, diffs = -np.inf, sizes[0], []
        for m in sizes:
            S = set(order[:m].tolist()); sel = np.isin(bid, list(S))
            nodes_S = set(ei[0, sel].tolist()) | set(ei[1, sel].tolist())
            p_minus = self._prob_of_nodes(d, nodes_S, sel, c)                       # only the chosen bonds and their atoms
            left = ~sel; nodes_left = set(ei[0, left].tolist()) | set(ei[1, left].tolist())
            p_plus = self._prob_of_nodes(d, nodes_left, left, c)                     # chosen bonds removed, bond-less atoms dropped
            diff = (p0 - p_plus) - (p0 - p_minus); diffs.append(diff)
            if diff > best:
                best, best_m = diff, m
        chosen = order[:best_m]
        node = np.zeros(n)
        for b in range(B):
            u, v = pairs[b]; node[u] += score[b]; node[v] += score[b]
        own = np.zeros(n)
        for b in chosen:
            u, v = pairs[b]; own[u] = own[v] = 1.0
        return {"node": node, "own": own, "meta": {"target": c, "bonds": B, "chosen_bonds": int(best_m), "max_bonds": M,
                                                   "bond_score": score.tolist(), "search_diffs": diffs}}


# ----------------------------------------------------------------------------------------------------- GStarX
class GStarXExplainer:
    """The authors' GStarX class (gstarx.py, commit 770b286) with their molecular-dataset config, through IdAdapter."""
    name = "gstarx"
    CFG = dict(payoff_type="norm_prob", max_sample_size=10, tau=0.01, subgraph_building_method="remove",
               superadditive_ext=False, sample_method="random", num_samples=-1, num_hops=4)

    def __init__(self, model, device, payoff_avg):
        mods = load_repo("GStarX", ["gstarx"], shims=True)
        self.adapter = IdAdapter(model.eval(), device).to(device)
        c = self.CFG
        self.gx = mods["gstarx"].GStarX(self.adapter, device, max_sample_size=c["max_sample_size"], tau=c["tau"],
                                        payoff_type=c["payoff_type"], payoff_avg=list(payoff_avg),
                                        subgraph_building_method=c["subgraph_building_method"])
        self.device = device; self.model = model

    def explain(self, d, budget=None):
        self.adapter.set_graph(d)
        c = pred_class(self.model, d, self.device)
        g = self.adapter.augment(d, y=c)
        s = self.gx.explain(g, superadditive_ext=self.CFG["superadditive_ext"], sample_method=self.CFG["sample_method"],
                            num_samples=self.CFG["num_samples"], k=self.CFG["num_hops"])
        return {"node": np.asarray(s, dtype=np.float64), "own": None, "meta": {"target": c}}


# ----------------------------------------------------------------------------------------------------- SAME
SAME_BLOCKS = {
    "mutag": dict(rollout=20, high2low=True, c_puct=10.0, min_atoms=4, expand_atoms=20, local_raduis=5, sample_num=100,
                  reward_method="mc_l_shapley", subgraph_building_method="split", single_explanation_size=2, candidate_size=7,
                  explanation_exploration_method="permutation"),
    "bbbp": dict(rollout=20, high2low=True, c_puct=10.0, min_atoms=5, expand_atoms=12, local_raduis=4, sample_num=100,
                 reward_method="mc_l_shapley", subgraph_building_method="split", single_explanation_size=5, candidate_size=7,
                 explanation_exploration_method="permutation"),
}


def same_block_for(dkey: str) -> str:
    return "bbbp" if dkey.startswith("bxaic") else "mutag"


class SAMEExplainer:
    """The authors' SAME pipeline (SAME.py, commit f0b4170): MCTS initialisation + mc_l_shapley + permutation exploration."""
    name = "same"

    def __init__(self, model, device, dkey):
        mods = load_repo("same", ["initialization_mcts_GC", "shapley", "utils"])
        self.MCTS = mods["initialization_mcts_GC"].MCTS; self.reward_func = mods["initialization_mcts_GC"].reward_func
        self.value_func = mods["shapley"].GnnNets_GC2value_func; self.find_explanations = mods["utils"].find_explanations
        self.block = same_block_for(dkey); p = SAME_BLOCKS[self.block]
        self.config = types.SimpleNamespace(explainers=types.SimpleNamespace(param=types.SimpleNamespace(**p)),
                                            models=types.SimpleNamespace(param=types.SimpleNamespace(graph_classification=True)))
        self.adapter = IdAdapter(model.eval(), device).to(device); self.device = device; self.model = model

    def explain(self, d, budget):
        """budget: 'own' (ceil(0.5 N), the config sparsity 0.5) or 'k' (floor(0.25 N))."""
        n = d.num_nodes
        size = int(math.ceil(n * (1 - 0.5))) if budget == "own" else max(1, int(0.25 * n))
        self.adapter.set_graph(d)
        c = pred_class(self.model, d, self.device)
        g = self.adapter.augment(d, y=c)                    # data.y = predicted class -> their value functions use it
        p = self.config.explainers.param
        payoff = self.reward_func(p, self.value_func(self.adapter, target_class=g.y))
        mcts = self.MCTS(g.x, g.edge_index, score_func=payoff, n_rollout=p.rollout, min_atoms=size, c_puct=p.c_puct,
                         expand_atoms=p.expand_atoms, high2low=p.high2low)
        results = mcts.mcts(verbose=False)
        coal = list(self.find_explanations(results, max_nodes=size, gnnNets=self.adapter, data=g, config=self.config).coalition)
        node = np.zeros(n); node[coal] = 1.0
        return {"node": node, "own": node.copy() if budget == "own" else None,
                "meta": {"target": c, "budget": budget, "size": size, "coalition": [int(i) for i in coal], "block": self.block}}


# ----------------------------------------------------------------------------------------------------- GNNShap
class GNNShapExplainer:
    """Our graph-level wrapper around the authors' GNNShapSampler (CUDA) and WLSSolver (commit f967229)."""
    name = "gnnshap"

    def __init__(self, model, device, num_samples=10000, batch_size=1024):
        import os
        cwd = os.getcwd(); os.chdir(os.path.join(__import__("src.explain.t5_common", fromlist=["TP"]).TP, "GNNShap"))
        try:                                               # the sampler compiles cppextension/cudagnnshap.cu relative to cwd
            mods = load_repo("GNNShap", ["gnnshap.samplers._gnnshap", "gnnshap.solvers._wls"])
        finally:
            os.chdir(cwd)
        self.Sampler = mods["gnnshap.samplers._gnnshap"].GNNShapSampler; self.Solver = mods["gnnshap.solvers._wls"].WLSSolver
        self.model = model.eval(); self.device = device; self.ns = num_samples; self.bs = batch_size

    def explain(self, d, budget=None):
        dev = self.device; n = d.num_nodes
        c = pred_class(self.model, d, dev)
        pairs, bid = bonds_of(d.edge_index); B = len(pairs)
        if B < 2:
            return {"node": np.zeros(n), "own": None, "meta": {"target": c, "bonds": B, "skipped": "fewer than 2 bonds"}}
        mask, kw = self.Sampler(nplayers=B, nsamples=self.ns).sample()
        bid_t = torch.as_tensor(bid, device=mask.device)
        yhat = []
        for i in range(0, mask.shape[0], self.bs):
            m = mask[i:i + self.bs][:, bid_t].cpu().numpy()
            yhat.append(torch.softmax(batch_logits(self.model, keep_edges_batch(d, list(m), dev)), -1)[:, c])
        yhat = torch.cat(yhat).double()
        none = np.zeros(len(bid), dtype=bool); full = np.ones(len(bid), dtype=bool)
        f2 = torch.softmax(batch_logits(self.model, keep_edges_batch(d, [none, full], dev)), -1)[:, c].double()
        phi, _ = self.Solver(mask, kw, yhat, float(f2[0]), float(f2[1]), device=dev).solve()
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        node = np.zeros(n)
        for b in range(B):
            u, v = pairs[b]; node[u] += phi[b]; node[v] += phi[b]
        return {"node": node, "own": None, "meta": {"target": c, "bonds": B, "fnull": float(f2[0]), "ffull": float(f2[1]),
                                                    "sum_phi": float(phi.sum())}}
