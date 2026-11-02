"""
RQ1 Tier 3, item E: where the edge mask enters each model, and the variants all / first / last.

PyG's set_masks switches on explain mode for every MessagePassing module; each propagate() call then multiplies its
messages by the mask in explain_message. `use_variant` wraps explain_message with a per-forward call counter, so the mask
acts on every call (all), the first call (first) or the last call (last) of one model forward; the calls run in step order.
The trained weights are never touched.

    python -m src.analysis.tier3_mask check          # documents the path on one graph per model; exits 1 if a check fails
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import MessagePassing

from . import reliance_fidelity as rf
from . import tier2_common as C

VARIANTS = ("all", "first", "last")
_S = {"n": 0, "depth": 0, "variant": "all", "calls": 0}
_ORIG = MessagePassing.explain_message


def _explain_message(self, inputs, dim_size):
    i = _S["n"]; _S["n"] += 1; _S["calls"] += 1
    v = _S["variant"]
    if v == "all" or (v == "first" and i == 0) or (v == "last" and i == _S["depth"] - 1):
        return _ORIG(self, inputs, dim_size)
    return inputs


@contextlib.contextmanager
def use_variant(model, variant):
    """Mask only at the chosen step(s) of each forward of `model`."""
    assert variant in VARIANTS
    _S.update(variant=variant, depth=int(model.cfg.depth), n=0)
    h = model.register_forward_pre_hook(lambda m, a: _S.update(n=0))
    MessagePassing.explain_message = _explain_message
    try:
        yield
    finally:
        MessagePassing.explain_message = _ORIG
        h.remove()


def _explain_message_cyclic(self, inputs, dim_size):
    i = _S["n"] % _S["depth"]; _S["n"] += 1
    v = _S["variant"]
    if v == "all" or (v == "first" and i == 0) or (v == "last" and i == _S["depth"] - 1):
        return _ORIG(self, inputs, dim_size)
    return inputs


# PyG's Inspector looks the function up by __name__ when a model is built or explain mode is set
_explain_message.__name__ = _explain_message_cyclic.__name__ = "explain_message"


@contextlib.contextmanager
def global_variant(variant, depth=3):
    """The same variant when the model is built elsewhere (run_phase3): every masked forward makes exactly `depth` mask calls,
    so the call number modulo depth is the step. Raises if the total is not a positive multiple of depth."""
    assert variant in VARIANTS
    _S.update(variant=variant, depth=int(depth), n=0)
    MessagePassing.explain_message = _explain_message_cyclic
    try:
        yield
        if _S["n"] == 0 or _S["n"] % _S["depth"]:
            raise RuntimeError(f"mask calls {_S['n']} not a positive multiple of depth {_S['depth']}")
    finally:
        MessagePassing.explain_message = _ORIG


def output(model, d, edge_mask, variant):
    """Logits with a constant / given edge mask under a variant (no sigmoid: the value is the mask)."""
    set_masks(model, edge_mask, d.edge_index, apply_sigmoid=False)
    _S["calls"] = 0
    with use_variant(model, variant), torch.no_grad():
        out = model(d.x, d.edge_index, getattr(d, "edge_attr", None)).detach().cpu()
    calls = _S["calls"]; clear_masks(model)
    return out, calls


def check(path=f"runs/rq1_tier3/E/mask_path.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rep = {}; ok = True
    dl, _ = C.load_data("mutagenicity_graphxai")
    for arch in ("dmpnn", "gin", "gine", "gcn"):
        unit = f"mutagenicity_{arch}_s0"; model, _ = rf.load_model(C.orig_spec(unit)["ckpt"], device); model.eval()
        mi = int(C.stored_cache(unit)["key"]["test_idx"][0]); d = dl[mi].clone().to(device); E = d.edge_index.size(1)
        with torch.no_grad():
            clean = model(d.x, d.edge_index, getattr(d, "edge_attr", None)).detach().cpu()
        one, zero = torch.ones(E, device=device), torch.zeros(E, device=device)
        r = {"mol": mi, "edges": E, "depth": int(model.cfg.depth), "mp_modules": sum(isinstance(m, MessagePassing) for m in model.modules())}; zs = {}
        for v in VARIANTS:
            a, calls = output(model, d, one, v); b, _ = output(model, d, zero, v); zs[v] = b
            diff, ident = float((a - b).abs().max()), float((a - clean).abs().max())
            r[v] = {"mask_calls_per_forward": calls, "max_abs_diff_ones_vs_zeros": diff, "max_abs_diff_ones_vs_clean": ident}
            ok &= diff > 1e-6 and ident < 1e-4 and calls == r["depth"]
        # constant zero mask: on the D-MPNN all and last are identical by construction (the readout input is zeroed, nothing upstream reaches it)
        r["zero_mask_all_vs_first"] = float((zs["all"] - zs["first"]).abs().max()); r["zero_mask_all_vs_last"] = float((zs["all"] - zs["last"]).abs().max())
        # fractional mask (fixed seed): the three variants must differ pairwise, and "all" must equal PyG's unpatched behaviour
        fr = torch.as_tensor(np.random.default_rng(86000).random(E), dtype=torch.float, device=device); fs = {v: output(model, d, fr, v)[0] for v in VARIANTS}
        set_masks(model, fr, d.edge_index, apply_sigmoid=False)
        with torch.no_grad():
            stock = model(d.x, d.edge_index, getattr(d, "edge_attr", None)).detach().cpu()
        clear_masks(model)
        r["frac_all_vs_stock_pyg"] = float((fs["all"] - stock).abs().max()); r["frac_first_vs_stock_pyg"] = float((fs["first"] - stock).abs().max()); r["frac_last_vs_stock_pyg"] = float((fs["last"] - stock).abs().max())
        r["frac_all_vs_first"] = float((fs["all"] - fs["first"]).abs().max()); r["frac_all_vs_last"] = float((fs["all"] - fs["last"]).abs().max()); r["frac_first_vs_last"] = float((fs["first"] - fs["last"]).abs().max())
        ok &= r["frac_all_vs_stock_pyg"] < 1e-5 and min(r["frac_all_vs_first"], r["frac_all_vs_last"], r["frac_first_vs_last"]) > 1e-6
        rep[arch] = r; print(arch, json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True); json.dump({"all_checks_pass": ok, **rep}, open(path, "w"), indent=1)
    print("all checks pass" if ok else "CHECK FAILED", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
