"""
RQ1 Tier 4, item L2: BBBP, 5 seeds (exploratory; runs/rq1_tier4/PLAN.md). Fid+ / Fid- / GEF under R1, R2, R3, both spaces, for
GNNExplainer, PGExplainer (float64 logits), SubgraphX fixed budget, Saliency and IG on the 30 explained test molecules, with 20
random rankings per molecule (Part-E protocol, tier2_fid).

    python -m src.analysis.tier4_L2 --unit bbbp_sS      -> runs/rq1_tier4/L2/fid/<unit>.npz
"""
from __future__ import annotations

import argparse

from . import tier2_common as C
from . import tier4_common as T
from .tier2_fid import run_fid

METHODS = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig"]


def run(unit):
    sp = C.orig_spec(unit); L = f"{T.T4}/L2"; ti = T.test_idx_of(unit); g = f"{L}/grad/{unit}.npz"
    src = {"gnnexplainer": T.cache_scores(sp["cache"], "gnnexplainer"), "pgexplainer": T.pg64(unit), "sx_fixed": C.sx_shard_scores(f"{L}/sx/{unit}"),
           "sal": C.npz_scores(g, "sal"), "ig": C.npz_scores(g, "ig")}
    missing = [(w, m) for w in METHODS for m in ti if m not in src[w]]
    if missing:
        raise RuntimeError(f"{unit}: {len(missing)} (method, molecule) scores missing")
    run_fid(unit, sp["ckpt"], f"{L}/gp/{unit}_out.json", "bbbp", src, ti, f"{L}/fid/{unit}.npz", with_randnode=True)
    print(f"{unit}: L2 Fid on {len(ti)} molecules", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit", required=True); run(a.parse_args().unit)
