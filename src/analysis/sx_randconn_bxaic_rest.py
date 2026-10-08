"""
SubgraphX vs its size-matched connected null (RandConn, 20 draws) on the three
B-XAIC tasks not covered by connectivity_baseline.py: PAINS, X, P. Same
construction as F15/F16-ADDENDUM. Per-seed paired Wilcoxon; Holm over all 15
(task, seed) tests. Arrays saved to runs/mutagenicity/exploratory/.

    python -m src.analysis.sx_randconn_bxaic_rest
"""
from __future__ import annotations
import json
import numpy as np
from .connectivity_baseline import run_dataset, _wpair, OUT, SEEDS

SAVE = "runs/mutagenicity/exploratory"


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1]); out, run, m = {}, 0.0, len(items)
    for r, (k, p) in enumerate(items):
        run = min(max(run, (m - r) * p), 1.0); out[k] = run
    return out


def main():
    per = {t: run_dataset(t, f"bxaic_{t}", lambda s, t=t: f"{OUT}/cache_bxaic_{t}_s{s}.pt", save_dir=SAVE)
           for t in ("PAINS", "X", "P")}
    raw = {(t, s): _wpair(per[t][s]["sx"], per[t][s]["rand_conn"]) for t in per for s in SEEDS}
    adj = holm(raw)
    rep = {}
    print("\nSubgraphX (mean-threshold) vs size-matched RandConn, Holm over 15 tests")
    for t in per:
        wins = 0; rows = []
        for s in SEEDS:
            d = per[t][s]; a, b = float(d["sx"].mean()), float(d["rand_conn"].mean())
            ok = a > b and adj[(t, s)] < 0.05; wins += ok
            rows.append({"seed": s, "sx": a, "rand_conn": b, "p_holm": adj[(t, s)], "n": int(len(d["sx"])),
                         "mol_wins": int(d["win_vs_conn"].sum()), "sig_above": bool(ok)})
            print(f"  {t:<6} seed {s}: SX {a:.3f}  RandConn {b:.3f}  p_Holm {adj[(t, s)]:.1e}  "
                  f"molecule wins {int(d['win_vs_conn'].sum())}/{len(d['sx'])}  {'ABOVE' if ok else '-'}")
        print(f"  {t:<6} -> SubgraphX significantly above RandConn in {wins}/5 seeds")
        rep[t] = {"units_sig_above": int(wins), "per_seed": rows}
    json.dump(rep, open(f"{SAVE}/sx_randconn_bxaic_rest.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
