"""
Decoupled-protocol verification (mutag_graphxai only). Re-checks the three
headline claims against runs/decoupled/ (model-seed and explainer-init-seed
now independent, ES=S+1000) instead of runs/prio/ (the original coupled
sweep, model-seed==explainer-seed):

  1. F15: SX at/below a connectivity-matched random null (was 5/5 seeds).
  2. F3-REVISION v2: GNN >> {PG, SX} (GNN beats both significantly, was 5/5
     seeds; strict GNN>PG>SX ordering was only 3/5).
  3. F10: PGExplainer collapse rate (was 1/5 seeds -- seed 3 -- on this
     dataset under the coupled protocol).

Reuses seed_correct_significance.build_arrays_mutag_graphxai (already takes
a prio_dir arg) and connectivity_baseline.run_dataset (already takes a
path_fn) -- just pointed at runs/decoupled/ instead of runs/prio/.

    python -m src.analysis.decoupled_verify
"""
from __future__ import annotations
import json
from .seed_correct_significance import (
    build_arrays_mutag_graphxai, per_seed_test, seed_level_test, EXPL, SH,
)
from .connectivity_baseline import run_dataset

DECOUPLED = "runs/decoupled"
SEEDS = [0, 1, 2, 3, 4]


def check_collapse():
    print("\n" + "=" * 100)
    print(" 3. PGExplainer collapse rate, decoupled protocol (explainer_health.pgexplainer)")
    print("=" * 100)
    n_collapsed = 0
    for s in SEEDS:
        j = json.load(open(f"{DECOUPLED}/out_mutag_graphxai_s{s}.json"))
        h = j["explainer_health"]["pgexplainer"]
        n, nd = h.get("n", 188), h.get("n_degenerate", 0)
        collapsed = nd / max(n, 1) >= 0.5
        n_collapsed += int(collapsed)
        print(f"  seed {s} (model-seed={s}, explainer-seed={s+1000}):  "
              f"n_degenerate={nd}/{n}  {'COLLAPSED' if collapsed else 'clean'}")
    print(f"\n  Decoupled collapse rate: {n_collapsed}/5")


def main():
    print("=" * 100)
    print(" 1. F15 re-check: SX vs connectivity-matched random null, decoupled protocol")
    print("=" * 100)
    run_dataset("mutag_graphxai [DECOUPLED]", "mutag_graphxai",
                lambda s: f"{DECOUPLED}/cache_mutag_graphxai_s{s}.pt")

    print("\n" + "=" * 100)
    print(" 2. F3-REVISION v2 re-check: GNN >> {PG, SX}, decoupled protocol")
    print("=" * 100)
    by_seed = build_arrays_mutag_graphxai(prio_dir=DECOUPLED, seeds=tuple(SEEDS))
    per_seed_test(by_seed)
    seed_level_test(by_seed)

    check_collapse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
