"""
RQ1 revision, T4 follow-up fixes (before DONE):
  1. collapse by predicted class under the old float32 rule and the new logit rule (the absolute
     std < 1e-6 rule is withdrawn), with the sign of the saturation (all logits > 16: masks all 1;
     all logits < -16: masks all 0);
  2. per unit: fraction of explained molecules whose re-run PGExplainer edge mask matches the stored
     one (max |difference| <= 1e-5), to separate tie effects from re-run differences in the
     logit-ranked GEA counts;
  3. T5: MUTAG-GT seeds 0-4 under R1 against the earlier 5/0/0.
Writes runs/rq1_revision/t4_fixes.txt / .json.

    python -m src.analysis.revision_t4_fixes
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

R = "runs/rq1_revision"; L = f"{R}/logits"
LOG = []


def P(s=""):
    print(s); LOG.append(s)


def degenerate_logits(lg):
    lg = np.asarray(lg, dtype=np.float64)
    return len(lg) == 0 or (lg.max() - lg.min()) <= 1e-6 * max(1.0, abs(lg.mean()))


def main():
    rep = {"by_class": {}, "reproduction": {}}
    P("[1] collapsed units: degenerate molecules by predicted class, old float32 rule and new logit rule; saturation sign of old-rule degenerate molecules")
    P(f"  {'unit':<30}{'old coll':>9}{'new coll':>9}   class: old-degenerate / new-degenerate / n   [old-degenerate: all logits > +16 | all < -16 | other]")
    for f in sorted(glob.glob(f"{L}/*.npz")):
        u = os.path.basename(f)[:-4]; z = np.load(f, allow_pickle=True)
        lgs = [np.asarray(a, dtype=np.float64) for a in z["logits"]]; tg = z["target"]
        old = np.array([float(torch.as_tensor(m, dtype=torch.float).std()) == 0.0 for m in z["stored_node_mask"]])
        new = np.array([degenerate_logits(a) for a in lgs])
        if not (old.mean() >= 0.5 or new.mean() >= 0.5):
            continue
        hi = np.array([len(a) > 0 and a.min() > 16 for a in lgs]); lo = np.array([len(a) > 0 and a.max() < -16 for a in lgs])
        parts = []
        for c in np.unique(tg):
            s = tg == c
            parts.append(f"{int(c)}: {int(old[s].sum())}/{int(new[s].sum())}/{int(s.sum())} [{int((old & s & hi).sum())}|{int((old & s & lo).sum())}|{int((old & s & ~hi & ~lo).sum())}]")
        P(f"  {u:<30}{('yes' if old.mean() >= 0.5 else 'no'):>9}{('yes' if new.mean() >= 0.5 else 'no'):>9}   " + "   ".join(parts))
        rep["by_class"][u] = {"old_collapsed": bool(old.mean() >= 0.5), "new_collapsed": bool(new.mean() >= 0.5),
                              "per_class": {int(c): {"n": int((tg == c).sum()), "old_deg": int(old[tg == c].sum()), "new_deg": int(new[tg == c].sum()),
                                                      "old_deg_all_high": int((old & (tg == c) & hi).sum()), "old_deg_all_low": int((old & (tg == c) & lo).sum())}
                                            for c in np.unique(tg)}}
    P("\n[2] reproduction of the stored PGExplainer masks by the re-run: fraction of explained molecules with max |edge-mask difference| <= 1e-5")
    units = [f"mutag_graphxai_s{s}" for s in range(10)] + [f"bxaic_indole_s{s}" for s in range(5)] + \
            [f"mutagenicity_{a}_s{s}" for a in ("dmpnn", "gin", "gine", "gcn") for s in range(5)]
    from .revision_logits import units as lunits
    U = lunits()
    for u in units:
        new = torch.load(f"{L}/rerun/{u}_cache.pt", weights_only=False)["results"]["pgexplainer"]
        old = torch.load(U[u]["cache"], weights_only=False)["results"]["pgexplainer"]
        ok = [len(a["edge_importance"]) == 0 or float(np.max(np.abs(np.asarray(a["edge_importance"]) - np.asarray(b["edge_importance"])))) <= 1e-5
              for a, b in zip(new, old)]
        rep["reproduction"][u] = float(np.mean(ok))
    for g in ("mutag_graphxai", "bxaic_indole", "mutagenicity_dmpnn", "mutagenicity_gin", "mutagenicity_gine", "mutagenicity_gcn"):
        P(f"  {g:<22}" + "  ".join(f"s{u.rsplit('_s', 1)[1]} {v:.2f}" for u, v in rep["reproduction"].items() if u.rsplit("_s", 1)[0] == g))
    t5 = json.load(open(f"{R}/t5.json"))["MUTAG-GT (10 seeds)"]["R1"]["signs"]
    P(f"\n[3] T5, MUTAG-GT under R1 (Holm over 3 rules), seeds 0-9: {' '.join(t5)}; seeds 0-4: {''.join(t5[:5])} "
      f"({t5[:5].count('+')}/{t5[:5].count('-')}/{t5[:5].count('.')}) vs earlier Part A 5/0/0 (uncorrected, other random draws)")
    rep["t5_mutag_gt_r1"] = t5
    open(f"{R}/t4_fixes.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{R}/t4_fixes.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
