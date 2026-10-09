"""
Fold file for the full-set Mutagenicity experiment: stratified 5-fold split of
the 1,768 graphs (fold seed 0). Fold f: test = fold f; the other four folds
are split 90/10 (stratified, seed f) into train/validation for model selection.
Writes runs/mutagenicity_cv/split_f<f>.json and folds.json. Never overwrites.

    python -m src.analysis.mutagenicity_cv_splits
"""
from __future__ import annotations
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from ..data import LOADERS

R = "runs/mutagenicity_cv"
FOLD_SEED = 0


def main():
    os.makedirs(R, exist_ok=True)
    if os.path.exists(f"{R}/folds.json"):
        print("folds.json exists -- not regenerating"); return 0
    dl, _ = LOADERS["mutagenicity_graphxai"]()
    y = np.array([int(d.y.view(-1)[0]) for d in dl]); n = np.array([d.num_nodes for d in dl])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    folds = {}
    for f, (rest, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = train_test_split(rest, test_size=0.10, random_state=f, stratify=y[rest])
        sp = {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(te.tolist())}
        json.dump(sp, open(f"{R}/split_f{f}.json", "w"))
        folds[f] = sp["test"]
        print(f"fold {f}: train {len(tr)}  val {len(va)}  test {len(te)}  | test positives {int(y[te].sum())}"
              f"  test <=80 atoms {int((n[te] <= 80).sum())}  GT-bearing <=80 {int(((n[te] <= 80) & (y[te] == 1)).sum())}")
    allte = sorted(i for v in folds.values() for i in v)
    assert allte == list(range(len(y))), "folds must partition the dataset"
    json.dump({"fold_seed": FOLD_SEED, "folds": folds}, open(f"{R}/folds.json", "w"))
    print(f"every graph is in exactly one test fold: {len(allte)} graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
