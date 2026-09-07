# RQ1 — Quantifying the Faithfulness Gap

Benchmarking study comparing Fidelity+/-, GEF, and GEA across GNNExplainer,
PGExplainer, and SubgraphX on D-MPNN models trained on MUTAG, BBBP, Tox21
(SR-p53, phase 1), B-XAIC, and GraphXAI's ground-truth-labeled MUTAG.

See `configs/rq1_metric_spec.md` for the locked metric definitions, decisions,
and the running results log (Phase 2 model bring-up, Phase 3 per-variant R3
numbers + significance). This README stays high-level; findings live there
until the full R3/R2/R1 sweep is complete.

## Status

- **Phase 2** (datasets + D-MPNN): complete, 5/5 variants trained.
- **Phase 3 / R3 + R2 masking**: complete, 5/5 variants (one seeded run per
  dataset, explanations cached and scored under R3 mean-fill, R3 mode-fill, R2
  zero-fill). Headline: the explainer ranking is metric-, masking- *and*
  dataset-dependent — no single "most faithful" explainer.
  - R3 Fidelity/GEF is noise-dominated on 4/5 (diffuse-decision datasets).
  - GEA ranking *inverts* between the GT datasets: GNN ≈ PG ≫ SX on
    mutag_graphxai, SX ≈ PG ≫ GNN on B-XAIC (GNNExplainer's flip significant).
  - R2 zero-fill's perturbation artifact is featurisation-dependent (severe on
    one-hot MUTAG family, mild on raw-integer BBBP/Tox21); it does restore
    discriminative power for SubgraphX's compact connected explanations.
  - Details + paired Wilcoxon significance in `configs/rq1_metric_spec.md`.
- **Next**: R1 (hard removal, topology broken), then the same cached sweep.

## Structure

- `src/data/`    — dataset loaders (MUTAG, GraphXAI-GT MUTAG, BBBP, Tox21 SR-p53, B-XAIC) + `sanity_check.py`
- `src/train/`   — PyG-native D-MPNN (`dmpnn.py`, Yang et al. 2019) + 10-fold CV training (`train.py`)
- `src/explain/` — GNNExplainer / PGExplainer / SubgraphX wrappers, uniform interface
- `src/metrics/` — Fidelity+/-, GEF, GEA implementations + R1/R2/R3 masking references
- `src/analysis/`— rank correlation (Spearman/Kendall), B-XAIC ground-truth validation
- `configs/`     — per-dataset/per-explainer run configs (YAML)
- `data/`        — dataset cache (gitignored)
- `third_party/` — GraphXAI + B-XAIC clones, reference only (gitignored)

## Build order

1. `src/metrics/masking.py` — R3 (distribution-aware) reference only, first  ✔
2. `src/metrics/fidelity.py`, `gef.py`, `gea.py` — validated against R3  ✔
3. R2 (zero-fill) ✔ — same 5-variant sweep re-scored;  R1 (hard removal) next
4. Tox21: SR-p53 only until Phase 1 results validate the pipeline, then all 12 endpoints
