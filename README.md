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
- **Phase 3 / masking-reference sweep**: complete — **R3 + R2 + R1**, 5/5
  variants, 3 explainers, 4 metrics. One seeded run per dataset; explanations
  cached and re-scored under R3 (mean+mode fill), R2 (zero-fill), R1 (hard
  removal). Headline: **no masking-/metric-invariant "most faithful"
  explainer.**
  - R3 Fidelity/GEF is noise-dominated on 4/5 (diffuse-decision datasets).
  - GEA ranking *inverts* between the GT datasets: GNN ≈ PG ≫ SX on
    mutag_graphxai, SX ≈ PG ≫ GNN on B-XAIC (GNNExplainer's flip significant).
  - R2 restores signal but only via a featurisation-dependent OOD artifact on
    one-hot data; R1 over-perturbs and washes even that out.
  - Across GEA + Fid+ × {R3,R2,R1}, the explainer ranking agrees with GEA in
    exactly 1 of 8 (dataset × masking) cells.
  - Full tables + paired Wilcoxon significance (F1–F8) in
    `configs/rq1_metric_spec.md`; `src/analysis/phase3_significance.py`.
- **Next**: scale-out — Tox21 all 12 endpoints, other B-XAIC tasks.

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
3. R2 (zero-fill) ✔  ·  R1 (hard removal) ✔ — full R3+R2+R1 × 5-variant sweep done
4. Tox21: SR-p53 only until Phase 1 results validate the pipeline, then all 12 endpoints
