# RQ1 — Quantifying the Faithfulness Gap

Benchmarking study comparing Fidelity+/-, GEF, and GEA across GNNExplainer,
PGExplainer, and SubgraphX on D-MPNN models trained on MUTAG, BBBP, Tox21
(SR-p53, phase 1), B-XAIC, and GraphXAI's ground-truth-labeled MUTAG.

See `configs/rq1_metric_spec.md` for the locked metric definitions and
decisions (mirrors the advisor-approved spec doc).

## Structure

- `src/train/`   — D-MPNN training per dataset (Chemprop-based)
- `src/explain/` — GNNExplainer / PGExplainer / SubgraphX wrappers, uniform interface
- `src/metrics/` — Fidelity+/-, GEF, GEA implementations + R1/R2/R3 masking references
- `src/analysis/`— rank correlation (Spearman/Kendall), B-XAIC ground-truth validation
- `configs/`     — per-dataset/per-explainer run configs (YAML)
- `data/`        — dataset cache (gitignored)

## Build order

1. `src/metrics/masking.py` — R3 (distribution-aware) reference only, first
2. `src/metrics/fidelity.py`, `gef.py`, `gea.py` — validated against R3
3. Extend masking to R2, then R1 (see spec, Section 4)
4. Tox21: SR-p53 only until Phase 1 results validate the pipeline, then all 12 endpoints
