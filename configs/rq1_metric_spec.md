# RQ1 Metric Spec (locked) -- mirrors RQ1_Metric_Spec.docx v3

Datasets: MUTAG (standard), MUTAG (GraphXAI GT-labeled), BBBP, Tox21 (SR-p53
phase 1, then all 12 endpoints), B-XAIC
Explainers: GNNExplainer, PGExplainer, SubgraphX
Base model: D-MPNN (Chemprop)

## Metrics (verified against GraphXAI source, graphxai/metrics/metrics_graph.py)

- Fid+  = (1/N) sum_i [ f(G_i)_yi - f(G_i \ E_i)_yi ]
- Fid-  = (1/N) sum_i [ f(G_i)_yi - f(E_i)_yi ]
- GEF   = 1 - exp( -KL( f(G_i) || f(E_i) ) )        # bounded [0,1), lower = more faithful
- GEA   = JAC(M_gt, M_pr) = TP / (TP+FP+FN)          # only where ground truth exists

## Masking references -- build order R3 -> R2 -> R1

- R3 (build first): distribution-aware, replace masked features with training-set
  per-feature mean. Implemented: src/metrics/masking.py::mask_r3_distribution_aware
- R2 (build second): zero-fill, topology kept. STUB, not yet implemented.
- R1 (build third): hard removal, topology broken. STUB, not yet implemented.

## GEA applicability (ground truth required)

| Dataset                     | GEA |
|------------------------------|-----|
| MUTAG (standard)             | No  |
| MUTAG (GraphXAI GT-labeled)  | Yes |
| BBBP                         | No  |
| Tox21                        | No  |
| B-XAIC                       | Yes |
