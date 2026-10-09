
## GraphXAI Mutagenicity, full set by 5-fold CV (runs/mutagenicity_cv/; runs/mutagenicity/ untouched)

### Design and registered predictions -- written 2026-10-09 11:12 IST, BEFORE the run is launched

Design: stratified 5-fold split of the 1,768 graphs (fold seed 0). Fold f: the
D-MPNN is trained on the other four folds (10% of them held out, stratified,
for model selection; model seed f), then every graph in fold f is explained
(explainer seed 1000 + f). Each graph is explained exactly once, by a model
that never saw it. Model config, explainer configs and the 80-atom exclusion
are those of the 5-seed test-split run. Outputs only under runs/mutagenicity_cv/.

Registered predictions:
(a) GNNExplainer beats RandNode in at least 4 of 5 folds.
(b) SubgraphX beats its connectivity-matched null in at least 4 of 5 folds.
(c) PGExplainer is not reliably above RandNode.

How they will be scored (fixed now):
- GEA is max-over-set; top-k 0.25 is the primary binarization.
- (a) GNNExplainer top-k GEA vs RandNode at the same budget (20 draws). A fold
  counts when the fold mean is higher AND the paired Wilcoxon test over that
  fold's GT-bearing molecules is significant after Holm.
- (b) SubgraphX vs RandConn sized to the SubgraphX mask (20 draws): the
  SubgraphX mean-threshold mask against random connected subgraphs of the same
  size, the construction used for the earlier connectivity checks. Same
  counting rule. The top-k version (RandConn at the budget) is reported too.
- (c) holds if PGExplainer is significantly above RandNode (top-k, same rule)
  in FEWER than 4 of 5 folds.
- Holm is applied within each fold over four tests: the three explainers vs
  RandNode and SubgraphX vs RandConn. The pooled test (one per comparison,
  over all unique molecules, Holm over the same four) is reported separately
  and does not score the predictions; the model differs by fold.
