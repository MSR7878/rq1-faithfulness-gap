# RQ1 Tier-1 revision analyses (T1-T6): plan, predictions, time estimate

Written 2026-10-26 and committed BEFORE any number of these analyses is
computed. Builds on commit f5b42b2. Outputs only under runs/rq1_revision/.
No .tex file is edited, no stored result is overwritten. Code is committed
before it runs. Pushed at the end.

## Common protocol (applies everywhere unless an item says otherwise)

- Top-k cut: k = max(1, int(0.25 N)), random tie-breaking: 20 seeded tie orders
  per molecule (random rank within exact ties only; ties_random.py; tag
  "<unit>|<explainer>|fid", i.e. the Phase 2b stream), each order scored, then
  averaged per molecule. GNNExplainer has no ties.
- Margin m(H) = logit_c - logit_other, c = the model's predicted class on G
  (= the class every cached explanation was made for).
- Fid+ = score(G) - score(G with the mask removed); Fid- = score(G) -
  score(only the mask kept); GEF = 1 - exp(-KL(p(kept graph) || p(G))) on the
  2-class softmax (src/metrics/gef.py convention). "score" = probability of c
  (probability space) or m (logit space).
- Hiding rules: R1 = delete nodes (and their edges); R2 = zero the node
  features; R3 = replace node features with the run's stored training-mean fill
  vector (out json, fills/R3/mean). Topology kept under R2 and R3.
- Random references: 20 draws per molecule; random-node = uniform node set of
  the same size; random-connected = connectivity_baseline.random_connected_subset
  of the same size. Each draw is scored separately (GEF per draw), then averaged.
- Tests: per seed (unit), paired two-sided Wilcoxon over molecules,
  zero_method "wilcox"; Holm within the family stated for each item;
  "significant" = Holm p < 0.05; direction = the rank test (W+ vs W-).
  Molecules are never pooled across seeds. MUTAG-GT uses all 10 seeds
  (0-4 priority sweep, 5-9 rq1_followups/phase1).
- "Better" means: Fid+ higher, Fid- lower, GEF lower, GEA higher.

## Predictions (wording fixed by the user; scoring rule underneath)

P-T1a: on Mutagenicity and B-XAIC (model uses the group), the ground-truth mask
beats size-matched random masks on Fid+ under R1 and R2 in >= 4/5 seeds per
dataset.
- Scored separately for 5 datasets: Mutagenicity (D-MPNN), B-XAIC indole,
  PAINS, X, P. Null: random-connected (primary). Space: probability (primary).
  Held if, in EVERY one of the 5 datasets, GT Fid+ is significantly above the
  random-connected Fid+ in >= 4/5 seeds under R1 AND in >= 4/5 under R2.
- Random-node and logit space are reported but do not score it. Disclosure:
  logit-space R1/R2 Fid+ of the GT mask is exactly Part A's delta_GT, already
  observed 5/5 in every one of these datasets (f53b612); only the probability
  space part is a fresh test.

P-T1b: under R3 the ground-truth mask beats random in fewer seeds than under R2.
- Count = number of seeds where GT Fid+ is significantly above random-connected
  (probability space), summed over the 5 datasets of P-T1a (25 seed cells).
  Held if the R3 sum is strictly smaller than the R2 sum. Equal = not held.
  Per-dataset counts are reported too.

P-T1c: under R2, explainer Fid+/GEF inflation (R2 minus R3) is NOT larger than
the random-mask inflation in most cells.
- Cell = explainer (GNN, PG, SX) x dataset (Mutagenicity D-MPNN, indole,
  PAINS, X, P, MUTAG-GT) x metric (Fid+, GEF), probability space: 36 cells.
- Per molecule d = (explainer R2 - explainer R3) - (random-node R2 - random-node R3),
  random-node at the explainer's top-k size (20 draws). Per seed: Wilcoxon of d
  vs 0; Holm family per (dataset, seed) = 3 explainers x 2 metrics = 6 tests.
- A cell is "larger" if d is significantly ABOVE 0 in a majority of seeds
  (>= 3/5; MUTAG-GT >= 6/10). Held if fewer than half of the 36 cells are
  "larger" (<= 17).

P-T2: per-seed rankings under R1, R2, R3 agree weakly (median Kendall tau < 0.5).
- With 3 explainers tau is in {-1, -1/3, 1/3, 1}, so "< 0.5" means <= 1/3.
- Taus computed for the same metric across rule pairs (R1-R2, R1-R3, R2-R3),
  for Fid+, Fid-, GEF (probability space), in every unit with explanations:
  MUTAG, BBBP, MUTAG-GT (10), Tox21 (60), B-XAIC (20), Mutagenicity D-MPNN /
  GIN / GINE / GCN (20). Primary median: over all (unit, metric, rule pair),
  each unit weighted equally. Held if that median <= 1/3. Per-dataset medians
  reported (Tox21 has 60 units and could dominate the pooled median).

P-T4: PGExplainer collapse rate on the collapsed Tox21 units drops by at least
half with 100+ epochs. (No direction assumed for the DIG comparison.)
- Units: the 24 Tox21 units collapsed in the priority sweep. Baseline = their
  collapse rate under the re-run with setting (a) (not the stored count).
- Collapse status of a run: degenerate molecule = node-importance std < 1e-6
  (float64); collapsed if >= 50% of the explained molecules are degenerate.
- Held if the collapse rate under (b) 100 epochs AND under (c) 300 epochs are
  each <= half the (a) rate. If (a) gives 0 collapsed units, P-T4 is "not
  testable" and reported as such.

P-T5: keeping only the annotated group preserves the prediction better than a
random connected set of the same size on Mutagenicity and B-XAIC (>= 4/5 seeds),
and also on MUTAG-GT.
- Measure: m(only S kept) vs mean m(only a random connected set of size |S|
  kept); comparing kept margins on the same molecule is equivalent to comparing
  drops (m(G) cancels). Primary rule: R1 (the earlier sufficiency test); R2 and
  R3 reported.
- Holm family per (model, seed) = R1, R2, R3 (3 tests).
- Held if significantly better under R1 in >= 4/5 seeds for Mutagenicity
  (D-MPNN) and for each of the 4 B-XAIC tasks, AND in >= 8/10 seeds for
  MUTAG-GT. GIN/GINE/GCN reported.

## Items

### T1 -- oracle test of the hiding rules (+ R3 reliance, inflation)
- Units: Mutagenicity D-MPNN (5), B-XAIC indole/PAINS/X/P (20), MUTAG-GT (10),
  plus Mutagenicity GIN/GINE/GCN (15) for the reliance/sufficiency parts.
  GT-bearing explained molecules; GT masks as in Part A (Mutagenicity: GraphXAI
  motif set; MUTAG-GT and B-XAIC: connected components of the merged mask);
  every quantity per mask, averaged over the molecule's masks.
- GT mask Fid+, Fid-, GEF under R1/R2/R3, probability and logit, vs 20
  random-node and 20 random-connected sets of |S|. Holm family per
  (dataset, seed, space) = 3 metrics x 3 rules x 2 nulls = 18 tests.
  Direction counts per seed and mean gaps.
- R3 reliance (group removal vs random-connected) for every model (MUTAG-GT,
  B-XAIC x4, Mutagenicity x4 architectures): Holm family per (model, seed) =
  {R1, R2, R3} x {random-node, random-connected} = 6 tests (the stored Part A
  used 4 tests without R3; both are reported).
- Inflation check: as in P-T1c, reported for every cell with both raw
  inflations.
- One pass computes T1 and T5 (same masked graphs).

### T2 -- rankings under each rule and cut
- Per unit: rank GNN/PG/SX by Fid+, Fid-, GEF under R1/R2/R3 (probability;
  random ties), and by GEA top-k and mean-threshold where GT exists. Kendall tau
  between every pair of the (metric, rule) rankings; median and range over seeds.
- Pairwise significance: per (unit, metric, rule), Wilcoxon on the 3 explainer
  pairs, Holm over the 3 pairs. A ranking "changes" between two rules only if a
  pair is significant in opposite directions under the two rules.
- Mutagenicity (D-MPNN): per seed, SubgraphX vs GNNExplainer GEA under top-k
  and under mean-threshold; Holm family per seed = the 2 cuts. The flip is
  "significant" if SX < GNN significantly under one cut and SX > GNN
  significantly under the other.

### T3 -- effect sizes with confidence intervals
- Estimand: mean over seeds of the per-seed mean difference (equal seed weight).
- Hierarchical bootstrap: resample seeds with replacement, then molecules with
  replacement within each drawn seed; 2,000 replicates; numpy default_rng(2026);
  95% percentile intervals.
- Quantities: (explainer top-k GEA - random-node GEA), random-tie values; and
  normalised GEA = (mean GEA - mean null) / (ceiling - mean null) computed as a
  ratio of means inside each replicate. Ceilings as before: MUTAG-GT 0.751,
  indole 0.724, Mutagenicity 0.551 (all four architectures share the same test
  molecules). Cells: main table (MUTAG-GT seeds 0-4, indole, Mutagenicity
  D-MPNN), architecture table (GIN, GINE, GCN; D-MPNN), MUTAG-GT 10 seeds.
- SubgraphX mean-threshold GEA vs size-matched random-connected (no ties), same
  bootstrap, same cells.
- GCN SubgraphX check (mean-threshold vs RandConn, Mutagenicity): per seed n,
  fraction of zero differences, number of tied |differences|, Wilcoxon p (raw
  and Holm within the registered 4-test family), exact sign-test p on nonzero
  differences, and a per-seed bootstrap CI of the mean difference.
- Tox21 TOST: metric = probability-space random-node-corrected Fid+ and Fid-
  under R3 (Part B values with Phase 2b random ties), per endpoint x explainer x
  metric (72 cells). "No-separation cell" = n.s. in >= 3/5 seeds in the stored
  Part B test (logit space, random-node, Holm over 18 within unit). For those
  cells: hierarchical-bootstrap 90% CI of the mean difference; equivalent if
  inside (-0.05, +0.05); "different" if the 95% CI excludes 0; otherwise
  undecided. Per-unit two one-sided t-tests (margin 0.05, alpha 0.05) counted too.

### T4 -- PGExplainer sanity checks
- Collapse by predicted class: for every collapsed unit (all datasets and
  architectures), degenerate molecules split by the explained class.
- Strict rule changed to node-importance std < 1e-6 (float64); recount all
  units per dataset: priority sweep (Tox21 60, B-XAIC 20, MUTAG-GT 5),
  MUTAG-GT 5-9, decoupled MUTAG-GT 5, Mutagenicity 5-seed and 5-fold CV,
  GIN/GINE/GCN (15), F10-mechanism runs (18), MUTAG and BBBP single splits.
- Re-training on the 24 collapsed Tox21 units and their clean partners (the
  lowest-seed clean unit of the same endpoint; 11 partners), each with the
  unit's original command, seed and subsample (PGExplainer only):
  (a) 30 epochs, lr 0.003 (current = PyG defaults); (b) 100 epochs;
  (c) 300 epochs; (d1) lr 0.0009; (d2) lr 0.009 (30 epochs each).
  Note: PyG's temperature schedule depends on epoch / epochs, so longer runs
  anneal more slowly. Logged per epoch: mean loss, gradient norm of the
  explainer MLP (read after each train call; PyG zeroes gradients at the start
  of the call), every 5 epochs and at the end: fraction of sigmoid edge outputs
  < 1e-3 or > 1 - 1e-3 on 50 training graphs; final mask std; collapse status.
- DIG: shown to be infeasible or run, with evidence: DIG's PGExplainer calls
  model.get_emb (our models have none) and injects masks via module._explain /
  module.__edge_mask__; a one-graph test checks whether that changes the output
  under PyG 2.8. PyG's original defaults equal setting (a).
- Edge embeddings: document and verify (number of get_embeddings outputs per
  model; which tensor [-1] is) for D-MPNN, GIN, GINE, GCN.

### T5 -- sufficiency (keep only the group)
- Same models and seeds as the reliance test; keep only S, hide the rest under
  R1 / R2 / R3; m(kept) vs random-connected of |S|; computed in the T1 pass.
  Compared with the earlier "sufficiency 5/0/0" (Part A: R1 only, uncorrected).

### T6 -- GCN negative null drop
- Checks: keep_nodes(G, all nodes) reproduces G's logits; sign convention in
  the code; per-molecule null-drop distribution split by predicted class;
  removal of nodes vs removal of their edges only (features kept); change of the
  graph-embedding norm; same quantities for D-MPNN/GIN/GINE. A bug is declared
  only if the identity test or the sign check fails; then it is fixed and the
  change reported.

## Time estimate

- T4 re-training: 35 units x about 490 epochs; measured 139 s per 30 epochs
  (Phase 3 re-runs, Tox21, 200 graphs, 4 concurrent) = about 38 min per unit;
  6 workers -> about 4-4.5 h wall-clock. Plus collapse-by-class and recount
  (~10 min) and the DIG / embedding checks (~15 min).
- T1/T5 pass (GT masks, 20+20 draws, 3 rules) on 40 units, and the explainer
  pass (random ties, GEF per order, random-node references) on 112 units:
  about 2-3 h wall-clock with 4 workers, run alongside T4 on GPU 1.
- T2, T3, T6 analyses: about 1 h.
- Total compute about 8 h; wall-clock about 5-6 h. Under the 10-hour limit.
  GPU 1 only; GPU 0 stays free.
