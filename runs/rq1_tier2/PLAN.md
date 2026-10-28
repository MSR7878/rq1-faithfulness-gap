# RQ1 Tier 2 (items H, G, R, S, M): plan, settings, predictions, time estimate

Written 2026-10-28, after a timing test (timings only) and BEFORE any GEA / Fid number of Tier 2 is computed.
No paper .tex file is edited. Nothing under runs/ outside runs/rq1_tier2/ is overwritten.
GPU 1 only (GPU 0 stays free), 12 single-thread workers.

## Order (approved by the user, run without waiting between items)

H -> G -> R -> S priority (MUTAG-GT 10 seeds, Mutagenicity D-MPNN, B-XAIC indole) -> S sensitivity
(one setting at a time) -> M -> S remainder (B-XAIC PAINS, X, P; Mutagenicity GIN, GINE, GCN).
Commit code before running it; commit, push and a short note after each item; DONE + one full report at the end.
If anything fails or a check is wrong: stop and report.

## Final protocol (every Tier-2 number; code: src/analysis/tier2_common.py)

- Units: Mutagenicity D-MPNN / GIN / GINE / GCN (5 seeds each), B-XAIC indole / PAINS / X / P (5 seeds each)
  = the valid benchmarks; MUTAG-GT (10 seeds) = the contrast case. Molecules: each unit's explained set
  (the stored cache's test_idx; Mutagenicity 350, B-XAIC 400, MUTAG-GT 188 per seed). GEA on molecules with
  ground truth only; Fid on all explained molecules.
- GEA: top 25%, k = max(1, floor(0.25 N)); exact ties at the k boundary broken at random, 20 seeded orders
  averaged per molecule (stream "<unit>|<method>|gea"). Jaccard against the merged ground truth (MUTAG-GT,
  B-XAIC); max-over-set over the GraphXAI motifs (Mutagenicity) -- as in every earlier table.
- Nulls: random-node (20 uniform k-sets) and random-connected (20 connected k-sets), mean GEA per molecule,
  drawn ONCE per unit (seeded by the unit name) and shared by every method in every item. The three explainers
  are re-scored on the same nulls, so Tier-2 numbers for GNNExplainer / PGExplainer / SubgraphX can differ
  slightly from T3; only a change of CI category (above 0 / below 0 / includes 0) counts as a contradiction.
- PGExplainer ranked by logits. Stored caches: ranked by the stored mask; exact ties split by the float64
  logit-based node scores of the T4-addendum re-run, but only on molecules whose re-run reproduces the stored
  edge mask (max |diff| <= 1e-5); remaining ties random. The number of tie-boundary molecules that fall back to
  random ties is reported. New PGExplainer runs (R, M) rank by their own float64 logits.
- Main number: hierarchical bootstrap of the mean difference (method - null, or method - method):
  seeds, then molecules within seed; 2,000 reps; seed 2026; 95% percentile CI. CI category: above / below /
  includes 0. Support: per-seed paired Wilcoxon (two-sided) + Holm; counts better / worse / n.s.
  Molecules are never pooled across seeds.
- Fid: Fid+ and Fid- (and GEF) under R1, R2, R3, probability space (primary) and logit space, with the Part-E
  protocol (top-k with random tie orders; R3 = training-mean fill). Null: the random-node ranking of Part E
  (runs/rq1_revision/pass/E, same molecules); new units (M) draw their own 20 random rankings.
  Holm family per (unit, method, space) = Fid+/Fid- x R1/R2/R3 = 6 tests.

## Decisions and deviations recorded before running

- Captum was not installed on ada and ada cannot reach PyPI. captum-0.7.0 (pure-Python wheel) was downloaded on
  the local machine and installed with --no-deps; torch 2.6.0 and numpy 1.26.4 unchanged.
- SubgraphX budget fix (src/explain/dig_subgraphx.py, option budget="score", default unchanged = legacy):
  old: max_nodes = max(3, round(0.25 N)), MCTS leaf size min_atoms = 3;
  new: max_nodes = max(1, floor(0.25 N)) = the scoring k, min_atoms = min(3, k).
  Asking for k does not force |S| = k (DIG returns the best coalition with <= k atoms); per molecule the
  coalition size is recorded and the shares |S| = k, < k, > k are reported for old and new. In the timing test
  (12 molecules each): |S| = k in 3/12 Mutagenicity, 10/10 indole, 10/12 MUTAG-GT; never > k.
- SubgraphX in Tier 2 is seeded per molecule (crc32 of "<tag>|<unit>|<molecule>"), so results do not depend on
  sharding. rollout 20, sample_num 30 unless stated (as before).
- R and M: GNNExplainer and PGExplainer are run through src.explain.run_phase3 with each unit's original
  arguments and explainer seed (so the explained set and the PGExplainer training set are the original ones);
  the PGExplainer MLP is saved and evaluated in float64.
- Heuristics: Mutagenicity has explicit H atoms (they are neither heteroatoms nor counted in the heavy-atom
  degree); B-XAIC "Unk" counts as a heteroatom. Note: the B-XAIC X and P ground truths are single heteroatoms by
  construction, so the heteroatom heuristic is near the ceiling there by definition.
- 100-molecule subsets (R; S sensitivity): fixed random 100 ground-truth molecules of the unit's explained set,
  rng([83000, crc32(unit)]); written to runs/rq1_tier2/subsets.json and committed with this plan. (MUTAG-GT s0
  has 188 GT molecules; Mutagenicity D-MPNN about 127 per seed; indole about 360.)

## Items

### H. Heuristic baselines (no model needed for the mask)
- hetero (1 for any atom not C or H), deg_high (number of heavy-atom neighbours), deg_low (minus that).
- GEA per group vs random-node and random-connected: bootstrap CI + counts; Holm family per seed =
  3 heuristics x 2 nulls = 6. Heuristic minus each explainer (GNN, PG, SX legacy budget): bootstrap CI + counts;
  Holm family per seed = 3 heuristics x 3 explainers = 9. Fid vs random-node as above.

### G. Gradient methods (Captum 0.7.0)
- Saliency (|grad|), Integrated Gradients (50 steps, zero baseline, Captum's default Gauss-Legendre rule),
  Input x Gradient; target = the predicted-class logit; model in eval mode; node attribution = sum over atom
  features of |attribution|. Wrapping (all four models): Captum sees the graph as one example x[None] of shape
  (1, N, F); the forward function reshapes a (B, N, F) input into a disjoint batch of B copies of the graph
  (same bonds, same bond features) and calls model(x, edge_index, edge_attr, batch) -> (B, 2).
- Checks (stop if they fail): every attribution finite and non-zero on every molecule; IG completeness:
  median over molecules of |sum attr - (f(x) - f(0))| / |f(x) - f(0)| <= 0.05 per unit (max reported).
- GEA vs nulls: Holm family per seed = 3 methods x 2 nulls = 6; vs each explainer: family per seed = 3 x 3 = 9.

### R. Randomised-model sanity check (Adebayo et al. style)
- Mutagenicity D-MPNN s0-4 and B-XAIC indole s0-4. "full" = reset_parameters() on every module;
  "head" = only the output FFN; seed 5000 + model seed; saved as new checkpoints; test accuracy / AUROC reported.
- Re-explain: GNNExplainer, PGExplainer (all explained molecules, via run_phase3), IG and Saliency (all explained
  molecules), SubgraphX (scoring budget) on the fixed 100-molecule subset per seed.
- GEA vs random-node and random-connected (bootstrap CI; Holm family per (dataset, randomisation, seed) =
  5 methods x 2 nulls = 10). Spearman correlation of trained vs randomised atom scores per GT molecule (mean
  per seed; bootstrap CI) and the Jaccard of the two top-k sets. SubgraphX column: trained = the fixed-budget
  S run on the same molecules, so it is filled after S priority (said in the R note).

### S. SubgraphX budget fix and search sensitivity
- Re-run SubgraphX (scoring budget) on every explained molecule: priority MUTAG-GT s0-9, Mutagenicity D-MPNN
  s0-4, indole s0-4; remainder PAINS, X, P, GIN, GINE, GCN (last).
- Old (legacy budget, stored) vs new GEA side by side on the same nulls: CI, counts, every change of CI
  category or per-seed count flagged; coalition-size shares; Fid of the new runs. Holm family per seed =
  SX vs {random-node, random-connected} = 2 (old and new each).
- Sensitivity on the fixed 100 GT molecules of Mutagenicity D-MPNN s0 and MUTAG-GT s0: base = rollout 20 /
  sample 30 (the S run on those molecules); rollout 50 and 100 at sample 30; sample 100 at rollout 20; plus a
  repeat of 20/30 with a different seed tag as the Monte-Carlo noise floor. Report mean GEA, mean run time per
  molecule, and the share of molecules whose selected subgraph differs from the base.

### M. More seeds
- Train seeds 5-9 with the exact original commands (kept; test accuracy and AUROC reported):
  Mutagenicity D-MPNN: --depth 3 --epochs 120 --hidden 300 --val-frac 0.10 --test-frac 0.20 --max-nodes 80
  explanation pool, explainer seed 1000 + S; indole: --depth 3 --epochs 25 --batch-size 512 --hidden 256,
  explanation args --limit 400 --stratify-frac 0.9 --max-nodes 80 --pg-train-limit 200, explainer seed S
  (as for seeds 0-4, which used the coupled seed).
- Explain: GNNExplainer, PGExplainer (run_phase3; logits saved), SubgraphX (scoring budget), IG, heteroatom.
- 10-seed CIs, counts (Holm family per seed = 5 methods x 2 nulls = 10) and a sign test over seeds (binomial,
  seeds with mean difference > 0) next to the 5-seed numbers (seeds 0-4, same protocol, SX = the S run).

## Predictions (wording fixed by the user; scoring rule underneath; fixed before running)

P-H1: the heteroatom heuristic is above random-node GEA (CI excludes 0) on Mutagenicity (all four models) and on
MUTAG-GT.
- Held if the 95% CI of hetero minus random-node is entirely above 0 in all five groups (Mutagenicity D-MPNN,
  GIN, GINE, GCN; MUTAG-GT 10 seeds).

P-H2: on Mutagenicity D-MPNN the heteroatom heuristic's mean GEA is at least as high as one of the three
explainers'.
- Mean GEA = mean over seeds of the per-seed mean (the bootstrap point estimate). Held if hetero >= the lowest of
  GNNExplainer, PGExplainer, SubgraphX. Scored with the stored (legacy-budget) SubgraphX, the only one that
  exists when H runs; the value with the fixed-budget SubgraphX is given as a secondary line after S.

P-G1: Integrated Gradients is above random-node GEA on Mutagenicity D-MPNN (CI excludes 0).
- Held if the 95% CI of IG minus random-node is entirely above 0.

P-R1: on randomised models, every explainer's GEA is within +/-0.05 of random (sanity check passes).
- 20 cells = 5 methods (GNN, PG, SX, IG, Saliency) x 2 datasets x 2 randomisations. Held if, in every cell, the
  bootstrap point estimate of (method - random-node) lies in [-0.05, +0.05]. The 90% CI (equivalence view) is
  reported next to it.

P-S1: with the matched SubgraphX budget, every SubgraphX direction on valid benchmarks stays the same, and
MUTAG-GT stays below random.
- Direction = CI category of (SX - random-node) top-k GEA for each of the 8 valid groups, old (legacy, stored)
  vs new (scoring budget), on the same nulls. Held if all 8 categories are unchanged AND the new MUTAG-GT CI upper
  bound is below 0. A partial score is given after S priority (Mutagenicity D-MPNN, indole, MUTAG-GT); the final
  score after the remainder.

P-S2: going from 20 to 100 rollouts changes SubgraphX mean GEA by less than 0.03.
- On each of the two sensitivity subsets: |mean GEA (rollout 100, sample 30) - mean GEA (rollout 20, sample 30)|
  < 0.03. Held if true on both.

P-M1: with 10 seeds, the sign of every CI on Mutagenicity D-MPNN and indole matches the 5-seed result.
- Per (dataset, method) for GNN, PG, SX (fixed budget), IG, heteroatom vs random-node: CI category (above /
  below / includes 0) with seeds 0-9 equals the one with seeds 0-4. Held if all 10 match.

## Time estimate (from the timing test: 12 single-thread SubgraphX workers, fixed budget)

Measured on ground-truth molecules: 212 s/molecule Mutagenicity D-MPNN (mean N 31), 152 s indole (N 27),
94 s MUTAG-GT (N 17); that is about 1.4-2x the per-molecule times of the old runs, with 12 running at once.
Gradients: 0.5 s/molecule. Estimates use the old per-molecule means x 1.7, 12 workers:

| Item | Work | Wall-clock |
|---|---|---|
| H | nulls, re-scored explainers, heuristics, Fid on 50 units | about 0.5 h |
| G | gradients + Fid on 50 units | about 0.5 h |
| R | 20 randomised models; GNN/PG/IG/Saliency on all explained molecules; SubgraphX on 20 x 100 | about 10 h |
| S priority | SubgraphX on 1,750 Mutagenicity + 2,000 indole + 1,880 MUTAG-GT molecules | about 20 h |
| S sensitivity | 4 extra settings on 2 x 100 molecules (about 12x the base cost) | about 7 h |
| M | train 10 models, explain 1,750 + 2,000 molecules with all methods | about 18 h |
| S remainder | SubgraphX on 6,000 B-XAIC + 5,250 Mutagenicity GIN/GINE/GCN molecules | about 41 h |
| analysis and reports | | about 4 h |
| Total | | about 100 h (range 80-125 h), about 4 days |

ada is shared (about 14 of 28 cores busy with other users), so the per-molecule time may drift.
