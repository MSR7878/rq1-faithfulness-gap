# RQ1 Tier 5: newer explainers. Plan

Written 2026-11-15, BEFORE any Tier-5 GEA or Fid number is computed. Looked up only:
- the authors' code (repositories, defaults);
- the interfaces of our models;
- which cached scores and checkpoints exist;
- molecule counts per unit.

Rules:
- Output goes to runs/rq1_tier5/ only.
- No paper .tex file is edited, and no existing result is overwritten (earlier tiers' files are read-only, including the Tier-2 R reset checkpoints).
- Code is committed before it runs. Push after each item and at the end.
- GPU 1 only (CUDA_VISIBLE_DEVICES=1), 12 workers.
- Order: N0, N4, N1, N2, N3, with no waiting between items. Stop only if a check fails or something breaks.

## Final protocol (everywhere)

- **GEA:** top-25% (k = max(1, floor(0.25 N))), random tie-breaking (20 orders, averaged per molecule; src/analysis/tier2_common.py).
- **Nulls:** random-node and random-connected, 20 draws (the Tier-2/Tier-4 null files where they exist, read only; otherwise the same rule into runs/rq1_tier5/nulls).
- **Second baseline:** the best trivial rule, fixed by name per dataset (below), with all three rules (hetero, high degree, low degree) shown in the tables.
- **Main number:** hierarchical bootstrap of the mean difference (seeds, then molecules; 2,000 reps; seed 2026; 95% percentile CI).
- **Support:** per-seed paired Wilcoxon + Holm counts (better / worse / n.s.). Molecules are never pooled across seeds.
- **Old methods:** GNNExplainer (stored cache), PGExplainer ranked by float64 logits, SubgraphX fixed budget, Saliency, IG, Input x Gradient, all from the existing Tier-2 / Tier-4 files.
- **Target:** the model's predicted class everywhere (as for every earlier method).
- **Seeding:** every new-explainer run on a molecule is seeded by crc32("<tag>|<unit>|<mol>") for python, numpy and torch. Main tag "T5"; N3 tags "T5a", "T5b", "T5c".

**Best trivial rule (fixed now, not re-selected per cell):** indole and PAINS: high degree (as in Tier 4 M1). Mutagenicity and MUTAG-GT: hetero.

## Explainers to add: code source and exact settings

The four repositories are kept outside git in third_party/ (gitignored). runs/rq1_tier5/third_party_manifest.json is committed with:
- the commit of each repository;
- the sha256 of every file used;
- the license status (GStarX and GNNShap have a LICENSE file; EiG-Search and SAME have none).

The authors' files are not edited. Every replaced or added function is in our adapter files (src/explain/t5_*.py) and listed in runs/rq1_tier5/patches.md, together with any import fix needed for PyG 2.8 (the repositories target PyG 2.1). The GStarX and SAME repositories both have top-level modules called utils, gnnNets and shapley. Each is loaded under its own package name, so they never collide.

**Model adapter (GStarX, SAME).** The authors' code passes node subsets around as `Data(x, edge_index)` and drops bond features. Our D-MPNN and GINE need them. Before calling their code, we append one column to x holding each atom's original index. Their code only slices or keeps rows of x, so the column follows the atoms through removal, relabelling and batching. The adapter strips the column, looks up each remaining bond's features by its (original u, original v) pair, and calls our `model(x, edge_index, edge_attr, batch)`. No model sees the extra column. N0 checks that the adapter's output equals the plain model on the full graph (max |difference| <= 1e-6) and on a removed-node subgraph built by hand.

### A. EiG-Search (Lu et al., ICML 2024). github.com/sluxsr/EiG-Search, commit b1e975d9061e8013a0911c69b9b120c53222313e

- **The authors' code is public.** The released code does not compute linear gradients. Its graph-classification branch (main.py) does the following:
  1. **Score each edge.** Remove one bond at a time (both directions; epsilon = 0) from the real model, and score it by log p(c | G) - log p(c | G without the bond).
  2. **Search the explanation size.** Rank edges by score and test the top 4, 6, 8, ... directed edges (2, 3, 4, ... bonds), up to max(2, int(0.3 E)) directed edges (sparsity 0.7, their default). Each candidate is scored by Fid+ minus Fid- in probability:
     - Fid+: the selected bonds are removed; atoms left without bonds are dropped.
     - Fid-: only the selected bonds and their atoms are kept.
  3. **Pick the best.** The candidate with the largest Fid+ - Fid- is the explanation.
- **Our port.** The authors' scoring needs a per-model `fwd()` written for their own GIN, GCN and SAGE. So we port the main.py graph-classification branch to our models, removing bonds physically (equivalent to edge weight 0 for removal). The search runs over undirected bonds: candidate sizes m = 2, 3, ..., M bonds with M = max(1, floor(0.3 B)), where B = number of bonds. The authors' directed-edge ordering assumes the two directions of a bond sit next to each other, and with our edge order they would not. An empty graph (possible only in very small molecules) gets probability 1/2 for the target.
- **Deviations:**
  - target = predicted class (the authors use the label);
  - every molecule is explained (the authors skip misclassified ones).
- **Node scores at our budget:** sum of the scores of the bonds incident to the atom, then top-25% with random ties.
- **Own size:** the atoms of the chosen bonds as a 0/1 set, scored as a set (as in Tier 4 L6).
- **Caveat:** EiG's search maximises a Fid objective, so its Fid numbers are not an independent check.
- **Randomness:** none expected (deterministic). N3 confirms identical output.

### B. SAME (Ye et al., NeurIPS 2023). github.com/same2023neurips/same, commit f0b4170b66b267f326451bd30eb46df063d7b595

- **Code used:** the authors' MCTS initialisation (initialization_mcts_GC.MCTS), the mc_l_shapley reward (shapley.py) and the permutation exploration (utils.find_explanations), with their explainer config (config/explainers/same.yaml).
- **Config block, fixed by a rule:** the authors' block for the closest dataset by source.
  - Mutagenicity and MUTAG-GT -> the "mutag" block: rollout 20, high2low True, c_puct 10, expand_atoms 20, local_raduis 5, sample_num 100, reward mc_l_shapley, subgraph_building split, single_explanation_size 2, candidate_size 7, exploration permutation.
  - B-XAIC indole and PAINS (drug-like molecules) -> the "bbbp" block: as above but expand_atoms 12, local_raduis 4, single_explanation_size 5.
- **Explanation size:** SAME.py sets the MCTS min_atoms and the final max_nodes to ceil(N (1 - sparsity)).
  - Own size: the config sparsity 0.5, i.e. ceil(0.5 N) atoms.
  - Our budget: a separate full run with min_atoms = max_nodes = k = floor(0.25 N).
  - So SAME is run twice per molecule, and both runs are budgeted.
- **Deviations:**
  - target = predicted class (the code uses data.y in the value function and in find_explanations);
  - no self-loops are added to edge_index (the authors add them for their own GCN/GIN; our models do not use them).
- **Node scores:** 0/1 on the chosen atoms. At our budget, ties are broken at random (20 orders) as for SubgraphX. At own size, scored as a set.
- **Randomness:** yes (Monte Carlo sampling in mc_l_shapley).

### C. GStarX (Zhang et al., NeurIPS 2022). github.com/ShichangZh/GStarX, commit 770b2864e9adcacbf044c2bdc27fc85825e646e5

- **Code used:** the authors' GStarX class (gstarx.py) and its helpers (utils.py).
- **Defaults:** their config for the molecular datasets (mutag, bbbp and bace are identical): payoff_type norm_prob, max_sample_size 10, tau 0.01, subgraph_building_method remove, superadditive_ext False, sample_method random, num_samples -1 (= N).
- **payoff_avg:** the mean predicted class probabilities over the unit's own test split (as run_gstarx.py does), recomputed for the reset models in N2.
- **Node scores:** GStarX returns a real score per atom; we take top-25% with random ties. It has no own size: the authors' sparsity 0.5 is only used in their evaluation.
- **Randomness:** yes (randperm in random sampling, seeded per molecule).

### D. GNNShap (Akkas and Azad, WWW 2024). github.com/HipGraph/GNNShap, commit f9672297394493ebe1ea9cf60bd14530e06d4916 (lowest priority)

- **The released code explains only node classification.** We use the authors' GNNShapSampler (CUDA extension cppextension/cudagnnshap.cu) and WLSSolver inside **our own graph-level wrapper**:
  - players = undirected bonds; a coalition keeps its bonds (both directions), and the other bonds are removed (atoms stay);
  - value = predicted-class probability;
  - fnull = all bonds removed; ffull = the full graph;
  - num_samples 10000 (their default), batch_size 1024.
- **Node scores:** sum of the Shapley values of the incident bonds, then top-25% with random ties. No own size.
- **Half-day cap:** if the CUDA extension does not build on ada with the available nvcc (11.0 / 11.3 vs torch built for CUDA 11.8), or the wrapper does not pass N0 within half a day (4 working hours from the start of the GNNShap integration), GNNShap is skipped and the report says so. The sampler is not re-implemented in Python.
- **Small molecules:** when 2^B <= num_samples, the behaviour of the sampler is reported in N0. Molecules where it fails are listed, not dropped silently.
- **Randomness:** yes (sampling).

## Scale rule (fixed now; applied mechanically after the N0 timings)

**Levels** (molecules per unit, ground-truth-bearing explained molecules only):
- **L0:** all GT molecules, all seeds.
- **L1:** the fixed 100-molecule subsets, all seeds.
- **L2:** the 100-molecule subsets, seeds 0-4.
- **L3:** the first 50 molecules of each subset (subset order shuffled by rng([83500, crc32(unit)])), seeds 0-4.
- **L4:** the first 25 molecules of each subset in that order, seeds 0-4.

**Projection.** In N0 each new explainer is timed on 6 molecules per model (the 2 smallest, 2 median and 2 largest GT molecules of seed 0 of each dataset), one worker. The projected N1 hours of a method are (sum over its N1 molecules of the mean seconds for that dataset and model) / (12 x 3600). For SAME this counts both runs.

**Selection.** Each method gets the first level whose projection is <= 24 h. If even L4 is over 24 h, L4 runs anyway and the report says so. The chosen level is written into runs/rq1_tier5/scale.json and committed (still before any GEA number) as an addendum to this PLAN. N2 and N3 use the same molecules as the method's level (the Tier-2 R subsets, or the same slice of them).

**Subsets.**
- Existing: indole and PAINS seeds 0-4 from runs/rq1_tier4/subsets.json; Mutagenicity seeds 0-4 and MUTAG-GT seed 0 from runs/rq1_tier2/subsets.json. The Mutagenicity GIN, GINE and GCN units explain exactly the same molecules as the D-MPNN in every seed (checked), so they share its subset, as M1 did.
- New, by the same subset100 rule into runs/rq1_tier5/subsets.json: indole seeds 5-9, PAINS seeds 5-9, Mutagenicity seeds 5-9, MUTAG-GT seeds 1-9.

**Molecule counts per unit (GT-bearing):**
- B-XAIC: 360;
- Mutagenicity: 124-128;
- MUTAG-GT: 188 (all molecules, as stored).

## Units

**N1 units:**
- B-XAIC indole and PAINS: D-MPNN seeds 0-9 (Tier 2 S/M and Tier 4 M2 models), plus GIN, GINE and GCN seeds 0-4 (Tier 4 M1 models; 19 of 30 miss the accuracy bar and are flagged).
- Mutagenicity: D-MPNN seeds 0-9, and GIN, GINE and GCN seeds 0-4.
- MUTAG-GT: seeds 0-9 (the invalid-benchmark contrast).

That is 85 units in total (fewer at L2-L4).

**Old-method sources:**

| Method | Source |
|---|---|
| GNNExplainer | stored caches |
| PGExplainer | runs/rq1_revision/logits or the unit's Tier-4 gp file |
| SubgraphX fixed | runs/rq1_tier2/S/sx, runs/rq1_tier2/M/sx, runs/rq1_tier4/M1/sx, runs/rq1_tier4/M2/sx |
| Saliency, IG, IxG | runs/rq1_tier2/G/grad, runs/rq1_tier2/M/grad, runs/rq1_tier4/M1/grad, runs/rq1_tier4/M2/grad |

## Items and exact settings

### N0. Integration check (before any scoring)

For each new explainer x each model (D-MPNN, GIN, GINE, GCN; seed-0 Mutagenicity models, plus the B-XAIC D-MPNN), on the 6 timing molecules:
- scores are finite and not constant;
- the perturbation reaches the model: the output for the empty input differs from the full input, under the method's own perturbation (removal for EiG and GStarX, split for SAME, bond removal for GNNShap);
- run time per molecule.

Checks that the PLAN requires to pass (a failure stops the chain for that explainer x model; it is reported, and the rest continue):
- the adapter equals the plain model on the full graph (<= 1e-6), and on a removed-node subgraph built by hand;
- D-MPNN and GINE with zero bonds and with a single atom, under remove and split;
- a batched coalition evaluation with bond features carried through equals one-by-one evaluation;
- EiG's two-graph evaluation equals removing the atoms directly.

If a method cannot be made to work with a model, this is reported, and N1-N3 run without that cell.

Output: runs/rq1_tier5/N0/integration.json and N0.txt; then scale.json.

### N4. Combinations from cached scores (no new explanations)

- **Cells:** indole, PAINS and Mutagenicity, each x D-MPNN, GIN, GINE, GCN (12 cells), on the molecules where SubgraphX exists:
  - Tier-2 S/M D-MPNN units: all GT molecules;
  - M2: all;
  - M1 units: the 100-molecule subset.
- **Per molecule** (GEA at top-25%): share of molecules where GNNExplainer > SubgraphX, SubgraphX > GNNExplainer, ties; and where IG > both. Per seed, then the mean over seeds with a bootstrap CI.
- **Combinations:**
  - rank-average of (GNNExplainer, SubgraphX, IG) and of (GNNExplainer, SubgraphX). Per molecule, each method's atom scores are turned into average ranks (scipy rankdata "average") divided by N, then averaged over the methods; top-25% with random ties.
  - IG smoothed over neighbours: s'_i = mean of s_i and s_j over the bonded neighbours j (one step).
- **Each combination** is compared with random-node, with the dataset's best trivial rule, and with the best single method.
- **Best single method:** among the six old methods (GNNExplainer, PG logit, SubgraphX fixed, Saliency, IG, IxG), the one with the highest mean GEA in that cell on the same molecules. Seeds: all that exist (indole and PAINS D-MPNN 10, Mutagenicity D-MPNN 10, others 5).
- Output: runs/rq1_tier5/N4/<unit>.npz, N4.txt / N4.json.

### N1. Main comparison

For every N1 unit, on the molecules of each new method's level:
- **New explainers:**
  - EiG at our budget and at its own size;
  - SAME at our budget and at its own size;
  - GStarX;
  - GNNShap.
- **Old methods**, on the same molecules: GNNExplainer, PG (logit), SubgraphX fixed, Saliency, IG.
- **Each method** is compared with random-node, random-connected and each trivial rule. Ordering by mean GEA per dataset and model.
- **Fid+ and Fid-, probability and logit, R2 and R3:**
  - methods: all methods at our budget (not the own-size sets), plus random nodes (20 random rankings);
  - molecules: each unit's 100-molecule subset, or the method level's slice if smaller, the same molecules for all methods in that group;
  - code: src/analysis/tier2_fid.run_fid (Part-E protocol, the unit's R3 fill);
  - layout as in Tier 2 G/H.
- Output: runs/rq1_tier5/N1/{expl,gea,fid}/, N1.txt / N1.json.

### N2. Sanity check (full weight reset; output-layer reset as a secondary view)

- **Models:** Mutagenicity D-MPNN and B-XAIC indole, seeds 0-4. The Tier-2 R checkpoints runs/rq1_tier2/R/ckpt/<unit>_{full,head}.pt are read only.
- **Molecules:** the Tier-2 R 100-molecule subsets (or the method level's slice).
- **Methods:** each new explainer, re-run on the reset models with the same settings and the same per-molecule seeds. GStarX's payoff_avg comes from the reset model's own test predictions.
- **Reported:**
  - GEA(trained) - GEA(reset) with a 95% CI (trained = the N1 run on the same molecules);
  - per-seed Wilcoxon + Holm;
  - Spearman between trained and reset atom scores (mean per seed; CI);
  - Jaccard of the two top-k sets, needed because Spearman is degenerate for 0/1 sets.
- Output: runs/rq1_tier5/N2/, N2.txt / N2.json.

### N3. Stability

- **Subsets:** the fixed 100-molecule subsets of Mutagenicity D-MPNN s0, indole s0 and MUTAG-GT s0 (or the method level's slice).
- **Methods with randomness** (SAME at our budget, GStarX, GNNShap): 3 runs with tags T5a, T5b, T5c. Reported as in Tier 4 L6:
  - per-molecule Jaccard of the top-k sets between runs;
  - share of atoms picked in >= 2 of 3 runs;
  - consensus GEA.
- **EiG:** 2 runs; the output must be identical.
- Output: runs/rq1_tier5/N3/, N3.txt / N3.json.

## Predictions (fixed now; not changed later; failures reported as failures)

"New explainer" means the fixed-budget (top-25%) version of EiG, SAME, GStarX and GNNShap. The own-size versions of EiG and SAME are reported next to them but do not count for any prediction. A method that could not run in a cell is reported as missing; a prediction that needs it is then "not decided" for that part.

- **P-N1:** on B-XAIC with the D-MPNN (all seeds of the method's level), at least one new explainer has, on BOTH indole AND PAINS, CI(method - random-node) entirely above 0 AND CI(method - high degree) entirely above 0. It must be the same explainer on both datasets.
- **P-N2:** on Mutagenicity D-MPNN, for EVERY new explainer, CI(method - hetero) includes 0 or is entirely below 0.
- **P-N3:** cells: indole, PAINS and Mutagenicity, each x D-MPNN, GIN, GINE, GCN (12). Explainers:
  - old: GNNExplainer, PGExplainer (logit), SubgraphX fixed, Input x Gradient;
  - new: the four fixed-budget versions.

  IG and Saliency are excluded. **Held** if NO explainer has CI(method - random-node) entirely above 0 in all 12 cells. Old methods use their existing primary numbers, each on its registered molecules: Tier 2 S/G/M, Tier 4 M1/M2 (SubgraphX on the 100-molecule subset for M1). New methods use N1. "New methods only" is reported as a secondary row and does not replace the prediction.
- **P-N4:** every new explainer has CI(GEA trained - GEA full reset) entirely above 0, on BOTH Mutagenicity D-MPNN and indole (N2).
- **P-N5:** in each of the 12 valid cells (as in P-N3), CI(rank-average of GNNExplainer, SubgraphX, IG - best single method) includes 0 or is entirely above 0 (N4).

## Holm families

- **N1 GEA:** per unit, (methods scored in that unit, including the own-size versions) x 2 nulls; a separate family per unit for "minus trivial rule" (methods x 3 rules).
- **N1 Fid:** per (unit, space), methods x {Fid+, Fid-} x {R2, R3}.
- **N2:** per (dataset, reset kind, seed), the new-explainer versions (trained - reset).
- **N4:** per unit, 3 combinations x 3 comparisons (random-node, trivial rule, best single).
- **N3 and the N4 win shares:** descriptive (bootstrap CIs only).

## Time estimate (12 workers on GPU 1; ada is shared, so this may drift)

The N1-N3 times depend on the N0 timings and are capped by the scale rule at about 24 h per method for N1.

| Item | Work | Estimate |
|---|---|---|
| N0 | integration, adapter checks, timing; GNNShap build (capped at half a day) | 4-8 h |
| N4 | scoring from cached files | 1 h |
| N1 | four explainers (SAME twice) on up to 85 units; scoring; Fid | 30-90 h (scale rule) |
| N2 | four explainers on 2 x 5 reset models x 2 reset kinds x <= 100 molecules | 5-20 h |
| N3 | 3 runs x 3 subsets for SAME, GStarX, GNNShap; 2 runs for EiG | 3-15 h |
| Reports | | 3 h |
| Total | | about 45-135 h |
