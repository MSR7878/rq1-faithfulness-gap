# RQ1: ground-truth reliance (Part A) and logit / random-corrected fidelity (Part B)

Plan, registered predictions and scoring rules. Written 2026-10-19 and
committed BEFORE any value of either analysis was computed (including the
smoke test). Outputs go only to runs/rq1_reliance_fidelity/. No training, no
new explanations, no changes to existing results or splits.

## Common definitions

- Margin of a graph H for class c: m(H) = logit_c(H) - logit_other(H) (raw
  logits of the 2-class head, never softmax). c is the model's predicted class
  on the full graph G (= the class every cached explanation was made for).
- Models, splits, seeds, explained molecules (test_idx in each explanation
  cache) and mol_idx are exactly those of the existing runs.
- Masked graphs use the repo's own masking code: R1 = hard removal (nodes
  and incident edges deleted, survivors reindexed), R2 = zero-fill of node
  features (topology kept), R3 = training-set mean fill (topology kept; the
  fill vector is the one stored in each run's output json, "fills"/R3/mean).
- Random draws: 20 per molecule; random-node sets are uniform; random
  connected sets use connectivity_baseline.random_connected_subset (frontier
  growth, as for the existing RandConn nulls). Seeded per unit with
  numpy default_rng([60000, unit hash, seed]) -- separate streams for the
  node and the connected draws.
- Statistics: per seed (per unit), paired two-sided Wilcoxon over molecules,
  zero_method="wilcox"; Holm within the unit over the family stated below;
  direction = the direction of the rank test (W+ vs W-). Never pooled over
  seeds. Units are counted above / below / n.s. over the 5 seeds.

## Part A -- does the model rely on the annotated group?

Datasets and models: MUTAG-GT (D-MPNN, the 5 priority-sweep checkpoints,
188 explained graphs each); Mutagenicity (D-MPNN, GIN, GINE, GCN; test split
<= 80 atoms, GT-bearing); B-XAIC indole, PAINS, X, P (D-MPNN, priority-sweep
checkpoints; GT-bearing explained molecules = the 360 positives). 5 seeds each.

Valid GT masks S of a molecule:
- Mutagenicity: each toxicophore occurrence in meta.extras["motifs"] (the
  GraphXAI set used for max-over-set GEA).
- MUTAG-GT and B-XAIC: the loaders give one merged node mask; its connected
  components (on the molecular graph) are taken as the separate masks.
- Every quantity is computed per mask and averaged over the masks of the
  molecule. A mask covering the whole molecule is skipped (cannot be removed
  under R1); counts of skipped masks are reported.

Per molecule:
- delta_GT = m(G) - m(G with S removed), under R1 and under R2.
- Nulls: the same quantity for 20 random node sets and for 20 random
  connected sets of size |S|; null = their mean.
- Reliance gap = delta_GT - null (per molecule), for each null.
- Exploratory (not registered): sufficiency = m(G keeping only S, R1) vs the
  mean over 20 random connected sets of size |S| kept alone (R1).

Tests per unit: gap vs 0 for {R1, R2} x {random-node, random-connected} =
4 tests, Holm over the 4. Sufficiency (exploratory): 1 test per unit,
uncorrected, reported separately.

REGISTERED PREDICTIONS
(A1) Mutagenicity, D-MPNN: delta_GT is significantly above the
     random-connected null in at least 4 of 5 seeds, under R1 AND under R2
     (each counted separately; A1 holds only if both reach 4/5).
(A2) MUTAG-GT, D-MPNN: delta_GT is NOT significantly above the
     random-connected null in at least 3 of 5 seeds. Operationalisation fixed
     here: counted separately under R1 and under R2; A2 holds only if it holds
     under both. (Rationale given with the prediction: the model may rely on
     the aromatic scaffold, not the nitro/amino group.)
Nothing else in Part A is a registered prediction.

## Part B -- logit-based and random-corrected fidelity

Cached explanations used (GNNExplainer, PGExplainer, SubgraphX), all existing:
- D-MPNN: MUTAG (single split, runs/ckpt_mutag.pt + runs/expl_cache_mutag.pt);
  BBBP (single split, runs/ckpt_bbbp.pt + runs/expl_cache_bbbp.pt);
  MUTAG-GT, 5 seeds (runs/prio/, the caches behind the paper's GEA figures);
  Tox21, 12 endpoints x 5 seeds (runs/prio/); B-XAIC indole, PAINS, X, P,
  5 seeds (runs/prio/); Mutagenicity, 5 seeds (runs/mutagenicity/).
- GIN, GINE, GCN on Mutagenicity, 5 seeds each (runs/arch_<arch>/).

Per molecule, per explainer, under R1, R2 and R3 (same top-k 0.25 node
selection and the same masking calls as src/explain/run_phase3.py):
- Fid+ = m(G) - m(G \ E), Fid- = m(G) - m(E) (logit margin in place of the
  softmax probability; definitions otherwise unchanged). The probability
  versions are recomputed in the same pass and checked against the stored
  per-molecule values of each run; a mismatch stops the run.
- Random references: the same two quantities with the explainer's importance
  replaced by (i) 20 uniformly random node rankings and (ii) 20 rankings that
  put a random connected set of the top-k size first (ties beyond it broken
  at random). So the kept / removed sets have exactly the explainer's sizes.
- Random-corrected value = explainer value - mean over the 20 draws, against
  each reference.

Tests per unit: random-corrected value vs 0, for 3 explainers x 3 maskings x
2 metrics = 18 tests per reference; Holm within the unit over those 18 (one
family per reference). Direction counts over seeds. MUTAG and BBBP (single
split): means and per-molecule values only, reported descriptively. Tox21:
counts over the 5 seeds of each endpoint, summarised over the 12 endpoints.

REGISTERED PREDICTION
(B1) The probability-space result "R3 is inert on diffuse rules" is caused
     by softmax saturation: on the four diffuse datasets (MUTAG-GT, BBBP,
     Tox21, Mutagenicity) the std/mean of logit-based Fid+ under R3 is below
     1 for each explainer, against 2-5 in probability space.
     Operationalisation fixed here: per unit, CV = sd / |mean| of the
     per-molecule logit Fid+ under R3 (mean fill) over the explained
     molecules; the dataset value is the median CV over its units (MUTAG-GT
     5, BBBP 1, Tox21 60, Mutagenicity 5 = the D-MPNN runs). B1 holds if this
     value is < 1 for all 3 explainers on all 4 datasets (12 cells). The
     probability-space CV is computed the same way and reported next to it;
     B1's "against 2-5" part is reported as observed, not scored.
Everything else in Part B is descriptive.

## Comparison with the GEA result (report item 3, descriptive)

For each unit that has GEA (MUTAG-GT, B-XAIC x4, Mutagenicity x4 models)
and each explainer: the GEA direction vs RandNode at the top-k budget
(existing per-molecule arrays; Holm over the 3 explainers within the unit)
is set against the direction of random-node-corrected logit Fid+ under R3.
Every cell whose three-way direction (+ / . / -) differs is listed.

## Rules

No tuning to results; no amendments after scoring starts. If anything looks
wrong (e.g. the stored probability fidelity does not reproduce), the run
stops and is reported. Per-molecule arrays with mol_idx per seed are saved.
Nothing is pushed. A DONE file is written when the run finishes.
