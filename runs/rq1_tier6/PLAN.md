# RQ1 Tier 6 (small). Plan

Written 2026-11-19, BEFORE any Tier-6 number is computed. Looked up only data facts (atom counts, aromatic atoms, where the
ground truth sits), which scores and checkpoints exist, and past run times.

Rules:
- Output goes to runs/rq1_tier6/ only. Earlier tiers' files are read only. No paper .tex file is edited.
- Code is committed before it runs. Push after each item and at the end.
- GPU 1 only (CUDA_VISIBLE_DEVICES=1), 8 workers (the N1 setting after its out-of-memory restart).
- Order: T1, T2, T3, T4, with no waiting between items. Stop only if a check fails or something breaks.

## Final protocol (everywhere, as in Tier 5)

- **GEA:** top-25% (k = max(1, floor(0.25 N))), random tie-breaking (20 orders, averaged per molecule).
- **Main number:** hierarchical bootstrap of the mean difference (seeds, then molecules; 2,000 reps; seed 2026; 95% percentile CI).
- **Support:** per-seed paired Wilcoxon + Holm counts (better / worse / n.s.). Molecules are never pooled across seeds.
- **Target:** the model's predicted class (for T2(b): the predicted class of the graph with the scaffold hidden).
- **New-explainer settings:** exactly as in Tier 5 (runs/rq1_tier5/PLAN.md), per-molecule seed crc32("<tag>|<unit>|<mol>").
- **Old methods:** GNNExplainer (200 epochs), SubgraphX fixed budget (DIG, rollout 20, sample 30, budget "score"),
  Saliency / IG (50 steps, zero baseline) / Input x Gradient (Captum), as in Tier 2.

## Data facts used by the rules (looked up, not results)

- MUTAG-GT: all 188 molecules are explained in every seed (10 seeds), and all 188 have ground-truth atoms.
- Aromatic scaffold = all atoms on an aromatic bond (bond-type column 0, as in Tier 4 L4).
  - 14 molecules have no aromatic atom (the same 14 in every seed: 12-16 atoms, 3 ground-truth atoms each).
  - No molecule is all aromatic. 174 molecules have both.
  - In all 174, **no ground-truth atom is aromatic**. So the ground truth is entirely in the remaining atoms.
  - After hiding the scaffold, 3-14 atoms remain (mean 5.9). With k = max(1, floor(0.25 n_remaining)), k = 1 for 78% of
    molecules.
  - Under R1 every remaining graph has at least 2 bonds; 55 of the 174 have at least one atom left with no bond.

## Predictions (fixed now; not changed later; failures reported as failures)

**P-T2** (given in the request): "EiG and GNNShap move above random once the rings are unavailable."

Operational version: on MUTAG-GT D-MPNN (10 seeds), GEA minus random-node has a CI entirely above 0 for EiG and for GNNShap
in each of the three conditions (a) molecules with no aromatic atom, (b)-R1 and (b)-R2 (primary k). Held = all 6 cells.
The "before" is the Tier 5 N1 result on all molecules (full graph).

T1, T3 and T4 are exploratory: no prediction.

## T1. Aromatic share of the chosen atoms (MUTAG-GT D-MPNN, 10 seeds; Tier 4 L4 overlap format)

- Methods: EiG, GNNShap, SAME, GStarX, Saliency, IG. Chosen atoms = the top-k set (random ties, 20 orders, averaged).
  EiG's own-size set is shown as a secondary row (scored as a set).
- Molecules: each method's molecules, minus the 14 with no aromatic atom.
  - EiG, GNNShap, Saliency, IG: all 174, seeds 0-9 (N1 runs; Tier 2 G gradient files).
  - GStarX: the 100-molecule subset; SAME: the 25-molecule slice of it. Seeds 0-4 are the N1 runs.
    **Seeds 5-9 are new runs** (runs/rq1_tier6/T1/expl/): same subsets (runs/rq1_tier5/subsets.json, slice rule of Tier 5),
    same settings, tag T5. SAME runs at budget k only.
- Number: share of chosen atoms that are aromatic, minus the mean share of 20 random-connected sets of the same size
  (stream "<unit>|T6|conn"; for EiG's own set the size is its own size).
- Rows: all molecules, and test-only molecules (each seed's 29 test molecules). Test-only GStarX and SAME rows have only about
  15 and 4 molecules per seed; they are shown, marked small.
- Holm per seed: 6 methods = 6 (test-only rows separately). The EiG own-size row is not in the family.
- For reference, the Tier 4 L4 rows (SubgraphX fixed, GNNExplainer, IG) are quoted next to it, not recomputed.

## T2. Rings unavailable (MUTAG-GT D-MPNN, 10 seeds)

Methods: EiG, GNNShap, GNNExplainer, SubgraphX fixed, IG. ("GNN" in the request is read as GNNExplainer.)

**(a) Molecules with no aromatic atom.** The 14 molecules, existing full-graph scores (N1 EiG and GNNShap; stored GNNExplainer;
Tier 2 S SubgraphX fixed; Tier 2 G IG). The same 14 molecules in every seed, so seed-to-seed variation here is the model only.
Random-node null: the Tier 2 null file (subset).

**(b) Scaffold hidden first, R1 and R2, only the remaining atoms explained.** The 174 molecules with an aromatic scaffold.
- R1: the scaffold atoms and their bonds are deleted; the explainer runs on the graph of the remaining atoms.
- R2: the scaffold atoms' features are set to 0 and the bonds kept; the explainer runs on that graph, and the scaffold atoms
  are dropped from its ranking before top-k.
- Every method is re-run on the hidden graph (new runs, runs/rq1_tier6/T2/expl/), per-molecule seed crc32("T6|<rule>|<unit>|<mol>").
- Ranking and k: over the remaining atoms only.
  - **Primary k** = max(1, floor(0.25 n_remaining)).
  - Secondary k = min(n_remaining, max(1, floor(0.25 N))) with N the full molecule (the full-graph set size).
- Ground truth: the molecule's ground-truth atoms (all are among the remaining atoms, see above).
- Random-node null: 20 random k-sets among the remaining atoms (stream "<unit>|T6|rn|k<k>"), the same for R1 and R2.
- Also reported:
  - the share of molecules with k = 1 and the GEA a random set reaches on average;
  - how often the predicted class of the hidden graph differs from the full graph;
  - how often a method's scores on the remaining atoms are all tied (for R2 SubgraphX: the coalition lies inside the scaffold);
  - the full-graph numbers on the same 174 molecules (N1 / Tier 2 scores), as "before".
- A molecule on which a method raises an error is listed and dropped for that method only.

Holm per seed and condition: 5 methods = 5 ((a), (b)-R1, (b)-R2; primary and secondary k separately).

## T3. Full-reset sanity check (as N2)

- Units: B-XAIC PAINS D-MPNN, indole GCN, PAINS GCN, Mutagenicity GCN; seeds 0-4 (20 units).
- Reset: reset_parameters() on every module of the trained model, torch seed 5000 + model seed (the Tier 2 R rule), written to
  runs/rq1_tier6/T3/ckpt/<unit>_full.pt with the reset model's test accuracy and AUROC. Check: every parameter tensor changed.
- Molecules: the unit's 100-molecule subset (the D-MPNN owner's subset, as in Tier 5).
- Methods: EiG, GNNShap (re-run on the reset model, same settings and per-molecule seeds as N1, tag T5); Saliency, IG
  (new gradient run on the reset model). Trained = the N1 run (EiG, GNNShap) or the existing gradient file (Saliency, IG),
  on the same molecules.
- Reported as N2: GEA(trained) - GEA(reset) with CI; per-seed Wilcoxon + Holm (4 methods = 4 per seed); Spearman of the two
  atom-score vectors; Jaccard of the two top-k sets. Passes = CI entirely above 0.

## T4. Seconds per molecule for the old methods

- Units: seed 0 of the 12 summary-table cells (indole, PAINS, Mutagenicity x D-MPNN, GIN, GINE, GCN) plus MUTAG-GT s0;
  the unit's 100-molecule subset.
- Methods and what is timed:
  - GNNExplainer: per molecule (the unit's stored epochs);
  - PGExplainer: training once on the unit's stored training molecules (pg_train_idx and epochs from the stored cache key),
    total seconds and seconds per training molecule; inference per molecule;
  - Saliency, IG, Input x Gradient: each method timed on its own, per molecule.
  - Times use torch.cuda.synchronize(); one untimed warm-up molecule per method.
- Load: GPU 1, 8 concurrent jobs (one job per unit and method). The Tier 5 timings were measured with 8-12 concurrent jobs on a
  shared GPU. As a load check, EiG and GNNShap are re-timed on the same molecules in the same run; their times are shown next
  to the N1 times. If they differ clearly, the table says so.

## Time estimate (8 workers on GPU 1; ada is shared)

- T1 new runs: GStarX 5 x 100 x ~96 s, SAME 5 x 25 x <= 329 s: about 3 h.
- T2(b): SubgraphX on R2 graphs (full size, ~123 s x 174 x 10) dominates, about 60 GPU-hours; the rest is a few GPU-hours.
  About 8-9 h.
- T3: under 1 h. T4: under 1 h.
- Total about 13 h.

## Output

runs/rq1_tier6/T1..T4/, T1.txt ... T4.txt (+ .json), FINAL_REPORT.md, DONE.

## Addendum, 2026-11-24 (written before these numbers are computed; T1, T3 and T4 numbers were known, T2 numbers were not)

Requested after the runs finished. Existing outputs only: no new explainer runs. No prediction is added and P-T2 is unchanged.

**T2 report additions.**
- The P-T2 table shows, for each of the 6 cells, the "before" next to the "after". "Before" = Tier 5 N1, full graph, all 188
  molecules (quoted from runs/rq1_tier5/N1.txt and recomputed here), plus the full graph on the same molecules as the "after"
  (for (a) this is the same number; for (b) the 174 molecules).
- How often the predicted class changes when the scaffold is hidden: per rule, per seed, and by direction, from the stored
  pred_full / pred_hidden of the T2 runs. The 5 methods' stored predictions are checked to agree.

**T5. What the chosen atoms are (MUTAG-GT D-MPNN, 10 seeds, all 188 molecules, full graph).**
- Methods: EiG, GNNShap (N1 runs), GNNExplainer (stored), SX fixed (Tier 2 S), IG, Saliency (Tier 2 G), and random-node.
- Chosen atoms = the top-k set, k = max(1, floor(0.25 N)), random ties (20 orders, stream "<unit>|<method>|T6comp").
- Each chosen set is split into (i) ground-truth atoms, (ii) aromatic non-GT atoms, (iii) other non-GT atoms; (iii) is split by
  element (C / N / O / other = F, I, Cl, Br; one-hot column of x). Also: the share of (iii) atoms bonded to a GT atom.
- Number = share of the chosen atoms in each part (averaged over molecules). Random-node = the exact expected share of a
  uniformly random k-set (the part's share of the molecule's atoms).
- Main rows: method minus random-node, with the hierarchical bootstrap CI and per-seed Wilcoxon + Holm (6 methods per part).

**T6. Does the model need the nitro / amino group once the rings are hidden? (Tier 4 L4 / Part G style test; forward passes only.)**
- MUTAG-GT D-MPNN, 10 seeds, the 174 molecules with an aromatic scaffold. Group = all GT atoms of the molecule (union of the
  NO2 and NH2 matches).
- Base graph: the scaffold hidden with R1 (induced graph of the remaining atoms) or R2 (scaffold rows of x set to 0). In the base
  graph, the group (or a null set) is hidden with the same rule: necessity = group hidden (Fid+), sufficiency = only the group kept
  among the remaining atoms (Fid-). Target = the predicted class of the base graph. Logit margin (probability also reported).
- Nulls: 20 random-connected sets (connected in the graph of the remaining atoms) and 20 random-node sets of the group's size,
  drawn from the remaining atoms; rng seed [86600, crc32("<unit>|<rule>"), mol].
- Reference ("before"): the same test on the full graph (pool = all atoms), R1 and R2.
- Molecules where the group is all the remaining atoms are skipped and counted.
- Holm per seed, space, null and base (full graph, scaffold hidden): 2 rules x {Fid+, Fid-} = 4.
