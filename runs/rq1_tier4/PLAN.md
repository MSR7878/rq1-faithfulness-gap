# RQ1 Tier 4: settle the limits. Plan

Written 2026-11-04 BEFORE any Tier-4 number is computed. Only settings were looked up: parameter counts,
the MUTAG-GT bond encoding and past run times. Output goes to runs/rq1_tier4/ only. No paper .tex file is edited
and no existing result is overwritten. Code is committed before it runs. Order: L1, L3, L4, L5, L7, L2, L6, M1, M2,
with no waiting between items. Stop only if a check fails or something breaks. Push after each item. GPU 1 only.

## Final protocol (everywhere)

Top-25% GEA, k = max(1, floor(0.25 N)), random tie-breaking (20 orders, src/analysis/tier2_common.py). PGExplainer is
ranked by its float64 logit-based node scores (runs/rq1_revision/logits for stored units; new runs by their own).
SubgraphX uses the fixed (scoring) budget. Nulls: random-node and random-connected, 20 draws each. Main number:
hierarchical bootstrap of the mean difference (seeds, then molecules; 2,000 reps; seed 2026; 95% percentile CI).
Support: per-seed paired Wilcoxon + Holm counts (better / worse / n.s.). Molecules are never pooled across seeds.
Fid: Part-E protocol (src/analysis/tier2_fid.py); "random-node" for Fid = 20 random rankings per molecule.

## Predictions (fixed now; not changed later; failures reported as failures)

**P-L1:** on MUTAG-GT TEST molecules only (10 seeds), (a) SubgraphX (fixed budget) minus random-node GEA has a CI
entirely below 0, (b) heteroatom minus random-node has a CI entirely above 0, and (c) ground-truth group removal changes
the margin less than random connected sets under R1. (c) is measured as the logit-space Fid+ of the GT mask minus that of
random-connected sets of the same size (Part-G definition), Holm-significant "worse" in >= 8/10 seeds (T1 logit family
of 12 per seed). Held = (a), (b) and (c).

**P-L3:** on Mutagenicity, restricted to ground-truth molecules whose PGExplainer float32 edge mask is NOT constant,
PGExplainer GEA (logit-ranked) minus random-node gives four bootstrap point estimates (D-MPNN, GIN, GINE, GCN). Held if
max minus min of the four is < 0.10. The float32 mask and the float64 logit scores come from the SAME run: the T4 re-run
(runs/rq1_revision/logits/rerun/<unit>_cache.pt and runs/rq1_revision/logits/<unit>.npz). The stored run reproduces that
mask only on some molecules, so a split on the stored mask is reported as a secondary row only.

**P-L4:** on MUTAG-GT (10 seeds; molecules with at least one aromatic atom), (a) removing the aromatic atoms under R1
changes the logit margin more than removing a random connected set of the same size (scaffold Fid+ logit > random-connected,
Holm-significant "better" in >= 8/10 seeds; family 6 per seed), and (b) the share of SubgraphX's chosen atoms that are
aromatic is higher than for random connected sets of the same size (Holm-significant "better" in >= 8/10 seeds; family 3
per seed). Held = (a) and (b).

**P-L5:** under deterministic settings, three repeated PGExplainer runs give bitwise-identical float32 edge masks on every
explained molecule of all 10 units (Mutagenicity GINE and D-MPNN, seeds 0-4).

**P-M1:** on B-XAIC indole and PAINS, for every architecture (D-MPNN, GIN, GINE, GCN), both SubgraphX minus the best trivial
rule and IG minus the best trivial rule have CIs entirely above 0 (8 task-architecture cells x 2 methods; all 16 must hold).
The best trivial rule is fixed now as **high degree** on both tasks: it has the highest mean GEA among hetero / high degree /
low degree on both tasks in Tier 2 item H, and the rules do not depend on the model. SubgraphX is compared on the fixed
100-molecule subset (all four architectures, D-MPNN included, on the same molecules). IG is compared on all GT molecules.
Models that miss the accuracy bar stay in and are flagged.

**P-M2:** PAINS D-MPNN with 10 seeds keeps the CI category (above / includes 0 / below) of the 5-seed result in every cell.
Cells: GNNExplainer, PGExplainer (logit-ranked for all 10 seeds), SubgraphX fixed, IG, hetero, each vs random-node (5 cells).

E (Tier 3) and L2, L6 are exploratory (no prediction).

## Items and exact settings

### L1. MUTAG-GT test-only re-score (no new explanations)
- Per seed s (0-9), the test molecules = the checkpoint's split["test"] (29). All 188 molecules were explained, so this is a subset.
- GEA under the final protocol for GNNExplainer (stored), PGExplainer (logit), SubgraphX fixed (Tier 2 S), Saliency, IG,
  Input x Gradient (Tier 2 G), hetero, high degree, low degree. Nulls = the unit's Tier-2 null arrays (subset).
- Reliance and sufficiency tests (T1 oracle): Part-G arrays (runs/rq1_revision/pass/G), GT mask vs random-node and
  random-connected, R1/R2/R3, prob (Fid+, Fid-, GEF) and logit (Fid+, Fid-), subset to test molecules.
- Fid+ and Fid- under R1, R2, R3 for the 9 methods, computed fresh on all 188 molecules (with 20 random rankings), then
  subset. Main table: Fid+ under R3.
- Comparison: the same numbers on all 188 molecules next to the test-only ones; CI categories that change are listed.
  The like-for-like comparison is fresh all-188 vs fresh test-only. The fresh all-188 Fid (PG logit-ranked, SX fixed budget) is
  not the computation behind the currently reported all-188 Fid (revision pass E: stored PG, legacy SX); where they differ,
  this is noted in one line.
- Holm: GEA per seed, 9 methods x 2 nulls = 18. Oracle: T1 families (prob 18, logit 12 per seed). Fid per (seed, space):
  9 methods x 2 metrics = 18.

### L3. PGExplainer saturation and the model ordering
- Mutagenicity D-MPNN, GIN, GINE, GCN, seeds 0-4; GT-bearing explained molecules.
- "Constant" = the T4 re-run's float32 PGExplainer edge mask has max == min on that molecule (same run as the logit scores).
  Secondary: the same split on the stored float32 mask.
- Per model and part: split sizes; PGExplainer (logit-ranked) GEA minus random-node (CI, counts); final-rule numbers shown too.
- Holm per (model, seed): 2 parts = 2.

### L4. What the MUTAG-GT model uses (scaffold test)
- Scaffold = all atoms on an aromatic bond (bond-type column 0; checked: it is the most common bond type, 4,708 of 7,442 directed
  bond entries). Molecules with no aromatic atom (14 of 188) or with every atom aromatic are skipped and counted.
- Necessity (scaffold hidden) and sufficiency (only scaffold kept, i.e. everything else hidden) vs 20 random-connected sets of the
  same size (and 20 random-node sets, reported), under R1, R2, R3, logit margin (prob also reported). Original models, 10 seeds,
  all 188 explained molecules; a test-only row is reported as well.
- Overlap = share of chosen atoms that are aromatic, for SubgraphX's chosen coalition (fixed budget), GNNExplainer top-25% and
  IG top-25% (random ties, 20 orders), vs 20 random-connected sets of the same size.
- Holm per seed: necessity/sufficiency 3 rules x 2 = 6 (logit space; prob separately 6); overlap 3 methods = 3.

### L5. PGExplainer reproducibility
- Mutagenicity GINE and D-MPNN, seeds 0-4 (10 units). PGExplainer only, original arguments (--max-nodes 80) and explainer seed
  (1000 + s); 3 repeats each.
- Deterministic settings: CUBLAS_WORKSPACE_CONFIG=:4096:8 set in the job's environment before Python starts;
  torch.use_deterministic_algorithms(True), cudnn deterministic, seeds fixed, set before the model is loaded.
- Device rule: a feasibility smoke test first runs PGExplainer twice on GPU for mutagenicity_dmpnn_s0 under these settings. If
  either run raises an error, OR the two float32 edge masks are not bitwise identical, all 30 repeats run on CPU (1 thread);
  otherwise all run on GPU. (Custom scatter kernels can stay non-deterministic without raising, hence the identity test.)
  The smoke test is a feasibility check and does not count towards P-L5. The device used is reported.
- Report: per unit, bitwise equality of the float32 edge masks and the max |difference| of the float64 logits across repeats;
  GEA minus random-node (CI) per repeat. Earlier non-deterministic runs: stored, T4 re-run (runs/rq1_revision/logits/rerun) and the
  Tier-3 E "all" re-run, compared pairwise (max |edge mask difference|, mean Spearman of node scores) with the GEA CI per run.

### L7. Fine-tuned copies that drifted
- The 10 copies outside the 0.02 limit in Tier 3 F (mutagenicity_gin_s3; MUTAG-GT seeds 0, 1, 2, 3, 5, 6, 7, 8, 9).
- Re-tune from the original model with the same random-hiding rule. Ladder of learning rates 5e-5, 2e-5, 1e-5, up to 5 epochs
  each. After every epoch the clean test accuracy and AUROC are checked. Within one learning rate, the last epoch within 0.02 is
  kept; the ladder stops at the first learning rate that has such an epoch. If none qualifies, the copy is reported as not fixed
  and the 1e-5 / 1-epoch copy is scored anyway, marked. (MUTAG-GT has 29 test molecules, so one changed prediction = 0.034:
  any change of a test prediction fails the accuracy limit.)
- Re-score item F for these copies (same code and seeds as Tier 3) into runs/rq1_tier4/L7/; report the F tables, P-F1 part 2 and the
  MUTAG-GT oracle with the re-tuned copies next to the Tier-3 numbers. Tier-3 files are not changed.

### L2. Single-split datasets
- MUTAG (standard, single split, 1 model) is dropped from all results; MUTAG-GT (10 seeds) covers that dataset. Recorded here; no run.
- BBBP: 5 seeds (0-4), D-MPNN, hidden 300, depth 3 (from the stored checkpoint's cfg), random 70/15/15 split by seed. The checkpoint
  does not record epochs or batch size; 100 epochs and batch 64 come from the phase-2 BBBP command (runs/phase2_gpu.log). Explain 30 test molecules per seed (--limit 30 --pg-train-limit 150, explainer seed 1000 + s): GNNExplainer,
  PGExplainer (logit), SubgraphX fixed, Saliency, IG. Fid+ / Fid- under R1, R2, R3, both spaces, vs random-node (20 rankings),
  bootstrap CIs; TOST with margin 0.05 as for Tox21 (90% CI inside +/-0.05 = equivalent; 95% CI excluding 0 = different; otherwise
  undecided; per-seed TOST counts). Exploratory, no prediction. Holm per (seed, space): 5 methods x 2 metrics x 3 rules = 30.

### L6. SubgraphX per-molecule stability
- Fixed 100-molecule subsets (runs/rq1_tier2/subsets.json) of mutagenicity_dmpnn_s0, bxaic_indole_s0, mutag_graphxai_s0 (seed-0
  models). 3 extra fixed-budget runs with explainer seeds from the tags "L6a", "L6b", "L6c".
- Report: per-molecule Jaccard between runs (3 pairs; also each vs the Tier-2 S run), share of atoms picked in >= 2 of 3 among
  atoms picked at least once, GEA of the consensus set (atoms picked in >= 2 of 3) vs each single run. Both are scored as sets,
  and single runs also under the final protocol. Descriptive.

### M1. Other architectures on B-XAIC indole and PAINS
- GIN (hidden 183), GINE (182), GCN (258): parameter counts 204,596 / 203,136 / 204,080 vs the D-MPNN's 203,522 on B-XAIC.
  Same training as the B-XAIC D-MPNN (depth 3, 25 epochs, batch 512, lr 1e-3) on the D-MPNN's own split (--split-json), seeds 0-4.
- Accuracy bar (the Mutagenicity bar, fixed before the architecture runs): test accuracy >= 0.95 and AUROC >= 0.98. A model that
  misses it is reported and still explained; no tuning to reach it.
- Explanations: the same explained molecules as the D-MPNN unit (run_phase3 with the D-MPNN's arguments: --limit 400
  --stratify-frac 0.9 --max-nodes 80 --pg-train-limit 200, explainer seed = model seed s, as the stored B-XAIC runs, which predate
  the separate explainer seed). run_phase3 draws the sample from the split, the labels and the seed only, not from the model, so
  the same split and seed give the same molecules. Check: the explained list equals the D-MPNN cache's; stop if not.
- "All ground-truth molecules" is read as the D-MPNN's explained GT set (about 360 per seed), not every GT molecule among the 7,500
  test molecules, so the ordering is compared on the same molecules as the D-MPNN. GEA on all GT-bearing explained molecules (about 360 per seed, the D-MPNN's GT set) for GNNExplainer,
  PGExplainer (logit), Saliency, IG, Input x Gradient and the three rules. SubgraphX (fixed budget) on a fixed subset of 100
  GT molecules per seed: Tier 2 subset100 rule (seeded by the D-MPNN unit name), the same molecules for all architectures.
- Report: CIs vs random-node and random-connected, vs the best trivial rule (high degree); method ordering per model next to the
  D-MPNN (D-MPNN numbers from Tier 2, SubgraphX for the D-MPNN subset to the same 100 molecules).
- Holm per unit: vs random 9 methods x 2 nulls = 18; vs best rule 6 methods (GNN, PG, SX, Sal, IG, IxG) = 6.
- Nulls: the D-MPNN unit's Tier-2 null arrays (same molecules; the nulls do not depend on the model).

### M2. PAINS with 10 seeds
- D-MPNN seeds 5-9 on B-XAIC PAINS: training as Tier 2 M (--depth 3 --epochs 25 --batch-size 512 --hidden 256, random split by seed);
  run_phase3 with BXA arguments and explainer seed = model seed; SubgraphX fixed budget on all explained molecules; Saliency, IG,
  Input x Gradient; rules. New nulls written under runs/rq1_tier4/nulls (same rule as Tier 2, seeded by unit name). New subsets
  go to runs/rq1_tier4/subsets.json; nothing under runs/rq1_tier2 is written.
- Report: 5- vs 10-seed CIs, counts and seed sign tests for the 5 P-M2 cells, as Tier 2 M; other methods reported.
- Holm per seed: 5 methods vs random-node = 5.

## Time estimate (12 workers on GPU 1; ada is shared, so this may drift)

From past runs: SubgraphX fixed budget takes about 4 min per B-XAIC molecule, 5 min per Mutagenicity molecule and 2 min per MUTAG-GT
molecule (Tier 2 S). GNNExplainer + PGExplainer took about 36 min for 10 B-XAIC units, and Tier 2 M's SubgraphX on 10 units x 400
molecules took 23 h.

| Item | Work | Estimate |
|---|---|---|
| L1 | Fid for 9 methods on 10 x 188 molecules; re-scoring | 1 h |
| L3 | re-scoring | 0.25 h |
| L4 | 10 units, oracle-style passes and overlap | 1 h |
| L5 | 30 PGExplainer runs (GPU about 1 h; CPU fallback 3-4 h) | 1-4 h |
| L7 | 10 re-tunes and F re-score | 0.5 h |
| L2 | 5 BBBP models, 5 x 30 molecules explained, Fid | 1 h |
| L6 | 900 SubgraphX explanations | 5 h |
| M1 | 30 models, 30 GNN+PG units, gradients, 3,000 SubgraphX molecules, scoring | 20-22 h |
| M2 | 5 models, gp, 2,000 SubgraphX molecules, scoring | 12 h |
| Reports | | 2 h |
| Total | | about 44-48 h |
