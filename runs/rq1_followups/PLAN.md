# RQ1 follow-ups: plan, predictions, time estimate

Written 2026-10-19, committed BEFORE any new run. Builds on commit f53b612.
Everything new goes under runs/rq1_followups/. No stored result file is
edited or overwritten. paper_results_draft.tex is not edited (it is not in
this repo or on Ada). Nothing is pushed.

## Predictions (fixed now; not changed later)

Phase 1
- P1. SubgraphX is significantly below its size-matched random-connected
  null on MUTAG-GT GEA in at least 8 of 10 seeds.
- P2. Removing the annotated group changes the logit margin significantly
  LESS than a random connected set of the same size (R1 and R2) in at least
  8 of 10 seeds.
- P3. GNNExplainer is significantly above the random-node null on GEA in at
  least 7 of 10 seeds.
- No prediction for the PGExplainer collapse count; it is reported.

Phase 2
- P4. Re-scoring with a deterministic tie rule leaves all non-tie units
  identical to the stored values (within 1e-5).
  Note on the count: the earlier report said "103" other units; the correct
  number is 105 (107 Part B units minus MUTAG and BBBP). P4 is scored on all
  105 units plus the GNNExplainer cells of MUTAG and BBBP, i.e. on every cell
  except the four tie cells.

Phase 3 (exploratory, no p-values)
- P5. Within the same Tox21 endpoint, collapsed PGExplainer runs have lower
  class separability of the model's node embeddings than clean runs in most
  pairs.

## How each prediction is scored (fixed now)

Common: per seed (unit), paired two-sided Wilcoxon over GT-bearing
molecules (zero_method "wilcox"); "significantly above/below" = p (after
Holm where stated) < 0.05 with the direction of the rank test (W+ vs W-).
Molecules are never pooled across seeds.

- P1: GEA (Jaccard vs the merged GT mask, the MUTAG-GT convention) of the
  SubgraphX mean-threshold mask vs the mean over 20 random connected sets of
  the SAME size (connectivity_baseline.py; RNG default_rng(10000 + seed)).
  One test per seed, as in the paper's connectivity check. Held if
  significantly below in >= 8 of 10 seeds.
- P2: Part A reliance gap (delta_GT - random-connected null, same code as
  src/analysis/reliance_fidelity.py), Holm over the 4 Part A tests within the
  seed. Counted separately under R1 and R2; held only if significantly below
  in >= 8 of 10 seeds under BOTH.
- P3: GNNExplainer top-k 0.25 GEA vs the random-node null at the same budget
  (20 draws; exploratory_null_audit.py; RNG default_rng(30000 + seed)), Holm
  over the 3 explainers within the seed (as in the paper's matched-budget
  counts). Held if significantly above in >= 7 of 10 seeds.
- P4: max |deterministic-rule probability Fid+/Fid- - stored| <= 1e-5 for
  every molecule, masking (R1, R2, R3) and explainer outside the four tie
  cells. If this fails, the run stops and is reported (instruction).
- P5: pairs as defined in Phase 3. Quantity scored: linear-probe balanced
  accuracy on node embeddings (label = the model's predicted class of the
  graph the node belongs to). Held if collapsed < clean in more than half of
  the Tox21 pairs. Silhouette and the other quantities are reported as
  direction counts only.

## Phase 1 -- MUTAG-GT seeds 5-9

Protocol of seeds 0-4 (priority sweep P2, checked in runs/prio/logs and
scripts/priority_sweep.sh): train with
`src.train.train --dataset mutag_graphxai --seed S --depth 3 --epochs 120
--hidden 300` (batch 50, seeded stratified 70/15/15 split from S), then
`src.explain.run_phase3 --dataset mutag_graphxai --seed S --explainers
gnnexplainer,pgexplainer,subgraphx --masking all --explain-pool all`
(COUPLED: explainer seed = model seed, as in seeds 0-4). Seeds 5-9 use exactly
this, written to runs/rq1_followups/phase1/. All five units concurrently on
GPU 1.

Seeds 0-4 are not touched. Their files' SHA-256 (first 16 hex) before the
run, re-checked after:
ckpt s0-s4: 56d85aaecf94a053 07a3c1231cfddfce b7cf7ca769a9de8c d01510675acc81fa 4691f6c6b8ba0e62
cache s0-s4: 57fee6ec883e4597 d9deed6b8c50cdc1 277c5b7d4d0ac026 7bdc0a1b1c232f30 9b5cabf13367961a
out s0-s4: 64ceb411f1b9ef23 5d143aef57b7ee31 26f6f93b5c176931 860c2bcacc44cd72 0f3d0262e7039acd
Their numbers are recomputed with the same code and must equal the stored
arrays (null_audit_arrays/mutag_graphxai_s*.npz, connectivity_arrays_
mutag_graphxai.npz, rq1_reliance_fidelity/partA and partB); a difference
stops the run.

Analyses for all 10 seeds (counts for seeds 0-4, 5-9 and all 10):
a) top-k GEA of GNN, PG, SX vs RandNode (Holm over 3); SX mean-threshold GEA
   vs size-matched RandConn (one test).
b) Part A reliance, R1 and R2, random-connected null (Holm over 4).
c) random-corrected Fid+ and Fid- under R3 vs the random-node null (Part B
   code, Holm over its 18 tests).
d) PGExplainer degenerate masks, strict collapse rule (n_degenerate/n >= 0.5).
Exploratory: exact two-sided sign test over the 10 seeds on the per-seed
rank-test direction, for a) and b).

## Phase 2 -- deterministic tie rule

Rule: one total order of a molecule's atoms, sorted by (score descending,
atom index ascending) with numpy's stable sort. E = the first k atoms
(k = max(1, int(0.25 N))); the explanation-removed graph keeps the LAST
max(1, int(0.75 N)) atoms of the same order. Implemented by turning the order
into strictly decreasing scores and passing them to the unchanged masking
code, so no torch tie-breaking is involved.
Applied to MUTAG SubgraphX, BBBP SubgraphX, MUTAG PGExplainer, BBBP
PGExplainer: probability and logit Fid+/Fid- under R1, R2, R3, plus MUTAG
GEA (MUTAG-GT mask; graphs verified identical first). Then all other cells
(P4). Output table: stored, current torch, deterministic rule, 5-order range.
Written to runs/rq1_followups/phase2/.

## Phase 3 -- PGExplainer collapse diagnosis (descriptive)

Units: priority-sweep Tox21 (24 collapsed / 60), B-XAIC (PAINS s3, s4
collapsed), MUTAG-GT (coupled s3 collapsed; runs/decoupled s3 collapsed with
the same checkpoint), from runs/prio/collapse_table.csv.
Pairing (fixed now): each collapsed unit is paired with the clean unit of the
same endpoint / task / dataset and protocol that has the lowest seed.
Per unit:
- PGExplainer training loss per epoch: obtained by re-running that unit's
  explanation step (same command, seed and subsample; GNNExplainer first,
  as in the original order) into runs/rq1_followups/phase3/, with the loss
  of every epoch saved. The re-run's PG masks are compared with the stored
  ones; if they do not reproduce, the curve is reported as "re-run, not
  identical" with the size of the difference.
- final mask std and normalised entropy: from the STORED PG edge masks.
- node-embedding norm (mean, std) and class separability (linear-probe
  balanced accuracy, 5-fold, and silhouette) of node and graph embeddings on
  the checkpoint's test split; embeddings = the layer PGExplainer reads
  (get_embeddings, last message-passing layer); graph embedding = sum pool;
  label = model's predicted class.
Direction counts collapsed vs clean per quantity over the pairs.

## Time estimate

- Phase 1: training ~5 min; explanations ~2.5-3.5 h (SubgraphX 188
  molecules per seed; 47.9 s/mol at concurrency 4 in the priority sweep);
  analysis ~20 min. About 3.5-4 h.
- Phase 2: about 30-45 min (re-scoring all 107 units, no random draws).
- Phase 3: about 1-1.5 h (GNNExplainer + PGExplainer re-runs for ~43 units,
  4 workers; embeddings and probes are quick).
- Total of the phases: about 5-6.5 h; with Phases 2-3 overlapping Phase 1's
  SubgraphX, about 4-5 h wall-clock. Under the 8-hour limit.

## RESULT RECORD -- P4 (2026-10-19): FAILED

P4 (unchanged above) did not hold. Under the deterministic rule (score
descending, atom index ascending) 190 of 317 non-tie unit x explainer cells
differ from the stored values by more than 1e-5 (up to 1.0 on single
molecules): every SubgraphX cell, most PGExplainer cells, and one Tox21
GNNExplainer molecule. Phase 2 stopped as instructed (commit 48202e7). The
count "103" in P4 should have been 105 (noted above before the run).

## ADDENDUM (2026-10-19, user decision after P4 failed; written BEFORE any of this is scored)

Phases 1 and 3 continue as planned. The atom-index rule is not used as a
primary rule (atom order follows the input file, so it is not neutral).

### Phase 1: SubgraphX and PGExplainer scored three ways

(a) current torch order (as planned);
(b) PRIMARY -- random tie-break: per molecule, 20 seeded random tie orders
    (random rank within exact ties only; distinct scores keep their order;
    numpy default_rng([62000, crc32(unit), molecule position])); GEA scored
    per order and averaged per molecule before the per-seed test;
(c) mean-threshold cut (imp > mean; no ties), against size-matched nulls:
    random-node of the same size (RandUnc, the stored "_unc" arrays) and, for
    SubgraphX, random connected sets of the same size.
P1-P3 are reported under each:
- P1 as registered uses the SubgraphX mean-threshold mask vs size-matched
  RandConn, which has no ties; so P1 under (a) and (c) is the same test.
  Under (b): SubgraphX top-k (random ties) vs RandConn at the top-k size
  (20 draws, default_rng(40000 + seed)), one test per seed.
- P2 (reliance) does not use any explainer; identical under (a), (b), (c).
- P3 (GNNExplainer): GNNExplainer masks are continuous; under (a)/(b) it is
  the planned test (ties checked and reported); under (c) GNN mean-threshold
  vs RandUnc (Holm over the 3 explainers).
Thresholds for "held" are as registered. GNNExplainer is not re-scored with
random ties unless it has boundary ties (counted and reported).

### Phase 2b (after Phase 1): tie robustness of all stored top-k results, rule (b)

- For every unit: fraction of molecules with a tie at the top-k boundary
  (k-th and (k+1)-th largest importance equal), per explainer.
- Recompute, under (b), top-k GEA (MUTAG-GT, B-XAIC x4, Mutagenicity D-MPNN /
  GIN / GINE / GCN, Mutagenicity 5-fold CV) and fidelity Fid+/Fid- under
  R1, R2, R3 (all 107 Part B units and MUTAG-GT seeds 5-9) for SubgraphX and
  PGExplainer. Molecules with no tie at the relevant boundary keep their
  current value (all 20 orders are identical there). Nulls are unchanged.
- Redo the direction counts with each table's ORIGINAL test code, stored
  vs random-tie side by side, marking every cell whose direction changes:
  main count table (MUTAG-GT, indole, Mutagenicity) and all B-XAIC tasks
  (exploratory_null_audit rule: Holm over 3 explainers vs RandNode at top-k;
  SubgraphX top-k vs RandConn at the budget added); architecture table and
  5-fold CV (the registered 4-test family, Holm within seed); Part A/B
  GEA-vs-fidelity comparison (reliance_fidelity_report rule).
- GNNExplainer: boundary-tie fraction reported to confirm it is unaffected.
Estimate: about 1.5-2.5 h of compute (SubgraphX ties on most molecules;
about 240 masked graphs per tied molecule), under the 6-hour limit.
