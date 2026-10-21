# RQ1 final export: plan and time estimate

Written 2026-10-21, before any export number is computed. No training, no new
explanations: re-scoring from cached explanations and stored arrays only.
Outputs only under runs/rq1_final_export/ and figures/final/. No .tex file is
edited, no existing figure or result file is overwritten, nothing is pushed.

## Tie rule (everywhere)

Random tie-breaking at the top-k cut: 20 seeded tie orders per molecule
(random rank within exact ties only), score each order, average per molecule,
then the per-seed test. The tie stream is the one already used for the stored
re-scoring in Phase 2b (numpy default_rng([62000, crc32("<unit tag>|<explainer>"),
molecule position]), unit tag "audit_<dataset>_s<seed>", "arch_<model>_s<seed>",
"cv_f<fold>"). Existing Phase 2b arrays are reused; MUTAG-GT seeds 5-9 are
scored with the same scheme (tag "audit_mutag_graphxai_s<seed>").
GNNExplainer has no ties: stored values are reused after checking they are
identical. Mean-threshold results have no ties: recomputed once from the caches
to confirm they are unchanged, then reused.

## Part 1 -- MUTAG-GT PGExplainer, seeds 0-4: 2/2/1 (Phase 1) vs 2/3 (Phase 2b)

Checked before writing this plan (from committed arrays): same molecules
(188 per seed), same random-node null arrays, same Holm family (the 3
explainers vs random-node, per seed), same scoring path for GEA; the two runs
differ only in the tie-order random stream (Phase 1 tag
"mutag_graphxai_s<seed>", Phase 2b tag "audit_mutag_graphxai_s<seed>"), and the
results differ only at seed 3, the collapsed seed (all 188 PG masks constant,
so every top-k pick is a pure tie-order draw). Added check: seed 3 re-scored
under 50 further tie streams to show how often each sign occurs.
The main-table protocol (exploratory_null_audit.sig_counts): per-seed paired
two-sided Wilcoxon, Holm over the 3 explainers vs random-node at top-k,
direction = higher mean with Holm p < 0.05.

## Part 2 -- tables (CSV + plain text each, old | random-tie | change)

a) per-seed top-k GEA, seeds 0-4: MUTAG-GT, B-XAIC indole, Mutagenicity D-MPNN
b) MUTAG-GT seeds 5-9
c) normalised GEA (GEA - RandNode)/(ceiling - RandNode) per seed, mean +/- std;
   ceilings as before (exploratory_pg_and_normalised.py): MUTAG-GT 0.751,
   indole 0.724, Mutagenicity 0.551; RandNode = the 20-draw matched-budget null
d) Mutagenicity 5-fold CV: per fold + pooled GEA; folds above/below (registered
   4-test family) incl. SubgraphX vs connectivity-matched (mean-threshold)
e) GIN, GCN, GINE per-seed GEA
f) direction-count tables: main table, architecture table, B-XAIC four tasks
Flags: |change| > 0.01, and any direction change.

## Part 3 -- figures (figures/final/, PDF + PNG)

Fig 1 as in make_figures.py / figures/figures.py (current colour style; the
navy/orange/green "patched" palette is not present in this repo), per-molecule
value = mean over the 20 tie orders, seeds 0-4; Fig 2 SubgraphX win rate vs
the connectivity-matched null (tie-free; MUTAG-GT seeds 0-9, indole 0-4);
Fig 3 PGExplainer collapse rate, MUTAG-GT 2/10.

## Time estimate

Under 1 hour in total (GEA re-scoring needs no model; the seed-3 stream check
and the mean-threshold confirmation are a few minutes on Ada). Well under the
3-hour limit.
