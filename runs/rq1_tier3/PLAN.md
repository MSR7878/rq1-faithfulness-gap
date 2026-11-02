# RQ1 Tier 3: plan

Written 2026-11-03, after the mask-path check and a timing test (no GEA or Fid number computed), and BEFORE any GEA / Fid
number of Tier 3. Output goes to runs/rq1_tier3/ only (the new figure to figures/final/fig1_v2); no paper file is edited; no
existing result is overwritten. Order: E -> F -> X, without waiting; stop only if a check fails or something breaks. Push after each item.

## Final protocol (everywhere)

Top-25% GEA, k = max(1, floor(0.25 N)), random tie-breaking (20 orders). PGExplainer ranked by logits. Random-node and
random-connected nulls (20 draws). Main number: hierarchical bootstrap mean difference with 95% CI (seeds, then molecules;
2,000 reps; seed 2026). Support: per-seed Wilcoxon + Holm counts. Molecules are never pooled across seeds for inference
(pooled boxes in the figure are for display only).

## Withdrawn predictions

- **P-E1: withdrawn, wrong premise.** Original wording: "on GIN, applying the edge mask at ONE message-passing step only (as
  in the D-MPNN) raises GNNExplainer's GEA so that its CI vs random-node excludes 0."
- **P-E2: withdrawn, wrong premise.** Original wording: "on the D-MPNN, applying the edge mask at EVERY step lowers
  GNNExplainer's GEA by at least 0.05."

Explanation: the D-MPNN also applies the edge mask at every step. It has one MessagePassing module, called 3 times per
forward (2 message updates + readout), and PyG applies the mask on every call. Checked on one graph per model
(E/mask_path.json): the "all" variant equals unpatched PyG exactly. Withdrawn by the user before any number was computed.

## Item E (EXPLORATORY: no prediction)

Where the mask enters (checked; 3 steps per forward on all four models):
- D-MPNN: multiplies edge states in the edge-to-node sum at all 3 calls. The reverse-edge subtraction and the h0 residual
  bypass the mask. "last" is the readout aggregation, not a message update. With an all-zero mask, all and last are identical
  by construction.
- GIN / GINE / GCN: multiplies messages in each of the 3 convolutions (GCN: including self-loops).
- GNNExplainer's atom scores come from its NODE mask (which multiplies input features), trained jointly with the edge mask,
  so the variants act on them only indirectly and it may barely respond. PGExplainer's atom scores come from the edge mask.

Variants: all / first / last, via a wrapper on explain_message; trained weights unchanged. Applied in PGExplainer training and
explanation and in GNNExplainer optimisation. Mutagenicity D-MPNN, GIN, GINE, GCN, seeds 0-4 = 60 jobs (GNNExplainer 200
epochs, PGExplainer 30 epochs, original arguments and explainer seeds). "all" is re-run next to the stored result. SubgraphX,
gradients and heuristics are not re-run.
Scores: GNNExplainer node mask; PGExplainer float64 logit-based node score; stored = final rule (items H, S).
Report: GEA vs random-node and random-connected (CI, counts); variant minus "all" (CI); "all" minus stored; Spearman with
"all"; ordering of GNN, PG, SX fixed per variant next to the stored ordering.
Holm family per (model, seed): 2 explainers x 3 variants x 2 nulls = 12; stored rows: 2 explainers x 2 nulls = 4.

## Item F (fine-tuned copy, F-Fidelity style)

Fine-tune: copy of each model; per training molecule and epoch f ~ U(0, 0.5), floor(f N) random atoms zero-filled (as in R2),
topology kept; Adam, lr 1e-4, 5 epochs, original batch size (B-XAIC 512, others 50), gradient clip 5.0, last epoch kept. Clean
test accuracy and AUROC must stay within 0.02 of the original; any copy outside this is reported and kept.
Units: Mutagenicity 4 models x 5 seeds, B-XAIC 4 tasks x 5 seeds, MUTAG-GT 10 seeds = 50.

Bounded hiding rule F, cap = max(1, floor(N / 2)), 20 draws, all atoms zero-filled:
- Fid+: hide the set, or a random subset of size cap if the set is larger.
- Fid-: hide a random subset of the atoms outside the set, of size min(cap, number of those atoms).
- Nulls are treated the same (same size as the set). Molecules cut down are counted and reported.
- **Fid- under F is not comparable with R2** (R2 hides all atoms outside the set; F hides at most 50% of N).
- Fid+ under F for top-k sets is the R2 operation on the fine-tuned copy.
- Predicted class = the copy's class on the clean graph.

Re-scored cached explanations (no re-explaining): GNNExplainer, PGExplainer (final rule), SubgraphX fixed budget, Saliency, IG,
Input x Gradient, heteroatom rule, on all explained molecules; top-k with random ties. Nulls: 20 random-node and 20
random-connected k-sets.
T1 repeat: ground-truth masks (Part G definition) vs size-matched nulls under F and under R1 removal on the copy (R1c), next to
R1, R2, R3 on the original models (recomputed from runs/rq1_revision/pass/G with the T1 family of 18 / 12).
T2 repeat: 3 explainers; F vs R1, R2, R3.
Holm families: methods per (unit, space): 7 methods x 2 metrics x 2 nulls = 28. Oracle per (group, seed): prob 3 metrics (Fid+,
Fid-, GEF) x 2 rules x 2 nulls = 12; logit 2 x 2 x 2 = 8.

## Predictions (fixed by the user; not changed later)

**P-F1:** under the fine-tuned hiding rule (F), the ground-truth mask beats random connected sets on Fid+ in >= 4/5 seeds on every
valid benchmark, and on MUTAG-GT it still does NOT beat random under R1-style removal.
- Fid+ in probability space; logit reported. Valid = the 8 groups (Mutagenicity 4 models, B-XAIC 4 tasks).
- Part 1 holds if the count of seeds significantly better (Holm, family above) is >= 4 of 5 in all 8.
- Part 2, MUTAG-GT, R1c: holds if the 95% CI of ground truth minus random-connected is not entirely above 0 AND fewer than 8
  of 10 seeds are significantly better. Held = both parts.

**P-F2:** on valid benchmarks, explainer rankings by Fid+ under the fine-tuned rule agree with GEA rankings more often than under
R2 (median Kendall tau higher).
- Per unit (40 units): tau between the ranking of GNN, PG, SX fixed by mean Fid+ (probability) and by mean top-k GEA. Held if the
  median under F is strictly higher than under R2. A tie counts as failed. R2 sources: runs/rq1_revision/pass/E for GNN, PG; item S
  for SX fixed; GEA from items H, S. Tau with 3 items takes only 4 values, so ties are plausible.
  Secondary, not scored: the same with all 7 methods.

## Item X

- Fid+ and Fid- under R1, R2, R3, both spaces, vs random-node (20 rankings drawn per unit) for: item-M seeds 5-9 (GNN, PG, SX
  fixed, IG, hetero; own fills) and item-R randomised models (GNN, PG, IG, Saliency on all explained molecules; SX fixed on its
  100-molecule subset). Holm family per (unit, method, space) = 6, as in existing reports.
- figures/final/fig1_v2 (PDF, PNG): panels MUTAG-GT (10 seeds), Mutagenicity D-MPNN (10), B-XAIC indole (10), B-XAIC PAINS (5, no
  more exist); GNNExplainer, PGExplainer, SubgraphX fixed budget, IG. PGExplainer: float64 logit-based scores for all seeds
  (runs/rq1_revision/logits for seeds 0-4 and MUTAG-GT; item M for 5-9); final-rule numbers reported next to them. Bands: random-node
  and the best trivial rule (highest mean GEA among hetero, deg-high, deg-low in that panel, chosen after seeing it and said so),
  each the range of seed means. Diamonds: per-seed means. Boxes: pooled, display only.
  Exact numbers in figures/final/fig1_v2_numbers.json and X.txt.

## Time estimate

Timing test: fine-tune 15 s training, 44 s total on the largest dataset (B-XAIC, 35,000 molecules), under 16 s on the others.
Earlier: 10 GNN + PG units took 36 min with 12 workers.

| Item | Work | Estimate |
|---|---|---|
| E | 60 GNNExplainer + PGExplainer jobs, 12 workers | 3-4 h |
| F | 50 fine-tunes (about 5 min), 50 re-scoring jobs | 1-2 h |
| X | 30 Fid jobs, figure | about 1 h |
| Reports | | about 1 h |
| Total | | about 6-8 h |

ada is shared, so this may drift. The 24-hour stop rule was removed by the user.
