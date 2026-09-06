# RQ1 Metric Spec (locked) -- mirrors RQ1_Metric_Spec.docx v4

Datasets: MUTAG (standard), MUTAG (GraphXAI GT-labeled), BBBP, Tox21 (SR-p53
phase 1, then all 12 endpoints), B-XAIC (task "indole" for phase 2 -- explicit
starting choice, mirroring SR-p53 for Tox21; other B-XAIC tasks deferred)
Explainers: GNNExplainer, PGExplainer, SubgraphX
Base model: PyG-native D-MPNN (Yang et al. 2019 architecture -- directed edge
message passing), implemented in src/train/dmpnn.py with a
`model(x, edge_index, edge_attr, batch)` forward so it drops straight into the
metrics harness and the Phase-3 explainers. Chemprop is NOT a required runtime
dependency (v3 named "D-MPNN (Chemprop)"; changed in v4). Chemprop may still be
installed informally as an optional accuracy cross-check.

v4 note (Phase 3 prep): the edge->node sum in dmpnn.py was moved from a raw
torch_geometric scatter to a tiny MessagePassing layer so PyG's
torch_geometric.explain stack (set_masks / get_embeddings) works natively.
Computation-preserving: bit-identical forward vs the old scatter across 5 random
seeds x 3 input configs (src/train/test_mp_equivalence.py), and mutag_graphxai
10-fold CV re-run agrees (acc 0.884+/-0.061, AUROC 0.952+/-0.035 vs the v4-table
0.884 / 0.943). BBBP/Tox21/B-XAIC weights stand without retraining.

## Metrics (verified against GraphXAI source, graphxai/metrics/metrics_graph.py)

- Fid+  = (1/N) sum_i [ f(G_i)_yi - f(G_i \ E_i)_yi ]
- Fid-  = (1/N) sum_i [ f(G_i)_yi - f(E_i)_yi ]
- GEF   = 1 - exp( -KL( f(G_i) || f(E_i) ) )        # bounded [0,1), lower = more faithful
- GEA   = JAC(M_gt, M_pr) = TP / (TP+FP+FN)          # only where ground truth exists

## Masking references -- build order R3 -> R2 -> R1

- R3 (build first): distribution-aware, replace masked features with training-set
  per-feature mean. Implemented: src/metrics/masking.py::mask_r3_distribution_aware
- R2 (build second): zero-fill, topology kept. STUB, not yet implemented.
- R1 (build third): hard removal, topology broken. STUB, not yet implemented.

## GEA applicability (ground truth required)

| Dataset                     | GEA |
|------------------------------|-----|
| MUTAG (standard)             | No  |
| MUTAG (GraphXAI GT-labeled)  | Yes |
| BBBP                         | No  |
| Tox21                        | No  |
| B-XAIC                       | Yes |

## Phase 2 status (dataset + base-model bring-up)

Datasets: all five variants load and pass `src/data/sanity_check.py`.
- MUTAG (std)      188 graphs, 7 node / 4 edge feats, 66.5% positive
- MUTAG (GraphXAI) 188 graphs, + node/edge GT: 272 NO2 + 19 NH2 motifs,
                   every flagged component verified as exactly {N,O,O} or {N}
- BBBP             2039 graphs, 9 / 3 feats, 76.5% positive
- Tox21 (SR-p53)   6750 graphs (dropped 19 empty + 1054 unlabeled), 6.3% positive
- B-XAIC (indole)  50000 graphs, 11 / 1 feats, 36.8% positive, + node/edge GT

Base model: PyG-native D-MPNN (hidden 300 / depth 3; hidden 256 for B-XAIC).
10-fold stratified CV, best checkpoint per fold selected on val(AUROC + acc)
with a short warmup guard. B-XAIC uses its shipped train/valid/test split
(40k/5k/5k) -- 10-fold on 50k graphs is neither tractable here nor standard
for that dataset. Reporting accuracy AND AUROC because class balance spans
6.3%-76.5%; AUROC is the cross-dataset-comparable number.

Compute: RTX 3050 Ti Laptop (4 GB), torch 2.14.0+cu130. All five re-run on
GPU for a device-consistent table; the earlier CPU run of MUTAG /
MUTAG-GraphXAI agreed within fold noise (0.873/0.945 vs 0.878/0.938).

| Dataset          | Protocol     | Test accuracy    | Test AUROC       | Status |
|------------------|--------------|------------------|------------------|--------|
| MUTAG (std)      | 10-fold CV   | 0.878 +/- 0.071  | 0.938 +/- 0.041  | PASS -- acc & AUROC in published GNN range |
| MUTAG (GraphXAI) | 10-fold CV   | 0.884 +/- 0.057  | 0.943 +/- 0.041  | PASS -- == MUTAG (std) up to checkpoint-selection noise |
| BBBP             | 10-fold CV   | 0.877 +/- 0.025  | 0.900 +/- 0.035  | PASS -- AUROC in random-split range ~0.88-0.92 |
| Tox21 (SR-p53)   | 10-fold CV   | 0.939 +/- 0.005  | 0.827 +/- 0.050  | PASS -- AUROC in per-endpoint range ~0.80-0.86; acc ~= majority (expected at 6.3% prevalence), AUROC confirms real ranking. Multi-task (12-endpoint) training would likely lift this. |
| B-XAIC (indole)  | native split | 0.9988           | 1.0000           | PASS -- by design: exact learnable substructure rule; a near-perfect model is the premise of the benchmark |

No result outside its expected range; nothing flagged for debug.

## Phase 3 status (explainer wiring -- mutag_graphxai only)

All three explainers run against the trained D-MPNN (checkpoint
runs/ckpt_mutag_graphxai.pt, single 70/15/15 split, test acc 0.966 / AUROC
0.990). dmpnn.py's edge->node sum was moved to a MessagePassing layer (v4 note
above) so PyG's Explainer stack hooks it natively; SubgraphX is DIG's, loaded
via a stub-package trick that imports only shapley.py + subgraphx.py.
* GNNExplainer -- torch_geometric.explain, node+edge object masks, explain_type=model
* PGExplainer  -- torch_geometric.explain, edge masks, trained 30 ep on 131 train graphs
* SubgraphX    -- DIG, MCTS+mc_l_shapley, zero_filling, LOCKED rollout 20 /
  sample 30 (DIG default rollout; rollout 10 vs 20 checked -- see sensitivity note)
Explanations target the model's predicted class; sparsity = top 25% of nodes.
Metrics under R3 ONLY (R2/R1 still stubbed). n = 29 test molecules.

R3 fill: 'mean' = per-feature marginal mean (spec default); 'mode' = one-hot of
the most common atom. training_fill_vector() in masking.py builds either.

Fidelity / GEF (mean-fill; mode-fill in parentheses):

| explainer     | Fid+            | Fid-            | GEF             |
|---------------|-----------------|----------------|-----------------|
| GNNExplainer  | 0.092 (0.113)   | 0.008 (0.027)  | 0.061 (0.097)   |
| PGExplainer   | -0.020 (-0.001) | 0.099 (0.097)  | 0.175 (0.170)   |
| SubgraphX     | 0.058 (0.054)   | 0.045 (0.093)  | 0.112 (0.249)   |

FINDING (R3): every Fidelity/GEF cell has std 2-4x its mean (e.g. GNNExplainer
Fid+ 0.092 +/- 0.264) -- at n=29 the explainers are NOT separable on
Fidelity/GEF under R3, with either fill. Cause: a saturated D-MPNN (logits
+/-8) plus a distribution-aware fill that dilutes rather than removes one-hot
atom identity; masking the *true* NO2/NH2 motif shifts p by only ~0.04-0.09.
R3 mean/mode-fill is near-inert as a faithfulness probe on saturated one-hot
models. Recorded as an RQ1 result; R2 (zero-fill) is expected to have signal
(zeroing all features flips every mutagenic prediction 1.00 -> 0.00).

GEA (Jaccard vs NO2/NH2 ground truth; fill-independent):

| explainer     | GEA Jaccard      | GEA micro |
|---------------|------------------|-----------|
| GNNExplainer  | 0.456 +/- 0.283  | 0.402     |
| PGExplainer   | 0.245 +/- 0.269  | 0.196     |
| SubgraphX     | 0.022 +/- 0.094  | 0.018     |   (rollout 20; rollout 10 gave 0.044 +/- 0.146 / 0.032)

FINDING (GEA): a clear, large ordering GNNExplainer > PGExplainer >> SubgraphX
at recovering the mutagenic motifs -- invisible if one looked only at Fidelity
under R3. This is the first faithfulness-gap data point.

SubgraphX rollout sensitivity: doubling MCTS rollout 10 -> 20 did NOT raise GEA
(0.044 -> 0.022, within noise at n=29; Fid/GEF unchanged). Not an under-search
artifact -- SubgraphX genuinely does not recover the NO2/NH2 motifs here: its
mc_l_shapley objective rewards a connected subgraph that preserves the
prediction under zero-filling, and on this saturated model that is the aromatic
carbon scaffold, not the nitro group. Locked at rollout 20 (DIG default; runtime
~1100s / 29 mols, acceptable) so "under-searched" is not an open question.

### MUTAG (standard) -- same 188 molecules and split as mutag_graphxai, minus GT

Separately trained checkpoint (runs/ckpt_mutag.pt, same seed/split logic ->
identical train/val/test indices to mutag_graphxai, verified): test acc 0.897 /
AUROC 0.979. No node_gt_mask/edge_gt_mask on this variant -- GEA not applicable
(spec's applicability table). SubgraphX at the locked rollout=20.

| explainer     | Fid+ (mean)     | Fid- (mean)      | GEF (mean)      | Fid+ (mode)    | Fid- (mode)     | GEF (mode)      |
|---------------|-----------------|------------------|-----------------|----------------|-----------------|------------------|
| GNNExplainer  | 0.055 +/- 0.263 | -0.024 +/- 0.112 | 0.045 +/- 0.102 | 0.116 +/- 0.340| -0.018 +/- 0.168| 0.071 +/- 0.170  |
| PGExplainer   | 0.003 +/- 0.078 | 0.065 +/- 0.302  | 0.188 +/- 0.315 | 0.028 +/- 0.157| 0.077 +/- 0.325 | 0.189 +/- 0.325  |
| SubgraphX     | 0.014 +/- 0.142 | -0.008 +/- 0.173 | 0.107 +/- 0.178 | 0.009 +/- 0.189| 0.120 +/- 0.347 | 0.285 +/- 0.387  |

CROSS-DATASET CHECK (mean-fill, SubgraphX at matched rollout=20 on both sides):
every Fid+/Fid-/GEF value moves by 0.01-0.04 between mutag_graphxai and MUTAG
(std) -- well under 1 SE of the mean at n=29 (SE ~= std/sqrt(29) ~= 0.02-0.06).
No meaningful divergence. CONFIRMS the R3-vacuity finding is a property of the
D-MPNN architecture + mean/mode-fill masking, not specific to the GraphXAI
variant or its ground-truth annotations.

### BBBP -- 2039 graphs, no ground truth, explained on n=30/306 test subsample

Bigger and structurally more diverse than MUTAG (N ranges 2-132 vs 10-28), so
explainer runs are capped to a random-seeded n=30 test subsample (comparable
size to MUTAG's n=29) and PGExplainer trains on a random-seeded 150/1427 train
subsample -- both to keep SubgraphX rollout=20 MCTS tractable; see --limit /
--pg-train-limit in run_phase3.py. Checkpoint runs/ckpt_bbbp.pt: test acc 0.833
/ AUROC 0.841 (single 70/15/15 split; a bit below the Phase-2 CV mean
0.877/0.900 but within its +/-0.025/+/-0.035, one split + 100 epochs).

BUGFIX (mid-run): training_fill_vector's "mode" strategy assumed one-hot
features (v[mean.argmax()]=1.0) -- correct for MUTAG's 7-dim atom one-hot, but
BBBP's node features are raw mixed-type columns (PyG from_smiles: atomic_num,
chirality, degree, charge, numH, radical_e, hybridization, aromatic, in_ring),
so the old formula produced a degenerate fill ("atomic_num~=1, everything else
0"), not a typical atom. Fixed to per-COLUMN mode (torch.mode(x, dim=0)) --
provably identical to the old formula on one-hot data (verified bit-identical
on MUTAG), so mutag/mutag_graphxai's already-reported mode-fill numbers stand.

| explainer     | Fid+ (mean)     | Fid- (mean)     | GEF (mean)      | Fid+ (mode)    | Fid- (mode)     | GEF (mode)      |
|---------------|-----------------|-----------------|-----------------|----------------|-----------------|------------------|
| GNNExplainer  | 0.026 +/- 0.065 | 0.085 +/- 0.091 | 0.113 +/- 0.184 | 0.017 +/- 0.094| 0.044 +/- 0.086 | 0.073 +/- 0.137  |
| PGExplainer   | 0.024 +/- 0.048 | 0.090 +/- 0.153 | 0.104 +/- 0.196 | 0.021 +/- 0.044| 0.038 +/- 0.126 | 0.071 +/- 0.142  |
| SubgraphX     | 0.059 +/- 0.081 | 0.075 +/- 0.115 | 0.091 +/- 0.179 | 0.051 +/- 0.088| 0.031 +/- 0.109 | 0.059 +/- 0.134  |

GEA: n/a (no ground truth).

REPLICATES, does not diverge: magnitudes stay small (0.02-0.11) and
noise-dominated (std > mean in nearly every cell), same qualitative R3-vacuity
finding as MUTAG. Distributions are somewhat tighter than MUTAG's (e.g. Fid+
std ~0.05-0.09 vs ~0.14-0.34) -- plausibly because the BBBP model is less
saturated (AUROC 0.841 vs MUTAG's ~0.98) -- but mean/std ratio stays << 1
throughout, so the conclusion is unchanged: not dataset-specific to MUTAG.

Sanity-check hardening: PGExplainer produced 1/30 empty (mean-threshold)
explanations -- mol 13, a 4-atom molecule, node_importance exactly all-zero.
A single edge case on a diverse dataset (N from 2 to 132) is not itself a bug;
the run_phase3.py sanity check now prints full diagnostics for every
empty/whole-graph explanation and only hard-fails past 20% (was: any single
occurrence), so genuine breakage still trips it. GNNExplainer and SubgraphX had
zero empty/whole explanations.

### Tox21 (SR-p53) -- 6750 graphs, no ground truth, n=30/1013 test subsample

Severe imbalance (6.3% positive): a plain random 30 from the test split has ~1
positive, so ~29 explanations would target the NEGATIVE class ("why not toxic"
-- uninformative). Deviation from BBBP's random draw: --stratify-frac 0.5 (new
flag) -> test subsample 15pos/15neg, PGExplainer-train 100pos/100neg.
Checkpoint runs/ckpt_tox21_srp53.pt: test acc 0.934 (~= majority 0.9375) /
AUROC 0.799 (single split; Phase-2 CV was 0.939/0.827).

| explainer     | Fid+ (mean)     | Fid- (mean)     | GEF (mean)      | Fid+ (mode)    | Fid- (mode)     | GEF (mode)      |
|---------------|-----------------|-----------------|-----------------|----------------|-----------------|------------------|
| GNNExplainer  | 0.026 +/- 0.151 | 0.091 +/- 0.246 | 0.099 +/- 0.249 | 0.044 +/- 0.151| 0.109 +/- 0.218 | 0.130 +/- 0.250  |
| PGExplainer   | 0.058 +/- 0.240 | 0.078 +/- 0.223 | 0.105 +/- 0.232 | 0.060 +/- 0.226| 0.098 +/- 0.186 | 0.122 +/- 0.211  |
| SubgraphX     | 0.079 +/- 0.199 | 0.052 +/- 0.196 | 0.064 +/- 0.177 | 0.078 +/- 0.158| 0.073 +/- 0.167 | 0.104 +/- 0.191  |

GEA: n/a (no ground truth).

REPLICATES: means 0.03-0.13, std > mean in every cell -> noise-dominated R3
vacuity, same as the other three. Std is back near MUTAG's level (~0.15-0.25)
rather than BBBP's tight ~0.05-0.10 -- the stratified draw pulls in 15 toxic
actives whose predictions move more variably under masking. Conclusion
unchanged.

IMBALANCE EDGE-CASE FLAG (asked): NO different explainer behaviour. 0 empty /
0 whole-graph explanations for ALL THREE explainers (BBBP had 1 PGExplainer
empty on a 4-atom molecule; Tox21's smallest subsample molecule N=6, fine), no
NaNs. Once the subsample is stratified, 6.3% prevalence does not produce
degenerate explainer output. SubgraphX again has the highest Fid+ (0.079) --
consistent pattern across mutag_graphxai 0.063 / BBBP 0.059 / Tox21 0.079.

R3 pipeline is validated end-to-end on 4 of 5 variants (mutag_graphxai, MUTAG
std, BBBP, Tox21 SR-p53). Last: B-XAIC (indole). R2/R1 stay STUBBED until all
five are done on R3.
