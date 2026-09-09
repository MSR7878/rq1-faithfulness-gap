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

- R3 (built): distribution-aware fill. src/metrics/masking.py::mask_r3_distribution_aware
  + training_fill_vector(strategy="mean"|"mode"). Keep top-k nodes, replace the
  rest with a training-set fill vector; topology kept.
- R2 (built): zero-fill, topology kept. mask_r2_zero_fill -- same top-k selection
  as R3, masked node features set to 0.
- R1 (built): hard removal, topology broken. mask_r1_hard_removal -- delete the
  masked nodes + incident edges, reindex survivors. Handles empty/isolated/
  disconnected results without fallback (see Phase 3 R1 note).

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

## Phase 3 status (explainer wiring + R3/R2/R1 masking -- all 5 variants COMPLETE)

Explainer wrappers in src/explain/: GNNExplainer + PGExplainer via
torch_geometric.explain (the D-MPNN's edge->node sum runs through a
MessagePassing layer so set_masks / get_embeddings hook it -- see v4 note),
SubgraphX via DIG (stub-package import of only shapley.py + subgraphx.py;
LOCKED rollout=20 / sample=30). One run per dataset (--masking both): torch +
numpy seeded, explanations computed ONCE and cached (runs/expl_cache_<ds>.pt,
keyed on ckpt+subsample+hyperparams), then scored under R3 mean-fill, R3
mode-fill, and R2 zero-fill. K_FRAC=0.25 (top-25% of nodes). Explanations
target the model's predicted class.

Per-variant setup:

| variant         | ckpt (70/15/15)        | test acc / AUROC | explained n | notes |
|-----------------|------------------------|------------------|-------------|-------|
| mutag_graphxai  | ckpt_mutag_graphxai.pt | 0.966 / 0.989    | 29/29 (all; 19pos) | node+edge GT; GEA over all 29 |
| MUTAG (std)     | ckpt_mutag.pt          | 0.897 / 0.979    | 29/29 (same split) | no GT |
| BBBP            | ckpt_bbbp.pt           | 0.833 / 0.841    | 30/306 random | no GT; PGExplainer 1/30 empty (mol 13, N=4) -- benign |
| Tox21 (SR-p53)  | ckpt_tox21_srp53.pt    | 0.934 / 0.799    | 30/1013 stratified 15pos/15neg | no GT; plain random -> ~1 pos, so --stratify-frac 0.5 |
| B-XAIC (indole) | ckpt_bxaic.pt          | 0.998 / 1.000    | 30/7500 random (13pos), --max-nodes 80 | node+edge GT; GEA over the 13 positives only |

training_fill_vector(x_train, strategy): "mean" = per-feature marginal mean;
"mode" = per-COLUMN most-frequent value (v4 fix -- the old global-argmax one-hot
was degenerate on BBBP/Tox21's raw mixed-type features; identical to old on
one-hot data, verified bit-identical on MUTAG).

### Results -- Fid+ / Fid- / GEF (mean; std ~= 2-4x mean, full in runs/phase3_<ds>.json)

R3 columns are mean-fill. mode-fill differs materially only where noted.

| dataset / explainer      | Fid+ R3 | Fid- R3 | GEF R3 | Fid+ R2 | Fid- R2 | GEF R2 | Fid+ R1 | Fid- R1 | GEF R1 | GEA Jacc / micro |
|--------------------------|---------|---------|--------|---------|---------|--------|---------|---------|--------|------------------|
| **mutag_graphxai** GNN   |  0.071  |  0.013  |  0.081 |  0.041  |  0.536  |  0.617 |  0.491  |  0.521  |  0.674 | **0.448** / 0.385 |
| mutag_graphxai PG        | -0.018  |  0.087  |  0.151 |  0.048  |  0.553  |  0.618 |  0.153  |  0.548  |  0.630 | 0.300 / 0.247 |
| mutag_graphxai SX        |  0.054  |  0.047  |  0.118 |  0.246  |  0.243  |  0.526 |  0.566  |  0.594  |  0.650 | 0.038 / 0.023 |
| **MUTAG (std)** GNN      |  0.048  | -0.008  |  0.091 |  0.041  |  0.486  |  0.594 |  0.461  |  0.487  |  0.663 | -- |
| MUTAG (std) PG           |  0.005  |  0.030  |  0.124 |  0.014  |  0.500  |  0.594 |  0.120  |  0.532  |  0.654 | -- |
| MUTAG (std) SX           | -0.000  |  0.007  |  0.123 |  0.192  |  0.189  |  0.516 |  0.515  |  0.530  |  0.603 | -- |
| **BBBP** GNN             |  0.025  |  0.076  |  0.097 |  0.063  |  0.095  |  0.175 |  0.100  |  0.105  |  0.212 | -- |
| BBBP PG                  |  0.002  |  0.119  |  0.147 |  0.053  |  0.033  |  0.102 |  0.008  |  0.080  |  0.198 | -- |
| BBBP SX                  |  0.057  |  0.074  |  0.097 |  0.146  |  0.027  |  0.137 |  0.139  |  0.082  |  0.210 | -- |
| **Tox21 SR-p53** GNN     |  0.029  |  0.085  |  0.104 |  0.030  |  0.110  |  0.195 |  0.069  |  0.100  |  0.207 | -- |
| Tox21 SR-p53 PG          |  0.083  |  0.015  |  0.020 |  0.121  |  0.099  |  0.151 |  0.096  |  0.092  |  0.202 | -- |
| Tox21 SR-p53 SX          |  0.083  |  0.051  |  0.066 |  0.156  |  0.063  |  0.136 |  0.131  |  0.089  |  0.211 | -- |
| **B-XAIC (indole)** GNN  |  0.411  |  0.104  |  0.206 |  0.335  |  0.465  |  0.574 |  0.433  |  0.431  |  0.439 | 0.216 / 0.204 |
| B-XAIC (indole) PG       |  0.061  |  0.149  |  0.244 |  0.370  |  0.276  |  0.354 |  0.432  |  0.434  |  0.440 | **0.536** / 0.514 |
| B-XAIC (indole) SX       |  0.067  |  0.051  |  0.119 |  0.401  |  0.098  |  0.155 |  0.400  |  0.294  |  0.341 | **0.680** / 0.564 |

mode-fill notables: B-XAIC GNN R3 Fid-/GEF collapse to ~0.005 (keep-only with a
pure-carbon fill leaves the indole intact); MUTAG/SX R3-mode GEF 0.32 vs 0.12
mean-fill; otherwise mode ~= mean within noise.

R1 note: on real small MUTAG-family molecules, E_i (keep top-25%) is almost
always ~2 ISOLATED atoms, 0 edges; G\E_i shatters into 2-5 components. The
D-MPNN forward handles both as-is (features-only, no message passing;
per-component aggregation) -- no zero-fill fallback, no skipped molecules.
Verified on synthetic 1-node / isolated / disconnected graphs + real small
molecules (all finite logits).

### Findings (paired Wilcoxon signed-rank, Holm-Bonferroni within block;
### src/analysis/phase3_significance.py)

F1. R3 Fidelity/GEF is NOISE-DOMINATED on the 4 diffuse-decision datasets
    (mutag_graphxai, MUTAG, BBBP, Tox21): means 0.00-0.15, std >> mean, no
    explainer separates. Cause: saturated D-MPNN (logits +/-8) + a fill that
    dilutes rather than deletes categorical identity -- masking even the TRUE
    NO2/NH2 motif shifts p by < 0.1.

F2. B-XAIC R3 Fidelity has a HEAVY RIGHT TAIL (GNNExplainer Fid+ mean 0.41,
    median 0.001): B-XAIC's model has a LOCALIZABLE rule (exact indole
    detection), so on a minority of molecules GNNExplainer's top-25% lands on
    it and p flips. Still not a reliable discriminator -- GNN vs PG/SX Fid+
    differences n.s. at n=30.

F3. GEA ranking (masking-independent) is INVERTED between the two GT datasets,
    and GNNExplainer's inversion is significant:
      mutag_graphxai:  GNN 0.448 ~= PG 0.300  >>  SX 0.038
                       GNN-PG p_Holm=0.084 n.s. ; GNN-SX p<1e-3 *** ; PG-SX p=0.0015 **
      B-XAIC:          SX 0.680 ~= PG 0.536   >>  GNN 0.216
                       GNN-PG p_Holm=7e-4 *** ; GNN-SX p_Holm=1e-3 *** ; PG-SX p=0.15 n.s.
    GNNExplainer: top tier on mutag_graphxai, significantly WORST on B-XAIC.
    SubgraphX: the mirror -- significantly worst on mutag_graphxai, top tier on
    B-XAIC. PGExplainer: top tier on both (the stable method). Mechanism:
    SubgraphX's "connected prediction-preserving subgraph" objective == the
    indole GT (B-XAIC by design) but != the NO2/NH2 motif (not the
    prediction-preserving scaffold for mutagenicity). GNNExplainer's soft masks
    over-select (20-35 of ~40 nodes) -> poor Jaccard on compact motifs.

F4. R2 zero-fill artifact is FEATURISATION-DEPENDENT, not a clean fix.
    One-hot datasets (MUTAG family): R3->R2 inflates |Fid-| and GEF massively
    and significantly for GNN & PG (mutag_graphxai GNN GEF 0.081->0.617,
    p<1e-4) -- but this is the ARTIFACT of zeroing 75% of a one-hot graph
    (OOD), not explanation quality: GNN 0.54 ~= PG 0.55.
    Raw-integer datasets (BBBP, Tox21): R2 barely moves anything (BBBP PG/SX
    R3->R2 n.s.; a zero vector in the 9-dim mixed feature space is far less
    OOD). R2's severity depends on the featuriser, not the model.

F5. R2 DOES separate SubgraphX where R3 could not, on Fid+/Fid-: SubgraphX's
    compact CONNECTED explanation survives zero-fill (its E_i keeps a real
    substructure), so under R2 it has significantly higher Fid+ and lower Fid-
    than GNN/PG on mutag_graphxai, MUTAG, BBBP (GNN-SX, PG-SX p<0.01). On
    B-XAIC, R2 separates ALL THREE on Fid-/GEF (GNN 0.465 > PG 0.276 > SX
    0.098, all pairs sig) -- and there R2-Fidelity AGREES with GEA (SX best,
    GNN worst). On mutag_graphxai R2-Fidelity DISAGREES with GEA (SX best Fid-,
    worst GEA) -- a further faithfulness-gap instance.

F6. R1 hard-removal is the MOST artifact-dominated. MUTAG family: Fid+ AND Fid-
    AND GEF all ~0.5-0.67 for every explainer (removing 25% of a ~18-atom
    molecule shatters it regardless of which 25%). The R2 Fid-/GEF separation
    (F5) is WASHED OUT under R1 -- on mutag_graphxai/MUTAG the GNN-SX and PG-SX
    Fid-/GEF pairs that were *** under R2 become n.s. under R1. BBBP/Tox21: R1
    ~= R2 in magnitude (mild, ~0.1-0.2), inflation vs R3 now consistently
    significant (was marginal under R2).

F7. Under R1, SubgraphX loses its Fid- advantage. R2 spared SubgraphX's
    connected E_i (it survived zero-fill); R1 DELETES nodes, so even a
    connected subgraph gets isolated -- mutag_graphxai SX |Fid-| R3 0.11 -> R2
    0.31 -> R1 0.63. The one signal R1 keeps: PGExplainer has the LOWEST Fid+
    on 3/5 datasets (its scattered soft masks, hard-removed, perturb the
    prediction least) -- GNN-PG and PG-SX Fid+ significant on
    mutag_graphxai/MUTAG/BBBP.

F8. CROSS-METRIC RANKING IS ESSENTIALLY NEVER CONSISTENT (block D). On the two
    GT datasets the explainers are ranked by GEA and by Fid+ under each of R3,
    R2, R1 -- 4 orderings per dataset:
      mutag_graphxai  GEA: GNN>PG>SX | R3-Fid+: GNN>SX>PG | R2: SX>PG>GNN | R1: SX>GNN>PG
      B-XAIC          GEA: SX>PG>GNN | R3-Fid+: GNN>SX>PG | R2: SX>PG>GNN | R1: GNN>PG>SX
    Only ONE (dataset, masking) cell agrees with GEA: B-XAIC / R2-Fid+. Every
    other combination gives a different "best explainer".

### MASKING-REFERENCE SWEEP COMPLETE -- R3 + R2 + R1, all 5 variants, 3 explainers, 4 metrics

Pipeline runs cleanly under all three references (explanations computed once
per dataset, cached, re-scored). BOTTOM LINE for RQ1: there is no
masking-/metric-invariant "most faithful" explainer.
- GEA ranking inverts between the two GT datasets (GNN <-> SX swap ends; F3),
  driven by whether the model's learned rule coincides with the annotated
  motif.
- R3 Fidelity is near-vacuous on 4/5 (F1); R2 restores signal but only via a
  featurisation-dependent OOD artifact on one-hot data (F4-F5); R1
  over-perturbs and washes even that out (F6-F7).
- Across GEA + Fid+ x {R3,R2,R1}, the explainer ranking agrees with GEA in
  exactly 1 of 8 (dataset x masking) cells (F8).

R1 also has R2/R1-style edge cases: none broke (F6 note in table; verified).
The masking sweep is done; Tox21 SR-p53 -> all 12 endpoints and the B-XAIC
other tasks remain as scale-out per the datasets line at the top.

## Scale-out -- Tox21 12 endpoints + B-XAIC 4 tasks (server: 2x V100, torch 2.6.0+cu118)

Run on an IIT-Gn V100 box (driver 525 -> capped at cu118, so torch 2.6.0
not 2.14.0 -- results comparable, not bit-identical to the tables above; the
MP-equivalence test still passes bit-identical on 2.6). Same protocol: single
70/15/15 split D-MPNN per endpoint/task, run_phase3 --masking all, n=30
stratified 15/15, K_FRAC=0.25, SubgraphX rollout=20. Analysis:
src/analysis/scaleout_summary.py. Two pipeline fixes shipped mid-run:
  * gef.py: clamp softmax to >=1e-12 before log -- a near-perfect model
    (B-XAIC, AUROC ~1.0) drove some softmax entries to exactly 0 -> log(0) ->
    NaN GEF. (B-XAIC P/X hit this; re-scored after the fix.)
  * run_phase3.sanity_check: warn + record `explainer_health` in the JSON
    instead of hard-asserting on ">20% empty" -- see F10.

### B-XAIC GEA across 4 task types (Jaccard vs substructure GT, R3/mean, masking-independent)

| task    | substructure kind       | GNN   | PG    | SX    | order         | within-task Wilcoxon |
|---------|-------------------------|-------|-------|-------|---------------|----------------------|
| indole  | fused ring system       | 0.216 | 0.536 | 0.680 | SX > PG > GNN | GNN<PG *** , GNN<SX *** , PG-SX ns (n=13) |
| PAINS   | reactive-group alert set| 0.209 | 0.150 | 0.327 | SX > GNN > PG | all ns (n=7) |
| X       | halogen atom present    | 0.066 | 0.000 | 0.555 | SX > GNN > PG | GNN>PG ** , SX>GNN *** , SX>PG *** (n=13) |
| P       | phosphorus atom present | 0.125 | 0.361 | 0.382 | SX > PG > GNN | GNN<PG ** , GNN<SX ** , PG-SX ns (n=15) |

F9. SubgraphX is top (or tied-top) on GEA for EVERY B-XAIC task type -- ring,
    alert-set, single-atom-presence alike -- and GNNExplainer is worst or
    near-worst on all four. This is the exact OPPOSITE of mutag_graphxai
    (GNN top, SX ~0). So the Phase-3 "inversion" (F3) is not an indole quirk:
    whenever the annotated substructure IS the model's decision rule (B-XAIC
    by construction), SubgraphX's prediction-preserving-subgraph objective
    wins; when it is NOT (mutag_graphxai: nitro group != mutagenicity
    scaffold), GNNExplainer's soft node masks win. The explainer ranking is a
    function of model-rule / annotation alignment, not of the explainer.
    (PGExplainer at GEA 0.000 on X: soft edge masks cannot localise a single
    halogen atom at all.)

### Tox21 -- 12 endpoints, aggregate (mean +/- sd of per-endpoint means, n=12)

| masking | expl | Fid+            | Fid-            | GEF             |
|---------|------|-----------------|-----------------|-----------------|
| R3      | GNN  | 0.039 +/- 0.049 | 0.054 +/- 0.068 | 0.088 +/- 0.068 |
| R3      | PG   | 0.034 +/- 0.046 | 0.063 +/- 0.076 | 0.097 +/- 0.075 |
| R3      | SX   | 0.068 +/- 0.068 | 0.031 +/- 0.055 | 0.068 +/- 0.059 |
| R2      | GNN  | 0.068 +/- 0.054 | 0.096 +/- 0.078 | 0.162 +/- 0.082 |
| R2      | PG   | 0.049 +/- 0.035 | 0.113 +/- 0.083 | 0.184 +/- 0.089 |
| R2      | SX   | 0.165 +/- 0.100 | 0.055 +/- 0.065 | 0.124 +/- 0.067 |
| R1      | GNN  | 0.044 +/- 0.091 | 0.058 +/- 0.086 | 0.189 +/- 0.069 |
| R1      | PG   | 0.037 +/- 0.086 | 0.057 +/- 0.088 | 0.189 +/- 0.071 |
| R1      | SX   | 0.075 +/- 0.083 | 0.053 +/- 0.091 | 0.182 +/- 0.073 |

The Phase-3 findings REPLICATE at scale across all 12 endpoints:
  - R3 noise-dominated (F1): every cell ~0.03-0.10, sd ~= mean.
  - R2 mild on raw-integer features (F4): GEF only ~0.12-0.18 (vs ~0.6 on
    one-hot MUTAG); SubgraphX separates -- highest Fid+ (0.165), lowest Fid-
    (0.055) (F5).
  - R1 washout (F6): GEF converges to ~0.19 for all three explainers.

F10. PGExplainer COLLAPSES to a degenerate uniform edge mask (node_importance
     std = 0 on all 30 molecules -> empty mean-threshold explanation) on
     exactly the 4 Tox21 endpoints whose D-MPNN barely beats majority
     (test AUROC <= ~0.80: NR-AR, NR-Aromatase, NR-ER-LBD, SR-ARE).
     GNNExplainer and SubgraphX never degenerate. PGExplainer's edge-mask MLP
     needs informative base-model embeddings (get_embeddings) to train; a
     near-majority model gives it nothing to learn from. These 4 endpoints'
     PGExplainer rows are recorded but flagged unreliable in explainer_health.

### FULL RQ1 PICTURE (5 core variants + 12 Tox21 endpoints + 4 B-XAIC tasks)

No masking-, metric-, or dataset-invariant "most faithful" explainer:
  - GEA ranking is set by model-rule / annotation alignment (F3, F9):
    SubgraphX wins when they coincide (all B-XAIC), GNNExplainer when they
    don't (mutag_graphxai).
  - R3 Fidelity/GEF is near-vacuous wherever the decision rule is diffuse
    (F1) -- 4 core datasets + all 12 Tox21 endpoints.
  - R2/R1 add "signal" that is mostly perturbation artifact, featurisation-
    and reference-dependent (F4-F7).
  - Explainer robustness itself varies: PGExplainer degenerates on weak
    models (F10); SubgraphX's compact connected explanations are the most
    stable across masking references.
