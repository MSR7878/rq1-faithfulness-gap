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
- R1 (build third): hard removal, topology broken. STUB (NotImplementedError).

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

## Phase 3 status (explainer wiring + R3/R2 masking -- all 5 variants COMPLETE)

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

### Results -- Fid+ / Fid- / GEF (mean; std ~= 2-4x mean, in runs/phase3_<ds>.json)

R3 columns are mean-fill. mode-fill differs materially only where noted below.

| dataset / explainer      | Fid+ R3 | Fid- R3 | GEF R3 | Fid+ R2 | Fid- R2 | GEF R2 | GEA Jacc / micro |
|--------------------------|---------|---------|--------|---------|---------|--------|------------------|
| **mutag_graphxai** GNN   |  0.071  |  0.013  |  0.081 |  0.041  |  0.536  |  0.617 | **0.448** / 0.385 |
| mutag_graphxai PG        | -0.018  |  0.087  |  0.151 |  0.048  |  0.553  |  0.618 | 0.300 / 0.247 |
| mutag_graphxai SX        |  0.054  |  0.047  |  0.118 | **0.246** | 0.243 |  0.526 | 0.038 / 0.023 |
| **MUTAG (std)** GNN      |  0.048  | -0.008  |  0.091 |  0.041  |  0.486  |  0.594 | -- |
| MUTAG (std) PG           |  0.005  |  0.030  |  0.124 |  0.014  |  0.500  |  0.594 | -- |
| MUTAG (std) SX           | -0.000  |  0.007  |  0.123 | **0.192** | 0.189 |  0.516 | -- |
| **BBBP** GNN             |  0.025  |  0.076  |  0.097 |  0.063  |  0.095  |  0.175 | -- |
| BBBP PG                  |  0.002  |  0.119  |  0.147 |  0.053  |  0.033  |  0.102 | -- |
| BBBP SX                  |  0.057  |  0.074  |  0.097 |  0.146  |  0.027  |  0.137 | -- |
| **Tox21 SR-p53** GNN     |  0.029  |  0.085  |  0.104 |  0.030  |  0.110  |  0.195 | -- |
| Tox21 SR-p53 PG          |  0.083  |  0.015  |  0.020 |  0.121  |  0.099  |  0.151 | -- |
| Tox21 SR-p53 SX          |  0.083  |  0.051  |  0.066 |  0.156  |  0.063  |  0.136 | -- |
| **B-XAIC (indole)** GNN  |  0.411  |  0.104  |  0.206 |  0.335  |  0.465  |  0.574 | 0.216 / 0.204 |
| B-XAIC (indole) PG       |  0.061  |  0.149  |  0.244 |  0.370  |  0.276  |  0.354 | **0.536** / 0.514 |
| B-XAIC (indole) SX       |  0.067  |  0.051  |  0.119 |  0.401  |  0.098  |  0.155 | **0.680** / 0.564 |

mode-fill notables: B-XAIC GNN R3 Fid-/GEF collapse to ~0.005 (keep-only-
explanation with a pure-carbon fill leaves the indole intact); MUTAG/SX R3-mode
GEF 0.32 vs 0.12 mean-fill; otherwise mode ~= mean within noise.

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
    GNN worst). On mutag_graphxai R2-Fidelity DISAGREES with GEA (SX has best
    Fid- but worst GEA) -- a further faithfulness-gap instance.

### R3 + R2 PHASE COMPLETE

Pipeline runs cleanly on all 5 variants under R3 (mean+mode) and R2 (zero),
explanations cached and shared across masking references. Headline: the
explainer ranking is metric- AND masking- AND dataset-dependent -- no single
"most faithful" explainer. GNNExplainer <-> SubgraphX swap ends of the GEA
ranking between the two GT datasets (F3); R2 restores discriminative power for
SubgraphX but on one-hot features only via an OOD artifact (F4, F5).

READY FOR R1 (hard removal -- explanation nodes/edges deleted, topology
broken). src/metrics/masking.py::mask_r1_hard_removal is still
NotImplementedError. Build it, add "R1" to run_phase3.py --masking (the harness
already dispatches by reference). NB: R1 changes edge_index, so
_reverse_edge_index / the MessagePassing forward must tolerate the reduced
graph -- verify on a single molecule first. Then re-run the cached-explanation
sweep. R1 is the aggressive upper-bound reference; expect the largest artifact
on every dataset.
