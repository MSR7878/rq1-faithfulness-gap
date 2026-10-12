# RQ1 Metric Spec (locked) -- mirrors RQ1_Metric_Spec.docx v5

**>>> Before citing ANY number from this file (esp. in a deck), read the
"## CONSOLIDATED CURRENT-NUMBERS TABLE" at the very end first. Many figures
in the body below are SUPERSEDED (F3, F9, F17's Fid-/GEF) -- the body is a
running log kept for provenance, not a source of current truth on its own. <<<**

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

## Metrics (checked against GraphXAI source; see the GEF note below for what matches which file)

- Fid+  = (1/N) sum_i [ f(G_i)_yi - f(G_i \ E_i)_yi ]
- Fid-  = (1/N) sum_i [ f(G_i)_yi - f(E_i)_yi ]
- GEF   = 1 - exp( -KL( f(E_i) || f(G_i) ) )        # bounded [0,1), lower = more faithful
  GEF note (documented 2026-10-12; comments and text only, no code or result changed):
  (1) The 1 - exp(-KL) transform and the zero-fill match
      graphxai/metrics/metrics_graph.py, graph_exp_faith_graph.
  (2) Which nodes are hidden follows graphxai/formal/metrics.py, graph_exp_faith
      (explanation kept, the rest zeroed; lines 176-185) and the paper's
      definition -- NOT metrics_graph.py line 192, which builds rem_nodes from the
      nodes that ARE in the top-k and so zeroes the explanation itself, despite its
      comments. Our GEF always compares G_i with the explanation-only graph E_i.
  (3) The KL argument order is that of F.kl_div(log q, p), i.e. KL( f(E_i) || f(G_i) ),
      the same as GraphXAI's code. This line previously read KL( f(G_i) || f(E_i) ),
      which was reversed relative to the code (src/metrics/gef.py); the code was
      never different, only the notation.
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

### F1-REVISION: seed-aware re-test (src/analysis/gt_datasets_seed_audit.py,
### section_f1_f2, 5 priority-sweep seeds) -- CONFIRMED on mutag_graphxai

Per-seed count of (explainer, metric) cells with std > |mean| under R3-mean,
out of 9 possible (3 explainers x {fid_plus, fid_minus, gef}):

| dataset         | s0  | s1  | s2  | s3  | s4  |
|-----------------|-----|-----|-----|-----|-----|
| mutag_graphxai  | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| indole          | 6/9 | 7/9 | 6/9 | 7/9 | 7/9 |
| PAINS           | 5/9 | 6/9 | 5/9 | 5/9 | 5/9 |
| X               | 5/9 | 5/9 | 5/9 | 5/9 | 6/9 |
| P               | 6/9 | 6/9 | 6/9 | 6/9 | 6/9 |

mutag_graphxai: FULLY ROBUST, noise-dominated (std > mean) on all 9 cells in
all 5/5 seeds -- exactly the pattern F1 describes. The 4 B-XAIC tasks were not
part of F1's original scope (they are GT-localizable by construction, not
"diffuse-decision"), but even there a majority (5-7/9) of cells are
noise-dominated under R3 in every seed -- incidental support that R3's noise
problem is broader than the 4 originally-named datasets, not narrower.
MUTAG-standard / BBBP / Tox21 std-vs-mean were not re-tested here (Tox21's
R3 vs R2/R1 behavior is covered separately by F11-REVISION/F17-REVISION).
VERDICT: CONFIRMED on mutag_graphxai, the one dataset directly re-testable.

F2. B-XAIC R3 Fidelity has a HEAVY RIGHT TAIL (GNNExplainer Fid+ mean 0.41,
    median 0.001): B-XAIC's model has a LOCALIZABLE rule (exact indole
    detection), so on a minority of molecules GNNExplainer's top-25% lands on
    it and p flips. Still not a reliable discriminator -- GNN vs PG/SX Fid+
    differences n.s. at n=30.

### F2-REVISION: seed-aware re-test (src/analysis/gt_datasets_seed_audit.py,
### section_f1_f2, 5 priority-sweep seeds) -- WITHDRAWN AS STATED

The specific mean-vs-median ratio for indole R3 GNNExplainer Fid+, per seed:

| seed | mean  | median | ratio (mean/median) | shape |
|------|-------|--------|----------------------|-------|
| 0    | 0.780 | 0.964  | 0.8x                 | left-skew (median > mean) |
| 1    | 0.570 | 0.951  | 0.6x                 | left-skew |
| 2    | 0.242 | 0.016  | 15.3x                | heavy RIGHT tail (matches original) |
| 3    | 0.507 | 0.579  | 0.9x                 | left-skew (mild) |
| 4    | 0.648 | 0.907  | 0.7x                 | left-skew |

Only 1/5 seeds (seed 2) reproduces the claimed "heavy right tail, median near
0" shape. In the other 4/5 seeds the distribution is the OPPOSITE shape --
left-skewed, median above the mean, i.e. most molecules get near-perfect Fid+
and a minority of low outliers drags the mean down. The original numeric
example (mean 0.41, median 0.001) reads as a single seed/run's extreme case,
not a stable property of indole R3 Fid+.
VERDICT: WITHDRAWN as stated. Corrected claim: indole R3 Fid+ for
GNNExplainer has a seed/run-dependent distributional shape (right-skewed in a
minority of seeds, left-skewed/ceiling-clustered in most) rather than a
consistent heavy-right-tail. The "GNN vs PG/SX Fid+ n.s. at n=30" clause was
not re-tested here (F5-REVISION below covers R2/R1 Fid+ pairwise significance
per seed, not R3).

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
    REVISION (5-seed, seed-aware -- see "F3-REVISION v2" below, NOT the
    pseudo-replicated pooled-n=940 v1): the mutag_graphxai "GNN ~= PG" tie
    and the "PGExplainer stable" claim BOTH change. GNN significantly beats
    PG and SX in 5/5 independent seeds (robust). But the exact "PG > SX"
    2nd-place ordering only reproduces in 3/5 seeds -- report "GNN >> {PG,
    SX}", not the strict 3-way "GNN > PG > SX". PGExplainer turns out to be
    the least RELIABLE of the three (worst CV + the only catastrophic-
    collapse mode), not the most stable -- though seed-to-seed swing itself
    is NOT PG-specific: GNN's absolute range across seeds is larger than
    PG's.

F4. R2 zero-fill artifact is FEATURISATION-DEPENDENT, not a clean fix.
    One-hot datasets (MUTAG family): R3->R2 inflates |Fid-| and GEF massively
    and significantly for GNN & PG (mutag_graphxai GNN GEF 0.081->0.617,
    p<1e-4) -- but this is the ARTIFACT of zeroing 75% of a one-hot graph
    (OOD), not explanation quality: GNN 0.54 ~= PG 0.55.
    Raw-integer datasets (BBBP, Tox21): R2 barely moves anything (BBBP PG/SX
    R3->R2 n.s.; a zero vector in the 9-dim mixed feature space is far less
    OOD). R2's severity depends on the featuriser, not the model.

### F4-REVISION: seed-aware re-test, one-hot side (src/analysis/gt_datasets_
### seed_audit.py, section_f4, mutag_graphxai, 5 priority-sweep seeds) -- CONFIRMED

R3->R2 |Fid-| / GEF, GNN & PG, per seed (all Holm-free single paired tests,
matching the original method):

| seed | GNN fid_minus        | GNN gef              | PG fid_minus          | PG gef                |
|------|-----------------------|-----------------------|------------------------|------------------------|
| 0    | 0.11->0.65 p=3e-25 *** | 0.13->0.66 p=5e-25 *** | 0.15->0.66 p=2e-22 *** | 0.19->0.66 p=1e-21 *** |
| 1    | 0.10->0.23 p=1e-22 *** | 0.08->0.28 p=1e-24 *** | 0.13->0.39 p=2e-26 *** | 0.12->0.50 p=8e-27 *** |
| 2    | 0.06->0.09 p=6e-06 *** | 0.04->0.10 p=5e-06 *** | 0.14->0.16 p=4e-02 *   | 0.14->0.21 p=1e-02 *   |
| 3    | 0.05->0.13 p=4e-13 *** | 0.03->0.17 p=2e-15 *** | 0.14->0.21 p=6e-05 *** | 0.16->0.28 p=2e-07 *** |
| 4    | 0.05->0.10 p=4e-09 *** | 0.04->0.15 p=2e-12 *** | 0.10->0.20 p=4e-15 *** | 0.09->0.26 p=4e-18 *** |

Direction (R3->R2 inflation) and significance are robust in all 5/5 seeds,
all 4 cells -- 20/20. Magnitude varies (seed 2 is a visibly weaker effect for
both explainers, ~2x inflation vs 3-6x elsewhere), but the qualitative claim
("R3->R2 inflates |Fid-|/GEF massively and significantly on the one-hot
side") is not seed-fragile. The raw-integer side (Tox21) is separately
confirmed per-seed by F11-REVISION (R3->R2/R1 inflation on |Fid-|/GEF, same
mechanism, opposite featurisation).
VERDICT: CONFIRMED, robust to seed.

F5. R2 DOES separate SubgraphX where R3 could not, on Fid+/Fid-: SubgraphX's
    compact CONNECTED explanation survives zero-fill (its E_i keeps a real
    substructure), so under R2 it has significantly higher Fid+ and lower Fid-
    than GNN/PG on mutag_graphxai, MUTAG, BBBP (GNN-SX, PG-SX p<0.01). On
    B-XAIC, R2 separates ALL THREE on Fid-/GEF (GNN 0.465 > PG 0.276 > SX
    0.098, all pairs sig) -- and there R2-Fidelity AGREES with GEA (SX best,
    GNN worst). On mutag_graphxai R2-Fidelity DISAGREES with GEA (SX best Fid-,
    worst GEA) -- a further faithfulness-gap instance.

### F5-REVISION: seed-aware re-test, R2 Fid+ pairwise (src/analysis/gt_datasets_
### seed_audit.py, section_f5_f6_f7, mutag_graphxai + 4 B-XAIC tasks, 5 seeds) -- CONFIRMED

Re-ran the GNN-SX / PG-SX pairwise Holm-corrected test on R2 Fid+, per seed
(proxy for the original Fid-/Fid+ separation claim -- Fid- is not directly
re-run here, see F4-REVISION for the |Fid-| side):
- mutag_graphxai: PG-SX significant (SX higher) in 5/5 seeds; GNN-SX
  significant (SX higher) in 4/5 seeds (ns only at seed 2).
- indole: GNN-SX and PG-SX both significant (SX higher) in 5/5 seeds.
- PAINS/X/P: SX-involving pairs significant in the large majority of seeds
  (occasional ns, mostly on the PG-SX pair, never enough to flip the
  direction).
VERDICT: CONFIRMED. R2's SX-separation on Fid+ is robust to seed on every GT
dataset checked. (MUTAG-standard / BBBP portions of the original claim remain
single-seed, unverified -- see the spec limitations note below F7-REVISION.)

F6. R1 hard-removal is the MOST artifact-dominated. MUTAG family: Fid+ AND Fid-
    AND GEF all ~0.5-0.67 for every explainer (removing 25% of a ~18-atom
    molecule shatters it regardless of which 25%). The R2 Fid-/GEF separation
    (F5) is WASHED OUT under R1 -- on mutag_graphxai/MUTAG the GNN-SX and PG-SX
    Fid-/GEF pairs that were *** under R2 become n.s. under R1. BBBP/Tox21: R1
    ~= R2 in magnitude (mild, ~0.1-0.2), inflation vs R3 now consistently
    significant (was marginal under R2).

### F6-REVISION: seed-aware re-test, R1 vs R2 pairwise significance (src/
### analysis/gt_datasets_seed_audit.py, section_f5_f6_f7, mutag_graphxai + 4
### B-XAIC tasks, 5 seeds) -- DOES NOT REPRODUCE as a general phenomenon

Counted, per dataset, how many of the 15 (3 pairs x 5 seeds) R1 fid_plus/
fid_minus tests stay significant (the "washout" claim predicts many should
drop to n.s. relative to R2):
- mutag_graphxai: 13/15 R1 fid_plus pairs still sig, 11/15 R1 fid_minus --
  only a handful wash out (mostly GNN-PG or PG-SX, never a clean sweep).
- indole: ~14/15 sig on both metrics.
- X: 15/15 sig on both metrics -- ZERO washout.
- P: 15/15 sig on fid_plus, ~13/15 on fid_minus -- almost no washout.
- PAINS: the weakest case, 11/15 fid_plus, 12/15 fid_minus -- some genuine
  washout, but still a minority of pairs.
On every GT dataset with 5-seed data, pairwise significance under R1 remains
nearly as high as under R2 -- the "R2's separation is washed out under R1"
pattern from the original (MUTAG-family) claim does NOT generalize to
mutag_graphxai or any B-XAIC task.
VERDICT: DOES NOT REPRODUCE as a general R1 phenomenon on the datasets we can
check. Restrict F6 to MUTAG-standard/BBBP specifically, where it remains
single-seed and unverified (see limitations note below F7-REVISION) rather
than stating it as a general property of R1 masking.

### F6-ADDENDUM: reconsidered against a proper NULL (F17-REVISION, Tox21) --
### RESTATE, do not drop. Washout is real, but only against RANDOM, only for
### Fid-/GEF, and only on Tox21 -- not a pairwise explainer-vs-explainer effect

The test above asks "does explainer A stay significantly different from
explainer B under R1?" -- pairwise, no null. F17-REVISION asks a different,
arguably more principled question: "does SX stay significantly ABOVE RANDOM
under R1?" -- and that test, already run and seed-verified for a different
purpose (the F15-17 random-baseline work), reaches F6's original intuition
independently, with a proper null, on Tox21 (block C, 5-seed per-metric
counts):
| masking | Fid+ (SX vs RandN) | Fid- (SX vs RandN) | GEF (SX vs RandN) |
|---|---|---|---|
| R3 | 5/5 | 0/5 | 0/5 |
| R2 | 5/5 | 0/5 | 0/5 |
| R1 | 4/5 | 0/5 | 0/5 |
Fid-/GEF's separation from random is essentially ABSENT at every masking
(0/5 even under R3/R2, so there is nothing for R1 to "wash out" there --
those two metrics never had a reproducible random-null signal to lose).
Fid+ is the metric that actually shows the R3->R1 pattern F6 described: it
stays significant under R3 and R2 (5/5) and MOSTLY survives R1 (4/5, not a
full washout) -- R1 erodes but does not eliminate it. This is a real,
seed-verified, properly-nulled signal, but it is (a) Tox21-specific -- GEA
is masking-independent by construction, so mutag_graphxai/B-XAIC have no
analogous "R1 vs random" test to run; a vs-random washout check is only
possible where a masking-dependent metric (Fidelity) exists, i.e. Tox21 --
and (b) a partial erosion of Fid+, not the "all 3 metrics wash out" claim
the original single-seed F17 entry stated (that overstatement is itself
corrected by the 5-seed F17-REVISION).
VERDICT: RESTATE, don't drop. F6 as "explainer-vs-explainer separation
washes out under R1" does not reproduce (confirmed above). F6 as "R1
partially erodes an explainer's separation from a random null" DOES
reproduce, but only for SX's Fid+ on Tox21 (4/5 seeds, down from 5/5 under
R2/R3) -- report it in those specific terms, sourced to F17-REVISION block C,
not as a general R1/masking-sweep claim spanning all datasets and metrics.

F7. Under R1, SubgraphX loses its Fid- advantage. R2 spared SubgraphX's
    connected E_i (it survived zero-fill); R1 DELETES nodes, so even a
    connected subgraph gets isolated -- mutag_graphxai SX |Fid-| R3 0.11 -> R2
    0.31 -> R1 0.63. The one signal R1 keeps: PGExplainer has the LOWEST Fid+
    on 3/5 datasets (its scattered soft masks, hard-removed, perturb the
    prediction least) -- GNN-PG and PG-SX Fid+ significant on
    mutag_graphxai/MUTAG/BBBP.

### F7-REVISION: seed-aware re-test, PG-lowest-Fid+ under R1 (src/analysis/
### gt_datasets_seed_audit.py, section_f5_f6_f7, mutag_graphxai + 4 B-XAIC
### tasks, 5 seeds) -- DATASET-DEPENDENT, NOT A GENERAL R1 SIGNATURE

Counted, per dataset, how many of 5 seeds have PGExplainer as the strict
lowest-Fid+ explainer under R1:
- mutag_graphxai: 4/5 seeds (breaks at seed 3, where GNN is lowest instead).
- indole: 5/5 seeds -- fully robust.
- PAINS: 4/5 seeds (breaks at seed 1).
- P: 2/5 seeds only.
- X: 0/5 seeds -- FLAT CONTRADICTION. PGExplainer has the HIGHEST R1 Fid+ in
  all 5/5 seeds on X (PG beats both GNN and SX on every seed).
VERDICT: RESTATED, not withdrawn outright. "PG has the lowest Fid+ under R1"
is common (mutag_graphxai, indole, PAINS: 12/15 seed-dataset cells) but not
universal, and B-XAIC/X is a clean, seed-stable counterexample in the
opposite direction. The original "3/5 datasets" framing undersold how
dataset-dependent this is -- report it as a common-but-not-universal pattern
with a named, robust exception (X), not a general R1 signature.

### Spec limitation (recorded per the 2026-09 seed-audit request): F5, F6 and
### F7 were originally stated as claims about the MUTAG family (mutag_graphxai
### AND MUTAG-standard AND BBBP) or about "3/5 datasets" spanning both GT and
### non-GT sets. MUTAG-standard and BBBP were NEVER multi-seeded (no
### priority-sweep equivalent exists for them) and so the portions of F5/F6/F7
### resting on those two datasets remain single-seed (n=29/30) and unverified
### by this audit. Only the mutag_graphxai and B-XAIC portions above have been
### re-tested across 5 seeds.

F8. CROSS-METRIC RANKING IS ESSENTIALLY NEVER CONSISTENT (block D). On the two
    GT datasets the explainers are ranked by GEA and by Fid+ under each of R3,
    R2, R1 -- 4 orderings per dataset:
      mutag_graphxai  GEA: GNN>PG>SX | R3-Fid+: GNN>SX>PG | R2: SX>PG>GNN | R1: SX>GNN>PG
      B-XAIC          GEA: SX>PG>GNN | R3-Fid+: GNN>SX>PG | R2: SX>PG>GNN | R1: GNN>PG>SX
    Only ONE (dataset, masking) cell agrees with GEA: B-XAIC / R2-Fid+. Every
    other combination gives a different "best explainer".

### F8-REVISION: seed-aware re-test (src/analysis/gt_datasets_seed_audit.py,
### section_f13_f8, mutag_graphxai + indole, R3-mean/R2-zero/R1-hard x 5 seeds)
### -- CONFIRMED (qualitative pattern), specific "agreeing cell" NOT reproducible

Re-ran GEA-order vs Fid+-order agreement per seed (3 maskings x 2 datasets x
5 seeds = 30 cells; NOTE this only covers R3-mean, not the R3-mode condition
that contributed 2 of the original "8" cells -- a partial re-test, flagged as
a gap below):
- mutag_graphxai: R3 agrees in 2/5 seeds (seeds 2,3), R2 agrees in 0/5, R1
  agrees in 0/5. 2/15 cells agree.
- indole: R3 agrees in 1/5 seeds, R2 agrees in 2/5 seeds (seeds 1,4), R1
  agrees in 0/5. 3/15 cells agree.
Combined: 5/30 = 17% agreement, close to the original "1/8" = 12.5%.
Crucially, the ONE cell the original single-seed run flagged as the
exception -- indole/R2 -- is NOT reliably the agreeing cell either: it only
agrees in 2/5 seeds. No (dataset, masking) cell agrees with GEA in all 5
seeds, and no cell is reliably the exception across seeds.
VERDICT: CONFIRMED as a qualitative pattern (agreement between GEA-order and
Fid+-order is rare, ~13-20% of cells, regardless of seed) -- this is the
core, paper-facing claim and it holds. But the SPECIFIC "indole/R2 is the one
cell that agrees" is not reproducible; it is one fragile instance among many
equally-fragile non-agreements, not a stable exception. Report F8 as "cross-
metric agreement is rare and unsystematic" without naming a specific
agreeing cell. GAP: R3-mode was not re-tested per seed (would need a small
script extension); given R3-mean and R3-mode gave the same GEA orderings in
the original single-seed run (F14), this gap is judged low-risk but is not
formally closed.

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
MP-equivalence test still passes bit-identical on 2.6; see ## Limitations).
Same protocol: single 70/15/15 split D-MPNN per endpoint/task, run_phase3
--masking all, n=30 stratified 15/15, K_FRAC=0.25, SubgraphX rollout=20.
Analysis: src/analysis/scaleout_summary.py (tables) +
src/analysis/scaleout_significance.py (full F1-F8-style Wilcoxon/Holm battery,
blocks A-E). Two pipeline fixes shipped mid-run:
  * gef.py: clamp softmax to >=1e-12 before log -- a near-perfect model
    (B-XAIC, AUROC ~1.0) drove some softmax entries to exactly 0 -> log(0) ->
    NaN GEF. (B-XAIC P/X hit this; re-scored after the fix.)
  * run_phase3.sanity_check: warn + record `explainer_health` in the JSON
    instead of hard-asserting on ">20% empty" -- see F10.

### B-XAIC GEA across 4 task types (Jaccard vs substructure GT, R3/mean, masking-independent)

| task    | substructure kind       | GNN   | PG    | SX    | point order   | paired Wilcoxon + Holm (within task) |
|---------|-------------------------|-------|-------|-------|---------------|-------------------------------------|
| indole  | fused ring system       | 0.216 | 0.536 | 0.680 | SX > PG > GNN | GNN<PG *** , GNN<SX *** , PG-SX ns (n=13) |
| X       | halogen atom present    | 0.066 | 0.000 | 0.555 | SX > GNN > PG | SX>GNN ** , SX>PG ** , GNN>PG ** (n=13) |
| P       | phosphorus atom present | 0.125 | 0.361 | 0.382 | SX > PG > GNN | GNN<SX ** , GNN<PG ** , PG-SX ns (n=15) |
| PAINS   | reactive-group alert set| 0.209 | 0.150 | 0.327 | SX > GNN > PG | ALL pairs ns after Holm (n=7) |

F9. GEA ranking by B-XAIC task type, under the full paired-Wilcoxon + Holm
    battery (src/analysis/scaleout_significance.py, block A).
    HOLDS on indole / X / P (3 of 4): SubgraphX is SIGNIFICANTLY top-or-tied-
    top on every one, and this is the exact mirror of mutag_graphxai (GNN top,
    SX ~0). So the Phase-3 "inversion" (F3) is confirmed as a real, repeatable
    pattern, not an indole quirk: whenever the annotated substructure IS the
    model's decision rule (B-XAIC by construction), SubgraphX's prediction-
    preserving-subgraph objective wins.
    WEAKENS on PAINS: with only n=7 GT-present molecules, NOTHING is significant
    after Holm, and the point-estimate order even puts GNNExplainer ABOVE
    PGExplainer. PAINS supports the "SX on top" reading by point estimate only.
    The "GNNExplainer always worst" half is LOOSER than F9's original wording:
    GNN is the significantly-worst method only on indole and P. On X the
    significantly-worst method is PGExplainer (GEA exactly 0.000 -- soft edge
    masks cannot localise a single halogen atom); on PAINS nothing is
    significant. Precise surviving claim: model-rule / annotation alignment,
    not the explainer, sets the GEA ranking -- SX wins that alignment on 3/4
    B-XAIC tasks at significance.
    BINARIZATION NOTE (F14): "3/4 tasks" is under mean-threshold GEA; matching
    the binarization to Fidelity's top-k=0.25 budget drops P to non-significant,
    so it is 2/4 (indole, X) under the matched metric. indole/X and F3 are
    binarization-robust.

### F9-REVISION: seed-aware re-test (src/analysis/bxaic_f9_seed_check.py, all
### 4 B-XAIC tasks x 5 priority-sweep seeds) -- MAJOR CORRECTION, X and P's
### rankings do NOT reproduce per seed; only indole is fully robust

F9's original per-task GNN-vs-PG-vs-SX significance was computed from a
SINGLE seed (the scale-out). Never checked against the other 4
priority-sweep seeds with the same per-seed + seed-level method F3-REVISION
v2 applied to mutag_graphxai -- until now.

**indole: FULLY ROBUST.** SX > PG > GNN in 5/5 seeds, and EVERY pairwise
comparison (GNN-PG, GNN-SX, PG-SX) is significant in EVERY one of the 5
seeds (n=360/seed). This is the single most reproducible 3-way ranking in
the whole project.

**PAINS: SX on top is robust; 2nd/3rd place is an artifact of PG's
collapse, not a ranking property.** SX beats both others in 5/5 seeds. But
GNN vs PG flips: PG > GNN in seeds 0-2 (PG healthy), GNN > PG in seeds 3-4
(PG COLLAPSED, GEA=0.000). Seed-level sign consistency for GNN-PG is only
3/5 -- not a real, seed-independent ordering; it's downstream of whether PG
happened to collapse that seed.

**X: DOES NOT REPRODUCE -- PG actually beats SX in most non-collapsed
seeds.** Point-estimate order flips across seeds: **PG > SX > GNN in seeds
0, 1, 2** (PG=0.44-0.54, actually the HIGHEST of the three when healthy);
SX > PG > GNN in seed 3 (close, PG-SX ns that seed); SX > GNN > PG in seed 4
(PG near-collapsed, PG=0.030). Seed-level test: PG-SX direction is
**"SX higher in only 3/5 seeds"** -- essentially a coin flip, means are
nearly tied (PG mean=0.368, SX mean=0.383). **F9's original framing of X as
an "SX top" task is WRONG for the 3-way ranking** -- it only looked that
way in the one seed sampled, where PG happened to be weaker. (F16's
separate finding that SX and PG BOTH decisively beat RANDOM on X, every
seed, is unaffected -- that's a different claim and still holds.)

**P: THE LEAST STABLE RANKING OF ANY B-XAIC TASK.** SX > PG > GNN in seeds
0, 2, 3; PG > SX > GNN in seed 1 (close); **GNN > PG > SX in seed 4** -- a
complete reversal, GNN's single best performance anywhere in the B-XAIC
table (0.327) happens to make it top-ranked on this one seed. Exact
ordering "SX > PG > GNN" holds in only 3/5 seeds; seed-level sign
consistency is ns on every pair (GNN-PG "PG higher 4/5", GNN-SX "SX higher
4/5", PG-SX "SX higher 3/5" -- none reach even the 5/5 threshold).

**Verdict: F9's "SX significantly top-or-tied-top on 3/4 B-XAIC tasks"
(and F14's binarization-adjusted "2/4") OVERSTATE how often SX is the
BEST-RANKED explainer.** Only indole is robust. PAINS's ranking is real for
SX-on-top but its 2nd-place is collapse-contingent. X and P show genuine
seed-to-seed reordering even setting collapse aside -- on X, PG is
frequently (3/5 seeds) the actual top scorer; on P, no explainer holds the
top rank consistently. **This does NOT contradict F16** (SX beats RANDOM on
every task, every seed, 20/20) -- "beats chance" and "ranks highest among
the three real explainers" are different claims, and only the first one is
robust across all 4 tasks. The GEA-inversion story (F3/F9: SX wins on
B-XAIC, GNN wins on mutag_graphxai) should be read as established for
INDOLE specifically (and PAINS's top rank), not asserted uniformly across
all four B-XAIC task types.

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

F10. PGExplainer sometimes COLLAPSES to a degenerate uniform edge mask
     (node_importance std = 0 on ~all 30 molecules -> empty mean-threshold
     explanation; GNNExplainer and SubgraphX never do this). The single-seed
     scale-out flagged this on 4/12 Tox21 endpoints (NR-AR, NR-Aromatase,
     NR-ER-LBD, SR-ARE) and F10 originally read it as a property of those
     endpoints / of "AUROC <= ~0.80". A 5-seed retest settles it:
     COLLAPSE IS SEED-DEPENDENT TRAINING NOISE, not an endpoint property.
     (src/analysis/f10_multiseed.py -- 6 endpoints x seeds 0-4; each seed is a
     fresh 70/15/15 split + D-MPNN init + PG-MLP init; PGExplainer only, same
     n=30 R3 protocol. #seeds (of 5) that collapsed:)
       NR-AR        3/5      NR-Aromatase          1/5
       NR-ER-LBD    1/5      SR-ARE                3/5
       NR-ER (ctrl) 0/5      NR-PPAR-gamma (ctrl)  1/5
     - NONE of the 4 "collapsed" endpoints collapse on every seed.
     - A control that did NOT collapse in the scale-out (NR-PPAR-gamma)
       collapses under 1/5 seeds here -- no endpoint is immune.
     - Even fixed-seed-0 retraining flipped all 4: they collapsed 30/30 in the
       scale-out but 0/30 on the rerun. D-MPNN training is GPU-nondeterministic
       on the V100 (cuDNN / scatter-add), so "seed 0" is not one fixed model,
       and the collapse follows the exact weights.
     - AUROC does not separate collapsed from non-collapsed seeds WITHIN an
       endpoint: NR-ER-LBD's one collapsed seed (AUROC 0.829) has a STRONGER
       model than 3 of its 4 non-collapsed seeds; NR-PPAR-gamma collapsed at
       its single strongest seed (0.847). Collapse only ever appears on
       weak-ish models (these 6 endpoints span AUROC ~0.65-0.86) but is
       otherwise ~1-in-3 per run (11/30 cells), ~independent of endpoint and
       of AUROC within that weak band.
     Mechanism: PGExplainer's edge-mask MLP optimisation is unstable when the
     base model's embeddings carry little class signal; whether a given
     (model init, MLP init) pair converges or dies is close to a coin flip.
     REVISED CLAIM: not "these 4 endpoints collapse", not "AUROC <= 0.80
     collapses" -- rather "PGExplainer has a ~30% per-run chance of training
     failure on any near-majority Tox21 model, seed-dependent". Practical
     consequence: a single PGExplainer run on a weak model is unreliable;
     trust its metrics only from multi-seed runs with degenerate-explanation
     screening (explainer_health). Block C's exclusion of PG pairs on the
     4 scale-out endpoints still stands for THAT run's cached explanations.

### F10-MECHANISM: decoupling the base-model seed from PGExplainer's own
### init/training seed -- REFUTES the pure "MLP-seed" hypothesis; collapse is
### a MODEL x MLP-INIT INTERACTION, not either factor alone

Motivated by F17-REVISION's observation that 9/12 Tox21 endpoints collapsed
specifically at seed 4 (vs ~4.8/12 expected by chance). `--seed` in
run_phase3.py controls BOTH the molecule subsample RNG and
`torch.manual_seed` (which seeds PGExplainer's edge-mask MLP init +
training stochasticity) -- but the base MODEL's weights come from a
separately-trained ckpt (`train.py --seed`), independent of run_phase3's
`--seed`. This lets the two be decoupled. scripts/f10_mechanism.sh, PG-only
(`--explainers pgexplainer`), on 2 endpoints (NR-AhR, SR-MMP):

**Experiment A -- fix the model at a clean (non-collapsing) train-seed,
sweep the explain-seed 0-4 (including 4, the worst offender):**
NR-AhR (model fixed at train-seed 0): 0/30, 1/30, 0/30, 0/30, 0/30
degenerate at explain-seed 0,1,2,3,4 respectively -- **NEVER collapses, not
even at explain-seed=4.**
SR-MMP (model fixed at train-seed 2): 0/30 at every explain-seed 0-4 --
**NEVER collapses either.**
**This REFUTES the pure "explain-seed=4 poisons PGExplainer's MLP
regardless of model" hypothesis** -- if that were the whole story, both
would have collapsed at explain-seed=4.

**Experiment B -- fix the explain-seed at 4, sweep which model (ckpt) is
loaded:**
NR-AhR (explain-seed fixed at 4): model-seed 0 -> 0/30 (clean), model-seed 1
-> **30/30 (collapsed)**, model-seed 2 -> **30/30 (collapsed)**, model-seed
3 -> 1/30 (clean), model-seed 4 (= the original matched-seed run) ->
**30/30 (collapsed)**. **3 of 5 models collapse at this fixed explain-seed,
2 don't.**
SR-MMP (explain-seed fixed at 4): model-seed 0 -> **29/30 (collapsed)**,
model-seed 1 -> **29/30 (collapsed)**, model-seed 2 -> 0/30 (clean),
model-seed 3 -> 0/30 (clean), model-seed 4 (original) -> 15/30 (borderline,
50%). **Also model-dependent, not universal.**

**VERDICT: collapse is a MODEL x MLP-INIT INTERACTION, not a pure effect of
either factor.** Neither "the model is fine, blame the MLP seed" (refuted
by Experiment A) nor "certain models are just bad regardless of who
explains them" (Experiment B shows the SAME explain-seed=4 collapses some
models and not others) holds alone. The most likely explanation for F17-
REVISION's seed-4 clustering: in the ORIGINAL sweep, seed VALUE 4 was used
for BOTH the model's own training (`train.py --seed 4`) and the explainer's
init (`run_phase3 --seed 4`) SIMULTANEOUSLY, by protocol construction (same
seed value reused for both). The apparent "seed 4 is cursed" pattern is
most plausibly an artifact of that PAIRING -- `torch.manual_seed(4)`
landing in a state that happens to produce both a somewhat harder-to-
explain model AND a badly-initialised MLP more often than other seed
values, not a property of PGExplainer's initialisation in isolation. This
turns F10 from "flaky" into a genuinely characterised (if not fully
resolved) mechanism, per the original ask: **the failure requires a bad
(model, MLP-init) combination, and either one alone is usually recoverable.**
Practical implication unchanged: multi-seed screening via
`explainer_health` remains the only reliable safeguard, since which
specific combinations are "bad" is not predictable from the model or the
seed value alone.

### Scale-out significance battery (src/analysis/scaleout_significance.py; F1-F8 method: paired two-sided Wilcoxon, Holm within block, bootstrap 95% CI)

Blocks B/C/D/E apply the F1-F8 rigor to all 16 scale-out cells (12 Tox21
endpoints + 4 B-XAIC tasks). What this adds / changes vs the core findings:

F11. R2/R1 inflation vs R3 (block B) IS statistically detectable on raw-integer
     features -- contra F4's "barely moves / n.s.". Across the 12 Tox21
     endpoints the R3->R2 |Fid-|/GEF increase is significant (p<0.05, often
     p<0.001) for all three explainers on ~8/12 endpoints, and R3->R1 is
     significant on nearly all. BUT the effect size stays small (|Fid-|/GEF
     ~0.05 -> 0.10 -> 0.15) -- an order of magnitude below one-hot MUTAG
     (GEF -> 0.6). F4's mechanism stands (a zero vector is far less OOD in the
     9-dim mixed feature space); only its "n.s." wording was an n=30 artifact
     that dissolves once 12 endpoints are pooled. The B-XAIC single-atom tasks
     X / P show LARGE significant R1 GEF inflation (P: 0.00 -> 0.37-0.70, all
     p<0.001) -- deleting the one discriminative atom is catastrophic
     regardless of which explainer chose it: F6 washout in its purest form.
     URGENT RE-TEST (below, F11-REVISION): unlike the SX Fid-/GEF pooled-n=60
     issue (F17-REVISION), this claim's ORIGINAL form was NOT cross-seed
     pooled to begin with (it's a per-endpoint, single-seed count, not one
     combined p-value) -- but it WAS single-seed, so re-checked independently
     against all 5 priority-sweep seeds. Verdict: CONFIRMED, not withdrawn.

### F11-REVISION: seed-aware re-test (URGENT, requested after the SX Fid-/GEF
### pooling artifact) -- CONFIRMED, reproduces in all 5 seeds independently

src/analysis/tox21_f11_seed_check.py. Re-runs F11's EXACT original test (one
paired Wilcoxon per endpoint, n=30 molecules, R3-mean vs R2-zero and vs
R1-hard on |Fid-| and GEF) independently for each of the 5 priority-sweep
seeds -- not one pooled test, 5 separate replications of the original
per-seed methodology. Counts of endpoints (out of 12) significant:

| expl | metric | R3->R2 per seed [s0..s4] | mean | R3->R1 per seed [s0..s4] | mean |
|------|--------|--------------------------|------|---------------------------|------|
| GNN | Fid- | [10,10,7,10,8] | 9.0/12 | [11,10,8,10,10] | 9.8/12 |
| GNN | GEF  | [10,10,8,10,8] | 9.2/12 | [12,10,9,10,11] | 10.4/12 |
| PG  | Fid- | [7,10,9,9,7]   | 8.4/12 | [11,10,10,10,8] | 9.8/12 |
| PG  | GEF  | [7,8,9,9,8]    | 8.2/12 | [11,10,10,10,10]| 10.2/12 |
| SX  | Fid- | [6,9,8,7,7]    | 7.4/12 | [10,11,9,10,11] | 10.2/12 |
| SX  | GEF  | [6,10,8,7,7]   | 7.6/12 | [12,11,9,11,11] | 10.8/12 |

**F11 is CONFIRMED across every seed independently -- no collapse to
near-zero in any seed, for any explainer or metric.** The R3->R2 count never
drops below 6/12 (SX, seed 0) and never exceeds 10/12; R3->R1 never drops
below 8/12 and is frequently 10-12/12. This is the OPPOSITE pattern from
SX's Fid-/GEF-vs-random claim (F17-REVISION), which was 0/5 in every
individual seed and only "worked" once pooled to n=60 -- here every seed
independently reproduces a result close to or above F11's original "~8/12"
estimate. **Why this one survives and SX's didn't**: F11's original test was
never pooled across a hidden replicate dimension to begin with (12
independent per-endpoint tests, not one combined test); it was simply run
on a single seed. Confirming it against 5 MORE independent seeds is exactly
the right check, and it passes. (Caveat: PG's counts (7-10/12) are computed
including its collapsed units, which don't obviously bias this specific
comparison since even a degenerate/uniform mask still gets pushed by the
R2/R1 perturbation; not separately re-checked with collapsed units excluded.)

F12. Per-endpoint explainer separability (block C) replicates F1 and F5 at
     scale. Under R3, essentially nothing separates on any Tox21 endpoint
     (block E: sd(per-molecule) > mean on 12/12 endpoints for every explainer
     x metric -- F1 noise-domination is universal). Under R2, SubgraphX takes a
     significantly higher Fid+ than GNN/PG on 5/12 endpoints (NR-AR **,
     NR-Aromatase ***, NR-ER-LBD **, SR-ARE ***, SR-MMP ***) -- the F5 "R2
     separates SX" effect. GNN vs PG separates on essentially no Tox21 cell
     under any masking. Pooling across the 12 per-endpoint means (block E,
     n=12): SubgraphX is significantly distinct from BOTH GNN and PG on R3 and
     R2 Fid+/Fid-/GEF, while GNN-PG is never significant -- so at the across-
     endpoint level SX is a separate cluster even though within any single
     endpoint per-molecule noise dominates. (Both statements hold; they are at
     different levels of aggregation.)
     SEED-AWARE RE-TEST (src/analysis/tox21_f12_seed_check.py, all 5
     priority-sweep seeds, same per-endpoint per-molecule method):
     CONFIRMED, reproduces every seed -- R2 "SX sig-beats both GNN and PG"
     count ranges 5-8/12 across the 5 seeds (mean ~6.8/12, if anything
     slightly stronger than the single-seed "5/12"), never collapsing to
     near-zero. R3 stays at 0-1/12 in every seed (matches F1). R1 sits at
     1-4/12 (partial washout, between R3's ~0 and R2's ~7, consistent with
     F6/F7). Unlike F9's B-XAIC 3-way ranking (F9-REVISION) or SX's
     Fid-/GEF-vs-random claim (F17-REVISION), this specific claim was never
     fragile -- it holds up cleanly at 5x the seed count.

F13. Cross-metric agreement (GEA order vs Fid+ order, block D) extends F8.
     Over the 4 B-XAIC tasks x {R3,R2,R1} = 12 cells, GEA order matches Fid+
     order in 5: indole/R2, PAINS/R2, and X under ALL of R3/R2/R1; P agrees
     nowhere. GEA and Fidelity coincide only when the decision rule is a single
     discrete localisable feature (halogen presence, X) -- there both metrics
     point the same way at every masking reference. For a ring system (indole)
     or a diffuse rule they agree at most under R2 and usually not at all. The
     faithfulness gap (F8) is the norm; X is the instructive exception that
     shows what full agreement requires, not a counterexample to it.

### F13-REVISION: seed-aware re-test (src/analysis/gt_datasets_seed_audit.py,
### section_f13_f8, all 4 B-XAIC tasks x {R3-mean,R2-zero,R1-hard}, 5 seeds)
### -- RESTATED, headline "X agrees under ALL 3 maskings" does not reproduce

Cross-metric (GEA-order vs Fid+-order) agreement rate per task, out of 15
cells (3 maskings x 5 seeds):
- indole: 3/15 (20%)
- PAINS: 5/15 (33%)
- P: 5/15 (33%)
- X: 9/15 (60%) -- clearly the highest of the 4, but full 3-masking
  agreement (all of R3+R2+R1 in the SAME seed) only happens in 2/5 seeds
  (seed 1, seed 2); the other 3/5 seeds have at least one masking disagree,
  and 2/5 seeds (3,4) have the underlying GEA ranking itself flip
  (PG<->SX swap, matching F9-REVISION's finding that X's GEA ranking is not
  seed-stable). The original claim was built from a single seed/run in
  which X happened to land in its highest-agreement state.
VERDICT: RESTATED, not fully withdrawn. The DIRECTIONAL claim survives --
X (single discrete halogen-presence rule) shows meaningfully higher
cross-metric agreement than the ring/diffuse tasks (60% vs 20-33% of cells)
-- consistent with "agreement is more likely when one feature decisively
sets the label." But the specific headline "X agrees under ALL of R3/R2/R1"
is NOT a reliable, reproducible property: it holds in only 2/5 seeds, not
by default. Report: "cross-metric agreement is elevated but not guaranteed
on the single-discrete-feature task (X, 60% of cells, full 3-masking
agreement in 2/5 seeds), versus rare on ring/diffuse tasks (20-33%)" --
drop the "X agrees under ALL of R3/R2/R1" framing entirely.

F14. GEA BINARIZATION SENSITIVITY (src/analysis/gea_binarization.py -- re-score
     the cached explanations with a top-k=0.25 node mask, the SAME budget
     Fidelity's E_i uses, instead of GraphXAI's mean-threshold).
     - F3 is ROBUST. mutag_graphxai GNN>PG>SX (GNN-SX ***, PG-SX **) and B-XAIC
       indole SX>PG>GNN (all 3 pairs sig under top-k, PG-SX tightens from ns to
       *) hold under BOTH binarizations. The GEA inversion is not a
       thresholding artifact.
     - F9 SPLITS BY GT SIZE. indole (GT ~11 atoms) and X (GT ~2) keep SX
       significantly top under top-k. P (GT ~1.3 atoms) does NOT: mean-threshold
       gave SX 0.38 ~= PG 0.36 >> GNN 0.13 with GNN-SX / GNN-PG **, but under a
       fixed ~10-node top-k budget (>> the 1-atom motif) all three collapse to
       GEA ~0.15 and nothing is significant. PAINS stays weak either way.
       So F9's "SX significantly top on 3/4 tasks" is 2/4 under the matched
       metric.
     - MECHANISM: mean-threshold GEA rewards COMPACT masks. Predicted |M_pr|
       (mean-threshold): GNN 6-15 nodes, PG 3-21 (wildly size-unstable), SX
       3-9 (always compact). On the single-atom tasks SX/PG emit ~3-4 node
       masks that nail the 1-2 atom motif (GEA 0.4-0.6) while GNN's ~14-node
       soft mask scores ~0.1; forcing every explainer to int(0.25*N) nodes
       erases that -- X's SX lead halves (0.55->0.30), P's vanishes. On
       mutag_graphxai / indole (GT 4 / 11, near the 25% budget) rankings barely
       move (<= 0.05).
     - F8 "1 of 8" is UNCHANGED. The GEA ORDERINGS don't change (only the
       values), so GEA-vs-Fid+ agreement is still exactly 1/8 (B-XAIC / R2-Fid+)
       whether or not GEA and Fid+ share a binarization -- matching the
       binarization does not rescue cross-metric agreement.
     TAKEAWAY: report GEA under both binarizations; on datasets whose GT motif
     is much smaller than the top-k budget, mean-threshold GEA is partly a
     mask-size proxy (B-XAIC X, P), and a fixed top-k GEA is partly a
     forced-over-selection penalty. mutag_graphxai and indole are stable under
     both, and F3 holds under both.

### F14-REVISION: seed-aware re-test (src/analysis/f14_seed_check.py,
### mutag_graphxai + 4 B-XAIC tasks, mean-threshold vs top-k=0.25, 5 seeds)
### -- MIXED: indole confirmed, P's instability confirmed and sharpened,
### mutag_graphxai/X weaker than originally framed

Per-seed count of seeds where the two binarizations give the SAME 3-way GEA
order:
- indole: 5/5 -- fully robust, matches the original "F3 is robust under both
  binarizations" claim exactly.
- PAINS: 4/5 (differs at seed 0).
- mutag_graphxai: 3/5 (differs at seeds 1, 3) -- this tracks F3-REVISION v2's
  own finding that the exact GNN>PG>SX order only reproduces in 3/5 seeds
  under mean-threshold alone; binarization choice adds no NEW fragility here,
  it inherits the same fragility already documented.
- X: 3/5 (differs at seeds 3, 4) -- weaker than the original "X keeps SX
  significantly top under top-k" framing suggested. Consistent with
  F9-REVISION/F13-REVISION: X's GEA ranking is independently seed-unstable
  (PG/SX swap) even before asking whether binarization agrees with itself.
- P: 0/5 -- ZERO agreement between mean-threshold and top-k order in every
  single seed. This CONFIRMS and SHARPENS the original claim ("P's SX lead
  vanishes under top-k") -- it is not a single-seed artifact, P's ranking is
  binarization-unstable in literally every seed tested.
VERDICT: indole and the core mechanism claim (mean-threshold rewards compact
masks; forcing a uniform top-k budget erases GT-size-dependent advantages,
worst on the smallest-GT task P) are CONFIRMED, P's instability is now on
firmer footing (0/5, not 1/1). mutag_graphxai and X's binarization-agreement
counts are lower than the single-seed report implied, but this is inherited
from GEA-ranking fragility already flagged in F3-REVISION/F9-REVISION/
F13-REVISION, not a new failure mode. F8's "1 of 8 unchanged under top-k"
sub-clause was not re-tested per seed here (would require pairing this with
per-seed Fid+ orderings) -- left as a gap.

### FULL RQ1 PICTURE (5 core variants + 12 Tox21 endpoints + 4 B-XAIC tasks)

No masking-, metric-, or dataset-invariant "most faithful" explainer:
  - GEA ranking is set by model-rule / annotation alignment (F3, F9):
    SubgraphX wins (significantly) when they coincide -- 3/4 B-XAIC tasks under
    mean-threshold GEA, 2/4 (indole, X) under a Fidelity-matched top-k GEA
    (F14); GNNExplainer when they don't (mutag_graphxai). PAINS (n=7) supports
    this by point estimate only (F9). F3 and the indole/X verdicts are
    binarization-robust; P is a mask-size artifact (F14).
  - R3 Fidelity/GEF is near-vacuous wherever the decision rule is diffuse
    (F1) -- 4 core datasets + all 12 Tox21 endpoints (noise-domination 12/12,
    F12).
  - R2/R1 add "signal" that is mostly perturbation artifact, featurisation-
    and reference-dependent (F4-F7). At scale the raw-integer inflation is
    small but statistically real (F11), not absent.
  - Explainer robustness itself varies: PGExplainer's edge-mask MLP has a ~30%
    per-run chance of a training failure (degenerate uniform mask) on ANY
    near-majority Tox21 model -- seed-dependent, not endpoint- or AUROC-tied
    (F10, confirmed by a 5-seed retest). SubgraphX's compact connected
    explanations are the most stable across masking references, and across-
    endpoint SX is a statistically separate cluster from GNN/PG (F12).
  - Cross-metric (GEA vs Fidelity) agreement happens only for a single
    discrete localisable rule (B-XAIC X); the faithfulness gap is otherwise
    the norm (F8, F13).

## Limitations

- **Two torch / CUDA stacks (comparable, not bit-identical).** The 5 core
  Phase-2/Phase-3 variants and the full R3/R2/R1 masking sweep (every table and
  finding F1-F8) ran on the laptop under **torch 2.14.0 + cu130** (RTX 3050 Ti
  Laptop). The scale-out -- Tox21's 12 endpoints and B-XAIC PAINS/X/P, findings
  F9-F13 -- ran on the Ada 2xV100 box under **torch 2.6.0 + cu118** (driver 525
  caps CUDA at 12.0, so the cu130 wheel will not load). The MP-equivalence test
  (src/train/test_mp_equivalence.py) is bit-identical on both stacks, but
  explainer fitting is stochastic (GNNExplainer/PGExplainer optimisation,
  SubgraphX MCTS rollouts) and the two BLAS/cuDNN builds differ in low-order
  bits, so per-molecule metric values are NOT reproducible across the boundary
  to full precision. Every finding F1-F13 is computed from cells collected
  entirely on ONE stack, so the within-block Wilcoxon/Holm tests are unaffected;
  only a direct numeric splice of a core-table cell against a scale-out-table
  cell would be invalid. Paper wording: "core experiments torch 2.14+cu130,
  scale-out torch 2.6+cu118 on 2xV100; comparable, not bit-identical".
- **Single seed per (dataset, endpoint, task)** for every table and for
  findings F1-F9, F11-F13. One 70/15/15 split, one model init, one explainer-
  training seed. EXCEPTION: F10 was retested with 5 seeds x 6 Tox21 endpoints
  (src/analysis/f10_multiseed.py) -- that is what established the collapse is
  seed noise, not an endpoint property. A broader multi-seed sweep of the
  Fidelity/GEF/GEA tables was not run (compute budget); those effect sizes and
  significance calls are single-seed and could shift a few points under
  re-seeding, though the qualitative findings (F1 noise-domination, F5/F12
  SX-separation direction, F9 GEA orderings) are large enough that a flip is
  unlikely. Also note D-MPNN training is GPU-nondeterministic on the V100, so
  even a re-run at the SAME seed does not reproduce the scale-out checkpoints
  bit-for-bit (see F10).
- **Model seed and explainer seed were COUPLED in the completed 5-seed
  priority sweep (recorded 2026-09, after the seed-aware re-test round that
  produced F1/F2/F4/F5/F6/F7/F8/F13/F14-REVISION and F9/F11/F12-REVISION).**
  scripts/priority_sweep.sh + scripts/_run_unit.sh passed the SAME value `$S`
  to both `src.train.train --seed` (base-model init/training/split RNG) and
  `src.explain.run_phase3 --seed` (PGExplainer's own MLP init/training RNG,
  plus GNNExplainer/SubgraphX stochasticity) for every one of the 25 (5
  B-XAIC/mutag_graphxai units) + 60 (Tox21) = 85 completed units. F10-
  MECHANISM's dedicated decoupling experiment showed collapse is a
  model-weights x MLP-init INTERACTION, not either factor alone -- so any
  "seed 4 collapses" pattern observed in the existing sweep (F9-REVISION,
  F17-REVISION) reflects THAT SPECIFIC (model, MLP-init) pairing, not a
  property of "seed 4" in the abstract, and cannot be cleanly attributed to
  either the model or the explainer alone. This does not invalidate the
  per-seed robustness checks above (each of the 5 units per dataset is still
  a fully independent draw), but it means "seed" in every seed-aware-
  REVISION table above should be read as "one coupled (model-init,
  explainer-init) draw," not as two independently-randomized factors.
  FIXED GOING FORWARD (not retroactive): scripts/_run_unit.sh now accepts an
  `ES` env var for the explainer seed, defaulting to `$S` only for
  back-compat; scripts/priority_sweep.sh now sets `ES=$((s + 1000))` in
  add_job, so any NEW sweep units draw the model seed and the explainer seed
  independently. The 85 already-completed units above were NOT re-run under
  the decoupled protocol (would require re-explaining, not just re-scoring,
  every unit -- out of scope for this audit); this bullet is the record that
  they remain coupled.
- **Small n on some cells.** B-XAIC PAINS GEA has only n=7 GT-present molecules
  (F9 has nothing significant there). Tox21 per-endpoint metrics use n=30
  stratified test molecules; cross-endpoint tests use n=12.
- **mutag_graphxai is the 188-graph / 2-toxicophore GraphXAI builder, not the
  1768-graph one.** GraphXAI ships two: `MUTAG.py` (class MUTAG) over
  TUDataset('MUTAG') = 188 graphs (Debnath 1991), GT = NO2 + NH2 only, one
  merged node mask per graph; and `mutagenicity.py` (class Mutagenicity) over
  TUDataset('Mutagenicity') = 4337 graphs (Kazius 2005) filtered to ~1768,
  GT over 5 toxicophores {NH2, NO2, aliphatic halide, nitroso, azo-type} and a
  combinatorial SET of sub-explanations per graph (the target of the paper's
  Eq. 2 max-over-set GEA). src/data/graphxai_mutag.py ports `MUTAG.py`. Audit
  (src/analysis/mutag_gt_audit.py): of the 188 graphs, all 188 have a NO2/NH2
  GT (none empty); 24/188 additionally contain a halide/azo instance our GT
  leaves unmarked (0 nitroso, 3 azo, 3 true aliphatic-halide, 35 aromatic-ring
  halide instances) -- a small omission in the Debnath nitroaromatic set, but a
  real scope limit vs the Mutagenicity benchmark where all 5 types are common.
  GEA-definition robustness: recomputing mutag_graphxai GEA from the cached
  explanations under merged / max-over-single-motif / max-over-motif-subset
  (Eq. 2) moves every mean by <= 0.06 and does NOT change F3 -- GNN > PG > SX,
  GNN-PG n.s., GNN-SX ***, PG-SX ** under all three. 10/29 test graphs (74/188
  overall) have >= 2 motifs; our single merged mask matches GraphXAI's own
  MUTAG.py exactly.
- **`--max-nodes` sampling exclusion (exact %, per dataset).** SubgraphX
  MCTS cost is highly non-linear in graph size (one 90-node Tox21 molecule
  alone cost 985s vs a 5-56s typical case), so every SubgraphX run excludes
  molecules above a size cap before subsampling -- this silently drops the
  largest few percent of each dataset from anything SubgraphX-derived (GEA,
  Fid, GEF for that explainer; GNN/PG rows are unaffected since they don't
  need the cap). Measured exclusion, `src/analysis/*` timing/stats scripts:
    B-XAIC, --max-nodes 80 (test split, full 7500):
      indole 2.4%  PAINS 3.0%  X 3.1%  P 3.0%   (all tasks share the same
      50k-graph pool + size distribution; exclusion is near-identical)
    Tox21, --max-nodes 60 (NEW as of the priority sweep below; test splits):
      NR-AR 1.5%  NR-AR-LBD 0.8%  NR-AhR 1.3%  NR-Aromatase 1.4%  NR-ER 0.4%
      NR-ER-LBD 1.4%  NR-PPAR-gamma 0.8%  SR-ARE 0.2%  SR-ATAD5 0.8%
      SR-HSE 0.3%  SR-MMP 1.1%  SR-p53 0.9%   (all <=1.5%)
    mutag_graphxai: no cap needed -- mean N=17.9, max N=28 across all 188
    graphs (no large-molecule tail at this dataset's scale).
  State this in the paper as: SubgraphX numbers are computed on the <=98.5-
  97.6% of each dataset within the size cap, not the literal full population.

## Priority scale-up sweep (revised scope -- src/explain/run_phase3.py
## --explain-pool, scripts/priority_sweep.sh, scripts/_run_unit.sh)

Cost-estimate-first (SubgraphX timing + concurrency tests on Ada, see prior
turn) narrowed the scope to 3 priority-ordered tiers, all at concurrency=4
(GPU 1 only), `runs/prio/out_<dataset>_s<seed>.json` per unit:

  P1 (headline finding). B-XAIC, 4 tasks (indole/PAINS/X/P) x 5 seeds.
     N=400 molecules per (task,seed), --stratify-frac 0.9 -> target 360
     positive + 40 negative (vs the original protocol's 13-15 GT-present
     molecules) so GEA/F3/F9's paired Wilcoxon gets real power. --max-nodes 80
     (existing). Estimated ~57h total (measured 4-way SubgraphX rate
     100.2 s/mol/worker on B-XAIC-sized graphs + GNN/PG/train overhead;
     derivation: 5 seeds x 4 tasks, ceil-batched over 4 concurrent slots).
  P2 (cost-confirmation gate, run first despite the P1/P2/P3 priority order
     purely because it is fast). MUTAG-GraphXAI, 5 seeds, --explain-pool all
     = the FULL 188 graphs (not the 29-molecule test split) -- all 188 have
     NO2/NH2 GT, mean N=17.9, max N=28 (no outlier tail; confirmed via a
     direct 20-molecule Ada timing run: 22.9 s/mol single-process). Estimated
     ~2.6-3h for all 5 seeds.
  P3 (cheapest, lowest marginal value -- no GT; F12 already shows R3
     separates nothing on 12/12 Tox21 endpoints, so this is a seed-robustness
     check of F1/F4/F10-F12, not a new sample size). Tox21, 12 endpoints x 5
     seeds, SAME n=30 --stratify-frac 0.5 protocol as the original scale-out,
     now with --max-nodes 60 (see exclusion table above). Estimated ~2.8h.

Execution order actually run: P2 -> P1 -> P3 (P2 first as the "confirm cost"
gate; P1 remains the substantive priority and is not meaningfully delayed by
~3h). `scripts/priority_sweep.sh` is a bash job-queue (`CONCURRENCY=4 bash
scripts/priority_sweep.sh all`) using `wait -n` to cap concurrent
`scripts/_run_unit.sh` workers at C=4; each worker trains a fresh ckpt (seed
drives split+init, same as F10's multi-seed design) then runs the full
gnnexplainer+pgexplainer+subgraphx sweep under --masking all. Units are
idempotent (skipped if their out_*.json already exists) so the sweep is
resumable. Results + exact GT-positive-molecule counts per unit: TODO once
the sweep completes -- see the next spec update.

### P2 COMPLETE (5/5 seeds): does --explain-pool all's full-188 GEA differ from held-out-only?

Measured concern before trusting a "full 188" GEA number: 131/188 molecules
are ones the model TRAINED on, so an explainer could look better there
(memorised decision boundary -> cleaner node-importance signal) than on the
29 held-out. `run_phase3.py` records each explained molecule's ckpt split
membership (`molecule_index`/`molecule_split` in the output JSON); measured
directly (src/analysis/mutag_full_split_gea.py), FINAL, all 5 seeds, n=188
per seed (all 5 explained cleanly -- seed 3's PGExplainer collapsed to a
degenerate all-0.000 mask on all 188, the same seed-dependent training-noise
failure mode as F10, included below since it's a real (if uninformative)
seed replicate, not excluded):

| expl | ALL 188 | TRAIN (n=655) | VAL (n=140) | TEST (n=145) | TRAIN-TEST | Mann-Whitney U |
|------|---------|---------------|-------------|---------------|------------|-----------------|
| GNN  | 0.431   | 0.439         | 0.420       | 0.405         | +0.034     | p=0.230 ns |
| PG   | 0.156   | 0.156         | 0.141       | 0.171         | -0.015     | p=0.347 ns |
| SX   | 0.065   | 0.062         | 0.067       | 0.076         | -0.014     | p=0.796 ns |

(pooled over all 5 seeds; n = molecule x seed pairs, 131/28/29 per seed x 5 = 655/140/145)

**No significant TRAIN-vs-TEST gap for any explainer, confirmed at full n.**
If anything TEST is numerically (non-significantly) HIGHER for PG and SX.
**Verdict: the full-188 pool is justified, not an in-sample-inflated number.**
mutag_graphxai's headline GEA is reported as ALL 188 (no train/test split
caveat needed) once the per-seed significance tables are built. (Contrast
with F10, where an identical "is this a real effect" check on Tox21 DID
reveal a seed-noise artifact -- this one measured clean at both 3/5 and 5/5
seeds.)

### F3-REVISION v1 (SUPERSEDED -- pseudo-replicated, do not cite the p-values below)

~~src/analysis/mutag_full_pooled_significance.py, n=940 molecule-seed pairs~~
~~GNN-PG p_Holm=7.8e-94 *** ; GNN-SX p_Holm=8.6e-133 *** ; PG-SX p_Holm=5.4e-20 ***~~
WRONG: the 940 "observations" are 188 molecules x 5 seeds, not 940
independent draws -- the same molecule recurs 5x, so Wilcoxon's n is
overstated by ~5x and the p-values are inflated (p~1e-94/1e-133 is itself
the tell -- real effects on n~200 basically never produce numbers that
small). Superseded by the seed-aware analysis below. The DESCRIPTIVE means
(GNN 0.431, PG 0.156, SX 0.065) are fine to keep -- pooling is only invalid
for the SIGNIFICANCE test, not for reporting a plain average.

### F3-REVISION v2 (seed-aware, correct): per-seed + seed-level tests

src/analysis/seed_correct_significance.py. The independent unit is the SEED
(a fresh model fit), not the (molecule, seed) pair. Two honest tests, both
reported (they trade power for validity in opposite directions):

**1. Per-seed test** (Wilcoxon+Holm within each seed, n=188/seed -- same
scope F3 originally used, just repeated 5x):

| seed | GNN-PG | GNN-SX | PG-SX | order (by mean) | GNN/PG/SX |
|------|--------|--------|-------|------------------|-----------|
| 0 | \*\*\* (GNN) | \*\*\* (GNN) | \* (PG) | GNN>PG>SX | 0.433/0.243/0.030 |
| 1 | \*\*\* (GNN) | \*\*\* (GNN) | \* (tie-ish) | GNN>PG>SX | 0.122/0.066/0.020 |
| 2 | \*\*\* (GNN) | \*\*\* (GNN) | \* (tie-ish) | **GNN>SX>PG** | 0.492/0.070/0.113 |
| 3 | \*\*\* (GNN) | \*\*\* (GNN) | \*\*\* (tie-ish) | **GNN>SX>PG** | 0.496/0.000/0.048 |
| 4 | \*\*\* (GNN) | \*\*\* (GNN) | \*\*\* (PG) | GNN>PG>SX | 0.612/0.400/0.113 |

GNN beats BOTH PG and SX significantly in 5/5 seeds -- that part of F3 is
robust. But "PG > SX" (the second/third-place ordering) reproduces in only
**3/5 seeds**; in seeds 2 and 3, SX numerically/significantly beats PG. So
the exact "GNN > PG > SX" ordering holds in 3/5 seeds, not all 5.

**2. Seed-level test** (paired Wilcoxon on the n=5 per-seed means -- the test
that actually respects independence; n=5 means the minimum attainable
two-sided p is 0.0625, so it structurally CANNOT reach p<0.05 even when
every seed agrees -- read the sign/consistency count, not the p-value):

| pair | direction consistency | p_raw | p_Holm |
|------|------------------------|-------|--------|
| GNN-PG | GNN higher in **5/5** seeds | 0.0625 | 0.1875 ns |
| GNN-SX | GNN higher in **5/5** seeds | 0.0625 | 0.1875 ns |
| PG-SX | PG higher in only 3/5 seeds | 0.4375 | 0.4375 ns |

Nothing is formally significant at n=5 (expected, by construction), but the
SIGN is perfectly consistent for GNN>PG and GNN>SX (5/5) and NOT consistent
for PG>SX (3/5) -- the same qualitative conclusion as the per-seed test,
now from the test that doesn't overstate n.

**REVISED F3 CLAIM: "GNN significantly beats both PG and SX on
mutag_graphxai, robustly (5/5 seeds, both tests). PG-vs-SX for 2nd place is
NOT robust (3/5 seeds) -- report 'GNN >> {PG, SX}' as the honest ordering,
not the strict 'GNN > PG > SX' from either the single-seed or the
(pseudo-replicated) pooled test."**

**Is the seed variance PG-specific?** NO. Per-explainer per-seed GEA:

| expl | seed0 | seed1 | seed2 | seed3 | seed4 | abs. range | CV (std/mean) |
|------|-------|-------|-------|-------|-------|------------|----------------|
| GNN  | 0.433 | 0.122 | 0.492 | 0.496 | 0.612 | **0.489** (largest) | 0.38 (lowest) |
| PG   | 0.243 | 0.066 | 0.070 | 0.000 | 0.400 | 0.400 | **0.94 (highest)** |
| SX   | 0.030 | 0.020 | 0.113 | 0.048 | 0.113 | **0.094 (smallest)** | 0.63 |

GNN has the LARGEST absolute seed-to-seed range (0.489, spanning nearly half
the [0,1] Jaccard scale) -- MORE than PG's (0.400). Seed variance is a
general property of this pipeline (small dataset, 5 independent model fits),
not something unique to PGExplainer. What IS unique to PGExplainer is not
the magnitude of its variance but its KIND: PG's low end is a literal
collapse to a degenerate all-0.000 mask (F10's failure mode) -- a
qualitatively distinct catastrophic mode never seen in GNN or SX, whose
variability stays "noisy but functioning" across the full range. By relative
dispersion (CV) PG is still the least stable (0.94, driven by the collapse
pulling its low end to exactly 0) and GNN is actually the MOST relatively
stable (0.38, because its mean is high enough to absorb the same absolute
swing) -- SX sits in between (0.63) despite having the smallest absolute
range, because its mean is so low that even a small absolute swing is a
large fraction of it.

**Bottom line for the paper:** (a) GNN's seed-to-seed swing is real and
larger in absolute terms than PG's -- F3's headline (GNN wins) rests on
consistent DIRECTION across seeds, not on GNN being some noise-free
explainer; a good chunk of GNN's apparent margin over PG/SX is itself seed
luck in magnitude, just never in sign. (b) PG's problem is not "high
variance" generically -- it's the collapse. (c) "PGExplainer is the stable
method" (B-XAIC-section wording) still does not stand: PG has the worst CV
of the three AND the only catastrophic-failure mode. Revised wording stands
as before: most cross-dataset-consistent RANKING among successful runs,
least run-to-run RELIABLE explainer overall.

**Apply this SAME correction (per-seed test + seed-level test on per-seed
means, never a naive pool-then-Wilcoxon across seeds) to P1's B-XAIC
significance once it lands, and to any other multi-seed significance
computed going forward.**

### F15. Random baseline (mutag_graphxai) -- the anchor GEA has been missing

src/analysis/mutag_random_baseline.py. GraphXAI itself flags trivially-
recoverable ground truth as a pitfall, and SX's 0.065 / PG's 0.156 meant
nothing without a chance-level number. Two matched-top-k=0.25 random
constructions, 5 seeds (n=188/seed, same seeds as everything else):
  random-node: uniform k-of-N node subset (baseline for GNN/SX, node-level).
  random-edge: uniform random per-edge score -> node_importance_from_edges
    (the SAME fold PGExplainer's real output goes through) -> top-k=0.25 on
    the folded importance (baseline for PG, edge-level).

|      | seed0 | seed1 | seed2 | seed3 | seed4 | mean  | std   |
|------|-------|-------|-------|-------|-------|-------|-------|
| GNN  | 0.433 | 0.122 | 0.492 | 0.496 | 0.612 | 0.431 | 0.165 |
| PG   | 0.243 | 0.066 | 0.070 | 0.000 | 0.400 | 0.156 | 0.146 |
| SX   | 0.030 | 0.020 | 0.113 | 0.048 | 0.113 | 0.065 | 0.041 |
| RandN| 0.149 | 0.145 | 0.141 | 0.140 | 0.132 | 0.141 | 0.005 |
| RandE| 0.194 | 0.158 | 0.176 | 0.184 | 0.165 | 0.175 | 0.013 |

(the random baselines are themselves extremely stable across seeds, CV
0.03-0.07 -- as expected, they don't depend on a trained model, only on
graph structure + the law of large numbers over 188 molecules; this is also
a sanity check that the implementation is doing what it says.)

**SubgraphX does NOT beat random-node selection on mutag_graphxai --
SX loses to RandN in 5/5 seeds** (per-seed Wilcoxon *** in 3/5, seed-level
sign count RandN-higher 5/5, the maximum possible consistency at n=5). In
seeds 1 and 3, SX (0.020, 0.048) is 3-7x BELOW RandN (0.145, 0.140). This is
the sharp statement the mean alone couldn't make: on mutag_graphxai, SX's
"connected prediction-preserving subgraph" objective is not just a poor
match for the NO2/NH2 motif (F3's mechanism) -- it is, per-seed,
indistinguishable from or worse than picking nodes at random.

**GNNExplainer robustly beats random-node -- GNN higher in 4/5 seeds**
(***, large margins, e.g. seed 4: 0.612 vs 0.132), losing only in its own
weakest seed (seed 1: 0.122 vs 0.145, RandN wins narrowly at **). This is a
real signal, distinct from SX and PG.

**PGExplainer does NOT consistently beat its own matched random-edge
baseline -- PG higher in only 2/5 seeds** (the same 2 seeds where PG's
absolute GEA is highest, 0.243 and 0.400); RandE beats PG in the other 3,
including the collapse seed but ALSO 2 non-collapsed ones (seed1: PG=0.066 <
RandE=0.158; seed2: PG=0.070 < RandE=0.176). Same conclusion vs random-node
(PG higher only 2/5). **PG's positive-looking pooled mean is driven by 2
lucky seeds, not a consistent above-chance signal** -- a materially weaker
claim than "PG beats random."

Full cross-battery (seed-level, n=5, Holm over 10 pairs) -- nothing clears
significance at this n (structural floor, as in F3-REVISION v2), but the
DIRECTION COUNTS are the finding: GNN beats RandN/RandE 4/5; SX loses to
RandN/RandE 5/5 (i.e. RandN/RandE beat SX 5/5); PG beats RandN 2/5 and loses
to RandE 3/5 (RandE beats PG 3/5). RandE > RandN in 5/5 (the edge->node fold
is itself a mild structural prior, worth noting but not the headline).

**Bottom line for the paper:** on mutag_graphxai, only GNNExplainer clears a
random baseline with any consistency. SubgraphX is at-or-below chance in
every seed -- sharper and more damaging than "SX scores low" (F3's original
framing). PGExplainer's above-chance appearance is seed-dependent, not
structural. This reframes F3's ranking: it is not "three explainers of
varying quality," it is "one explainer (GNN) with real signal, one
(SubgraphX) indistinguishable from noise, and one (PGExplainer) that is
sometimes real and sometimes not, unpredictably by seed." Random baselines
for B-XAIC (where GEA orderings look the opposite way, F9) and Tox21 Fidelity
are a natural next step -- not yet run.

### F15-ADDENDUM: connectivity-matched random baseline (src/analysis/
### connectivity_baseline.py, mutag_graphxai, 5 seeds) -- SURVIVES, and IS
### SHARPENED, not weakened, by the fair null

The objection: SubgraphX only ever proposes CONNECTED subgraphs, so
comparing it to F15's unconstrained random-node baseline (RandN, fixed
top-k=0.25 budget) may be unfair on compact motifs -- an unconstrained null
can freely pick any node subset, which could itself be a disadvantage (or
advantage) relative to SX's structural constraint. Built two NEW nulls,
matched per-molecule to SX's own real mask size (mean-threshold, not the
fixed 0.25 budget): `RandConn` (random CONNECTED node-induced subgraph, same
size, via random frontier growth) and `RandUnc` (random unconstrained subset,
same size -- the direct apples-to-apples control for the size change alone,
separate from F15's fixed-budget RandN). 20 random draws/molecule, averaged,
n=188 molecules (all of mutag_graphxai, matching P2's --explain-pool all):

| seed | mean\|mask\| | SX | RandConn | RandUnc | SX vs RandConn | SX vs RandUnc |
|------|------|-------|----------|---------|-----------------|-----------------|
| 0 | 4.1 | 0.030 | 0.102 | 0.136 | p=7e-19 *** (SX wins 20/188) | p=2e-24 *** (SX wins 17/188) |
| 1 | 4.4 | 0.020 | 0.109 | 0.144 | p=6e-28 *** (SX wins 17/188) | p=1e-31 *** (SX wins 9/188) |
| 2 | 4.2 | 0.113 | 0.110 | 0.142 | p=3e-03 ** (SX wins 49/188) | p=7e-06 *** (SX wins 44/188) |
| 3 | 4.4 | 0.048 | 0.106 | 0.144 | p=4e-18 *** (SX wins 24/188) | p=1e-21 *** (SX wins 19/188) |
| 4 | 4.4 | 0.113 | 0.111 | 0.141 | p=3e-03 ** (SX wins 51/188) | p=1e-05 *** (SX wins 48/188) |

SX loses to the FAIR (connectivity- AND size-matched) null, significantly,
in ALL 5/5 seeds -- and loses to it MORE often per-molecule (9-51/188 wins,
5-27%) than it loses to the unconstrained same-size null in only a slightly
smaller margin. The gap between RandConn and RandUnc themselves is small and
in the expected direction (RandConn always a bit lower than RandUnc -- the
real "cost" of the connectivity constraint on this dataset) but that cost is
tiny (0.03-0.04 absolute GEA) next to the SX-vs-RandConn gap (0.06-0.08).
VERDICT: the fairness objection does NOT rescue SX -- if anything it
sharpens the finding. A random CONNECTED subgraph of the exact size SX
itself chose still beats SX's actual choice of WHICH connected subgraph in
188/188 molecules' aggregate and in the large majority of individual
molecules. This rules out "connectivity was an unfair handicap" as an
explanation and rules IN "SX is choosing badly among connected subgraphs of
the right size and shape family," not merely "SX is structurally
disadvantaged by an unconstrained comparison." F15's headline stands.

### F16. Random baseline, B-XAIC -- ALL 4 TASKS COMPLETE (20/20 P1 units) --
### SX beats random with ZERO exceptions across all 20 seed-task units; PAINS
### and X each independently reveal a PGExplainer collapse instance; see
### "F16 WRAP-UP VERDICT, FINAL" below

src/analysis/bxaic_random_baseline.py, run against whatever P1 units had
completed at each check (no waiting for the full sweep), finalised once all
20 units (4 tasks x 5 seeds) landed. Same matched-top-k=0.25 construction as
F15.

| task | seed | GNN | PG | SX | RandN | RandE |
|------|------|-----|-----|-----|-------|-------|
| indole | 0 | 0.102 | 0.511 | 0.684 | 0.169 | 0.155 |
| indole | 1 | 0.257 | 0.537 | 0.679 | 0.166 | 0.145 |
| indole | 2 | 0.360 | 0.536 | 0.722 | 0.158 | 0.155 |
| indole | 3 | 0.361 | 0.563 | 0.640 | 0.169 | 0.151 |
| indole | 4 | 0.150 | 0.513 | 0.567 | 0.156 | 0.150 |
| indole | mean | 0.246 | 0.532 | 0.658 | 0.164 | 0.151 |
| PAINS | 0 | 0.331 | 0.361 | 0.465 | 0.162 | 0.161 |
| PAINS | 1 | 0.322 | 0.442 | 0.475 | 0.167 | 0.154 |
| PAINS | 2 | 0.252 | 0.277 | 0.501 | 0.169 | 0.163 |
| PAINS | 3 | 0.336 | **0.000 (collapsed)** | 0.520 | 0.165 | 0.165 |
| PAINS | 4 | 0.278 | **0.000 (collapsed)** | 0.497 | 0.159 | 0.158 |
| PAINS | mean | 0.304 | 0.216 | 0.492 | 0.165 | 0.160 |
| X | 0 | 0.066 | 0.435 | 0.329 | 0.053 | 0.080 |
| X | 1 | 0.062 | 0.535 | 0.382 | 0.056 | 0.076 |
| X | 2 | 0.061 | 0.525 | 0.438 | 0.053 | 0.073 |
| X | 3 | 0.061 | 0.313 | 0.336 | 0.056 | 0.069 |
| X | 4 | 0.064 | **0.030 (near-collapse)** | 0.430 | 0.060 | 0.072 |
| X | mean | 0.063 | 0.368 | 0.383 | 0.055 | 0.074 |
| P | 0 | 0.128 | 0.225 | 0.373 | 0.043 | 0.024 |
| P | 1 | 0.180 | 0.273 | 0.249 | 0.040 | 0.022 |
| P | 2 | 0.174 | 0.295 | 0.325 | 0.043 | 0.022 |
| P | 3 | 0.237 | 0.294 | 0.308 | 0.047 | 0.023 |
| P | 4 | 0.327 | 0.268 | 0.241 | 0.037 | 0.026 |
| P | mean | 0.209 | 0.271 | 0.299 | 0.042 | 0.023 |

**THE ANSWER TO THE SPECIFIC QUESTION: SubgraphX beats random-node decisively
on every B-XAIC task checked, with ZERO exceptions across all 15 completed
seed-task units** -- indole 5/5 (p~1e-58 to 1e-60, ~4x margin), PAINS 5/5
(p~1e-45 to 1e-52, ~3x, UNAFFECTED by PG's collapses in the same seeds), X
5/5 (p~1e-41 to 1e-51, ~6-8x, the widest margins of any task). **SX vs
RandN is 15/15.**

PGExplainer beats its matched random-edge baseline in indole 5/5, but only
**3/5 on PAINS** (seeds 3-4 fully COLLAPSED, all-0.000 mask, n_degenerate/
n=400/400 -- meets the formal collapse threshold) and **4/5 on X** (seed 4,
PG=0.030, still significantly BELOW RandE, p=3.7e-09).
CORRECTION (reconciled against runs/prio/collapse_table.csv, which parses
explainer_health mechanically): X seed 4 is NOT a collapse by the project's
own n_degenerate/n>=0.5 criterion -- its explainer_health shows
n_degenerate=0/400, n_empty=0, n_whole=0, i.e. a structurally valid,
non-constant mask. Calling it "near-collapsed" here conflated two different
things: (1) a literal degenerate/constant mask (PAINS's failure mode, a
binary training-failure diagnostic) and (2) a valid-but-poor explanation
that still scores significantly BELOW random -- a DIFFERENT, separately
important failure mode, now its own finding: see F18, which found this is
NOT unique to X seed 4 (10/75 tests flagged, common on mutag_graphxai,
rare on B-XAIC). Restate: PAINS seeds 3-4 are B-XAIC's only two formal
collapses (2/20 B-XAIC units, 10%, not 3/20/15%); X seed 4 belongs to F18,
not to the collapse-rate tally.
Across THREE different datasets: Tox21 (F10, ~20-30% estimated rate),
mutag_graphxai seed 3 (F3-REVISION v2), and B-XAIC PAINS seeds 3-4 (this
entry) -- 2 of 20 B-XAIC (task,seed) units formally collapsed (10%),
consistent with F10's estimated failure rate given small-n. PGExplainer's
training instability is a general property of the method, not tied to any
one dataset.

GNNExplainer beats random on indole in 3/5 seeds (loses 2, one ns), PAINS
5/5 (all ***), X 5/5 but only 3/5 SIGNIFICANTLY (seeds 3-4 are ns, p=0.340
and p=0.256) -- **X is, across its full 5 seeds, the one task where GNN's
margin over random is consistently thin and often not significant**, unlike
its solid, consistent wins on PAINS and (mostly) indole.

**This is the OPPOSITE of mutag_graphxai (F15), where SX failed to beat
random in 5/5 seeds.** The inversion is not trivially-recoverable ground
truth -- SX's B-XAIC wins (15/15 seed-task units across all 3 completed
tasks) and its mutag_graphxai loss (0/5) are both real against a proper
null. Supports, does not undermine, the model-rule-alignment reading of
F3/F9.

**X's flagged risk did NOT materialise, confirmed across all 5 seeds.** X's
RandN GEA sits at 0.053-0.060 across every seed -- consistently the LOWEST
of any B-XAIC task (vs indole's ~0.164, PAINS's ~0.165), not the highest as
the risk framing worried. SX clears it every time by 5-8x; GNN's margin is
thin but still positive-direction in 5/5. **F13's "X is where GEA and Fid+
agree under every masking" does NOT need a trivial-recoverability caveat --
settled, not just likely.**

**PAINS was flagged by F9 as "weak/non-significant among the real explainers
themselves"** (GNN vs PG vs SX don't separate well from EACH OTHER there,
n=7 in the original 1-seed protocol) -- **a DIFFERENT axis from beating
random.** At N=400/seed, GNN and SX beat random on PAINS in 5/5 seeds each;
PAINS has real signal for two of the three explainers, it just doesn't
distinguish GNN/PG/SX well from each other (and PG's own signal there is
now further complicated by its 2/5 collapse rate).

**P COMPLETE (5/5 seeds) -- clean sweep, zero collapses, ALL THREE explainers
beat random in ALL 5 seeds, and GNN's margin is STRONG throughout, not
thin:** GNN vs RandN p in [4.8e-35, 1.6e-57] all ***; SX vs RandN p in
[6.8e-56, 4.4e-59] all ***; PG vs RandE p in [1.4e-49, 2.3e-63] all ***. P's
random baselines are the LOWEST of any B-XAIC task (RandN 0.037-0.047,
RandE 0.022-0.026 -- even lower than X's), consistent with P having the
tiniest GT of the four tasks (~1.3 atoms, F9). **The X-specific hypothesis
("smallest GT -> thinnest GNN margin") does NOT generalise to P**, which has
an even smaller GT than X but a solid, unambiguous GNN signal (its best
seed, seed 4, hit GNN=0.327 -- P's own single highest value across every
task/seed in this table) -- so whatever makes GNN's B-XAIC margin thin on X
is task-specific to X (its particular halogen-detection rule), not simply
"GT size". **P is also the ONLY B-XAIC task with zero PG collapses across
all 5 seeds** (vs PAINS's 2 and X's 1).

### F16 WRAP-UP VERDICT, FINAL (20/20 P1 units -- indole, PAINS, X, P all complete)

1. **SubgraphX beats random-node with ZERO exceptions across all 20
   completed B-XAIC seed-task units** (indole 5/5, PAINS 5/5, X 5/5, P 5/5)
   -- the cleanest possible confirmation that SX's B-XAIC success is real
   and general across task TYPES (ring system, alert-set, halogen presence,
   phosphorus presence), not a property of any one task. Combined with F15
   (SX at/below random on mutag_graphxai, 0/5), the GEA inversion between
   the two GT datasets (F3, F9) is a real, structural difference in what
   SubgraphX's objective can recover, not a trivial-ground-truth artifact on
   either side. **Final SX vs RandN tally across both GT datasets: 20/20 on
   B-XAIC, 0/5 on mutag_graphxai -- as clean an inversion as this kind of
   result gets.**
2. **PGExplainer collapse is a FIVE-dataset-spanning phenomenon**, with 2
   formal collapses inside B-XAIC (out of 20 units, 10% -- CORRECTED from an
   earlier "3/20, 15%" that miscounted X seed 4 as a collapse; its
   explainer_health is clean, n_degenerate=0/400, see the correction note
   above under "PGExplainer beats its matched random-edge baseline...";
   X seed 4 IS still a real, separately-reported finding -- a valid mask
   that scores significantly below random -- just not a collapse): PAINS
   seeds 3-4 (full collapse, n_degenerate=400/400 both) -- P and indole show
   ZERO collapses across their 10 combined seeds -- plus Tox21 (F10) and
   mutag_graphxai seed 3 (F3-REVISION v2). 10% is consistent with F10's
   ~20-30% Tox21 estimate (small-n, same order of magnitude).
   When PG doesn't collapse, it clears random decisively every time (17/17
   non-collapsed instances).
3. **GNNExplainer's random-margin varies by TASK, not by GT size**: robust
   on PAINS (5/5 ***) and P (5/5 ***), mostly robust on indole (3/5 ***,
   2/5 losing -- one significantly), genuinely thin/inconsistent ONLY on X
   (3/5 *, 2/5 ns, though positive-direction in all 5). P has a SMALLER GT
   than X (~1.3 vs ~2.3 atoms) yet the strongest, cleanest GNN signal of any
   B-XAIC task -- the thin-margin phenomenon is specific to X's particular
   halogen-detection mechanism, not a general "tiny GT" rule.
4. **F13's "X is where GEA and Fid+ agree" needs no trivial-recoverability
   caveat** -- settled at 5/5 X seeds -- and P (even smaller GT, even lower
   random baseline, the cleanest signal of any task) confirms the same
   conclusion rather than contradicting it.
5. **Priority-sweep tally**: this closes out P1 (20/20 B-XAIC units) and,
   with F15 (mutag_graphxai, 5/5 seeds x full 188 graphs) and F17 (Tox21),
   completes the random-baseline check across every dataset in the priority
   sweep.

### F16-ADDENDUM: connectivity-matched random baseline (src/analysis/
### connectivity_baseline.py, indole, 5 seeds) -- SURVIVES cleanly

Same construction as F15-ADDENDUM, applied to indole (n=360 GT-present
molecules/seed -- the priority-sweep's larger N=400/stratify-0.9 sample, not
the original scale-out's n=13):

| seed | mean\|mask\| | SX | RandConn | RandUnc | SX vs RandConn | SX vs RandUnc |
|------|------|-------|----------|---------|-----------------|-----------------|
| 0 | 7.7 | 0.684 | 0.181 | 0.162 | p=2e-60 *** (SX wins 352/360) | p=2e-60 *** (SX wins 352/360) |
| 1 | 7.7 | 0.679 | 0.182 | 0.161 | p=3e-60 *** (SX wins 352/360) | p=2e-60 *** (SX wins 352/360) |
| 2 | 7.7 | 0.722 | 0.183 | 0.164 | p=1e-60 *** (SX wins 356/360) | p=1e-60 *** (SX wins 358/360) |
| 3 | 7.6 | 0.640 | 0.181 | 0.163 | p=4e-60 *** (SX wins 349/360) | p=3e-60 *** (SX wins 349/360) |
| 4 | 7.5 | 0.567 | 0.169 | 0.151 | p=7e-60 *** (SX wins 349/360) | p=4e-60 *** (SX wins 352/360) |

SX beats the fair, connectivity- and size-matched null overwhelmingly in
5/5 seeds, winning in 97-99% of individual molecules -- as clean as F16's
original unconstrained-baseline result. Note the connectivity "cost" here is
essentially negligible (RandConn ~0.17-0.18 vs RandUnc ~0.15-0.16, a small
gap in the OPPOSITE direction from a handicap -- being forced connected
helps the random null slightly on indole's large, contiguous ring motif).
VERDICT: F16's headline stands, unweakened by the fairness objection.
Combined with F15-ADDENDUM, the GEA inversion between mutag_graphxai (SX
loses even to a fair null) and indole (SX wins overwhelmingly against the
same kind of fair null) is confirmed to be a real property of what
SubgraphX's objective can recover on each dataset's motif -- not an artifact
of comparing it to an unconstrained random baseline.

### F15/F3/F10 DECOUPLED-PROTOCOL VERIFICATION (mutag_graphxai, 5 units,
### src/analysis/decoupled_verify.py, runs/decoupled/) -- ALL THREE HOLD

Every completed unit so far (85, across F1-F17) used the SAME value for the
model-training seed and the explainer-init seed (see the Limitations
bullet). This re-runs mutag_graphxai's 5 units under the now-decoupled
protocol (scripts/decoupled_mutag_check.sh): the EXISTING trained
checkpoints are reused unchanged (runs/prio/ckpt_mutag_graphxai_s{0-4}.pt --
training is not the factor under test, see the cost-estimate note below),
but each is now explained with an INDEPENDENT explainer-init seed
ES = S + 1000 instead of ES = S. Outputs in runs/decoupled/, originals in
runs/prio/ untouched. This isolates: do F15/F3/F10's findings depend on the
explainer's own stochastic init numerically coinciding with the model's
training seed, or do they hold regardless of that pairing?

**1. F15 (SX vs connectivity-matched random null) -- HOLDS, 5/5 seeds:**

| seed | model-seed | explainer-seed | SX | RandConn | p (SX vs RandConn) | SX wins |
|------|-----------|-----------------|-------|----------|---------------------|---------|
| 0 | 0 | 1000 | 0.025 | 0.105 | 4e-24 *** | 17/188 |
| 1 | 1 | 1001 | 0.020 | 0.111 | 6e-28 *** | 16/188 |
| 2 | 2 | 1002 | 0.120 | 0.108 | 7e-03 ** | 50/188 |
| 3 | 3 | 1003 | 0.063 | 0.108 | 8e-13 *** | 28/188 |
| 4 | 4 | 1004 | 0.123 | 0.111 | 2e-02 * | 53/188 |

SX loses to the fair, connectivity-matched null significantly in 5/5 seeds
-- same as the coupled result (F15-ADDENDUM: 9-51/188 wins there vs
16-53/188 here, same order of magnitude, same conclusion).

**2. F3-REVISION v2 (GNN >> {PG, SX}) -- HOLDS, 5/5 seeds; PG-vs-SX still
unstable, same 3/5-vs-2/5 split (different seeds flip, same overall pattern):**

| seed | GNN-PG | GNN-SX | order | means (GNN/PG/SX) |
|------|--------|--------|-------|---------------------|
| 0 | \*\*\* (GNN) | \*\*\* (GNN) | GNN>PG>SX | 0.443/0.197/0.025 |
| 1 | \*\*\* (GNN) | \*\*\* (GNN) | GNN>PG>SX | 0.108/0.051/0.020 |
| 2 | \*\*\* (GNN) | \*\*\* (GNN) | GNN>PG>SX | 0.495/0.135/0.120 |
| 3 | \*\*\* (GNN) | \*\*\* (GNN) | GNN>SX>PG | 0.490/0.000/0.063 |
| 4 | \*\*\* (GNN) | \*\*\* (GNN) | GNN>SX>PG | 0.609/0.039/0.123 |

GNN beats both PG and SX significantly in 5/5 seeds (identical to the
coupled result). Strict "GNN>PG>SX" holds in 3/5 seeds (0,1,2) here vs 3/5
seeds (0,1,4) in the coupled run -- the SAME 3-vs-2 split, just seed 2
(not seed 4) is now the one that stays GNN>PG>SX, and seed 4 (not seed 2)
now flips.
CORRECTION (this was mischaracterized in an earlier draft of this section as
a "genuine closeness-of-competing-signals effect" -- that framing is wrong
and has been struck): per F15/F15-ADDENDUM, SX is AT OR BELOW a fair random
null on mutag_graphxai in 5/5 seeds, and per the original F15 analysis PG
beats its own random baseline in only 2/5 (non-collapsed) seeds -- a coin
flip. Neither explainer has a reliable above-chance signal on this dataset.
Two near-chance quantities swapping order across seeds -- which specific
seed "wins" changing when the explainer-seed changes -- is what sampling
noise around a null looks like, not a real competition between two valid
signals. The unstable PG-vs-SX middle ranking is a CONSEQUENCE of both
explainers lacking signal here, not a separate, informative finding in its
own right. Report it as: "PG-vs-SX ordering on mutag_graphxai is noise,
consistent with neither explainer clearing a random baseline" -- not as
evidence of some third phenomenon. Seed-level test: GNN higher in 5/5 for
both GNN-PG and GNN-SX (same as coupled, ns by construction at n=5); PG
higher in 3/5 for PG-SX (same 3/5 as coupled) -- itself the expected rate
for a coin flip between two noise-level quantities.

**3. F10 (PGExplainer collapse rate) -- HOLDS: still exactly 1/5, still
seed 3, now under explainer-seed 1003 instead of 3:**

| seed | model-seed | explainer-seed | n_degenerate/n | status |
|------|-----------|-----------------|------------------|--------|
| 0 | 0 | 1000 | 0/188 | clean |
| 1 | 1 | 1001 | 0/188 | clean |
| 2 | 2 | 1002 | 0/188 | clean |
| 3 | 3 | 1003 | 188/188 | **COLLAPSED** |
| 4 | 4 | 1004 | 0/188 | clean |

Seed 3 collapses under BOTH its original coupled explainer-seed (3) and its
new decoupled explainer-seed (1003) -- two different, unrelated explainer
inits, same model checkpoint, same outcome. Seed 4 -- clean under coupled --
stays clean under decoupled too. This is direct, model-specific evidence for
F10-MECHANISM's claim: collapse on this dataset tracks ckpt_mutag_graphxai_s3's
particular weights, not the numeric coincidence of the shared seed value.

**VERDICT: all three headline claims (F15, F3-REVISION v2, F10's collapse
rate) reproduce under the decoupled protocol on mutag_graphxai, with numbers
close enough to the coupled run that the seed-coupling confound does not
appear to have been driving any of them here.** Cost note: this re-run
reused the existing checkpoints (training is not the factor under test) and
still finished in ~5h wall-clock (13:40-18:43, faster than the ~14h
estimate -- likely lower external node contention at the time), at
CONCURRENCY=4, GPU-only explain cost. Scope: mutag_graphxai only, per the
user's explicit request; B-XAIC's decoupled re-run (P1, 20 units, the
expensive tier) has NOT been run and is a separate decision -- see the
recommendation in the commit/report accompanying this section.

**PAPER PARAGRAPH (drop-in, Methods/Robustness subsection):**

> To rule out a protocol confound, we re-ran the mutag_graphxai evaluation
> (5 units) with the base model's training seed and the explainer's own
> initialization seed decoupled, rather than fixed to the same value as in
> the main sweep. Each unit reused its original trained checkpoint unchanged
> and was re-explained under a fresh, independent explainer-init seed. All
> three seed-sensitive claims we report for this dataset reproduced:
> SubgraphX remained at or below a connectivity- and size-matched random
> baseline in 5/5 seeds; GNNExplainer significantly outperformed both
> PGExplainer and SubgraphX in 5/5 seeds; and PGExplainer's collapse rate was
> unchanged at 1/5. The collapse result is the more informative one: the
> same checkpoint (seed 3) collapsed to a degenerate, uniform edge mask
> under two independent, unrelated explainer initializations (its original
> coupled seed and a newly assigned, unrelated one). Because the only factor
> these two runs share is the trained model's weights, this is direct
> evidence that PGExplainer's failure on this checkpoint is a property of
> the model, not an artifact of the seed value used to initialize the
> explainer -- ruling out the alternative explanation that our seed-coupled
> protocol was itself manufacturing the appearance of a stable collapse
> rate.
>
> We did not extend this check to B-XAIC or Tox21, which remain evaluated
> under the original coupled protocol. This is a judgment call, not only a
> cost one -- a comparable B-XAIC re-run is roughly an order of magnitude
> more expensive (20 units at several hours each). The seed-coupling
> confound's most concerning failure mode is a model-weights x
> initialization interaction masquerading as seed noise; that is exactly
> what the collapse mechanism check above tests, directly, and the result
> was decisive rather than ambiguous. We therefore treat the
> mutag_graphxai decoupling result as evidence that the mechanism
> generalizes, and report the coupled seeds on B-XAIC and Tox21 as a
> disclosed limitation rather than an unresolved threat to validity.

### F17. Random baseline, Tox21 Fidelity/GEF (all 12 endpoints, existing scale-out
### caches, seed 0) -- GNN and PG NEVER separate from random; only SX does, and
### only under R3/R2. SUPERSEDED/EXTENDED by "F17-REVISION" below (full 5-seed
### P3 sweep, 60 units) -- see there for the current numbers, especially the
### PGExplainer collapse rate (revised UP to 40%, not the ~20-30% cited here).

src/analysis/tox21_random_baseline.py. Uses the original scale-out caches
(not the 5-seed P3 re-run, which had not started) -- second, independent
confirmation of F1 alongside its within-explainer std>>mean signature.

AGGREGATE (mean of 12 per-endpoint means):

| masking | metric | GNN | PG | SX | RandN | RandE |
|---------|--------|-----|-----|-----|-------|-------|
| R3 | Fid+ | 0.039 | 0.034 | 0.068 | 0.032 | 0.032 |
| R3 | Fid- | 0.054 | 0.063 | 0.031 | 0.058 | 0.065 |
| R3 | GEF  | 0.088 | 0.097 | 0.068 | 0.093 | 0.096 |
| R2 | Fid+ | 0.068 | 0.049 | 0.165 | 0.054 | 0.049 |
| R2 | Fid- | 0.096 | 0.113 | 0.055 | 0.105 | 0.109 |
| R2 | GEF  | 0.162 | 0.184 | 0.124 | 0.178 | 0.174 |
| R1 | Fid+ | 0.044 | 0.037 | 0.075 | 0.053 | 0.041 |
| R1 | Fid- | 0.058 | 0.057 | 0.053 | 0.055 | 0.060 |
| R1 | GEF  | 0.189 | 0.189 | 0.182 | 0.189 | 0.188 |

Significance (Wilcoxon paired by ENDPOINT, n=12 -- the seed-aware lesson
applied here too: the independent unit is the endpoint, not the molecule;
Holm over the 3 real-vs-random pairs per cell):

| masking | metric | GNN vs RandN | SX vs RandN | PG vs RandE |
|---------|--------|--------------|-------------|-------------|
| R3 | Fid+ | ns | **\*\* (SX)** | ns |
| R3 | Fid- | ns | **\*\* (random lower = SX beats it)** | ns |
| R3 | GEF  | ns | **\*\* (SX beats it)** | ns |
| R2 | Fid+ | ns | **\*\* (SX)** | ns |
| R2 | Fid- | ns | **\*\* (SX beats it)** | ns |
| R2 | GEF  | ns | **\* (SX beats it)** | ns |
| R1 | Fid+ | ns | ns | ns |
| R1 | Fid- | ns | ns | ns |
| R1 | GEF  | ns | ns | ns |

**GNNExplainer and PGExplainer NEVER significantly separate from their
matched random baseline, in ANY of the 9 (masking, metric) cells on Tox21.**
This is a considerably stronger statement of F1 than "std >> mean within one
explainer" -- it says the mean itself is statistically indistinguishable
from chance, for two of the three explainers, everywhere. **SubgraphX DOES
separate from random -- but only under R3 and R2, significantly, on all 3
metrics; R1 washes it out to ns too**, an independent confirmation of F6's
"R1 washout" using a null baseline rather than a same-explainer R3-vs-R1
comparison. SX's Tox21 signal is real but small (e.g. R3 Fid+ 0.068 vs
RandN 0.032 -- both still tiny in absolute terms).

### F17-REVISION: full P3 5-seed sweep (60 units, 12 endpoints x 5 seeds) --
### seed-aware significance, and PGExplainer collapse revised UP to 40%

src/analysis/tox21_random_baseline_5seed.py -- the F17 entry above used the
ORIGINAL 1-seed scale-out caches (P3 had not run yet). Now extended to the
full 5-seed priority-sweep output (60 (endpoint,seed) units), applying the
SAME seed-aware lesson established for mutag_graphxai/B-XAIC: the
independent unit is the (endpoint,seed) pair (n=60), not a molecule, and NOT
simply "pool everything and run one test" -- two tests reported, one
primary (valid, higher power) and one secondary (lower power, checks
reproducibility per seed):

**A. AGGREGATE (mean of 60 unit means) -- unchanged shape from the 1-seed
version:**

| masking | metric | GNN | PG | SX | RandN | RandE |
|---------|--------|-----|-----|-----|-------|-------|
| R3 | Fid+ | 0.028 | 0.025 | 0.057 | 0.026 | 0.020 |
| R3 | Fid- | 0.057 | 0.059 | 0.032 | 0.059 | 0.063 |
| R3 | GEF  | 0.092 | 0.096 | 0.074 | 0.096 | 0.101 |
| R2 | Fid+ | 0.072 | 0.070 | 0.172 | 0.061 | 0.061 |
| R2 | Fid- | 0.114 | 0.118 | 0.060 | 0.122 | 0.123 |
| R2 | GEF  | 0.179 | 0.185 | 0.135 | 0.186 | 0.183 |
| R1 | Fid+ | 0.065 | 0.058 | 0.093 | 0.067 | 0.064 |
| R1 | Fid- | 0.074 | 0.073 | 0.071 | 0.073 | 0.075 |
| R1 | GEF  | 0.190 | 0.188 | 0.185 | 0.192 | 0.191 |

**B. PRIMARY significance -- paired Wilcoxon on the 60 unit-means** (each
already an aggregate over its own 30 molecules -- pairing at this level
respects independence, no pseudo-replication; Holm over the 3 real-vs-random
pairs per cell):

| masking | metric | GNN vs RandN | SX vs RandN | PG vs RandE |
|---------|--------|--------------|-------------|-------------|
| R3 | Fid+ | ns | **\*\*\* (SX)** | ns |
| R3 | Fid- | ns | **\*\*\* (SX beats it)** | ns |
| R3 | GEF  | ns | **\*\*\* (SX beats it)** | \* (RandE beats PG) |
| R2 | Fid+ | ns | **\*\*\* (SX)** | ns |
| R2 | Fid- | ns | **\*\*\* (SX beats it)** | ns |
| R2 | GEF  | ns | **\*\*\* (SX beats it)** | ns |
| R1 | Fid+ | ns | **\*\*\* (SX)** | ns |
| R1 | Fid- | ns | ns | ns |
| R1 | GEF  | ns | ns | ns |

**C. SECONDARY -- per-seed test, n=12 endpoints PER seed** (5 independent
tests, matching F17's original single-seed scope exactly -- does the effect
reproduce in each individual model fit, not just when pooled to n=60):

| masking | metric | GNN vs RandN | SX vs RandN | PG vs RandE |
|---------|--------|--------------|-------------|-------------|
| R3 | Fid+ | 0/5 | **5/5** | 0/5 |
| R3 | Fid- | 0/5 | 0/5 | 0/5 |
| R3 | GEF  | 0/5 | 0/5 | 0/5 |
| R2 | Fid+ | 0/5 | **5/5** | 1/5 |
| R2 | Fid- | 0/5 | 0/5 | 0/5 |
| R2 | GEF  | 0/5 | 0/5 | 0/5 |
| R1 | Fid+ | 0/5 | **4/5** | 0/5 |
| R1 | Fid- | 0/5 | 0/5 | 0/5 |
| R1 | GEF  | 0/5 | 0/5 | 0/5 |

**The honest, seed-aware reading: SX's Fid+ advantage over random is the
ONLY Tox21 finding that reproduces across individual seeds** (5/5 under R3
and R2, 4/5 under R1) -- a genuinely robust result. **SX's Fid-/GEF
"significance" in the n=60 pooled test (block B) does NOT reproduce in ANY
single seed's own n=12 test** (0/5 everywhere) -- that pooled significance
is a power artifact of aggregating many individually-small, non-significant
per-seed effects up to n=60, not evidence of a reliable per-fit effect.
Report Fid+ as SX's real, reproducible signal; do not cite the pooled
Fid-/GEF *** numbers as if they held up per-seed, because they don't. GNN
and PG show ZERO seeds significant on ZERO cells, by either test -- F1's
noise-domination claim is now confirmed at both the pooled (n=60) and the
per-seed (n=12 x5) level for two of the three explainers, everywhere.

**D. PGExplainer collapse, all 60 units: 24/60 = 40%** (`n_degenerate/n >=
0.5` per unit, from each unit's own `explainer_health`) -- HIGHER than the
prior ~20-30% estimate (f10_multiseed.py's 6-endpoint check, 9/30=30%) and
than B-XAIC's 10% (2/20, F16 -- corrected from an earlier "15%"/"3/20" that
miscounted X seed 4 as a collapse; see the correction note under F16).
Every endpoint except SR-HSE collapsed in at least
one of its 5 seeds. **A striking pattern: 9 of the 12 endpoints collapsed
specifically at seed 4** (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER,
NR-ER-LBD, SR-ARE, SR-MMP, SR-p53 all collapse at s4; expected count under a
uniform-across-seeds null would be ~4.8/12, ~2 std below the observed 9) --
**RESOLVED by the dedicated decoupling experiment, see "F10-MECHANISM"
below F10: NOT a pure MLP-seed effect** (fixing a clean model and sweeping
the explain-seed through 0-4, including 4, never collapsed) -- it's a
model x MLP-init INTERACTION, most likely arising because the original
protocol used the SAME seed value for both the model's training and the
explainer's init simultaneously, not because seed value 4 poisons
PGExplainer's own initialisation in isolation.

**Bottom line combining F15-F17, FINAL (all three datasets' random-baseline
checks complete: mutag_graphxai 5/5 seeds, B-XAIC 20/20 units, Tox21 60/60
units):**

**GNNExplainer clears random on BOTH GT/GEA datasets** -- mutag_graphxai
(4/5 seeds) and B-XAIC (indole 3/5, PAINS 5/5, X 3/5 sig, P 5/5 -- 16/20
B-XAIC units significantly) -- **but never separates from random on Tox21
Fidelity, at all, on any of 9 cells, by either the pooled (n=60) or the
per-seed (n=12 x5) test** (F17-REVISION). GNN's real signal is confined to
the two datasets with an actual chemical-substructure ground truth; it has
nothing on Tox21's raw Fidelity/GEF.

**SubgraphX is the most dataset-DEPENDENT of the three**: AT OR BELOW random
on mutag_graphxai (0/5 seeds beat it), decisively ABOVE random on EVERY
B-XAIC task with ZERO exceptions (20/20 seed-task units, all 4 task types,
all p<1e-40), and on Tox21 the ONLY explainer with a signal that survives
the seed-aware check -- specifically Fid+ (reproducible in 4-5/5 seeds
individually), while its Fid-/GEF "significance" only appears once molecule
counts are pooled to n=60 and does not reproduce in any single seed
(F17-REVISION block C). SX's GEA-vs-random result flips harder between the
two GT datasets than either other explainer's -- the same inversion F3/F9
describe, now shown real (not trivial-GT) on the B-XAIC side across all 4
task types.

**PGExplainer clears random reliably only when it doesn't collapse, and it
collapses often and unevenly by dataset**: B-XAIC 2/20 (10%), mutag_graphxai
1/5 (20%), **Tox21 24/60 (40%) -- the highest rate seen, and notably
clustered at seed 4 across 9 of 12 endpoints** (F17-REVISION block D, a
pattern possibly tied to `torch.manual_seed`'s effect on the edge-mask
MLP's init, not resolved). Total across the whole priority sweep: **27 of
85 completed (dataset,endpoint/task,seed) units where PGExplainer's own
GEA/Fid numbers are unreliable** (~32%; CORRECTED from an earlier "28/85" --
see runs/prio/collapse_table.csv, which mechanically applies the SAME
n_degenerate/n>=0.5 threshold cited above and is the authoritative count;
B-XAIC/X/seed4 was previously miscounted here as a collapse -- its
explainer_health is clean (n_degenerate=0/400); it is a real but different
finding, a valid mask that still scores significantly below random, kept
and reported as such, not folded into this tally) -- this is not a rare
edge case, it is a first-order property of the method that any paper using
PGExplainer without multi-seed screening needs to disclose.

**No explainer beats chance in every setting checked.** GNN is real only on
the two GT/GEA datasets. SX is real on B-XAIC (fully) and Tox21 (Fid+ only,
reproducibly) but actively fails on mutag_graphxai. PG is real wherever it
successfully trains, which is 60-85% of the time depending on dataset. The
faithfulness gap now has a proper null to be measured against on every
dataset in the priority sweep, not just inter-explainer comparisons.

### F18. VALID-BUT-BELOW-RANDOM explanations -- a failure mode distinct from
### collapse, undetectable by the degenerate-mask health check (src/analysis/
### below_random_scan.py, mutag_graphxai + all 4 B-XAIC tasks, 25 coupled
### units x 3 explainers = 75 tests; Tox21 OUT OF SCOPE, see below)

Promoted from a footnote on the B-XAIC collapse-count correction (X/seed4:
PGExplainer produced a structurally VALID mask -- n_degenerate=0/400,
n_empty=0, n_whole=0, would pass any health screen -- that nonetheless
scored significantly BELOW its random-edge baseline, GEA 0.030, p=3.7e-09).
The question: is that rare or common? Scanned every NON-collapsed
(unit, explainer) pair among mutag_graphxai (5 units) and all 4 B-XAIC tasks
(20 units) for a paired Wilcoxon test (real vs matched random, per unit,
n=molecules in that unit) that is BOTH significant (p<0.05) AND has the real
explainer's mean BELOW its random baseline's mean.

**TOTAL: 10/75 tests flagged** (GNN vs RandN, SX vs RandN, PG vs RandE; 3
of the 75 were excluded as already-collapsed -- PAINS PG seeds 3-4,
mutag_graphxai PG seed 3 -- leaving 72 valid non-collapsed tests, 10/72=14%):

| explainer | flagged | units |
|---|---|---|
| SX | 5 | mutag_graphxai s0, s1, s2, s3, s4 (ALL 5 seeds) |
| PG | 3 | mutag_graphxai s1, s2; B-XAIC/X s4 |
| GNN | 2 | mutag_graphxai s1; B-XAIC/indole s0 |

| dataset/task | flagged / non-collapsed tests | rate |
|---|---|---|
| mutag_graphxai | 8 / 14 | **57%** |
| B-XAIC/indole | 1 / 15 | 7% |
| B-XAIC/X | 1 / 15 | 7% |
| B-XAIC/PAINS | 0 / 13 | 0% |
| B-XAIC/P | 0 / 15 | 0% |
| **B-XAIC total** | **2 / 58** | **3%** |

**VERDICT: dataset-dependent, not a blanket "rare" or "common" -- COMMON on
mutag_graphxai (a MAJORITY, 8/14, of its non-collapsed explainer-unit pairs
score significantly below random), RARE on B-XAIC (2/58, ~3%).** X seed 4
is NOT the only instance -- it is 1 of 10, and not even the dominant
pattern (that's mutag_graphxai, 8 of the 10). Three things follow:
1. **This mostly re-describes F15's own finding, not a new dataset-level
   claim**: SX's 5/5 mutag_graphxai flags here are exactly F15's headline
   (SX at-or-below random, 5/5 seeds) restated per-unit. Not double-counted
   as new evidence for the paper's reliability claim.
2. **What IS new**: on mutag_graphxai, GNN (seed 1) and PG (seeds 1, 2)
   ALSO score significantly below random in specific seeds, despite neither
   being flagged as "collapsed" and despite GNN/PG both being described
   elsewhere as having real signal on this dataset overall (F15: GNN beats
   random 4/5 seeds, PG 2/5). This is the first time it's been shown, per
   unit rather than as an aggregate direction-count, that on mutag_graphxai
   EVERY explainer has at least one non-collapsed instance of scoring
   significantly worse than chance -- not just SX.
3. **On B-XAIC the phenomenon is genuinely rare and dataset/seed-specific**
   (indole/GNN/s0, X/PG/s4 only) -- consistent with B-XAIC being where all
   three explainers otherwise perform reliably (F16).
**Practical framing for the paper**: the degenerate-mask health check
(explainer_health, used for F10/F16/F17's collapse rates) catches ONE
failure mode (a literal constant/degenerate mask) but misses this second,
distinct one (a valid, non-constant mask that is still worse than chance).
A practitioner screening only for collapse would judge mutag_graphxai's
GNN/PG/SX outputs "healthy" in most of these 8 instances and be wrong to
trust them.
**SCOPE NOTE: Tox21 (60 units) is not covered.** Its per-molecule random
Fid+ baseline was never persisted (data gap 4, explicitly deferred by the
user -- unlike GEA's structural random baseline, Tox21's needs a GPU
forward pass through the trained model to score a random importance vector,
not a pure recompute from cached data). If/when gap 4 closes, this scan
should be extended to the remaining 60 units before citing "rare" or
"common" as a claim spanning the full 85-unit sweep.

## Pooling / single-seed audit, F1-F17 (requested after the SX Fid-/GEF and
## mutag n=940 pooling bugs surfaced)

Two DIFFERENT concerns get conflated under "pooling," worth separating:
  (a) TRUE POOLING BUGS -- treating repeated-measures observations (the same
      or correlated molecules/units recurring across SEEDS) as independent
      n, inflating a single p-value. Only 2 instances found, both already
      fixed: mutag_graphxai's original n=940 GEA significance (F3-REVISION
      v1, WITHDRAWN) and SX's Tox21 Fid-/GEF-vs-random significance
      (F17-REVISION, corrected from "***" to "not reproducible per seed").
  (b) SINGLE-SEED FINDINGS never stress-tested against the other 4
      priority-sweep seeds once they became available. Not a bug in the
      original test (each was a legitimate n=29/30 test on its one seed's
      data), but an open question of whether the SPECIFIC claim (a ranking,
      a "how many of the 12/8/4 cells" count) reproduces. Re-tested where
      cached data made it cheap; flagged where it wasn't.

| Finding | Pooling bug? | Seed-tested? | Verdict |
|---------|---------------|---------------|---------|
| F1 (R3 noise-domination) | No -- within-explainer std>>mean, not cross-unit | Partial (Tox21 via F12-REVISION R3 rows; mutag_graphxai via F15) | Supported by adjacent re-checks; **MUTAG-std and BBBP never re-tested at all (see note below)** |
| F2 (B-XAIC R3 heavy tail) | No | No | Single-seed (indole s0 only); B-XAIC now has 5-seed data at a different N (400 vs 29) that could check this -- not done |
| F3 (GEA inversion) | **YES, v1 (n=940) -- WITHDRAWN** | Yes, v2 | **CORRECTED** -- "GNN >> {PG,SX}" robust, strict "GNN>PG>SX" only 3/5 seeds |
| F4 (R2 featurisation-dependent) | No | Partial (Tox21 side via F11-REVISION) | One-hot (mutag family) side never re-seeded; MUTAG-std/BBBP raw side never re-seeded |
| F5 (R2 separates SX) | No | **No** | mutag_graphxai's "SX sig. higher Fid+/lower Fid- than GNN/PG under R2" and B-XAIC's "R2 separates all three" were never re-run per-seed even though 5-seed data exists for both -- **flagged, not yet checked, same design as F12-REVISION would apply directly** |
| F6 (R1 most artifact-dominated) | No | No | MUTAG-std/BBBP never multi-seeded |
| F7 (PG lowest Fid+ on 3/5 datasets under R1) | No (cross-DATASET count, not cross-seed) | No | Single-seed per dataset |
| F8 (cross-metric agreement "1 of 8") | No | No | Single-seed (mutag_graphxai + indole); could be re-checked against F9-REVISION's now-known-unstable X/P GEA orderings -- see F13 below, same risk applies |
| F9 (B-XAIC 3-way GEA ranking) | No | **Yes** | **MAJOR CORRECTION** -- indole robust, PAINS's 2nd place collapse-contingent, X and P DO NOT reproduce (F9-REVISION) |
| F10 (PGExplainer collapse) | No | Yes (dedicated f10_multiseed.py + F17-REVISION) | Confirmed and STRENGTHENED -- rate revised 20-30% -> 40% on Tox21 |
| F11 (R2/R1 inflation, Tox21) | No | **Yes** | **CONFIRMED** (F11-REVISION), reproduces every seed |
| F12 (Tox21 separability) | No | **Yes** | **CONFIRMED** (F12-REVISION), R2 count 5-8/12 every seed |
| F13 (cross-metric agreement, B-XAIC "5 of 12") | No | **No** | **NOT re-tested, HIGH RISK of not surviving** -- F9-REVISION just showed X and P's GEA 3-way order changes seed-to-seed, so a per-seed re-check of "does GEA order match Fid+ order" is very likely to show the same instability, especially for X (F13's headline "agrees under every masking" example). Flagged, not yet run. |
| F14 (GEA binarization) | No | No | Single-seed caches (mutag_graphxai + 4 B-XAIC tasks); not re-verified at 5 seeds |
| F15 (mutag_graphxai random baseline) | No | Yes, by design | Seed-aware from the start |
| F16 (B-XAIC random baseline) | No | Yes, by design | Seed-aware from the start |
| F17 (Tox21 random baseline + F11/F12 above) | **YES (Fid-/GEF-vs-random) -- CORRECTED** | Yes | Fid+ confirmed robust; Fid-/GEF pooled-only, now flagged as such |

**Net: 2 confirmed true pooling bugs (both fixed), 3 single-seed findings
re-tested and CONFIRMED (F11, F12, F17's Fid+), 2 re-tested and REVISED/
CORRECTED (F3, F9), and a meaningful backlog of single-seed findings never
stress-tested (F1 partial, F2, F4 partial, F5, F6, F7, F8, F13, F14) --
of these, F13 is flagged HIGH RISK given F9-REVISION's direct evidence, F5
is the next-cheapest to check (data already cached, same method as
F12-REVISION), and F6/F7/F1/F2 rest partly or wholly on MUTAG-std/BBBP,
which have NEVER been multi-seeded at all.**

**MUTAG-standard and BBBP were never multi-seeded.** Every core-Phase-3
number for these two datasets (F1, F4, F5, F6, F7, F8's underlying data)
rests on a SINGLE 70/15/15 split + single model fit + n=29 (MUTAG) or n=30
(BBBP) explained molecules -- there is no priority-sweep equivalent for
either (the priority sweep only covered mutag_graphxai, B-XAIC, and Tox21).
Any claim specific to MUTAG-std or BBBP (as opposed to the parts of F1/F4/
F6/F7 that also cite mutag_graphxai or Tox21, which DO now have multi-seed
support) should be read as single-replicate evidence, on the same footing
Phase-3's original findings were in before this whole multi-seed effort
started.

## CONSOLIDATED CURRENT-NUMBERS TABLE (read this before citing anything above)

### Status ledger, all findings

| # | Finding (one line) | Status | Current headline |
|---|---|---|---|
| F1 | R3 noise-dominated, 4 core datasets | **CONFIRMED** (mutag_graphxai) | 9/9 cells noise-dominated, 5/5 seeds; MUTAG-std/BBBP/Tox21 not re-tested here |
| F2 | B-XAIC R3 heavy right tail | **WITHDRAWN** as stated | reproduces in only 1/5 seeds; 4/5 show the OPPOSITE (left-skew) shape |
| F3 | mutag_graphxai GEA: GNN~=PG>>SX | **REVISED, decoupled-verified** | "GNN >> {PG,SX}" robust (5/5 seeds, coupled AND decoupled); strict "GNN>PG>SX" only 3/5 either way |
| F3-v1 | (pooled n=940 significance) | **WITHDRAWN** | do not cite p~1e-94 etc. |
| F4 | R2 featurisation-dependent | **CONFIRMED** (both sides) | one-hot side: R3->R2 inflation sig in 20/20 seed-cells; Tox21 side confirmed via F11-REVISION |
| F5 | R2 separates SX vs GNN/PG | **CONFIRMED** | SX-vs-GNN/PG Fid+ separation robust across all 5 seeds, mutag_graphxai + indole (+ PAINS/X/P mostly) |
| F6 | R1 most artifact-dominated (washout) | **RESTATED** (pairwise: does not reproduce; vs-random: does, narrowly) | pairwise explainer-vs-explainer washout absent on every GT dataset; SX-vs-random Fid+ washout IS real but partial (Tox21 only, 5/5->4/5 seeds, F6-ADDENDUM) |
| F7 | PG lowest Fid+ under R1, 3/5 datasets | **RESTATED** | common (mutag/indole/PAINS, 12/15 seed-cells) but flatly contradicted on B-XAIC/X (PG HIGHEST, 5/5 seeds) |
| F8 | Cross-metric agreement "1 of 8" | **CONFIRMED** (qualitative) | rare agreement (~15-20% cells) holds up; the specific "indole/R2 agrees" cell is NOT itself reproducible (2/5 seeds) |
| F9 | B-XAIC 3-way GEA ranking per task | **MAJOR REVISION** | only indole robust; X/P rankings do NOT reproduce per seed |
| F10 | PGExplainer collapse | Confirmed + mechanism found + decoupled-verified | rate 10-40% depending on dataset (B-XAIC corrected 3/20->2/20, see runs/prio/collapse_table.csv); model x MLP-init interaction (F10-MECHANISM); mutag_graphxai seed 3 collapses under BOTH its coupled and decoupled explainer-seed (same model, different inits) |
| F11 | Tox21 R2/R1 inflation, ~8/12 endpoints | **CONFIRMED** | reproduces every seed, 6-10/12 (R2), 8-12/12 (R1) |
| F12 | Tox21 R2 separates SX, 5/12 endpoints | **CONFIRMED** | reproduces every seed, 5-8/12 |
| F13 | B-XAIC GEA/Fid+ agreement "5 of 12" | **RESTATED** | "X agrees under ALL 3 maskings" holds in only 2/5 seeds; directional claim (X 60% vs others 20-33% cell agreement) survives |
| F14 | GEA binarization (mean-thr vs top-k) | **MIXED** | indole confirmed robust (5/5); P's instability confirmed+sharpened (0/5); mutag_graphxai/X weaker than framed (3/5, inherited GEA fragility) |
| F15 | mutag_graphxai vs random | **CONFIRMED, sharpened, decoupled-verified** | GNN beats random 4/5; SX 0/5 (loses to a connectivity+size-matched fair null too, 5/5 coupled AND 5/5 decoupled); PG inconsistent 2/5 |
| F16 | B-XAIC vs random, all 4 tasks | **CONFIRMED, sharpened** | SX beats random 20/20; indole also beats a connectivity+size-matched fair null 5/5 (F16-ADDENDUM) |
| F17 | Tox21 vs random | **PARTIALLY REVISED** | Fid+ CONFIRMED (SX, 4-5/5 seeds); Fid-/GEF significance was pooling-only, WITHDRAWN as stated |
| F18 | Valid-but-below-random explanations (new) | **CONFIRMED, dataset-dependent** | 10/75 non-collapsed tests flagged; COMMON on mutag_graphxai (8/14, 57%), RARE on B-XAIC (2/58, 3%); Tox21 not covered (gap 4 deferred) |

### Table A -- GEA, every (dataset, seed): mutag_graphxai + all 4 B-XAIC tasks

matched top-k=0.25 random baselines shown alongside; all numbers verified in
this session (F15/F16/F3-REVISION v2/F9-REVISION).

| dataset | seed | GNN | PG | SX | RandNode | RandEdge |
|---|---|---|---|---|---|---|
| mutag_graphxai | 0 | 0.433 | 0.243 | 0.030 | 0.149 | 0.194 |
| mutag_graphxai | 1 | 0.122 | 0.066 | 0.020 | 0.145 | 0.158 |
| mutag_graphxai | 2 | 0.492 | 0.070 | 0.113 | 0.141 | 0.176 |
| mutag_graphxai | 3 | 0.496 | 0.000 (collapsed) | 0.048 | 0.140 | 0.184 |
| mutag_graphxai | 4 | 0.612 | 0.400 | 0.113 | 0.132 | 0.165 |
| indole | 0-4 | 0.246 mean (3/5 sig>rand) | 0.532 mean (5/5 sig>rand) | 0.658 mean (5/5 sig>rand) | 0.164 mean | 0.151 mean |
| PAINS | 0-4 | 0.304 mean (5/5 sig>rand) | 0.216 mean (3/5, 2 collapsed) | 0.492 mean (5/5 sig>rand) | 0.165 mean | 0.160 mean |
| X | 0-4 | 0.063 mean (3/5 sig>rand) | 0.368 mean (4/5, 1 near-collapsed) | 0.383 mean (5/5 sig>rand) | 0.055 mean | 0.074 mean |
| P | 0-4 | 0.209 mean (5/5 sig>rand) | 0.271 mean (5/5, 0 collapsed) | 0.299 mean (5/5 sig>rand) | 0.042 mean | 0.023 mean |

Per-seed detail for the 4 B-XAIC tasks (all 20 rows) is in F16's table; per-
seed detail for mutag_graphxai is in F3-REVISION v2 / F15 above. **3-way
ranking (which of GNN/PG/SX scores highest) is SEED-STABLE only for
mutag_graphxai (GNN always #1) and indole (SX always #1, all pairs always
significant) -- PAINS/X/P's rankings move around by seed (F9-REVISION);
"beats random" (this table) is a separate, more robust claim from "ranks
highest among the three" (F9-REVISION), and the two should not be
conflated.**

### Table B -- Tox21 aggregate (60 units = 12 endpoints x 5 seeds, F17-REVISION)

| masking | metric | GNN | PG | SX | RandN | RandE | primary sig (n=60) | per-seed sig (n=12 x5) |
|---|---|---|---|---|---|---|---|---|
| R3 | Fid+ | 0.028 | 0.025 | 0.057 | 0.026 | 0.020 | SX *** | SX 5/5 |
| R3 | Fid- | 0.057 | 0.059 | 0.032 | 0.059 | 0.063 | SX *** | **SX 0/5 (pooling-only)** |
| R3 | GEF | 0.092 | 0.096 | 0.074 | 0.096 | 0.101 | SX *** | **SX 0/5 (pooling-only)** |
| R2 | Fid+ | 0.072 | 0.070 | 0.172 | 0.061 | 0.061 | SX *** | SX 5/5 |
| R2 | Fid-/GEF | -- | -- | -- | -- | -- | SX *** | **SX 0/5 (pooling-only)** |
| R1 | Fid+ | 0.065 | 0.058 | 0.093 | 0.067 | 0.064 | SX *** | SX 4/5 |
| R1 | Fid-/GEF | -- | -- | -- | -- | -- | ns | ns |
| all cells | -- | -- | -- | -- | -- | -- | GNN/PG always ns | GNN/PG always 0/5 |

**Only SX's Fid+ advantage is a genuinely reproducible, per-seed-verified
finding (4-5/5 seeds, every masking). All the *** results for Fid-/GEF are
n=60-pooling artifacts -- cite the Fid+ result, not those.** PGExplainer
collapse: 24/60 (40%) on Tox21, model x MLP-init interaction (F10-MECHANISM).

### Table C -- single-seed-only, never re-verified (cite with that caveat)

| dataset | protocol | status |
|---|---|---|
| MUTAG (standard) | 70/15/15, n=29, seed 0 | never multi-seeded; F1/F4/F5/F6/F7/F8 numbers for this dataset are single-replicate |
| BBBP | 70/15/15, n=30, seed 0 | never multi-seeded; same caveat |
| Tox21 SR-p53 (as its OWN single-dataset core-Phase-3 number, separate from the 12-endpoint aggregate) | 70/15/15, n=30, seed 0 | superseded in aggregate by Table B above (SR-p53 is 1 of the 12 endpoints, now has 5-seed data); the STANDALONE SR-p53-only Phase-3 table entry is not itself re-verified in isolation |

### Do-not-cite list (superseded, kept only for provenance)

- mutag_graphxai GEA significance at p~1e-94/1e-133 (F3-REVISION v1) -- pseudo-replicated, withdrawn.
- F2's "heavy right tail, median~0.001" framing for indole R3 Fid+ -- reproduces in only 1/5 seeds, withdrawn as a general claim (F2-REVISION).
- F13's "X agrees under ALL of R3/R2/R1" framing -- reproduces in only 2/5 seeds, restated as a directional (not universal) claim (F13-REVISION).
- F8's "indole/R2 is the one agreeing cell" -- that specific cell is not itself reproducible (2/5 seeds); cite only the qualitative "agreement is rare" claim (F8-REVISION).
- F6 as a general R1-washout claim beyond MUTAG-standard/BBBP -- does not reproduce on any GT dataset with 5-seed data (F6-REVISION).
- Every finding above computed from the completed 5-seed priority sweep (F1/F2/F4/F5/F6/F7/F8/F9/F11/F12/F13/F14/F15/F16/F17 and their REVISIONs) used a sweep where the model-training seed and the explainer/PGExplainer-init seed were the SAME value per unit (coupled) -- see the "Limitations" bullet above. Treat "seed" in those tables as one coupled draw, not two independent factors. Only NEW sweep units (post this commit) use the decoupled protocol (scripts/_run_unit.sh ES, scripts/priority_sweep.sh ES=S+1000).
- Tox21 Fid-/GEF "SX beats random ***" under any masking (F17, original + primary n=60 test) -- does not reproduce per seed (F17-REVISION block C), pooling artifact.
- F9's "SX significantly top-or-tied-top on 3/4 B-XAIC tasks" as a claim about the 3-way ranking -- only true for indole; PAINS/X/P do not reproduce (F9-REVISION). ("SX beats RANDOM on 4/4 tasks" is a different, still-true claim, F16.)
- "B-XAIC PGExplainer collapse: 3/20 (15%)" and the derived "28/85 (~33%) units unreliable" -- CORRECTED to 2/20 (10%) and 27/85 (~32%). The "3" miscounted B-XAIC/X/seed4 as a collapse; its explainer_health is clean (n_degenerate=0/400) -- it does not meet the n_degenerate/n>=0.5 threshold used for every other collapse count in this document. X seed 4 belongs to a different, now-promoted finding (F18: valid-but-below-random explanations), not the collapse tally. Reconciled against the mechanically-computed runs/prio/collapse_table.csv, which is authoritative for all collapse counts going forward.

## GraphXAI Mutagenicity -- third GT dataset (runs/mutagenicity/, separate from every result above)

### PREDICTIONS -- registered 2026-10-07 15:46 IST, BEFORE Step 3 (no explanation
### has been scored against ground truth on this dataset at the time of writing;
### the only runs so far are the Step 1 dataset check and the Step 2 timing run,
### whose GEA output has not been read)

1. GNNExplainer clears RandNode in at least 4 of 5 seeds.
2. No firm prediction for SubgraphX. The alignment mechanism says it should do
   better when the annotated toxicophores are what the model uses, and the
   dataset's label filter makes that more likely than in MUTAG-188. A SubgraphX
   win would support the mechanism, not contradict it.
3. PGExplainer collapse rate: report as observed.

Rule: report the results whichever way they go.

Note recorded with the predictions (a Step 1 fact, not a prediction): GraphXAI's
filter keeps graphs with has_toxicophore == y, and TUDataset's label map is
0 = mutagen, 1 = nonmutagen. The 1,768 kept graphs are therefore the 640
NON-mutagens that contain a toxicophore plus the 1,128 mutagens that contain
none. Within the kept set the label is still exactly "toxicophore present", so
prediction 2's premise (the label filter ties the label to the annotated
toxicophores) holds as a property of the data, with class 1 = "has toxicophore".

### Steps 1-2 (2026-10-07): dataset check and cost estimate. Step 3 NOT run.

Code: src/data/graphxai_mutagenicity.py, src/metrics/gea_set.py,
src/analysis/mutagenicity_step1.py (report: runs/mutagenicity/step1_report.json).

Dataset (GraphXAI logic followed exactly):
- TUDataset Mutagenicity: 4,337 graphs (2,401 y=0, 1,936 y=1). 2,401 isolated
  nodes removed across 121 graphs (this empties the Na/K/Li/Ca channels).
- Filter has_toxicophore == y keeps 1,768: 1,128 y=0 without a toxicophore,
  640 y=1 with one. TUDataset's label map is 0 = mutagen, 1 = nonmutagen, so
  the kept set is mutagens WITHOUT a toxicophore plus non-mutagens WITH one.
  Toxicophore prevalence is 53.0% in mutagens and 33.1% in non-mutagens; the
  opposite polarity would keep 2,569 graphs. Not corrected.
- Nodes: mean 29.1, median 28, max 103. 14-dim atom one-hot, 3-dim bond one-hot.
- Encoding matches GraphXAI's hard-coded 14-dim order (TU readme + degree
  histograms: H always degree 1, C <= 4, O <= 2). The port returns identical
  match lists to GraphXAI's own matcher code on 4,337/4,337 graphs.
- Motif composition: NH2 = {N,H,H}, NO2 = {N,O,O}, nitroso = {N,O},
  azo = {N,N}, halide = single F/Cl/Br/I.
- match_aliphatic_halide keeps a terminal halogen only if its neighbour is on
  a cycle-basis ring of length > 4: 913 ring-attached halogens kept, 956
  chain/small-ring halogens dropped (all 4,337 graphs). It also accepts F.
- Kept-set toxicophores (graphs / instances): NH2 254/309, NO2 104/128,
  nitroso 6/6, azo 45/52, halide 296/658. 63/640 graphs have 2+ types.
- GT set: 2^L - 1 masks for L instances; L mean 1.80, median 1, max 12.
  Enumeration cap 12 instances (4,095 masks) -> affects 0 graphs; every graph
  is scored exactly. 4/640 graphs have overlapping instances.
- Merged GT: mean 3.23 atoms (15.2% of the molecule); top-k 0.25 budget is
  6.37 atoms on GT-present graphs. GT-present = the 640 positives.
- Split 70/10/20, seeded and stratified (GraphXAI's own split is sequential,
  unshuffled): train 1,237 / val 177 / test 354 (128 positive). After
  --max-nodes 80: 345-351 test molecules per seed, 125-127 GT-present.
  Exclusion: 29/1,768 graphs (1.64%), 8/640 GT-present (1.25%).

Cost (Ada GPU 1, rollout 20, sample_num 30, seed-0 model: test acc 0.986,
AUROC 0.999):
- SubgraphX single run: 109.9 s/mol (20 molecules, 3.62 s/node).
- SubgraphX 4 concurrent: 80 molecules in 3,430 s; 102-171 s/mol per job
  (mean 4.65 s/node, 1.28x the single-run rate).
- GNNExplainer 2.0 s/mol. PGExplainer: 539 s to train on the full train
  split (1,215 graphs), explanation time negligible. Scoring negligible.
- Test split x 5 seeds (1,740 molecules) and full dataset x 1 seed (1,739
  molecules) cost the same: 16-21 h wall-clock at 4-way concurrency.

### PREDICTION 2 REVISED -- registered 2026-10-07 17:38 IST, BEFORE Step 3 (the 5-seed run has
### not started; the timing logs' GEA tables are still unopened)

Original text of prediction 2 (committed c39397b, 2026-10-07 15:46 IST), unchanged:
  "No firm prediction for SubgraphX. The alignment mechanism says it should do
   better when the annotated toxicophores are what the model uses, and the
   dataset's label filter makes that more likely than in MUTAG-188. A SubgraphX
   win would support the mechanism, not contradict it."
Added:
  "Because the kept set's label is exactly toxicophore presence, the model's
   rule is aligned with the annotation, so SubgraphX should beat RandConn in at
   least 4 of 5 seeds. If it does not, that counts against the alignment
   mechanism."
Prediction 1 (GNNExplainer clears RandNode in >= 4/5 seeds) and prediction 3
(PGExplainer collapse rate reported as observed) stand as committed.

### Label meaning in the kept set -- three independent checks (recorded before Step 3)

(a) TUDataset readme (data/TUDataset/Mutagenicity/raw/Mutagenicity_label_readme.txt),
    verbatim: "Class labels were converted to integer values using this map:
    0 mutagen / 1 nonmutagen". GraphXAI's repository ships the same file
    (formal/real_world/data/Mutagenicity/raw/) with the same map.
(b) Toxicophore prevalence by class, all 4,337 graphs, GraphXAI's matchers:
    y=0: 1,273/2,401 = 53.0%;  y=1: 640/1,936 = 33.1%. The class with the
    higher toxicophore rate is y=0, consistent with (a).
(c) GraphXAI's own material: no source file, docstring, README or test script
    in the repository (commit a11e65f) states which class index is mutagenic;
    its verification script only tabulates has_match against y. The paper
    (arXiv:2208.09339, read via ar5iv) says the graphs are "labeled into two
    different classes according to their mutagenic properties" and that "the
    dataset is trimmed from its original 4,337 graphs to 1,768, based on those
    whose labels directly correspond to the presence or absence of our chosen
    toxicophores". Neither names the mutagenic class index.
Result: (a) and (b) agree that y=0 is the mutagen class; (c) is silent.
How the kept classes are described from here on: label 1 = toxicophore present
(640 graphs), label 0 = no toxicophore (1,128 graphs). The label is a function
of toxicophore presence, which is the property the experiment relies on. The
polarity-corrected 2,569-graph set is not run: its label is also a function of
toxicophore presence, so it would double the cost without adding a regime.

### Step 3 protocol decisions (recorded before the run)
- Dataset: GraphXAI's 1,768-graph set exactly as is.
- 5 seeds, decoupled from the start: model seed s in 0..4, explainer seed 1000+s.
- Split 70/10/20, test split only, --max-nodes 80.
- PGExplainer trains on the FULL train split (about 1,215 graphs after the
  node cap), as on MUTAG-GT. B-XAIC used --pg-train-limit 200.
- Five seed jobs concurrently on GPU 1 (GPU 0 left free). Fallback, if
  SubgraphX per-molecule time in the first hour is worse than about 1.5x the
  4-way measurement (4.65 s/node, 138 s/mol): drop to 4 and shard molecules.
- D-MPNN: depth 3, hidden 300, 120 epochs, batch 50 (MUTAG-GT settings), input
  dims from the data (14 node, 3 edge).
- Additions to the report: GEA stratified by toxicophore type (halide-only vs
  NH2/NO2/azo/nitroso graphs) under both binarizations; oracle top-k ceiling
  next to the random floor, also for MUTAG-GT and B-XAIC indole.

### How the predictions will be scored -- fixed 2026-10-07 17:42 IST, while Step 3 is running
### and before any GEA from this dataset has been read

- GEA variant for the predictions: max-over-set (paper Eq. 2). Merged-mask GEA
  is reported next to it.
- Prediction 1 (GNNExplainer vs RandNode): same construction as F15 on MUTAG-GT
  -- GNNExplainer's mean-threshold mask against a uniform random top-k 0.25
  node mask. The size-matched top-k vs top-k comparison is reported as well.
- Prediction 2 (SubgraphX vs RandConn): same construction as F15-ADDENDUM --
  SubgraphX's mean-threshold mask against random connected subgraphs of the
  same size on the same molecule. The top-k version is reported as well.
- A seed counts as "clears"/"beats" when the explainer's per-seed mean GEA is
  higher AND the paired Wilcoxon test over that seed's GT-present test
  molecules is significant after Holm correction within the seed. Direction-
  only counts are reported separately. Nothing is pooled across seeds.
- Nulls use 20 draws per molecule, averaged (RandNode, RandEdge, RandConn,
  and a size-matched unconstrained RandUnc). MUTAG-GT/B-XAIC used one draw for
  RandNode/RandEdge.

### Oracle ceiling for top-k GEA (src/analysis/oracle_ceiling.py; no model)

Best Jaccard any explainer can reach with exactly k = max(1, int(0.25 N))
atoms: min(g, k) / max(g, k) for a GT of g atoms; for the set-valued GT, the
max over all unions of motif instances. Mean-threshold GEA has no size budget
(ceiling 1.0).

| dataset | n | ceiling (mean) | GT atoms | budget k | k > GT | k < GT | random floor |
|---|---|---|---|---|---|---|---|
| Mutagenicity, max-over-set | 632 | 0.551 | 3.21 | 6.17 | 75% | 16% | Step 4 |
| Mutagenicity, merged | 632 | 0.513 | 3.21 | 6.17 | 75% | 16% | Step 4 |
| -- halide-only, max-over-set | 246 | 0.453 | 2.30 | 5.83 | 81% | 11% | Step 4 |
| -- NH2/NO2/azo/nitroso, max-over-set | 386 | 0.614 | 3.79 | 6.38 | 72% | 19% | Step 4 |
| MUTAG-GT | 188 | 0.751 | 4.44 | 4.11 | 37% | 37% | RandNode 0.141, RandEdge 0.175 |
| B-XAIC indole (5 seeds) | 1,800 | 0.724 | 9.69 | 7.62 | 13% | 78% | RandNode 0.164, RandEdge 0.151 |

On Mutagenicity the top-k budget is about twice the ground truth, so a
perfect explainer scores about 0.55 under top-k; on indole the budget is
smaller than the ground truth in 78% of molecules.

### Steps 3-4 RESULTS (run 2026-10-07 17:39 -> 2026-10-08 09:09 IST, 15.5 h;
### src/analysis/mutagenicity_report.py; full output runs/mutagenicity/report.txt)

Protocol as registered above: 5 seeds, decoupled (explainer seed = 1000 +
model seed), test split only, --max-nodes 80, five concurrent jobs on GPU 1.
SubgraphX ran at 137-154 s/mol, at most 1.11x the 4-way measurement (138
s/mol), so the 1.5x fallback was not needed. (The first-hour check was not
made live; the figure is from the completed logs.)

Model: test accuracy 0.966 / 0.992 / 0.992 / 0.994 / 0.986 (mean 0.986),
AUROC 0.998-1.000 (mean 0.999). Explained 345-351 test molecules per seed,
125-127 GT-present.

GEA, max-over-set, per seed (GT-present test molecules; nulls = 20 draws):

| seed | GNN mean-thr | PG mean-thr | SX mean-thr | GNN top-k | PG top-k | SX top-k | RandNode | RandEdge | RandConn (top-k) | top-k ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.419 | 0.043 | 0.354 | 0.470 | 0.037 | 0.266 | 0.117 | 0.138 | 0.096 | 0.542 |
| 1 | 0.408 | 0.013 | 0.355 | 0.469 | 0.016 | 0.259 | 0.119 | 0.152 | 0.108 | 0.559 |
| 2 | 0.341 | 0.044 | 0.368 | 0.383 | 0.049 | 0.302 | 0.127 | 0.147 | 0.103 | 0.565 |
| 3 | 0.303 | 0.304 | 0.406 | 0.332 | 0.256 | 0.308 | 0.132 | 0.161 | 0.106 | 0.597 |
| 4 | 0.217 | 0.210 | 0.376 | 0.233 | 0.233 | 0.266 | 0.117 | 0.142 | 0.095 | 0.522 |
| mean | 0.338 | 0.123 | 0.372 | 0.378 | 0.118 | 0.280 | 0.122 | 0.148 | 0.102 | 0.557 |
| std | 0.074 | 0.114 | 0.019 | 0.089 | 0.104 | 0.020 | 0.006 | 0.008 | 0.005 | |

Size-matched RandConn under mean-threshold: 0.107 (GNN's size), 0.106 (PG's),
0.105 (SX's). Merged-mask GEA is 0.02-0.04 lower than max-over-set for every
explainer and null, and gives the same direction and significance in every
cell (report.txt section 3).

PREDICTIONS (scored by the rules fixed before the results were read):
- P1 HOLDS: GNNExplainer clears RandNode in 5/5 seeds (all p_Holm < 0.001).
- P2 HOLDS: SubgraphX beats its size-matched RandConn in 5/5 seeds (all
  p_Holm < 0.001; 0.354-0.406 vs 0.094-0.115). Also 5/5 under top-k.
- P3 (as observed): PGExplainer collapsed in 0/5 seeds by the strict rule
  (degenerate masks: 0, 0, 1, 0, 0 of ~348). But it scores significantly BELOW
  every null in 3/5 seeds (0, 1, 2: GEA 0.013-0.044) and significantly above
  in 2/5 (3, 4: 0.304, 0.210). That is the F18 failure mode -- a valid,
  non-degenerate mask that is worse than chance -- not collapse.

Other observations:
- Explainer ranking depends on the binarization. Mean-threshold: SX 0.372 vs
  GNN 0.338 (SX higher in seeds 2, 3, 4). Top-k: GNN 0.378 vs SX 0.280 (GNN
  higher in seeds 0-3). SX's mean-threshold mask is 4.0-4.5 atoms, close to
  the GT (3.0-3.5); the top-k budget forces 5.7-6.4 atoms.
- SX is the seed-stable explainer here (std 0.019); GNN ranges 0.217-0.419.
- Mask sizes (mean-threshold, atoms): GNN 8.1-11.1, PG 6.6-11.1, SX 4.0-4.5;
  top-k budget 5.7-6.4; GT 3.0-3.5.
- Under top-k, GNN reaches 0.47 against a ceiling of 0.54-0.56 in seeds 0-1.

Stratified by toxicophore type (max-over-set, mean over seeds; "clears" =
higher and Holm-significant, per seed):

| stratum | binarization | GNN | PG | SX | RandNode | RandConn (SX size / k) | SX clears RandNode | SX clears RandConn | GNN clears RandNode |
|---|---|---|---|---|---|---|---|---|---|
| halide-only (46-61/seed, GT 2.25 atoms) | mean-thr | 0.301 | 0.044 | 0.229 | 0.116 | 0.069 | 3/5 | 4/5 | 5/5 |
| halide-only | top-k (ceiling 0.450) | 0.342 | 0.046 | 0.187 | 0.116 | 0.068 | 3/5 | 4/5 | 5/5 |
| NH2/NO2/azo/nitroso (64-81/seed, GT 3.72) | mean-thr | 0.359 | 0.180 | 0.463 | 0.127 | 0.128 | 5/5 | 5/5 | 5/5 |
| NH2/NO2/azo/nitroso | top-k (ceiling 0.624) | 0.398 | 0.172 | 0.338 | 0.127 | 0.123 | 5/5 | 5/5 | 5/5 |

SubgraphX's result is carried by the multi-atom toxicophores; on single-atom
halide ground truth it is weaker and not significant against RandNode in 2/5
seeds. PGExplainer is significantly below RandNode on halide-only graphs in
4/5 (mean-thr) and 5/5 (top-k) seeds.

Across the three GT datasets (per-seed, significant):

| | MUTAG-GT (coupled) | B-XAIC indole (coupled) | Mutagenicity (decoupled) |
|---|---|---|---|
| SubgraphX vs connectivity-matched null | below, 5/5 | above, 5/5 | above, 5/5 |
| GNNExplainer vs RandNode | above, 4/5 | above, 3/5 | above, 5/5 |
| PGExplainer vs its null | above 2/5 | above 5/5 | above 2/5, below 3/5 |
| PGExplainer collapse | 1/5 | 0/5 | 0/5 |

Fid+/Fid-/GEF under R3/R2/R1 per seed: report.txt section 8.
Persisted: runs/mutagenicity/arrays_s<S>.npz (per-molecule GEA for every
explainer x binarization x GT variant, all nulls, win/loss vs RandConn, mask
sizes, split membership, seeds), report.json, seeds_s<S>.json.

### EXPLORATORY (2026-10-08) -- not part of the registered predictions, which stand as scored above

Scripts: src/analysis/exploratory_pg_and_normalised.py,
src/analysis/exploratory_saturation_scan.py. Output:
runs/mutagenicity/exploratory/.

**E1. What PGExplainer's below-random masks are.** PGExplainer, GT-present
test molecules; "range" = max - min node importance within a molecule; "GT
rank" = mean percentile rank of the GT atoms' importance (1 = highest, 0.5 =
chance); "inverted" = top-k GEA of the negated importance (merged GT).

| unit | n_empty | n_whole | degenerate | mean-thr mask size (mean / p10-p90) | std median | std min | range median | GT rank | GEA top-k | inverted | selected atoms | GT atoms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mutagenicity s0 | 0 | 0 | 0/350 | 8.4 / 2-16 | 0.0009 | 2e-16 | 0.003 | 0.19 | 0.026 | 0.369 | H 70%, C 16% | H 31%, N 28%, Cl 20% |
| Mutagenicity s1 | 0 | 0 | 0/345 | 9.1 / 3-16 | 0.0015 | 1e-27 | 0.005 | 0.21 | 0.012 | 0.342 | H 71%, C 18% | H 32%, N 27%, Cl 20% |
| Mutagenicity s2 | 1 | 0 | 1/348 | 6.6 / 2-12 | 0.057 | 0 | 0.234 | 0.28 | 0.032 | 0.246 | C 60%, H 26% | H 29%, N 27%, Cl 27% |
| Mutagenicity s3 | 0 | 0 | 0/351 | 8.5 / 3-16 | 0.271 | 7e-11 | 0.771 | 0.60 | 0.235 | 0.084 | H 54%, C 26% | H 29%, N 27%, Cl 25% |
| Mutagenicity s4 | 0 | 0 | 0/346 | 11.1 / 4-22 | 0.221 | 5e-26 | 0.707 | 0.60 | 0.221 | 0.066 | H 41%, C 38% | Cl 29%, H 25%, N 24% |
| B-XAIC X s0-s3 | 0 | 0 | 0/400 | 3.0-6.0 | 0.069-0.109 | >= 1e-4 | 0.35-0.50 | 0.98-0.99 | 0.25-0.26 | 0.000 | C 44-64%, F 18-31%, Cl 10-20% | F 52-58%, Cl 32-36% |
| B-XAIC X s4 | 0 | 0 | 0/400 | 6.2 / 4-10 | 0.075 | 0.012 | 0.256 | 0.70 | 0.093 | 0.015 | C 91%, N 5% | F 58%, Cl 33% |

Reading:
- Mutagenicity seeds 0-1 are near-constant masks (importance varies by about
  0.003-0.005 within a molecule; seeds 3-4 vary by 0.7-0.8). The strict rule
  misses them because the spread is tiny but not zero. The residual ordering
  is not noise: the GT atoms rank at the bottom, and the inverted mask scores
  0.34-0.37, the level SubgraphX reaches. The mask selects hydrogens.
- Mutagenicity seed 2 is not near-constant (range 0.23). It is a structurally
  valid mask that ranks the GT low and selects carbons; inverted GEA 0.246.
- B-XAIC X seed 4 is not near-constant either: its spread matches X seeds 0-3.
  It selects carbons (91%) where seeds 0-3 select halogens. Inverting it does
  not help (0.015). A structurally valid mask on the wrong atoms.
So the answer is mixed: two of the four units are near-collapses the strict
rule misses, two are valid masks that select the wrong atoms.

**E2. Can a revised criterion catch the near-collapses? Spread scan over all
95 units** (85 coupled, 5 decoupled MUTAG-GT, 5 Mutagenicity). PGExplainer's
median within-molecule range runs continuously from 0 to 1 with one gap,
between 6e-20 and 6e-8. Two candidate "saturated mask" criteria:

| criterion | all 95 units | 85 coupled | MUTAG-GT coupled | B-XAIC | Tox21 | Mutagenicity |
|---|---|---|---|---|---|---|
| strict (std == 0 in >= 50% of molecules) | 28 | 27 | 1/5 | 2/20 | 24/60 | 0/5 |
| range median < 1e-12 (below the gap) | 39 | 34 | 5/5 | 2/20 | 27/60 | 0/5 |
| range median < 0.01 | 61 | 54 | 5/5 | 2/20 | 47/60 | 2/5 |

Neither revised criterion is recommended as a replacement for the strict
rule, for two reasons visible in the scan:
- Spread does not predict failure. Every MUTAG-GT unit has a range below
  6e-20, including coupled seeds 0 and 4, where PGExplainer beats its random
  null. The soft mask is saturated but its ordering still carries signal.
- No threshold separates the cases. The 1e-12 cut sits in the only gap but
  does not flag Mutagenicity seeds 0-1 (range 0.005-0.007). The 0.01 cut flags
  them but has no gap behind it (B-XAIC P s0 at 0.011 and PAINS s2 at 0.012,
  both working units, sit just above it) and flags 47/60 Tox21 units.
What does identify all four units is the outcome test of F18: significantly
below the matched random null. That needs ground truth (or, on Tox21, the
per-molecule random Fid+ baseline that was deferred). For reference, GNN's
smallest median range over all units is 0.028 and SubgraphX's is 1.0.

**E3. Normalised top-k GEA, (GEA - RandNode) / (ceiling - RandNode), per seed.**
Ceilings fixed at 0.551 (Mutagenicity, max-over-set), 0.751 (MUTAG-GT), 0.724
(B-XAIC indole). 0 = random, 1 = the best the budget allows.

| dataset | GNN mean (std) | PG mean (std) | SX mean (std) | per seed: GNN | per seed: PG | per seed: SX |
|---|---|---|---|---|---|---|
| Mutagenicity | 0.595 (0.208) | -0.009 (0.238) | 0.369 (0.039) | 0.81, 0.81, 0.60, 0.48, 0.27 | -0.18, -0.24, -0.19, 0.30, 0.27 | 0.34, 0.33, 0.41, 0.42, 0.34 |
| MUTAG-GT | 0.355 (0.248) | 0.103 (0.183) | -0.121 (0.064) | 0.38, -0.11, 0.38, 0.55, 0.57 | 0.25, -0.10, -0.07, 0.06, 0.38 | -0.17, -0.20, -0.06, -0.15, -0.04 |
| B-XAIC indole | 0.064 (0.145) | 0.395 (0.035) | 0.816 (0.086) | -0.15, 0.08, 0.15, 0.27, -0.04 | 0.37, 0.43, 0.34, 0.42, 0.41 | 0.86, 0.85, 0.92, 0.79, 0.66 |

Caveats: top-k GEA for MUTAG-GT and indole is recomputed here from the cached
explanations (the tables above for those datasets are mean-threshold);
RandNode is one draw per molecule there and 20 on Mutagenicity; MUTAG-GT seed
3's PGExplainer value comes from a constant mask, so its top-k set is
arbitrary; the ceiling is one number per dataset (per-seed ceilings on
Mutagenicity range 0.52-0.60).

### EXPLORATORY (2026-10-08, second batch): null matching, matched-budget counts, edge-level GEA

Script: src/analysis/exploratory_null_audit.py (cached explanations only).
Output: runs/mutagenicity/exploratory/null_audit.{txt,json}. Registered
predictions untouched.

**E4. What each reported null was matched to.**

| comparison (where reported) | explainer mask | null | null size | draws | matched? |
|---|---|---|---|---|---|
| MUTAG-GT GNN/PG/SX vs RandNode, RandEdge (F15) | own mean-threshold mask | uniform nodes / folded random edge scores | top-k budget, int(0.25 N) | 1 | NO -- one null shared by all three explainers |
| B-XAIC, 4 tasks, same comparisons (F16) | own mean-threshold mask | same | top-k budget | 1 | NO |
| below-random scan (F18) | own mean-threshold mask | RandNode (GNN, SX), RandEdge (PG) | top-k budget | 1 | NO |
| SubgraphX vs RandConn / RandUnc, MUTAG-GT and indole (F15/F16-ADDENDUM, decoupled check) | SX mean-threshold mask | random connected / unconstrained | the SubgraphX mask size, per molecule | 20 | YES (SubgraphX only) |
| Tox21 Fid+/Fid-/GEF vs random (F17) | top-k 0.25 of the importance vector | top-k 0.25 of a random importance vector | top-k budget on both sides | 1 | YES |
| Mutagenicity, mean-threshold vs RandNode/RandEdge | own mean-threshold mask | as F15 | top-k budget | 20 | NO (kept to mirror F15) |
| Mutagenicity, mean-threshold vs RandConn/RandUnc | own mean-threshold mask | connected / unconstrained | each explainer's own mask size | 20 | YES |
| Mutagenicity, top-k comparisons | top-k | RandNode, RandEdge, RandConn | top-k budget | 20 | YES |

Mask sizes (atoms, mean over seeds) and what a random mask of that size scores:

| dataset | GT | budget k | GNN mask | PG mask | SX mask | RandNode at k | random at GNN size | at PG size | at SX size |
|---|---|---|---|---|---|---|---|---|---|
| MUTAG-GT | 4.44 | 4.11 | 6.31 | 3.22 | 4.30 | 0.146 | 0.178 | 0.113 | 0.146 |
| B-XAIC indole | 9.69 | 7.62 | 14.75 | 19.08 | 7.64 | 0.161 | 0.235 | 0.272 | 0.164 |
| B-XAIC PAINS | 9.75 | 7.10 | 14.85 | 8.53 | 6.97 | 0.164 | 0.253 | 0.137 | 0.162 |
| B-XAIC X | 1.80 | 7.32 | 15.16 | 4.49 | 2.78 | 0.054 | 0.058 | 0.052 | 0.046 |
| B-XAIC P | 1.26 | 7.64 | 9.53 | 4.59 | 4.41 | 0.043 | 0.042 | 0.043 | 0.042 |
| Mutagenicity | 3.14 | 5.94 | 9.23 | 8.73 | 4.26 | 0.122 | 0.128 | 0.130 | 0.118 |

The shared budget-size null understates chance for GNN and PG where the GT
is large (indole, PAINS: 0.24-0.27 at their own size against 0.16). It makes
little difference where the GT is 1-3 atoms. Re-testing mean-threshold GEA
against each explainer's own-size null (seeds significantly above / below,
of 5) changes two cells: GNN on B-XAIC X drops from 4/0 to 1/0, and PG on
MUTAG-GT goes from 2/3 to 2/2. Every other cell keeps its count.

**E5. Matched-budget counts: top-k 0.25 GEA vs RandNode at the same budget,
20 draws, per-seed Wilcoxon + Holm over the three explainers.**

Seeds (of 5) significantly ABOVE / significantly BELOW the null:

| | GNN | PG | SX |
|---|---|---|---|
| MUTAG-GT | 4 / 1 | 2 / 2 | 0 / 5 |
| B-XAIC indole | 3 / 2 | 5 / 0 | 5 / 0 |
| Mutagenicity | 5 / 0 | 2 / 3 | 5 / 0 |
| total of 15 | 12 / 3 | 9 / 5 | 10 / 5 |

Other B-XAIC tasks (above / below): PAINS GNN 5/0, PG 3/0, SX 5/0; X GNN 1/0,
PG 5/0, SX 5/0; P 5/0 for all three. PG vs RandEdge at the same budget:
MUTAG-GT 2/2, indole 5/0, PAINS 3/0, X 4/0, P 5/0, Mutagenicity 2/3.

Earlier claims against this:
- "GNN clears random on indole 3/5": unchanged at 3/5. New: it is
  significantly BELOW random in the other two seeds (0 and 4); seed 4 was
  n.s. with one draw.
- "PG clears RandEdge on indole" (4/4 when first written, 5/5 final): holds,
  5/5 against RandEdge and against RandNode.
- Tallies 12/15 (GNN), 10/15 (SX), 9/15 (PG), read as MUTAG-GT + indole +
  Mutagenicity: all three unchanged.
- Changed: GNN on B-XAIC X is significantly above random in 1/5 seeds at the
  matched budget (F16 said 3/5).
- Changed: B-XAIC X seed 4, PGExplainer (the F18 seed case). At the matched
  top-k budget it scores 0.093, significantly ABOVE RandNode (0.054) and n.s.
  against RandEdge (0.074). It is below random only under mean-threshold
  (0.030). That flag is binarization-specific.
- MUTAG-GT seed 3 (collapsed PG) is n.s. under top-k: the top-k set of a
  constant mask is arbitrary.

**E6. Edge-level GEA on native edge masks**, top 25% of directed edges, vs
uniform random edges at the same budget (20 draws). Edge GT exists for
MUTAG-GT, B-XAIC indole and PAINS, and Mutagenicity (edges incident to a
motif atom; max-over-set on Mutagenicity). Per seed; the random-edge null is
about 0.135 (MUTAG-GT), 0.162 (indole), 0.155 (PAINS), 0.110 (Mutagenicity):

| dataset | explainer | s0 | s1 | s2 | s3 | s4 | sig above | sig below |
|---|---|---|---|---|---|---|---|---|
| MUTAG-GT | PG | 0.261 | 0.210 | 0.154 ns | 0.118 | 0.294 | 3 | 1 |
| MUTAG-GT | GNN | 0.221 | 0.098 | 0.309 | 0.210 | 0.245 | 4 | 1 |
| B-XAIC indole | PG | 0.321 | 0.349 | 0.312 | 0.347 | 0.334 | 5 | 0 |
| B-XAIC indole | GNN | 0.182 | 0.094 | 0.125 | 0.207 | 0.168 ns | 2 | 2 |
| B-XAIC PAINS | PG | 0.196 | 0.273 | 0.185 | 0.167 ns | 0.157 ns | 3 | 0 |
| B-XAIC PAINS | GNN | 0.220 | 0.197 | 0.157 ns | 0.186 | 0.196 | 4 | 0 |
| Mutagenicity | PG | 0.030 | 0.035 | 0.067 | 0.195 | 0.164 | 2 | 3 |
| Mutagenicity | GNN | 0.103 ns | 0.183 | 0.159 | 0.177 | 0.135 | 4 | 0 |

- Mutagenicity: the native PGExplainer edge mask shows the same split as its
  folded node mask (below random in seeds 0-2, above in 3-4). The inverted
  edge mask scores 0.240 / 0.272 in seeds 0-1 against a null of 0.109. The
  inversion is in the edge mask itself, not created by folding.
- MUTAG-GT seed 1: the PGExplainer edge mask is significantly ABOVE random
  (0.210 vs 0.135) while its folded node mask is significantly below (0.086
  vs 0.148). Here the node-level verdict does depend on the fold.
- The GNNExplainer edge mask is much weaker than its node mask on
  Mutagenicity (0.10-0.18 against 0.23-0.47).

Degree of the atoms PGExplainer selects, Mutagenicity (terminal = degree 1):

| seed | mean degree, top-k | bottom-k | GT atoms | all atoms | terminal %, top-k | bottom-k | GT | all |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.43 | 1.83 | 1.43 | 2.00 | 77 | 58 | 74 | 51 |
| 1 | 1.32 | 1.84 | 1.42 | 2.02 | 83 | 58 | 76 | 51 |
| 2 | 2.32 | 1.93 | 1.42 | 2.01 | 35 | 54 | 74 | 51 |
| 3 | 1.60 | 2.01 | 1.42 | 2.01 | 71 | 48 | 75 | 50 |
| 4 | 1.84 | 1.58 | 1.35 | 2.03 | 60 | 70 | 78 | 49 |

In seeds 0-1 the selected atoms are as terminal as the GT atoms (77-83%
against 74-76%), so the mask does favour terminal atoms -- but so does the
working seed 3 (71%), and the GT itself is mostly terminal. The mask picks
the wrong terminal atoms (hydrogens). Refolding the same edge masks by sum
or by max instead of mean leaves top-k GEA unchanged in seeds 0-1
(0.037 / 0.029 / 0.033 and 0.016 / 0.013 / 0.014). The folding rule is not
what puts these two seeds below random.

## GraphXAI Mutagenicity, full set by 5-fold CV (runs/mutagenicity_cv/; runs/mutagenicity/ untouched)

### Design and registered predictions -- written 2026-10-09 11:12 IST, BEFORE the run is launched

Design: stratified 5-fold split of the 1,768 graphs (fold seed 0). Fold f: the
D-MPNN is trained on the other four folds (10% of them held out, stratified,
for model selection; model seed f), then every graph in fold f is explained
(explainer seed 1000 + f). Each graph is explained exactly once, by a model
that never saw it. Model config, explainer configs and the 80-atom exclusion
are those of the 5-seed test-split run. Outputs only under runs/mutagenicity_cv/.

Registered predictions:
(a) GNNExplainer beats RandNode in at least 4 of 5 folds.
(b) SubgraphX beats its connectivity-matched null in at least 4 of 5 folds.
(c) PGExplainer is not reliably above RandNode.

How they will be scored (fixed now):
- GEA is max-over-set; top-k 0.25 is the primary binarization.
- (a) GNNExplainer top-k GEA vs RandNode at the same budget (20 draws). A fold
  counts when the fold mean is higher AND the paired Wilcoxon test over that
  fold's GT-bearing molecules is significant after Holm.
- (b) SubgraphX vs RandConn sized to the SubgraphX mask (20 draws): the
  SubgraphX mean-threshold mask against random connected subgraphs of the same
  size, the construction used for the earlier connectivity checks. Same
  counting rule. The top-k version (RandConn at the budget) is reported too.
- (c) holds if PGExplainer is significantly above RandNode (top-k, same rule)
  in FEWER than 4 of 5 folds.
- Holm is applied within each fold over four tests: the three explainers vs
  RandNode and SubgraphX vs RandConn. The pooled test (one per comparison,
  over all unique molecules, Holm over the same four) is reported separately
  and does not score the predictions; the model differs by fold.

### RESULTS, Mutagenicity 5-fold CV (run 2026-10-09 11:13 -> 2026-10-10 02:10 IST;
### src/analysis/mutagenicity_cv_report.py; full output runs/mutagenicity_cv/report.txt)

Run as registered: fold seed 0, model seed f, explainer seed 1000 + f, five
folds concurrently on GPU 1. runs/mutagenicity/ was not modified. Folds 1-4
were started, stopped after 2.5 minutes at the user's request, and restarted
7 minutes later from their saved checkpoints; no explanations were lost.

Model and counts:

| fold | test acc | test AUROC | fold size | explained (<= 80 atoms) | GT-bearing | wall-clock |
|---|---|---|---|---|---|---|
| 0 | 0.9972 | 0.9999 | 354 | 346 | 127 | 14.96 h |
| 1 | 0.9944 | 0.9998 | 354 | 350 | 127 | 13.28 h |
| 2 | 0.9915 | 0.9962 | 354 | 350 | 127 | 14.46 h |
| 3 | 0.9688 | 0.9758 | 353 | 348 | 124 | 14.52 h |
| 4 | 0.9943 | 0.9997 | 353 | 345 | 127 | 13.13 h |
| mean / total | 0.9892 | 0.9943 | 1,768 | 1,739 | 632 | |

Top-k 0.25 GEA, max-over-set (primary), per-fold means over GT-bearing molecules:

| fold | GNN | PG | SX | RandNode | RandEdge | RandConn at k | ceiling |
|---|---|---|---|---|---|---|---|
| 0 | 0.462 | 0.094 | 0.255 | 0.145 | 0.165 | 0.112 | 0.606 |
| 1 | 0.359 | 0.061 | 0.243 | 0.131 | 0.154 | 0.104 | 0.553 |
| 2 | 0.337 | 0.290 | 0.246 | 0.123 | 0.151 | 0.101 | 0.555 |
| 3 | 0.395 | 0.078 | 0.232 | 0.117 | 0.136 | 0.095 | 0.530 |
| 4 | 0.400 | 0.196 | 0.286 | 0.104 | 0.123 | 0.092 | 0.511 |
| fold mean | 0.391 | 0.144 | 0.252 | 0.124 | 0.146 | 0.101 | |
| fold std | 0.043 | 0.087 | 0.018 | 0.014 | 0.015 | 0.007 | |
| pooled, 632 molecules | 0.390 | 0.144 | 0.252 | 0.124 | 0.146 | 0.101 | 0.551 |

Mean-threshold, max-over-set, pooled: GNN 0.337, PG 0.129, SX 0.362; RandConn
sized to the SubgraphX mask 0.107. Merged-mask values are 0.02-0.03 lower
throughout (report.txt section 2).

Per-fold paired Wilcoxon, Holm within fold over the four registered
comparisons (explainer / null; all p_Holm < 0.001 unless marked):

| comparison | f0 | f1 | f2 | f3 | f4 | sig above | sig below |
|---|---|---|---|---|---|---|---|
| GNN vs RandNode (top-k) | 0.462/0.145 | 0.359/0.131 | 0.337/0.123 | 0.395/0.117 | 0.400/0.104 | 5/5 | 0/5 |
| PG vs RandNode (top-k) | 0.094/0.145 | 0.061/0.131 | 0.290/0.123 | 0.078/0.117 | 0.196/0.104 ns | 1/5 | 3/5 |
| SX vs RandNode (top-k) | 0.255/0.145 | 0.243/0.131 | 0.246/0.123 | 0.232/0.117 | 0.286/0.104 | 5/5 | 0/5 |
| SX vs RandConn (SX mask size) | 0.331/0.128 | 0.366/0.108 | 0.350/0.109 | 0.324/0.099 | 0.439/0.090 | 5/5 | 0/5 |

Additional, outside the registered family: SX top-k vs RandConn at the budget
5/5 above; PG vs RandEdge 1/5 above, 3/5 below; mean-threshold vs own-size
RandUnc: GNN 5/5 above, PG 1/5 above and 3/5 below, SX 5/5 above.

Pooled test, one per comparison, over all 632 unique molecules (Holm over
the four). Each molecule appears once, but the explained model differs by
fold, so this is not a single-model test:

| comparison | mean, explainer / null | p_Holm | molecule wins / losses / ties | direction of the rank test |
|---|---|---|---|---|
| GNN vs RandNode | 0.390 / 0.124 | 1.0e-97 | 585 / 47 / 0 | above |
| PG vs RandNode | 0.144 / 0.124 | 7.4e-03 | 189 / 443 / 0 | BELOW |
| SX vs RandNode | 0.252 / 0.124 | 6.6e-35 | 410 / 221 / 1 | above |
| SX vs RandConn | 0.362 / 0.107 | 4.0e-50 | 416 / 200 / 16 | above |

For PGExplainer the pooled mean is above the null while the rank test points
below (median GEA 0.000 against 0.101). "Above"/"below" in this section is
the direction of the rank test. In every per-fold cell the mean and the rank
test agree. The first version of the report labelled direction by the mean
and printed this one cell as "above"; kept as report_v1_mean_direction.txt.

REGISTERED PREDICTIONS:
- (a) GNNExplainer beats RandNode in at least 4 of 5 folds: 5/5 -- HELD.
- (b) SubgraphX beats its connectivity-matched null in at least 4 of 5 folds:
  5/5 -- HELD.
- (c) PGExplainer is not reliably above RandNode: significantly above in 1/5
  folds, significantly below in 3/5, n.s. in 1/5 -- HELD.

PGExplainer collapse (strict rule): 0/5 folds. Degenerate masks per fold:
0/346, 0/350, 1/350, 0/348, 0/345. No empty or whole-graph masks for
GNNExplainer or SubgraphX.

EXPLORATORY -- GEA by toxicophore type, pooled over the 632 molecules,
max-over-set (a graph counts under every type it contains):

| type | graphs | GT atoms | top-k GNN | top-k PG | top-k SX | RandNode | ceiling | mean-thr GNN | mean-thr PG | mean-thr SX | RandConn (SX size) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NH2 | 248 | 3.90 | 0.425 | 0.204 | 0.354 | 0.131 | 0.631 | 0.369 | 0.193 | 0.505 | 0.135 |
| NO2 | 104 | 4.25 | 0.393 | 0.129 | 0.251 | 0.144 | 0.700 | 0.362 | 0.113 | 0.330 | 0.132 |
| halide | 295 | 2.75 | 0.398 | 0.146 | 0.171 | 0.132 | 0.513 | 0.339 | 0.129 | 0.220 | 0.091 |
| nitroso | 6 | 2.00 | 0.184 | 0.052 | 0.189 | 0.069 | 0.334 | 0.174 | 0.067 | 0.444 | 0.081 |
| azo | 44 | 3.20 | 0.239 | 0.059 | 0.243 | 0.084 | 0.407 | 0.183 | 0.078 | 0.431 | 0.092 |

Against the 5-seed test-split run (same four comparisons, same Holm family;
+ significantly above, - significantly below, . n.s.):

| comparison | seeds 0-4 | folds 0-4 | counts match? |
|---|---|---|---|
| GNN vs RandNode | + + + + + | + + + + + | yes (5/5 above) |
| PG vs RandNode | - - - + + | - - + - . | NO: 2 above / 3 below vs 1 above / 3 below / 1 n.s. |
| SX vs RandNode | + + + + + | + + + + + | yes (5/5 above) |
| SX vs RandConn | + + + + + | + + + + + | yes (5/5 above) |

The only cell that differs is PGExplainer vs RandNode. Mean GEA across units:
GNN 0.378 (seeds) vs 0.391 (folds); PG 0.118 vs 0.144; SX 0.280 vs 0.252;
RandNode 0.122 vs 0.124.

Fid+/Fid-/GEF under R3 (mean fill), R2 and R1, per fold: report.txt section 7.
Persisted: runs/mutagenicity_cv/arrays_s<fold>.npz (per-molecule GEA for every
explainer x binarization x GT variant, all nulls, win/loss, mask sizes,
mol_idx, explained_idx, split membership, seeds), report.json, folds.json,
split_f<fold>.json, seeds_s<fold>.json.

## GIN architecture check (runs/arch_gin/; nothing else in runs/ is written)

### Design, implementation and accuracy results -- 2026-10-10, before any GIN explanation exists

Design: plain PyG GIN (3 x GINConv with a 2-layer MLP, hidden 212, sum
pooling, the D-MPNN's 2-layer FFN head, no dropout, no BatchNorm; bond
features ignored) on MUTAG-GT and GraphXAI Mutagenicity, five seeds each.
Splits are the D-MPNN checkpoints' stored splits, exported to json and passed
through --split-json (MUTAG-GT 131/28/29 from runs/prio/, Mutagenicity
1,237/177/354 from runs/mutagenicity/). Training recipe as the D-MPNN: 120
epochs, batch 50, Adam lr 1e-3, same selection rule. Model seed S, explainer
seed 1000 + S. Parameters: GIN 273,058 (MUTAG-GT) and 274,542 (Mutagenicity)
against 276,602 and 280,502 for the D-MPNN.

Code: src/train/baselines.py (GIN, arch switch), --arch in src/train/train.py
(default dmpnn; D-MPNN behaviour unchanged, checkpoints now also carry an
"arch" key), src/explain/run_phase3.py load_model (no arch key = D-MPNN),
src/analysis/arch_gin_splits.py, scripts/arch_gin_sweep.sh,
src/analysis/arch_gin_report.py. Explainers, masking and metrics unchanged.

Smoke test (src/train/test_gin_explainers.py), passed locally and on Ada
before training: single == batched forward; bond features ignored; all-ones
edge mask reproduces the unmasked output and a random mask changes it;
get_embeddings returns 3 x [N, 212]; PGExplainer trains on 10 graphs;
GNNExplainer, SubgraphX (DIG stub) and R3 Fidelity/GEF run.

Accuracy bar, fixed in the plan before training: Mutagenicity every seed test
acc >= 0.95 and AUROC >= 0.98; MUTAG-GT 10-fold CV acc >= 0.83 and AUROC >=
0.90.

Mutagenicity, test split (354 graphs):

| seed | GIN acc | GIN AUROC | D-MPNN acc | D-MPNN AUROC | bar |
|---|---|---|---|---|---|
| 0 | 0.9915 | 0.9994 | 0.9661 | 0.9982 | pass |
| 1 | 0.9887 | 0.9975 | 0.9915 | 1.0000 | pass |
| 2 | 0.9802 | 0.9978 | 0.9915 | 0.9996 | pass |
| 3 | 0.9831 | 0.9709 | 0.9944 | 0.9990 | FAIL (AUROC) |
| 4 | 0.9831 | 0.9842 | 0.9859 | 0.9990 | pass |

MUTAG-GT, 10-fold CV (seed 0, same protocol as the D-MPNN Phase-2 table):
GIN acc 0.8143 +/- 0.1193, AUROC 0.8944 +/- 0.0861 (D-MPNN 0.884 +/- 0.057,
0.943 +/- 0.041) -- FAIL on both. Single-split test accuracy / AUROC of the
five trained GIN models (29 graphs): 0.8966/0.9579, 0.7931/0.9053,
0.8621/0.9158, 0.8276/0.8526, 0.7241/0.8789 (D-MPNN: 0.8966/0.9842,
0.9310/0.9368, 0.8621/0.9684, 0.7586/0.8632, 0.8966/0.9579).

RESULT: both datasets miss the bar as fixed. Nothing was explained or timed;
the run stopped here. Logs: runs/arch_gin/logs/.

### AMENDMENT to the GIN accuracy bar -- 2026-10-10, written BEFORE any GIN explanation is timed or scored

Original bar (fixed before training): Mutagenicity every seed test acc >=
0.95 and AUROC >= 0.98; MUTAG-GT 10-fold CV acc >= 0.83 and AUROC >= 0.90.

Observed: Mutagenicity seed 3 AUROC 0.9709; every other Mutagenicity number
passes (acc 0.9802-0.9915 on all five seeds, AUROC 0.9842-0.9994 on seeds 0,
1, 2, 4). MUTAG-GT 10-fold CV acc 0.814, AUROC 0.894: fails both.

Decision (user, after seeing the accuracy numbers and nothing else):
- Mutagenicity is explained with the bar amended to acc >= 0.95 and AUROC >=
  0.97 on every seed. All five seeds pass the amended bar.
- MUTAG-GT is NOT explained for GIN, and prediction (c) (MUTAG-GT: SubgraphX
  significantly below RandNode in at least 4 of 5 seeds) is dropped.
- The GIN config is not changed and nothing is retrained.
- Because the amendment was made after the accuracy was known, every count
  is also reported with seed 3 excluded.

### Registered predictions, GIN on Mutagenicity -- written 2026-10-10, BEFORE the run

(a) GNNExplainer significantly above RandNode in at least 4 of 5 seeds.
(b) SubgraphX significantly above its RandConn null in at least 4 of 5 seeds.
(d) PGExplainer not significantly above RandNode in at least 3 of 5 seeds.

How they will be scored (fixed now; the rules of the Mutagenicity 5-fold CV):
- Test split of each seed, <= 80 atoms, GT-bearing molecules; GEA is
  max-over-set; top-k 0.25 is the primary binarization; nulls use 20 draws
  per molecule.
- (a), (d): top-k GEA vs RandNode at the same budget. (b): the SubgraphX
  mean-threshold mask vs RandConn sized to GIN's own SubgraphX mask on that
  molecule.
- Per-seed paired two-sided Wilcoxon, Holm within seed over four tests (the
  three explainers vs RandNode, SubgraphX vs RandConn). "Significantly
  above/below" = Holm p < 0.05 with the direction of the rank test.
- (d) counts seeds in which PGExplainer is NOT significantly above RandNode
  (n.s. or below).
- The D-MPNN column for the cell-by-cell comparison is the same four tests on
  runs/mutagenicity/arrays_s<S>.npz (read-only), same molecules per seed.
- Explainer configs, seeds (model S, explainer 1000 + S), masking and the
  three-stage cached driver are those of the D-MPNN 5-seed run. Scoring is by
  src/analysis/arch_gin_report.py, committed with this entry.

### GIN run HELD -- 2026-10-10 15:11 IST (user request)

State at the hold: the five GIN Mutagenicity units were started at 14:34 and
stopped at 15:11. GNNExplainer and PGExplainer had finished on all five
seeds (cached in runs/arch_gin/mutagenicity_graphxai/cache_s<S>.pt on Ada;
run_phase3 also wrote its stage-1 output out_gp_s<S>.json there). SubgraphX
was 4-11 molecules into each seed and that progress is not saved. The
SubgraphX timing run (20 molecules, seed-0 model) measured 107.0 s/mol,
3.52 s/node, single process.

No GIN explanation result has been read, analysed or reported: the report
script has not been run and the stage-1 output files have not been opened or
copied off Ada. The amendment and the registered predictions above stand
unscored. MUTAG-GT was not explained.
