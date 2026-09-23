# RQ1 Metric Spec (locked) -- mirrors RQ1_Metric_Spec.docx v5

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
**3/5 on PAINS** (seeds 3-4 fully COLLAPSED, all-0.000 mask) and **4/5 on X**
(seed 4 near-collapsed, PG=0.030, still significantly BELOW RandE,
p=3.7e-09). That is FOUR independent collapse/near-collapse instances now,
across THREE different datasets: Tox21 (F10, ~20-30% estimated rate),
mutag_graphxai seed 3 (F3-REVISION v2), and B-XAIC PAINS seeds 3-4 + X
seed 4 (this entry) -- 3 of 25 B-XAIC (task,seed) units collapsed/
near-collapsed (12%), consistent with F10's estimated failure rate given
small-n. PGExplainer's training instability is a general property of the
method, not tied to any one dataset.

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
2. **PGExplainer collapse/near-collapse is a FIVE-dataset-spanning
   phenomenon**, with 3 confirmed instances inside B-XAIC (out of 20 units,
   15%): PAINS seeds 3-4 (full collapse) and X seed 4 (near-collapse) -- P
   and indole show ZERO collapses across their 10 combined seeds -- plus
   Tox21 (F10) and mutag_graphxai seed 3 (F3-REVISION v2). 15% is consistent
   with F10's ~20-30% Tox21 estimate (small-n, same order of magnitude).
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
than B-XAIC's 15% (F16). Every endpoint except SR-HSE collapsed in at least
one of its 5 seeds. **A striking pattern: 9 of the 12 endpoints collapsed
specifically at seed 4** (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER,
NR-ER-LBD, SR-ARE, SR-MMP, SR-p53 all collapse at s4; expected count under a
uniform-across-seeds null would be ~4.8/12, ~2 std below the observed 9) --
suggestive that collapse correlates with the SEED VALUE itself (likely via
`torch.manual_seed`'s effect on the edge-mask MLP's specific initialisation
draw) more than with independent per-(endpoint,seed) chance, though n=12 is
too small to treat this as settled. Worth a dedicated multi-seed-value check
if pursued further; flagged here, not resolved.

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
collapses often and unevenly by dataset**: B-XAIC 3/20 (15%), mutag_graphxai
1/5 (20%), **Tox21 24/60 (40%) -- the highest rate seen, and notably
clustered at seed 4 across 9 of 12 endpoints** (F17-REVISION block D, a
pattern possibly tied to `torch.manual_seed`'s effect on the edge-mask
MLP's init, not resolved). Total across the whole priority sweep: **28 of
85 completed (dataset,endpoint/task,seed) units where PGExplainer's own
GEA/Fid numbers are unreliable** -- this is not a rare edge case, it is a
first-order property of the method that any paper using PGExplainer without
multi-seed screening needs to disclose.

**No explainer beats chance in every setting checked.** GNN is real only on
the two GT/GEA datasets. SX is real on B-XAIC (fully) and Tox21 (Fid+ only,
reproducibly) but actively fails on mutag_graphxai. PG is real wherever it
successfully trains, which is 60-85% of the time depending on dataset. The
faithfulness gap now has a proper null to be measured against on every
dataset in the priority sweep, not just inter-explainer comparisons.
