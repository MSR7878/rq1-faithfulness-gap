# RQ1 Tier 3: final report

**Tier 3 complete.** Items E, F and X all ran (E -> F -> X, one chain, 2026-11-03 10:56 to 14:53 on ada, GPU 1). 210 jobs, 0 failed
in the final run. Of the 2 scored predictions, P-F1 held and P-F2 failed. E is exploratory (P-E1 and P-E2 were withdrawn before
any number). One registered limit was not met: 10 of the 50 fine-tuned copies are outside the 0.02 clean-metric limit (kept and
reported, as the PLAN says).

All numbers are method minus random-node (or random-connected) with a hierarchical bootstrap 95% CI, unless stated. Counts are
per-seed better / worse / n.s. after Holm. Full tables: `E.txt`, `F.txt`, `X.txt`; figure numbers: `figures/final/fig1_v2_numbers.json`.

## Predictions

| Prediction | Outcome | Result |
|---|---|---|
| P-E1, P-E2 | **withdrawn** (wrong premise, before any number) | the D-MPNN also applies the mask at every step |
| P-F1: under F, ground truth beats random-connected on Fid+ in >= 4/5 seeds on all 8 valid groups; on MUTAG-GT under R1c it does not | **HELD** | part 1: 5/5 in all 8 groups. Part 2: MUTAG-GT R1c -0.237 [-0.264, -0.210], 0/10 seeds better |
| P-F2: median Kendall tau (Fid+, top-k GEA; GNN, PG, SX fixed; 40 units) strictly higher under F than under R2 | **FAILED** | F +0.33, R2 +0.33 (tie counts as failed). Mean tau F +0.467, R2 +0.350; share with tau = 1: F 0.47, R2 0.40 |

P-F1 caveat: 9 of the 10 MUTAG-GT copies are outside the 0.02 limit (see F), so part 2 rests on copies whose clean accuracy moved by up to 0.069 (upwards on average: 0.831 -> 0.852).

## Checks

- Mask path, before the restart: with a random fractional mask, "all" equals unmodified PyG exactly (max difference 0.0) on D-MPNN, GIN, GINE, GCN; the three variants differ pairwise on every model. `E/mask_path_restart_check.json` (the earlier `E/mask_path.json` is kept).
- "all" re-run vs stored: GNNExplainer reproduces exactly (difference +0.000, Spearman 1.00) on all four models.
- Fine-tune clean metrics: **10 of 50 outside the 0.02 limit** (details under F). Kept.

## E. Where the edge mask enters (exploratory)

GEA minus random-node (full table with random-connected in `E.txt`):

| Model | Method | stored | all | first | last |
|---|---|---|---|---|---|
| D-MPNN | GNN | +0.255 [+0.176, +0.331] | +0.255 [+0.176, +0.331] | +0.169 [+0.117, +0.221] | +0.182 [+0.106, +0.263] |
| D-MPNN | PG | -0.004 [-0.091, +0.087] | -0.006 [-0.098, +0.102] | +0.062 [+0.026, +0.099] | -0.066 [-0.098, -0.036] |
| GIN | GNN | +0.039 [-0.021, +0.102] | +0.039 [-0.021, +0.102] | +0.014 [-0.027, +0.057] | +0.056 [-0.003, +0.113] |
| GIN | PG | -0.093 [-0.107, -0.075] | -0.074 [-0.100, -0.030] | -0.054 [-0.099, -0.000] | -0.093 [-0.101, -0.085] |
| GINE | GNN | +0.080 [+0.004, +0.154] | +0.080 [+0.004, +0.154] | +0.098 [+0.006, +0.223] | +0.118 [+0.004, +0.227] |
| GINE | PG | -0.087 [-0.122, -0.031] | -0.027 [-0.113, +0.127] | -0.032 [-0.108, +0.111] | -0.021 [-0.046, +0.003] |
| GCN | GNN | +0.137 [+0.071, +0.199] | +0.137 [+0.071, +0.199] | +0.148 [+0.106, +0.192] | +0.204 [+0.163, +0.247] |
| GCN | PG | +0.341 [+0.304, +0.373] | +0.343 [+0.299, +0.384] | +0.325 [+0.265, +0.380] | +0.175 [+0.127, +0.217] |

Stored PG = final rule; variant PG = float64 logit scores.

- GNNExplainer does respond, although its atom scores come from the node mask: on the D-MPNN, first - all -0.086 [-0.126, -0.043] and last - all -0.074 [-0.122, -0.032] (Spearman with "all" 0.18 and 0.34). On GCN, last - all +0.067 [+0.025, +0.120].
- PGExplainer on GCN: last - all -0.168 [-0.227, -0.107].
- Ordering of GNN, PG, SX fixed by mean GEA: the same under all, first and last on D-MPNN (GNN > SX > PG), GIN and GINE (SX > GNN > PG). On GCN it changes only under "last" (GNN 0.327 > PG 0.298; stored PG 0.464 > GNN 0.260).
- **D-MPNN seed 0, PG "first":** constant atom scores on all 127 molecules, so its GEA equals random-node (0.117) and Spearman is undefined. The +0.062 for D-MPNN PG "first" therefore comes from seeds 1-4 plus one seed at random. Other undefined-Spearman counts are small (1-12 molecules per seed; listed in `E.txt`).
- **PGExplainer re-runs are not exactly reproducible.** GNNExplainer's "all" re-run matches stored exactly, but PGExplainer's does not (Spearman with stored 0.89-0.95; GINE all - stored +0.050 [+0.000, +0.148]). See contradictions.

## F. Fine-tuned copy and bounded hiding rule

Clean test metrics, copy vs original (limit 0.02):

| Group | acc before -> after | max abs change acc | AUROC before -> after | max abs change AUROC |
|---|---|---|---|---|
| Mutagenicity D-MPNN | 0.986 -> 0.989 | 0.020 | 0.999 -> 0.999 | 0.002 |
| Mutagenicity GIN | 0.985 -> 0.980 | 0.008 | 0.990 -> 0.996 | **0.022** |
| Mutagenicity GINE | 0.992 -> 0.986 | 0.008 | 0.993 -> 0.996 | 0.009 |
| Mutagenicity GCN | 0.980 -> 0.977 | 0.008 | 0.991 -> 0.993 | 0.008 |
| B-XAIC indole | 0.998 -> 0.996 | 0.004 | 1.000 -> 1.000 | 0.000 |
| B-XAIC PAINS | 0.976 -> 0.972 | 0.006 | 0.996 -> 0.995 | 0.001 |
| B-XAIC X | 1.000 -> 1.000 | 0.000 | 1.000 -> 1.000 | 0.000 |
| B-XAIC P | 1.000 -> 1.000 | 0.000 | 1.000 -> 1.000 | 0.000 |
| MUTAG-GT (10 seeds) | 0.831 -> 0.852 | **0.069** | 0.925 -> 0.937 | **0.042** |

Outside the limit: mutagenicity_gin_s3 (AUROC 0.022) and MUTAG-GT seeds 0, 1, 2, 3, 5, 6, 7, 8, 9 (accuracy 0.034-0.069). MUTAG-GT has 188 molecules, so its test set is small and one molecule moves accuracy by about 0.034.

Molecules cut down by the 0.5 bound: no top-k explanation was cut for Fid+ (k = 25% of N). Fid- was cut on every top-k molecule (the complement is 75% of N), so Fid- under F always hides a random half of the molecule and is **not comparable with R2**. Ground-truth sets cut for Fid+: indole 80 of 1,800, PAINS 166 of 1,797, none elsewhere.

T1 repeat (ground truth minus size-matched null, Fid+, probability; counts and mean gap):

| Group | R1 | R2 | R3 | F | R1c |
|---|---|---|---|---|---|
| Mutagenicity D-MPNN (conn) | 5/0/0 +0.430 | 5/0/0 +0.421 | 5/0/0 +0.436 | 5/0/0 +0.418 | 5/0/0 +0.428 |
| Mutagenicity GIN (conn) | 5/0/0 +0.442 | 5/0/0 +0.372 | 5/0/0 +0.450 | 5/0/0 +0.355 | 5/0/0 +0.377 |
| Mutagenicity GINE (conn) | 5/0/0 +0.476 | 5/0/0 +0.402 | 5/0/0 +0.449 | 5/0/0 +0.398 | 5/0/0 +0.431 |
| Mutagenicity GCN (conn) | 5/0/0 +0.468 | 5/0/0 +0.379 | 5/0/0 +0.449 | 5/0/0 +0.381 | 5/0/0 +0.409 |
| B-XAIC indole (conn) | 5/0/0 +0.369 | 5/0/0 +0.397 | 4/1/0 +0.138 | 5/0/0 +0.084 | 5/0/0 +0.362 |
| B-XAIC PAINS (conn) | 5/0/0 +0.331 | 5/0/0 +0.373 | 5/0/0 +0.424 | 5/0/0 +0.340 | 5/0/0 +0.304 |
| B-XAIC X (conn) | 5/0/0 +0.527 | 5/0/0 +0.552 | 5/0/0 +0.552 | 5/0/0 +0.443 | 5/0/0 +0.492 |
| B-XAIC P (conn) | 5/0/0 +0.761 | 5/0/0 +0.623 | 5/0/0 +0.690 | 5/0/0 +0.203 | 5/0/0 +0.756 |
| MUTAG-GT (conn) | 0/10/0 -0.239 | 0/9/1 -0.023 | 2/1/7 +0.012 | 0/9/1 -0.016 | 0/10/0 -0.237 |

R1-R3 use the original models; F and R1c use the fine-tuned copy. Fid-, GEF, the random-node null and logit space are in `F.txt`.

T2 repeat (median tau with top-k GEA over 40 valid units; share with tau = 1):

| Metric | R1 | R2 | R3 | F |
|---|---|---|---|---|
| Fid+ | +0.33 (0.35) | +0.33 (0.40) | +1.00 (0.55) | +0.33 (0.47) |
| Fid- | +1.00 (0.53) | +0.33 (0.47) | +1.00 (0.70) | +1.00 (0.55) |
| GEF | +0.67 (0.50) | +0.33 (0.47) | +1.00 (0.72) | +1.00 (0.53) |

Secondary (all 7 methods, not scored): median tau(Fid+, GEA) F +0.62, R2 +0.43.

Methods under F (minus random-node, probability, Fid+): GNNExplainer, SubgraphX fixed and the gradient methods are above random in most groups. PGExplainer is below or at random on Mutagenicity D-MPNN (-0.011 [-0.033, +0.014], 0/5/0), GIN (-0.043 [-0.073, -0.007]) and GINE (-0.079 [-0.105, -0.049]), and above random on GCN and B-XAIC. The heteroatom rule is above random on Fid+ in all 8 valid groups. On MUTAG-GT only SX fixed and the gradient methods are above random (10/0/0), GNN +0.031 [+0.015, +0.049] (6/1/3), and PG and hetero are not.

## X. Small gaps and the new figure

**Fid for the new seeds (item M, seeds 5-9)**, compared with seeds 0-4 (`G.txt`, `H.txt`) for IG and hetero. In probability space all rows keep their CI category except two:
- IG, Mutagenicity D-MPNN, Fid+ under R1: seeds 0-4 +0.032 [-0.018, +0.093] -> seeds 5-9 +0.053 [+0.014, +0.095] (n.s. -> above).
- hetero, B-XAIC indole, Fid- under R3: seeds 0-4 +0.189 [-0.027, +0.347] -> seeds 5-9 +0.297 [+0.130, +0.464] (n.s. -> above).
GNN, PG and SX fixed for seeds 5-9 are in `X.txt`; I did not compare them line by line with seeds 0-4 (those tables are in `runs/rq1_revision`).

**Fid for the randomised models (item R).** Under R2 and R3 all full-reset cells are within +/-0.012 of random-node. Under R1 some full-reset cells are not:

| Cell (full reset, R1, probability) | Fid+ minus random-node | counts |
|---|---|---|
| Mutagenicity D-MPNN, Saliency | +0.066 [+0.034, +0.098] | 5/0/0 |
| Mutagenicity D-MPNN, PG | +0.032 [+0.016, +0.049] | 5/0/0 |
| B-XAIC indole, PG | -0.122 [-0.155, -0.092] | 0/5/0 |
| B-XAIC indole, Saliency | +0.033 [+0.018, +0.050] | 5/0/0 |
| B-XAIC indole, SX fixed | -0.034 [-0.066, +0.001] | 1/4/0 |

So Fid under R1 can also be away from random on a model with no learned signal, as GEA was in Tier 2 (P-R1).

**New Fig 1** (`figures/final/fig1_v2.pdf`, `.png`). GEA at top-k = 0.25; per-seed diamonds; boxes pool molecules across seeds for display only. PGExplainer uses float64 logit scores for all seeds. The best-trivial-rule band was chosen after seeing each panel, and the figure and report say so. Exact numbers shown:

| Panel | seeds | random-node band | best rule (band) | GNN | PG (logit) | SX fixed | IG |
|---|---|---|---|---|---|---|---|
| MUTAG-GT | 10 | 0.140-0.148 | hetero 0.644-0.650 | 0.359; +0.213 [+0.113, +0.303] | 0.210; +0.065 [+0.010, +0.125] | 0.068; -0.077 [-0.101, -0.053] | 0.122; -0.023 [-0.070, +0.023] |
| Mutagenicity D-MPNN | 10 | 0.113-0.134 | hetero 0.323-0.380 | 0.377; +0.255 [+0.206, +0.300] | 0.131; +0.009 [-0.061, +0.091] | 0.263; +0.141 [+0.114, +0.163] | 0.456; +0.334 [+0.301, +0.366] |
| B-XAIC indole | 10 | 0.160-0.164 | high degree 0.225-0.233 | 0.218; +0.056 [-0.014, +0.128] | 0.430; +0.268 [+0.207, +0.333] | 0.654; +0.491 [+0.462, +0.513] | 0.596; +0.434 [+0.410, +0.459] |
| B-XAIC PAINS | 5 | 0.159-0.167 | high degree 0.233-0.241 | 0.236; +0.073 [+0.049, +0.096] | 0.221; +0.058 [+0.012, +0.109] | 0.466; +0.303 [+0.289, +0.316] | 0.372; +0.209 [+0.186, +0.235] |

Each cell: mean of per-seed mean GEA; minus random-node [95% CI]. PG final rule, not shown: MUTAG-GT +0.065 [+0.010, +0.125], D-MPNN +0.014 [-0.061, +0.098], indole +0.265 [+0.204, +0.331], PAINS +0.064 [+0.019, +0.113]. Per-seed values and box medians are in `X.txt`. PAINS has 5 seeds because no more exist.

## Contradictions with current results

Rule: only a change of CI category counts. The paper's `.tex` files are not on this machine, so I compared with the results files, not with the paper's wording.

| Current result | New number | Why it matters |
|---|---|---|
| T3 / H: Mutagenicity GINE PG below random, -0.087 [-0.122, -0.031] (0/4/1) | Seeded "all" re-run of the same PGExplainer: -0.027 [-0.113, +0.127] (1/4/0); "all" - stored +0.050 [+0.000, +0.148] | **CI category changes (below -> n.s.) just from re-training PGExplainer.** GNNExplainer's re-run matches exactly, so this is run-to-run variation in PGExplainer, not a code change. "PG below random on GINE" is not stable |
| IG, Mutagenicity D-MPNN, Fid+ under R1, n.s. (seeds 0-4) | above random with seeds 5-9: +0.053 [+0.014, +0.095] | Category change with the new seeds |
| hetero, indole, Fid- under R3, n.s. (seeds 0-4) | above random with seeds 5-9: +0.297 [+0.130, +0.464] | Category change with the new seeds |
| Any claim that Fid shows reliance on the model | Full reset, R1: D-MPNN Saliency Fid+ +0.066 [+0.034, +0.098]; indole PG Fid+ -0.122 [-0.155, -0.092] | Fid under R1 is away from random on models with no learned signal (R2 and R3 are not) |
| Any claim that a fine-tuned hiding rule fixes Fid's disagreement with GEA | P-F2 failed: median tau F +0.33 = R2 +0.33 | F does not raise the median agreement (the mean and the share with tau = 1 are somewhat higher) |

Confirmed, no change: ground truth beats random on Fid+ on all 8 valid groups under every rule, including F; on MUTAG-GT it does not beat random under R1, R1c, R2 or F (R3: 2/1/7, +0.012). The Fig 1 numbers for GNN, SX fixed and IG equal the Tier 2 numbers (item M for D-MPNN and indole; H, S and G for MUTAG-GT); for PG, the final-rule numbers equal Tier 2 and the logit-ranked numbers shown differ by at most 0.005 in this table.

## What did not run or is limited

- Nothing registered was skipped; all 210 jobs (60 E explain, 20 E score, 50 F fine-tune, 50 F score, 30 X Fid) finished with 0 failures in the final run.
- E does not re-run SubgraphX, gradients or heuristics (by design).
- 10 of 50 fine-tuned copies are outside the 0.02 limit (kept, per PLAN).
- Seeds 5-9 Fid for GNN, PG, SX fixed were not compared line by line with seeds 0-4.
- The figure palette was not run through the colour validator (no node on this machine). The figure keeps the paper's existing palette and hatches, and IG gets its own fill and hatch.
- Fine-tuned checkpoints, PGExplainer MLPs and run_phase3 caches (`*.pt`) stay on ada, not in git.

## Things that broke

- First E launch: all 60 jobs crashed at model setup. The mask wrapper had a different function name, and PyG's Inspector looks up `explain_message` by name. No output was written. Fixed (f064d42), smoke-tested, mask check re-run on user request, then re-run. Record kept in `E/crash_f064d42_before/`.
- The E report printed `nan` for one mean Spearman, because a seed had all-constant scores. It now averages over the seeds where Spearman is defined and lists the undefined counts.
- First draw of Fig 1: the x-axis labels overlapped. Redrawn with the method names on two lines and the rule name in each panel title.

## Commits

| Commit | Content |
|---|---|
| f3781e6, d4a2ad4 | E code, mask-path check |
| 787f12a | F fine-tune code |
| e97289a | PLAN, E/F/X code |
| ba91e15 | Reports, Fig 1 code |
| f064d42 | E fix (wrapper name) |
| 7d21b2e | E results and report |
| c0a4882 | F results and report |
| a1dd24f | X results, Fig 1 v2 |
| (this commit) | Final report, DONE |
