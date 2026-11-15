# RQ1 Tier 4: final report

**Tier 4 complete.** All nine items ran (L1, L3, L4, L5, L7, L2, L6, M1, M2) as one chain on ada, GPU 1, from 2026-11-04 14:43 to
2026-11-06 03:03 (about 36 h; the PLAN estimated 44-48 h). Every stage finished with 0 failed jobs. The reports were run afterwards
(M1 and M2 on 2026-11-12). Of the 6 predictions, **3 held (P-L1, P-L5, P-M2) and 3 failed (P-L3, P-L4, P-M1)**. L2 and L6 are
exploratory and have no prediction. The TOST results for Tox21 and BBBP are re-labelled with four labels; only the stored numbers
were used, and nothing was re-run.

All numbers are method minus null with a hierarchical bootstrap 95% CI (seeds, then molecules), unless stated otherwise. Counts are
per-seed better / worse / n.s. after Holm correction. Molecules are never pooled across seeds. The PLAN (`PLAN.md`, commit 7506ea0)
was committed before any number was computed. Full tables are in `L1.txt` ... `M2.txt` and `tost_relabel.txt`.

## Predictions

| Prediction | Outcome | Result |
|---|---|---|
| P-L1: MUTAG-GT test molecules only: (a) SX < random-node, (b) hetero > random-node, (c) GT removal changes the margin less than random connected sets under R1 in >= 8/10 seeds | **HELD** | (a) -0.078 [-0.112, -0.030]; (b) +0.520 [+0.493, +0.548]; (c) R1 logit Fid+ GT - random-connected -3.120 [-3.968, -2.342], GT worse in 10/10 seeds |
| P-L3: on molecules with a non-constant PG mask, PG - random-node differs by < 0.10 across the 4 models | **FAILED** | D-MPNN -0.015, GIN -0.103, GINE -0.109, GCN +0.389; range 0.499 |
| P-L4: (a) removing aromatic atoms changes the margin more than random connected sets (R1) in >= 8/10 seeds, and (b) SX overlaps aromatic atoms more than random connected sets in >= 8/10 seeds | **FAILED** | (a) -0.357 [-0.681, -0.106], scaffold higher in 1/10 seeds; (b) +0.056 [+0.014, +0.094], higher in 6/10 seeds |
| P-L5: three deterministic PG runs give identical masks (10 units) | **HELD** | bitwise identical on all 10 units (max difference 0.0) |
| P-M1: on indole and PAINS, SX and IG are above the best trivial rule (high degree) for every architecture (16 cells) | **FAILED** | 15 of 16 hold; PAINS GCN IG +0.013 [-0.016, +0.041] includes 0 |
| P-M2: PAINS D-MPNN with 10 seeds keeps every CI category of the 5-seed result (5 cells) | **HELD** | 5 of 5 keep their category (also the 4 cells not scored); every cell 10/10 in the seed sign test |

## Limits: resolved / partly / not

| Limit | Status | One line |
|---|---|---|
| L1 MUTAG-GT scored on all 188 molecules, including training molecules | **resolved** | Test-only gives the same GEA categories; 14 of 108 Fid cells lose significance (none reverses a significant result) |
| L2 single-split datasets (MUTAG, BBBP) | **partly** | BBBP now has 5 seeds; MUTAG (standard) is dropped, not re-run (MUTAG-GT with 10 seeds covers it); BBBP models are below the Mutagenicity bar |
| L3 PG saturation behind the model ordering | **resolved** (answer: no) | On non-constant masks the GCN lead stays (+0.389 vs about -0.1 for GIN and GINE), so saturation does not explain it |
| L4 what the MUTAG-GT model uses | **partly** | The aromatic scaffold is sufficient (Fid- R1 10/10) but not more necessary than random sets; the scaffold covers 12.4 of 18.3 atoms, so the test has little room |
| L5 PG not reproducible run to run | **resolved** | Deterministic GPU runs are bitwise identical; GINE PG is below random in all three |
| L6 SubgraphX stability per molecule | **resolved** | Sets vary (Jaccard 0.63-0.82), but a consensus of 3 runs changes no category |
| L7 10 fine-tuned copies drifted > 0.02 | **partly** | 8 of 10 fixed; MUTAG-GT s1 and s9 not fixed; P-F1 still holds |
| M1 only D-MPNN on B-XAIC | **partly** | GIN, GINE and GCN all ran; 19 of 30 new models miss the accuracy bar (all PAINS, 4/5 indole GCN) |
| M2 PAINS with only 5 seeds | **resolved** | 10 seeds keep every category |

## L1. MUTAG-GT test molecules only (`L1.txt`)

Each seed has 29 test molecules; 10 seeds; no new explanations.

| Method | test only, minus random-node | all 188 | counts (test) |
|---|---|---|---|
| GNN | +0.201 [+0.097, +0.295] | +0.213 [+0.113, +0.303] | 7/2/1 |
| PG (logit) | +0.054 [+0.000, +0.107] | +0.065 [+0.010, +0.125] | 1/1/8 |
| SX fixed | -0.078 [-0.112, -0.030] | -0.077 [-0.101, -0.053] | 0/8/2 |
| Saliency | -0.125 [-0.134, -0.115] | -0.123 [-0.128, -0.116] | 0/10/0 |
| IG | -0.035 [-0.075, +0.011] | -0.023 [-0.070, +0.023] | 0/4/6 |
| IxG | -0.118 [-0.130, -0.102] | -0.114 [-0.127, -0.096] | 0/10/0 |
| hetero | +0.520 [+0.493, +0.548] | +0.503 [+0.494, +0.512] | 10/0/0 |
| high degree | -0.029 [-0.034, -0.023] | -0.029 [-0.032, -0.027] | 0/9/1 |
| low degree | +0.310 [+0.297, +0.323] | +0.308 [+0.304, +0.313] | 10/0/0 |

- No GEA category changes between test-only and all 188.
- Fid: 14 of 108 cells go from significant (all 188) to "includes 0" (test only). One point estimate flips sign (IxG Fid- R3, probability: +0.002 test vs -0.013 all, both near 0); none reverses a significant result. This is the loss of power from 29 instead of 188 molecules per seed.
- Freshly computed all-188 Fid matches the reported pass-E arrays (differences <= 0.002).

## L3. PG saturation (`L3.txt`)

Split on the T4 re-run's float32 mask (constant = max equals min); ranked by the same run's float64 logits.

| Model | not constant: molecules | PG - random-node | constant: molecules | PG - random-node |
|---|---|---|---|---|
| D-MPNN | 631 | -0.015 [-0.089, +0.062] | 1 | (1 molecule) |
| GIN | 391 (none in s4) | -0.103 [-0.111, -0.095] | 241 | -0.032 [-0.090, +0.020] |
| GINE | 496 (none in s4) | -0.109 [-0.128, -0.089] | 136 | +0.041 [-0.014, +0.113] |
| GCN | 563 | +0.389 [+0.341, +0.445] | 69 | -0.032 [-0.101, +0.038] |

In GIN s4 and GINE s4 the mask is constant on every molecule. The split on the stored mask (secondary) gives the same picture (GCN +0.389, GIN -0.082, GINE -0.094).

## L4. MUTAG-GT aromatic scaffold (`L4.txt`)

174 molecules per seed (14 without an aromatic atom are skipped); the scaffold has 12.4 of 18.3 atoms on average.

| Logit, scaffold - random-connected | R1 | R2 | R3 |
|---|---|---|---|
| Fid+ (necessity, higher = more needed) | -0.357 [-0.681, -0.106] 1/6/3 | -0.008 [-0.258, +0.332] 2/4/4 | -0.424 [-0.593, -0.261] 1/9/0 |
| Fid- (sufficiency, lower = more sufficient) | -2.213 [-2.820, -1.704] 10/0/0 | +0.103 [-0.126, +0.335] 4/4/2 | +0.414 [+0.227, +0.639] 1/8/1 |

| Share of chosen atoms that are aromatic, minus random-connected | difference | counts |
|---|---|---|
| SX fixed | +0.056 [+0.014, +0.094] | 6/0/4 |
| GNN | -0.306 [-0.417, -0.172] | 2/8/0 |
| IG | -0.073 [-0.155, +0.004] | 3/4/3 |

## L5. PG reproducibility (`L5.txt`)

The GPU smoke test (deterministic settings, two runs) gave no error and identical masks, so all repeats ran on the GPU.

| Model | 3 deterministic repeats | stored (final rule) | T4 re-run (stored logits) | Tier-3 E "all" re-run |
|---|---|---|---|---|
| D-MPNN | +0.007 [-0.087, +0.108] (all three identical) | -0.004 [-0.091, +0.087] | -0.014 [-0.087, +0.063] | -0.006 [-0.098, +0.102] |
| GINE | **-0.070 [-0.114, -0.011]** (all three identical) | -0.087 [-0.122, -0.031] | -0.077 [-0.121, -0.015] | -0.027 [-0.113, +0.127] |

The earlier, non-deterministic runs differ a lot: on many units the masks differ by up to 1.0 between runs, and the mean Spearman of node scores between runs is 0.85-0.97.

## L7. Re-tuned fine-tuned copies (`L7.txt`)

| Copy | lr / epochs | change in acc / AUROC | fixed |
|---|---|---|---|
| mutagenicity_gin_s3 | 5e-5 / 5 | 0.008 / 0.019 | yes |
| MUTAG-GT s0, s2, s3, s5, s6, s7, s8 | 1e-5 to 5e-5 / 1-5 | 0.000 / <= 0.016 | yes |
| MUTAG-GT s1 | 1e-5 / 1 | 0.069 / 0.000 | **no** |
| MUTAG-GT s9 | 1e-5 / 1 | 0.034 / 0.000 | **no** |

- Item F re-scored: 3 CI categories change relative to random-node (listed under contradictions).
- P-F1 part 2 is unchanged: MUTAG-GT R1c Fid+ GT - random-connected -0.237 [-0.265, -0.211], 0/10 seeds better.

## L2. BBBP, 5 seeds (`L2.txt`, exploratory)

- Models: accuracy 0.84-0.87, AUROC 0.86-0.90 (the stored single model: 0.833 / 0.841).
- SubgraphX is above random rankings: under R2, Fid+ +0.185 [+0.133, +0.243] and Fid- -0.176 [-0.259, -0.090], 5/5 seeds.
- GNN and PG are close to random (see the TOST table below).

## L6. SubgraphX stability (`L6.txt`, exploratory)

| Subset (seed-0 model, 100 molecules) | mean Jaccard of 3 runs | atoms in >= 2 of 3 runs | consensus - single (GEA) | single - random-node |
|---|---|---|---|---|
| Mutagenicity D-MPNN | 0.635 | 0.73 | +0.015 [-0.000, +0.030] | +0.240 [+0.180, +0.301] |
| B-XAIC indole | 0.815 | 0.87 | -0.000 [-0.007, +0.007] | +0.477 [+0.435, +0.514] |
| MUTAG-GT | 0.628 | 0.71 | -0.006 [-0.012, +0.001] | -0.118 [-0.131, -0.104] |

The Tier-2 run agrees with the new runs about as well as they agree with each other (0.634, 0.826, 0.596).

## M1. GIN, GINE, GCN on B-XAIC (`M1.txt`)

Models (test accuracy / AUROC; bar 0.95 / 0.98). All D-MPNN, indole GIN and indole GINE models meet the bar. Below the bar:
indole GCN in 4/5 seeds (0.94-0.95), PAINS GIN (about 0.935 / 0.98), PAINS GINE (0.93-0.95), PAINS GCN (about 0.85 / 0.92).
As the PLAN says, they are kept and flagged.

Minus the best trivial rule (high degree); SubgraphX on the same 100 molecules for every architecture:

| Task | Model | SX fixed | IG | Saliency | PG (logit) | GNN |
|---|---|---|---|---|---|---|
| indole | D-MPNN | +0.400 [+0.349, +0.441] | +0.351 [+0.328, +0.378] | +0.339 | +0.144 | -0.029 |
| indole | GIN | +0.146 [+0.027, +0.276] | +0.280 [+0.226, +0.333] | +0.285 | -0.054 | -0.114 |
| indole | GINE | +0.231 [+0.086, +0.340] | +0.201 [+0.141, +0.260] | +0.314 | -0.206 | -0.136 |
| indole | GCN (bar missed) | +0.226 [+0.173, +0.275] | +0.181 [+0.121, +0.230] | +0.207 | +0.174 | -0.122 |
| PAINS | D-MPNN | +0.237 [+0.210, +0.266] | +0.135 [+0.110, +0.161] | +0.088 | -0.016 | -0.002 |
| PAINS | GIN (bar missed) | +0.051 [+0.007, +0.107] | +0.122 [+0.053, +0.166] | +0.167 | -0.061 | -0.068 |
| PAINS | GINE (bar missed) | +0.159 [+0.116, +0.199] | +0.138 [+0.107, +0.168] | +0.185 | +0.045 | -0.073 |
| PAINS | GCN (bar missed) | +0.107 [+0.083, +0.134] | **+0.013 [-0.016, +0.041]** | +0.025 | -0.022 | -0.004 |

Ordering by mean GEA: SubgraphX is first on D-MPNN (both tasks) and GCN (both tasks); Saliency is first on GIN and GINE (both
tasks; SubgraphX means are on its 100-molecule subset, the others on all GT molecules). PG is far below random-node on indole GINE (-0.139 [-0.156, -0.113]) and near random on indole GIN (+0.012 [-0.093, +0.192]),
while it is +0.211 on the D-MPNN. GNNExplainer is at or below random-node on indole for GIN, GINE and GCN.

## M2. PAINS D-MPNN, 10 seeds (`M2.txt`)

| Cell (minus random-node) | 5 seeds | 10 seeds | counts (10) | category |
|---|---|---|---|---|
| GNN | +0.073 [+0.049, +0.096] | +0.074 [+0.060, +0.088] | 10/0/0 | above -> above |
| PG (logit) | +0.058 [+0.013, +0.109] | +0.042 [+0.017, +0.072] | 8/0/2 | above -> above |
| SX fixed | +0.303 [+0.289, +0.316] | +0.303 [+0.289, +0.317] | 10/0/0 | above -> above |
| IG | +0.209 [+0.186, +0.235] | +0.207 [+0.189, +0.225] | 10/0/0 | above -> above |
| hetero | +0.011 [+0.004, +0.016] | +0.011 [+0.007, +0.016] | 1/0/9 | above -> above |

New models (seeds 5-9) all meet the bar (accuracy 0.967-0.976).

## TOST re-labelled with four labels (`tost_relabel.txt`)

The labels are:
- **different:** the 95% CI excludes 0 and |effect| > 0.05.
- **small but real:** the 95% CI excludes 0 and |effect| <= 0.05.
- **equivalent:** the 95% CI includes 0 and the 90% CI lies within +/-0.05 (the same TOST criterion as before).
- **undecided:** otherwise.

Effect = the stored bootstrap point estimate, in probability space. Only the stored numbers are used.

| Dataset | different | small but real | equivalent | undecided | old labels (different / equivalent / undecided) |
|---|---|---|---|---|---|
| Tox21 (68 cells, R3) | 1 | 21 | 41 | 5 | 10 / 53 / 5 |
| BBBP (30 cells, R1-R3) | 9 | 4 | 11 | 6 | 11 / 13 / 6 |

- **Tox21:** 21 cells change label, and all of them become "small but real". 18 are SubgraphX cells; the other 3 are NR-AR-LBD GNN Fid+ and NR-ER-LBD PG Fid+ and Fid-.
  - 9 were "different", with effects of only 0.033-0.049.
  - 12 were "equivalent" although their 95% CI excludes 0.
  - Only NR-AR SubgraphX Fid- (-0.077) stays "different".
- **BBBP:** 4 cells change, all under R3: PG Fid+, SX Fid+, SX Fid- and IG Fid-. SX Fid- R3 was "equivalent" with a 95% CI that excludes 0 (-0.029 [-0.049, -0.009]); it is now "small but real".

## Contradictions with current results

Rule (as in Tier 3): a change of CI category counts. The paper's `.tex` files were not read; the comparison is with the results files.

| Current result | New number | Why it matters |
|---|---|---|
| Tier 3 contradiction: "GINE PG below random is not stable" (stored logit -0.077 [-0.121, -0.015] vs Tier-3 E "all" re-run -0.027 [-0.113, +0.127]) | **L5: three deterministic repeats give -0.070 [-0.114, -0.011] (below random), identical in all three**, matching the stored run (final rule -0.087 [-0.122, -0.031]) and the T4 re-run (logit -0.077 [-0.121, -0.015]) | **Correction:** GINE PG is below random. The one run that disagreed was the Tier-3 E "all" re-run. Of the three earlier trainings (all non-deterministic), it is the only one whose CI includes 0. Logit-ranked like for like: deterministic -0.070, T4 re-run -0.077, E "all" -0.027. The Tier-3 contradiction is withdrawn |
| Tox21 TOST: 53 of 68 no-separation cells "equivalent", 10 "different" | Four labels: 41 equivalent, 21 small but real, 1 different, 5 undecided | 12 "equivalent" cells have a 95% CI excluding 0; 9 of 10 "different" cells have effects <= 0.05. Only NR-AR SX Fid- is different by more than the margin |
| BBBP TOST (L2): SX Fid- R3 "equivalent" | -0.029 [-0.049, -0.009], now "small but real" | The old rule called a cell equivalent although its CI excludes 0 |
| MUTAG-GT Fid on all 188 molecules: 14 cells significant (e.g. SX fixed R1 logit Fid+ +0.278 [+0.136, +0.417]) | Test only: includes 0 (e.g. +0.321 [-0.013, +0.621]); full list in `L1.txt` | These 14 cells rest on training molecules for power; one point estimate flips sign near 0 (IxG Fid- R3 prob +0.002 vs -0.013), none reverses a significant result |
| Tier 3 F: MUTAG-GT PG Fid+ vs random-node -0.003 [-0.012, +0.006] (n.s.) | Re-tuned copies: -0.013 [-0.027, -0.001] (below) | Category change from re-tuning (the re-tuned set includes s1 and s9, which are still outside the limit) |
| Tier 3 F: MUTAG-GT hetero Fid+ vs random-node -0.008 [-0.016, +0.000] (n.s.) | Re-tuned copies: -0.018 [-0.037, -0.003] (below) | Category change from re-tuning (includes s1 and s9, still outside the limit) |
| Tier 3 F: Mutagenicity GIN PG Fid+ vs random-node -0.043 [-0.073, -0.007] (below) | Re-tuned copy: -0.036 [-0.071, +0.000] (includes 0) | Category change from re-tuning |
| SX and IG beat the best trivial rule on B-XAIC (D-MPNN, Tier 2 / Tier 3 X) | PAINS GCN IG - high degree +0.013 [-0.016, +0.041] | Does not hold for every architecture (this model misses the accuracy bar) |

Not a category change, but it bears on how results are read:
- **L3:** the GCN PG lead on Mutagenicity is not a saturation effect. On non-constant masks GCN is +0.389 and GIN and GINE are about -0.10.
- **L4:** the aromatic scaffold is not more necessary than random connected sets under R1 (-0.357 [-0.681, -0.106]). It is much more sufficient (Fid- -2.213 [-2.820, -1.704], 10/10).
- **M1, PG:** on indole, PG is +0.211 [+0.188, +0.237] on the D-MPNN, but -0.139 [-0.156, -0.113] on GINE and +0.012 [-0.093, +0.192] on GIN, so PG's result depends on the architecture (these are new models, not a re-test of a current cell).
- **M1, ordering:** SubgraphX is first on the D-MPNN and GCN; Saliency is first on GIN and GINE. SubgraphX means are on the 100-molecule subset and Saliency on all GT molecules; the gap is narrow on PAINS GINE (0.422 vs 0.390).
- **L6:** SubgraphX sets change between runs with different seeds (Jaccard 0.63 on Mutagenicity and MUTAG-GT), but no conclusion changes.

Confirmed, no change:
- P-F1 holds with the re-tuned copies.
- PAINS D-MPNN with 10 seeds keeps every category.
- Test-only MUTAG-GT keeps every GEA category.

## What did not run or is limited

- Nothing registered was skipped; all stages finished with 0 failed jobs.
- MUTAG (standard) was dropped (recorded in the PLAN), not re-run.
- MUTAG-GT copies s1 and s9 are still outside the 0.02 limit (any changed test prediction moves accuracy by 0.034); scored and marked.
- 19 of 30 M1 models miss the accuracy bar; no tuning was done to reach it (PLAN). P-M1 rests partly on these models, and its one failed cell (PAINS GCN) is the weakest model (accuracy about 0.85).
- The BBBP models (L2) are below the Mutagenicity bar; no bar was set for BBBP.
- The four-label TOST is a re-labelling of stored numbers. Its "equivalent" uses the 90% CI and "excludes 0" uses the 95% CI; the effect compared with 0.05 is the point estimate.
- After the chain finished (2026-11-06), the M1 and M2 reports were not run until 2026-11-12, because the watcher that should have reported the finish failed. No result is affected.
- `.pt` files (checkpoints, caches) stay on ada and are not in git.

## Commits

| Commit | What |
|---|---|
| 7506ea0 | PLAN (predictions, settings, Holm families, time estimate), before any number |
| 505d2e9 | compute code for L1-L7, M1, M2, job lists and chain scripts |
| 0771555 | report code for L1 and L3 |
| e642c94 | L1 done, P-L1 HELD |
| c162bce | L3 done, P-L3 FAILED |
| 8ea96a7 | L4 report code |
| 9d0eae8 | L4 done, P-L4 FAILED |
| deed3c4 | L5 report code |
| 486008f | report code for L7, L2, L6, M1, M2 |
| cb4db5f | L5 done, P-L5 HELD |
| bfa202e | L7 done |
| d1d5b0f | L2 done (exploratory) |
| ef4b4e2 | L6 done (exploratory) |
| 8e502b0 | four-label TOST re-labelling code |
| 1d54417 | TOST re-labelled |
| 5dd5afd | M1 done, P-M1 FAILED |
| e453dc0 | M2 done, P-M2 HELD |
