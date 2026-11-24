# RQ1 Tier 6: final report

**Tier 6 complete.** T1–T4 ran as registered in `PLAN.md`, with 0 failed jobs. The requested additions (T2 before/after and class changes, T5, and the optional T6) were registered in the PLAN addendum of 2026-11-24 before any of their numbers. They use existing outputs only, with no new explainer runs.

**P-T2 FAILED: 0 of 6 cells above random-node.**

All numbers are a method minus a null (or trained minus reset), with a hierarchical bootstrap 95% CI (seeds, then molecules; 2,000 reps). Counts are per seed: better / worse / n.s. after Holm. "Above" means the whole CI is above 0. GEA is at top-25% (k = max(1, floor(0.25 N))) with random ties (20 orders).

Source files: `T1.txt` … `T6.txt` (+ `.json`), `PLAN.md`.

## 1. Commits

| Commit | What |
|---|---|
| 0d8846c | PLAN with P-T2, before any number |
| 67b0f08, c62c42e | Explain / score / job code; report code (before running) |
| 58d94e5 | T1 results |
| 39d6e42 | T3 results |
| f996593 | T4 results |
| 53d922f | PLAN addendum (T2 additions, T5, T6), before those numbers |
| 139b258, 34022cc | Addendum code; post-hoc T2(b) row in the T6 report |
| 90e40eb | T2 results. P-T2 FAILED |
| 743e79a | T5 results |
| 5fbd4c4 | T6 results |
| (this commit) | FINAL_REPORT.md and DONE |

Run times on ada (GPU 1, 8 workers): T1 12:04–13:26, T2 13:26–19:05, T3 19:05–19:18, T4 19:18–19:45 (all 2026-11-19); T5 and T6 a few minutes each (2026-11-24).

## 2. T1: are the chosen atoms on the aromatic rings? (MUTAG-GT D-MPNN, 10 seeds)

Share of the top-k atoms that are aromatic, minus random-connected sets of the same size. 174 molecules with an aromatic scaffold. GStarX uses the 100-molecule subset and SAME the 25-molecule slice; seeds 5–9 of both were new runs.

| Method | Share | Null | Difference [95% CI] | Seeds better/worse/n.s. |
|---|---|---|---|---|
| GNNShap | 0.848 | 0.732 | +0.115 [+0.092, +0.140] | 10/0/0 |
| SAME | 0.801 | 0.730 | +0.071 [+0.029, +0.112] | 2/0/8 |
| Saliency | 0.735 | 0.732 | +0.003 [−0.030, +0.038] | 4/0/6 |
| EiG | 0.691 | 0.732 | −0.042 [−0.080, −0.001] | 2/2/6 |
| IG | 0.660 | 0.732 | −0.073 [−0.157, +0.006] | 3/4/3 |
| GStarX | 0.497 | 0.735 | −0.238 [−0.300, −0.172] | 0/9/1 |
| EiG own size (not in Holm) | 0.751 | 0.742 | +0.009 [−0.022, +0.038] | – |

- GNNShap and SAME prefer ring atoms. EiG does not: its top-k picks slightly fewer ring atoms than random connected sets, and its own-size set picks about the same share.
- IG matches Tier 4 L4 (0.660; difference −0.073).
- Test-only rows are in `T1.txt`. GNNShap is still above 0 there; SAME and GStarX have only about 4 and 15 test molecules per seed.

## 3. T2: when the rings are not available (MUTAG-GT D-MPNN, 10 seeds)

**P-T2** (registered): EiG and GNNShap move above random once the rings are unavailable. All 6 cells must have a CI above 0.

| Method | Cell | Before: N1, all 188, full graph | Before: same molecules, full graph | After | Above? |
|---|---|---|---|---|---|
| EiG | (a) no aromatic atom, 14 molecules | −0.050 [−0.064, −0.034] | +0.070 [−0.015, +0.162] | +0.070 [−0.015, +0.162] | no |
| GNNShap | (a) | −0.106 [−0.117, −0.096] | −0.107 [−0.140, −0.066] | −0.107 [−0.140, −0.066] | no |
| EiG | (b)-R1, scaffold deleted | −0.050 [−0.064, −0.034] | −0.060 [−0.077, −0.043] | −0.050 [−0.071, −0.026] | no |
| GNNShap | (b)-R1 | −0.106 [−0.117, −0.096] | −0.106 [−0.116, −0.096] | −0.013 [−0.027, +0.004] | no |
| EiG | (b)-R2, scaffold features 0 | −0.050 [−0.064, −0.034] | −0.060 [−0.077, −0.043] | −0.044 [−0.056, −0.029] | no |
| GNNShap | (b)-R2 | −0.106 [−0.117, −0.096] | −0.106 [−0.116, −0.096] | −0.030 [−0.052, −0.007] | no |

**P-T2 FAILED (0 of 6).**

- The recomputed "before" on all 188 molecules matches N1: EiG −0.050 [−0.064, −0.035], GNNShap −0.106 [−0.116, −0.096].
- In (a), the 14 molecules are full graphs, so the two right-hand columns hold the same number. EiG's point estimate is positive, but its CI includes 0 (2/0/8 seeds).
- In (b), GNNShap moves from clearly below random to near random, but not above it. EiG stays below random under both rules.
- With the scaffold hidden, GNNExplainer is at random and SX fixed is below random under both rules. IG is at random. Full tables, including the secondary k, are in `T2.txt`.
- **The test in (b) is weak.** On average 5.9 atoms remain, and k = 1 for 78% of molecules.
- **On 82 of the 174 molecules, the only atoms left after hiding the rings are the GT atoms** (found in T6). On those molecules every chosen set is all GT, so every method equals random.
- Post hoc, on the other 92 molecules (not part of the prediction):

  | Rule | EiG | GNNShap |
  |---|---|---|
  | R1 | −0.094 [−0.137, −0.049] | −0.025 [−0.052, +0.011] |
  | R2 | −0.082 [−0.105, −0.055] | −0.057 [−0.098, −0.010] |

  So the failure is not caused by those 82 molecules diluting the effect.

**Predicted class after hiding the scaffold.** The 5 methods' stored predictions agree.

| Rule | Molecules whose class changed (mean over seeds) | Per seed (of 174) | Direction (summed over seeds) |
|---|---|---|---|
| R1 | 62.0% | 54 to 139 | 1→0: 1,012; 0→1: 67 |
| R2 | 33.3% | 16 to 125 | 1→0: 463; 0→1: 117 |

The target in (b) is the hidden graph's predicted class. Hiding the rings mostly turns "mutagenic" predictions into "not mutagenic", so for many molecules (b) explains a different class than the full graph.

Other notes:
- No method raised an error.
- R2 SX fixed scores were all tied on the remaining atoms for 23% of molecules: its coalition lay inside the hidden scaffold.
- R2 SX fixed took 84 s per molecule.

## 4. T5: what the chosen atoms are (addendum; MUTAG-GT D-MPNN, 10 seeds, all 188 molecules, full graph)

Each method's top-k atoms (mean k = 4.1), split into three parts. No GT atom is aromatic, so the three parts add up to 1. Random-node = the expected share for a random k-set.

| Method | (i) GT | (ii) aromatic, non-GT | (iii) other non-GT | of (iii): C / N / O / F,I,Cl,Br |
|---|---|---|---|---|
| EiG | 0.150 | 0.639 | 0.211 | 0.58 / 0.10 / 0.09 / **0.23** |
| GNNShap | 0.068 | 0.785 | 0.148 | 0.69 / 0.15 / 0.06 / 0.10 |
| GNNExplainer | 0.484 | 0.395 | 0.121 | 0.58 / 0.10 / 0.21 / 0.11 |
| SX fixed | 0.111 | 0.730 | 0.159 | 0.66 / 0.20 / 0.06 / 0.08 |
| IG | 0.218 | 0.611 | 0.172 | 0.65 / 0.15 / 0.09 / 0.12 |
| Saliency | 0.049 | 0.680 | 0.271 | 0.51 / 0.15 / 0.12 / 0.22 |
| random-node | 0.252 | 0.620 | 0.129 | 0.68 / 0.10 / 0.11 / 0.10 |

Method minus random-node [95% CI] (better/worse/n.s.):

| Method | (i) GT | (ii) aromatic | (iii) other | Halogens (F/I/Cl/Br) |
|---|---|---|---|---|
| EiG | −0.102 [−0.121, −0.079] 0/10/0 | +0.020 [−0.017, +0.058] 2/1/7 | **+0.082 [+0.051, +0.113] 7/0/3** | **+0.036 [+0.028, +0.043] 8/0/2** |
| GNNShap | −0.184 [−0.202, −0.166] 0/10/0 | **+0.165 [+0.142, +0.188] 10/0/0** | +0.019 [+0.001, +0.039] 1/1/8 | +0.001 [−0.004, +0.006] 0/1/9 |
| GNNExplainer | +0.233 [+0.114, +0.332] 8/2/0 | −0.225 [−0.330, −0.100] 2/8/0 | −0.008 [−0.020, +0.006] 1/1/8 | −0.000 [−0.004, +0.004] 0/0/10 |
| SX fixed | −0.140 [−0.171, −0.109] 0/10/0 | +0.110 [+0.075, +0.144] 9/0/1 | +0.030 [+0.015, +0.044] 3/0/7 | −0.001 [−0.008, +0.007] 1/4/5 |
| IG | −0.034 [−0.108, +0.038] 4/6/0 | −0.009 [−0.083, +0.064] 4/3/3 | +0.043 [+0.020, +0.064] 6/0/4 | +0.006 [−0.004, +0.019] 3/2/5 |
| Saliency | −0.202 [−0.214, −0.188] 0/10/0 | +0.060 [+0.030, +0.094] 6/0/4 | +0.142 [+0.113, +0.169] 10/0/0 | +0.047 [+0.039, +0.054] 10/0/0 |

**What EiG picks instead of the ground truth.** EiG's GT share is 0.10 below random, and its share of aromatic atoms is at random. The missing GT share goes to other non-GT atoms (+0.082): mostly carbon (+0.034) and halogens. Halogens make up 23% of EiG's "other" atoms, against 10% for random. Atoms bonded directly to a GT atom are not over-picked.

GNNShap and SX fixed lose their GT share to ring atoms instead. Saliency behaves like EiG but more strongly.

T1 and T5 use different nulls (random-connected vs random-node) and different molecules (174 vs 188). That is why EiG's aromatic share reads −0.042 in T1 and +0.020 in T5. Both say EiG is not ring-biased.

## 5. T6 (optional): does the model need the nitro/amino group once the rings are hidden? (addendum; forward passes only)

Group = all GT atoms of the molecule. The group (or a random set of the same size from the remaining atoms) is hidden in the base graph. Logit margin, against random-connected sets. Full tables for probability and random-node are in `T6.txt`.

| Base graph, rule | Fid+: group − null (needed?) | Fid−: null − group (enough alone?) |
|---|---|---|
| Full graph, R1 | −3.95 [−4.89, −3.09] 0/10/0 | +1.06 [+0.56, +1.62] 9/1/0 |
| Full graph, R2 | −0.64 [−1.03, −0.32] 0/8/2 | −0.27 [−0.80, +0.11] 1/5/4 |
| Scaffold hidden R1, R1 | −0.48 [−0.65, −0.30] 0/10/0 | −0.58 [−0.82, −0.33] 1/8/1 |
| Scaffold hidden R2, R2 | −0.15 [−0.33, −0.01] 0/4/6 | −0.16 [−0.39, +0.02] 1/5/4 |

**Answer: no.** Once the rings are hidden, removing the nitro/amino group changes the margin *less* than removing a random set of the same size. Keeping only the group preserves the prediction *worse* than random sets.

On the full graph the model also does not need the group (as in Tier 4 L1). Under R1, though, the group alone keeps the prediction better than random sets.

Caveats:
- The scaffold-hidden rows use 92 of the 174 molecules. In the other 82, the group is all that remains, so they are skipped.
- The target is the base graph's predicted class. Hiding the rings changed it for 62% (R1) and 33% (R2) of molecules (section 3).

## 6. T3: full-reset sanity check (as N2)

GEA(trained) − GEA(full reset), on the 100-molecule subset, seeds 0–4. Every parameter tensor changed in all 20 resets, and the reset models are at chance. **All 16 cells pass** (CI above 0):

| Cell | EiG | GNNShap | Saliency | IG |
|---|---|---|---|---|
| PAINS D-MPNN | +0.339 (5/0/0) | +0.272 (5/0/0) | +0.054 (2/0/3) | +0.186 (5/0/0) |
| indole GCN | +0.343 (5/0/0) | +0.337 (5/0/0) | +0.213 (5/0/0) | +0.230 (4/0/1) |
| PAINS GCN | +0.110 (4/0/1) | +0.105 (4/0/1) | +0.102 (4/0/1) | +0.051 (2/0/3) |
| Mutagenicity GCN | +0.187 (3/0/2) | +0.173 (4/0/1) | +0.393 (5/0/0) | +0.166 (4/0/1) |

The smallest margins are PAINS D-MPNN Saliency (CI from +0.010) and PAINS GCN IG (CI from +0.008).

Spearman correlations between trained and reset atom scores are −0.10 to 0.27, and top-k Jaccard is 0.14 to 0.33. CIs and accuracies are in `T3.txt`.

## 7. T4: seconds per molecule, old methods

Seed 0 of the 12 summary-table cells plus MUTAG-GT s0: 100 molecules per unit, GPU 1, 8 concurrent jobs, one untimed warm-up.

| Method | Seconds per molecule (min / median / max over 13 units) |
|---|---|
| GNNExplainer | 2.44 / 2.89 / 4.34 |
| PGExplainer training, total | 85 / 144 / 883 s (131 to 1,215 training molecules) |
| PGExplainer training, per training molecule | 0.52 / 0.59 / 0.95 |
| PGExplainer inference | 0.007 / 0.015 / 0.022 |
| Saliency | 0.008 / 0.010 / 0.014 |
| IG (50 steps) | 0.018 / 0.020 / 0.024 |
| Input x Gradient | 0.008 / 0.010 / 0.013 |

**Load check.** GNNShap re-timed in the same run is 0.90–1.03× its N1 time (median 0.97), so the load matches Tier 5. EiG's ratio is 0.44–1.16 (median 0.68), but its times are only 0.05–0.19 s per molecule, where small fixed costs dominate.

GCN units are the slowest for GNNExplainer and PG training. The per-unit table is in `T4.txt`.

## 8. Deviations and limits

- **Reset checkpoints:** the 20 T3 reset checkpoints (20 MB) stay on ada (`runs/rq1_tier6/T3/ckpt/`). Earlier tiers never committed model checkpoints either.
- **Post-hoc row:** the T2(b) row on 92 molecules was added after seeing T6's skip count. It is labelled post hoc and does not re-judge P-T2.
- **Small test for (b):** T2(b) has little room to show a difference (k = 1 for 78% of molecules, and 82 of 174 molecules have only GT atoms left).
- **Reading of the request:** "GNN" in T2 was read as GNNExplainer (stated in the PLAN).
- **Empty stage log:** `stages.log` is empty. Stage times come from the STARTED / FINISHED / ITEM_DONE files.
- No paper .tex file was edited, and no earlier result was overwritten.
