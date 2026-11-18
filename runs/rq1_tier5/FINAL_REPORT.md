# RQ1 Tier 5: final report

**Tier 5 complete.** Items N0–N4 ran as registered in `PLAN.md`, and the six requested follow-ups (F1–F6, `PLAN_followups.md`) are done. Section 8 lists what did not run or ran smaller than planned.

All numbers are method minus a null (or minus another method), with a hierarchical bootstrap 95% CI (seeds, then molecules). Counts are per-seed better / worse / n.s. after Holm. "Above" means the whole CI is above 0. GEA is at top-k = 25% under the final protocol: random ties with 20 orders, and nulls of 20 draws.

Source files: `N0.txt`, `N4.txt`, `N1.txt`, `N2.txt`, `N3.txt`, `F.txt`, `summary_table.md/.csv`, `fig1_v3.pdf/.png`.

## 1. Commits

| Commit | What |
|---|---|
| 2e3a295 | PLAN, before any number |
| 2fdc52f, 1d3fbd0, bde3afb, 3f7792a | N0 integration code, colorlog shim, GNNShap wrapper fix, queue scripts |
| ff3472c, 7b5e0a4, 1f89d5d | N4 code; N1–N3 code; report code (all before running) |
| c8537f6 | GStarX PyG 2.8 device fix (wrapper only), before re-timing |
| bad9436 | N0 done; scale rule written to `scale.json` (PLAN addendum, before any GEA number) |
| 333a867 (+ 2260124 message correction) | N4 done. P-N5 FAILED |
| bbbb9e1 | Report: "not decided" when a method is missing |
| daebfad | N1 done. P-N1 HELD, P-N2 FAILED, P-N3 FAILED |
| cdb9481 | N2 done. P-N4 FAILED |
| 5e46def | N3 done. Registered EiG identity check FAILED (not bitwise) |
| c4ab175, 845a483, fa1784b, cdecce9 | Follow-ups: PLAN addendum and code, fig code, report code, job-list fix |
| aeb9177 (+ 7d6faa5 message correction) | Follow-up results F1, F3–F6 |

## 2. Integration status

Code sources: the authors' repositories at fixed commits. File hashes are in `third_party_manifest.json`; every change around the authors' code is in `patches.md`. Target = the predicted class for every explainer. No setting was tuned on ground truth.

| Explainer | Code (commit) | How atom scores are obtained | Own size | Level (scale rule) | Status |
|---|---|---|---|---|---|
| EiG-Search | authors' graph-classification branch, ported (b1e975d) | drop in log p(class) when a bond is removed; atom = sum over its bonds | atoms of the bonds chosen by the Fid+ − Fid− search (2 to floor(0.3·bonds) bonds) | L0: all GT molecules, all seeds | ran on all 85 units; 0 molecules skipped |
| SAME | authors' MCTS + mc_l_shapley + find_explanations (f0b4170) | 0/1 on the chosen atoms; a separate run at k = floor(0.25 N) | ceil(0.5 N) atoms (config sparsity 0.5) | L4: 25 molecules × seeds 0–4. **Over the 24 h budget even at L4** (31.4 h projected) | ran on all 65 units of the level |
| GStarX | authors' `GStarX`, molecular config (770b286) | the per-atom score GStarX returns | none | L2: 100 molecules × seeds 0–4 (22.9 h projected) | ran; needed the PyG 2.8 device fix |
| GNNShap | authors' sampler (CUDA) + WLS solver (f967229); our graph-level wrapper | Shapley value per bond; atom = sum over its bonds | none | L0 | ran on all 85 units; 0 molecules skipped; built well within the half-day cap |

N0 (`N0.txt`):

- All adapter checks pass on 6 models.
- All four explainers run on all 13 timing units, and all outputs are finite.
- On a few saturated molecules, removing atoms or bonds does not change the model output. This holds for at most 2 of 6 timing molecules per unit (4 of 12 SAME runs), mostly on Mutagenicity GCN, and is listed in `N0.json`.

## 3. Predictions (registered before any number; not changed)

| Prediction | Outcome | Key numbers |
|---|---|---|
| **P-N1:** one new explainer beats random-node AND high degree on both indole and PAINS (D-MPNN) | **HELD** | EiG, SAME and GNNShap do. EiG − high degree: indole +0.487 [+0.479, +0.496], PAINS +0.335 [+0.320, +0.348]. GStarX does not (−0.008 and −0.028) |
| **P-N2:** on Mutagenicity D-MPNN no new explainer beats hetero | **FAILED** | EiG +0.050 [+0.024, +0.075] and GNNShap +0.072 [+0.044, +0.099] are above. SAME includes 0, GStarX is below |
| **P-N3:** no explainer above random-node in all 12 cells | **FAILED** | Above in all 12: SX fixed, IxG (old methods) and EiG, SAME, GNNShap (new). GStarX misses PAINS GCN |
| **P-N4:** every new explainer passes the full reset on both Mutagenicity D-MPNN and indole | **FAILED** | EiG and GNNShap pass both. SAME on indole: −0.034 [−0.158, +0.098]. GStarX on indole: −0.014 [−0.025, −0.005] |
| **P-N5:** rank-average of GNN, SX and IG is not below the best single method in any of the 12 cells | **FAILED** | Below in all 12 cells (N4) |
| N3 registered check: EiG's two runs must be identical | **FAILED** (not bitwise) | Identical scores on 60 / 9 / 40 of 100 molecules; max difference 7.2e-7 / 1.0e-5 / 1.2e-6 (GPU floating-point noise; see 6.1) |

No N3 prediction was registered beyond this check. The follow-ups add no predictions.

## 4. Tables per item

### N4: combinations from cached scores (`N4.txt`)

- rank-avg(GNN, SX, IG) is below the best single old method in all 12 cells. Examples:
  - indole D-MPNN, vs SX fixed: −0.079 [−0.117, −0.041];
  - Mutagenicity D-MPNN, vs Saliency: −0.036 [−0.069, −0.005].
- IG smoothed over bonded neighbours is level with or above the best single method in 10 of 12 cells. It is below in PAINS GCN (−0.069 vs SX fixed) and Mutagenicity GCN (−0.058 vs PG logit).

### N1: GEA, new explainers next to SX fixed and IG (D-MPNN; minus random-node | minus the best rule)

| Cell | EiG | GNNShap | SAME | GStarX | SX fixed | IG |
|---|---|---|---|---|---|---|
| indole | +0.554 \| +0.487 | +0.534 \| +0.467 | +0.334 \| +0.272 | +0.056 \| −0.008 (0) | +0.491 \| +0.424 | +0.434 \| +0.367 |
| PAINS | +0.407 \| +0.335 | +0.328 \| +0.255 | +0.282 \| +0.220 | +0.040 \| −0.028 | +0.303 \| +0.230 | +0.207 \| +0.134 |
| Mutagenicity | +0.278 \| +0.050 | +0.299 \| +0.072 | +0.199 \| −0.039 (0) | +0.125 \| −0.112 | +0.141 \| −0.087 | +0.334 \| +0.107 |
| MUTAG-GT | −0.050 \| −0.553 | −0.106 \| −0.609 | −0.097 \| −0.582 | +0.109 \| −0.385 | −0.077 \| −0.580 | −0.023 (0) \| −0.526 |

- Best rule: high degree for indole and PAINS, hetero for Mutagenicity and MUTAG-GT.
- (0) means the CI includes 0; every other entry has its CI on one side of 0.
- Old methods here are on the union of the new methods' molecules (N1). All 13 cells, GIN, GINE and GCN included, are in `N1.txt`.

### N2: sanity check, GEA trained minus reset (`N2.txt`)

| Explainer | Mutag. D-MPNN full | Mutag. D-MPNN head | indole full | indole head | indole Spearman, trained vs full reset |
|---|---|---|---|---|---|
| EiG | +0.365 | +0.222 | +0.285 | +0.453 | +0.47 |
| GNNShap | +0.381 | +0.253 | +0.271 | +0.364 | +0.47 |
| SAME | +0.284 | +0.197 | −0.034 (0) | +0.195 | +0.40 |
| GStarX | +0.216 | +0.186 | **−0.014 (below)** | −0.003 (0) | **+0.94** |

On indole, GStarX's atom scores barely change when the model is reset (top-k Jaccard 0.83). Its output there is mostly independent of the model.

### N3: run to run, seed-0 models (`N3.txt`; Tier 4 L6 format in `F.txt` F1a)

| Explainer | Mutagenicity D-MPNN | indole | MUTAG-GT |
|---|---|---|---|
| EiG (2 runs) | Jaccard 0.960 | 0.987 | 0.973 |
| GNNShap (3 runs) | 0.982 | 1.000 | 0.997 |
| GStarX (3 runs) | 0.556 | 0.356 | 0.556 |
| SAME (3 runs, 25 molecules) | 0.383 | 0.649 | 0.377 |

- A consensus of 3 runs helps SAME on Mutagenicity: +0.136 [+0.082, +0.203]. Elsewhere the effect is small or below 0.
- Single run, final protocol, minus random-node: above 0 for all four on Mutagenicity and indole. On MUTAG-GT, SAME (−0.103), GNNShap (−0.103) and EiG (−0.050) are below.

## 5. Summary table: method × benchmark

The full table, with 12 cells × 5 fields, is in `summary_table.md` and `summary_table.csv`. The fields are: above random, above best rule, sanity, stable across seeds, and seconds per molecule. The condensed D-MPNN view:

| Method | indole | PAINS | Mutagenicity | MUTAG-GT (N1) |
|---|---|---|---|---|
| EiG | random ✓ rule ✓ sanity ✓ stable ✓ | ✓ ✓ n.t. ✓ | ✓ ✓ ✓ ✓ | below random |
| GNNShap | ✓ ✓ ✓ ✓ | ✓ ✓ n.t. ✓ | ✓ ✓ ✓ ✓ | below random |
| SAME | ✓ ✓ 0 ✓ | ✓ ✓ n.t. ✓ | ✓ 0 ✓ ✓ | below random |
| GStarX | ✓ 0 ✗ ✓ | ✓ ✗ n.t. ✗ | ✓ ✗ ✓ ✗ | above random, below rule |
| SX fixed | ✓ ✓ ✓ ✓ | ✓ ✓ n.t. ✓ | ✓ ✗ ✓ ✗ | below random |
| IG | ✓ ✓ ✓ ✓ | ✓ ✓ n.t. ✓ | ✓ ✓ ✓ ✓ | includes 0 |
| GNNExplainer | 0 0 0 ✗ | ✓ 0 n.t. ✓ | ✓ 0 ✓ ✓ | above random, below rule |

- ✓ means above and ✗ means below; 0 means the CI includes 0; n.t. means not tested.
- "Stable" means the same Holm verdict in every seed.

Seconds per molecule (new methods, measured with 8–12 concurrent jobs on one shared 32 GB V100):

| Method | Seconds per molecule |
|---|---|
| EiG | 0.1–0.2 |
| GNNShap | 1.1–1.3 |
| GStarX | 130–175 |
| SAME (both budgets) | 530–745 |

Old methods' timings were not recorded.

## 6. Follow-ups (`PLAN_followups.md`, exploratory; details in `F.txt`)

### 6.1 F1 stability

There is no Tier 4 GNNExplainer or cross-training-seed stability report to copy. Tier 4 L6 covered SubgraphX only, and only run to run. So F1 reports three blocks.

- **(a) Run to run:** N3, in the L6 format (see N3 above).
- **(b) Determinism.** Same model, same seed (tag T5), the first 20 molecules of each of the three N3 units, compared with N1:

  | Explainer | Bitwise identical | Max \|difference\| | Top-k sets vs N1 |
  |---|---|---|---|
  | SAME | 20/20 in all three units | 0 | identical |
  | GStarX | 13, 17, 20 of 20 | 3e-8 | identical |
  | EiG | 10, 2, 11 of 20 | 4.3e-6 | identical except one molecule (Jaccard 0.992 on Mutagenicity) |
  | GNNShap | 0, 0, 1 of 20 | 1.9e-8 | identical |

  The differences are GPU floating-point noise and almost never change the explanation.
- **(c) Across training seeds.** Top-k Jaccard between two models' explanations of the same molecule. Only molecules in both seeds' test sets count. The random baseline is about 0.14.

  | Method | Range over cells |
  |---|---|
  | EiG | 0.56–0.89 |
  | GNNShap | 0.50–0.89 |
  | Saliency | 0.42–0.79 |
  | IG | 0.40–0.69 |
  | SX fixed | 0.33–0.72 |
  | GNNExplainer | 0.18–0.47 |
  | GStarX (Mutagenicity and MUTAG-GT only) | 0.33–0.49 |
  | SAME | not computable (fewer than 20 shared items), except MUTAG-GT 0.35 (37 items) |

### 6.2 F3 fig1_v3

`fig1_v3.pdf/.png` uses fig1_v2's layout and data, plus EiG and GNNShap, with the random-node band and the best-rule band marked.

- EiG and GNNShap use the same molecules as fig1_v2; the per-seed arrays were checked equal.
- The PAINS panel keeps fig1_v2's 5 seeds.
- On MUTAG-GT both new methods sit below the random-node band.
- On indole and PAINS they are the two highest boxes, well above the best rule.
- On Mutagenicity they are close to IG and above the best rule (hetero).

### 6.3 F5 EiG at its own size vs at top-25%

| Cell | Top-25%, minus random-node | Own size, minus random-node | Own minus size-matched null | Own size / atoms |
|---|---|---|---|---|
| indole D-MPNN | +0.554 | +0.601 | +0.576 | 0.31 |
| indole GIN / GINE / GCN | +0.528 / +0.501 / +0.355 | +0.502 / +0.524 / +0.340 | +0.497 / +0.488 / +0.303 | 0.27 / 0.34 / 0.34 |
| PAINS D-MPNN | +0.407 | +0.466 | +0.427 | 0.33 |
| PAINS GIN / GINE / GCN | +0.329 / +0.343 / +0.117 | +0.383 / +0.404 / +0.186 | +0.352 / +0.360 / +0.137 | 0.32 / 0.34 / 0.37 |
| Mutagenicity D-MPNN | +0.278 | +0.231 | +0.221 | 0.32 |
| Mutagenicity GIN / GINE / GCN | +0.274 / +0.256 / +0.287 | +0.334 / +0.331 / +0.298 | +0.334 / +0.330 / +0.294 | 0.26 / 0.27 / 0.29 |
| MUTAG-GT | −0.050 | −0.061 | −0.076 | 0.29 |

- All CIs are on one side of 0; they are in `F.txt`.
- EiG's own size is 26–37% of atoms, a little above our 25% budget. It picks the maximum allowed bonds, floor(0.3 × bonds), for 33–74% of molecules.
- The size-matched null removes the size advantage. EiG stays clearly above it in every B-XAIC and Mutagenicity cell.

### 6.4 F6 EiG fidelity, not independent

EiG's search maximises Fid+ − Fid−, so its Fid numbers are not an independent check. They are in `F.txt` F6 and `N1.txt`; for example, Fid+ R2 on indole D-MPNN is +0.420 [+0.382, +0.452]. GEA-only rankings are given in two ways:

1. minus random-node, on each method's own molecules;
2. mean GEA on a common set (the SAME L4 slice, which every method covers).

Results:

- The two GEA rankings agree on the top method in 11 of 13 cells:
  - EiG first in 7 of the 8 B-XAIC cells, and SX fixed first on PAINS GCN;
  - GNNShap or Saliency first on Mutagenicity GIN and GINE;
  - GNNExplainer first on MUTAG-GT.
- They disagree on Mutagenicity D-MPNN (Saliency vs IG) and Mutagenicity GCN (PG logit vs Saliency).
- The Fid+ ranking often disagrees. On indole D-MPNN it puts IG first and EiG fourth, even though Fid is EiG's own objective.

## 7. Contradictions and points to watch

1. **EiG identity check failed, but only by floating-point noise.** The PLAN expected identical output; GPU message passing is not bitwise deterministic. In N3, most of EiG's top-k disagreements (Jaccard 0.96–0.99) are on molecules with identical scores. They come from the random tie order, which is drawn per run tag, not from the noise. The final protocol averages over 20 tie orders, so GEA is barely affected: single-run GEA minus random-node differs from the final-protocol value by at most 0.001.
2. **MUTAG-GT reverses the ranking.** EiG, GNNShap, SAME, SX fixed, Saliency and IxG are below random-node on MUTAG-GT. Only GNNExplainer (+0.213), GStarX (+0.109) and PG (+0.065) are above. On the other benchmarks, EiG and GNNShap are at or near the top. This fits Tier 4 L4: the MUTAG-GT model's prediction is already sufficiently explained by the aromatic scaffold, not only by the ground-truth atoms.
3. **GStarX looks model-independent on indole.** Trained and reset scores correlate at +0.94. It is not above high degree in any B-XAIC cell: below in 6, includes 0 in 2. Its results against random-node there are small (+0.01 to +0.06).
4. **Fid and GEA disagree** on the ranking of methods (F6). This repeats the Tier 2–4 finding with the new methods.
5. **Cross-seed stability is not "per molecule across many seeds".** Each seed has its own test split, so pairs share only about 8 (B-XAIC), 26 (Mutagenicity) or 188 (MUTAG-GT) molecules. The Jaccards in 6.1(c) come from those shared molecules only.
6. **Commit messages corrected after the fact (results unaffected):**
   - 333a867 said 8 of 12 cells; the correct count is 10 of 12 (2260124).
   - aeb9177 gave SX fixed's range as 0.38–0.72; the correct range is 0.33–0.72 (7d6faa5).
7. **Incidents (no result changed):**
   - **N1 out of memory:** the first N1 explain pass had 47 CUDA out-of-memory failures with 12 workers. It was rerun with 8 workers; finished molecules were kept and partial shards resumed (logs in `N1/attempt1_oom/`).
   - **Cross-seed job list:** the first list read `|` as a shell pipe, so all 13 jobs failed and nothing was written (`F/attempt1_cross_pipe/`).
   - **third_party copy:** early on, a copy of third_party/ replaced ada's B-XAIC and GraphXAI checkouts with identical clean copies at the same commits; the loaders gave identical data.

## 8. What did not run, or ran smaller than planned

- **SAME at L4 only:** 25 molecules × seeds 0–4. Its projected time was over 24 h even at L4 (31.4 h), so its CIs are wide and its cross-seed Jaccard cannot be computed (no shared molecules on B-XAIC).
- **GStarX at L2:** 100 molecules × seeds 0–4.
- **Sanity check:**
  - Done only on the D-MPNN for Mutagenicity and indole; the other 10 cells are "not tested".
  - IxG has no reset check (it was not in Tier 2 R).
  - The old PG in Tier 2 R is the final-rule version, not logit.
- **Old methods' seconds per molecule:** not recorded.
- **Determinism reruns:** 20 molecules per method and unit, not every molecule.
- **fig1_v3:** keeps fig1_v2's 5 PAINS seeds; N1 has all 10.
- No paper .tex file was edited, and no existing result was overwritten.
