# RQ1 Tier 2: final report

**Tier 2 complete.** Items H, G, R, S and M all ran. 6 of 7 predictions held; P-R1 failed. One registered check failed (IG completeness, 1 of 50 units; the user decided to keep 50 steps).

All numbers are GEA minus random-node with a hierarchical bootstrap 95% CI, unless stated. Counts are per-seed better / worse / n.s. after Holm. Full tables, including all Fid+ / Fid- under R1, R2, R3: `H.txt`, `G.txt`, `R.txt`, `R_rescore.txt`, `S_prio.txt`, `S_rest.txt`, `S_sensitivity.txt`, `M.txt`.

## Predictions

| Prediction | Outcome | Result |
|---|---|---|
| P-H1: hetero above random-node on Mutagenicity (4 models) and MUTAG-GT | **HELD** | +0.230 [0.212, 0.249], +0.231 [0.212, 0.250], +0.231 [0.213, 0.249], +0.232 [0.213, 0.251]; MUTAG-GT +0.503 [0.494, 0.512] |
| P-H2: hetero mean GEA at least that of one explainer, Mutagenicity D-MPNN | **HELD** | hetero 0.352; GNN 0.378, PG 0.118, SX legacy 0.275 (SX fixed 0.281, same outcome) |
| P-G1: IG above random-node, Mutagenicity D-MPNN | **HELD** | +0.373 [+0.344, +0.401] |
| P-R1: every explainer within +/-0.05 of random on randomised models | **FAILED** | 9 of 20 cells within; 90% CI inside +/-0.05 in 2 of 20 |
| P-S1: SubgraphX directions unchanged on valid benchmarks, MUTAG-GT below random | **HELD** | 8 of 8 groups unchanged; MUTAG-GT new -0.077 [-0.101, -0.053] |
| P-S2: rollout 20 to 100 changes mean GEA by less than 0.03 | **HELD** | +0.027 Mutagenicity D-MPNN s0 (narrow), +0.003 MUTAG-GT s0 |
| P-M1: CI category with 10 seeds equals 5 seeds | **HELD** | 10 of 10 cells match (caveat under M) |

## H. Heuristic baselines

Difference from random-node:

| Group | hetero | deg-high | deg-low |
|---|---|---|---|
| Mutagenicity D-MPNN | +0.230 [+0.212, +0.249] | -0.091 [-0.102, -0.081] | +0.074 [+0.065, +0.084] |
| Mutagenicity GIN | +0.231 [+0.212, +0.250] | -0.091 [-0.100, -0.083] | +0.073 [+0.064, +0.082] |
| Mutagenicity GINE | +0.231 [+0.213, +0.249] | -0.092 [-0.102, -0.082] | +0.071 [+0.063, +0.078] |
| Mutagenicity GCN | +0.232 [+0.213, +0.251] | -0.092 [-0.102, -0.082] | +0.074 [+0.066, +0.081] |
| B-XAIC indole | -0.081 [-0.083, -0.079] | +0.067 [+0.063, +0.070] | -0.105 [-0.108, -0.103] |
| B-XAIC PAINS | +0.011 [+0.004, +0.016] | +0.074 [+0.070, +0.079] | -0.042 [-0.048, -0.037] |
| B-XAIC X | +0.149 [+0.144, +0.153] | -0.053 [-0.055, -0.052] | +0.190 [+0.185, +0.196] |
| B-XAIC P | +0.071 [+0.068, +0.073] | +0.148 [+0.143, +0.153] | -0.043 [-0.045, -0.042] |
| MUTAG-GT (10 seeds) | +0.503 [+0.494, +0.512] | -0.029 [-0.032, -0.027] | +0.308 [+0.304, +0.313] |

On MUTAG-GT, hetero (mean GEA 0.648) is above all three explainers (GNN 0.359, PG 0.210, SX 0.066). On every Mutagenicity model, the mean GEA of hetero (about 0.35) is above SX and, except on GCN PG (0.464), above PG; it beats GNN on GIN, GINE and GCN and is just below it on D-MPNN (0.378).

## G. Gradient methods

| Group | Saliency | IG | Input x Gradient |
|---|---|---|---|
| Mutagenicity D-MPNN | +0.357 [+0.331, +0.385] | +0.373 [+0.344, +0.401] | +0.348 [+0.307, +0.385] |
| Mutagenicity GIN | +0.262 [+0.232, +0.291] | +0.073 [-0.016, +0.153] | +0.195 [+0.163, +0.223] |
| Mutagenicity GINE | +0.308 [+0.239, +0.371] | +0.201 [+0.113, +0.287] | +0.285 [+0.238, +0.338] |
| Mutagenicity GCN | +0.336 [+0.304, +0.369] | +0.167 [+0.121, +0.211] | +0.205 [+0.163, +0.245] |
| B-XAIC indole | +0.406 [+0.340, +0.467] | +0.418 [+0.394, +0.445] | +0.185 [+0.168, +0.205] |
| B-XAIC PAINS | +0.163 [+0.147, +0.180] | +0.209 [+0.186, +0.235] | +0.170 [+0.152, +0.191] |
| B-XAIC X | +0.207 [+0.199, +0.214] | +0.207 [+0.199, +0.214] | +0.207 [+0.199, +0.214] |
| B-XAIC P | +0.148 [+0.144, +0.153] | +0.148 [+0.144, +0.153] | +0.148 [+0.144, +0.153] |
| MUTAG-GT (10 seeds) | -0.123 [-0.128, -0.116] | -0.023 [-0.070, +0.023] | -0.114 [-0.127, -0.096] |

**Check failed:** IG completeness median 0.073 > 0.05 on Mutagenicity GINE s4 (49 of 50 units pass). IG GEA does not depend on the step count (step table in `PLAN.md`). Kept at 50 steps by user decision. `G.txt` still prints "checks: FAILED".

## R. Randomised models

The full reset is the main check. The head reset keeps the trained encoder, so above-random scores there are expected.

| Dataset | Reset | GNN | PG | SX fixed | IG | Saliency |
|---|---|---|---|---|---|---|
| Mutagenicity D-MPNN | full | +0.020 [+0.003, +0.038] | +0.036 [+0.014, +0.056] | -0.037 [-0.082, +0.009] | +0.026 [-0.055, +0.116] | **-0.100** [-0.113, -0.084] |
| Mutagenicity D-MPNN | head | -0.011 [-0.035, +0.017] | **+0.076** [+0.026, +0.145] | **+0.070** [-0.013, +0.156] | **+0.218** [+0.128, +0.310] | **+0.217** [+0.156, +0.285] |
| B-XAIC indole | full | +0.018 [-0.025, +0.075] | **-0.132** [-0.141, -0.123] | -0.016 [-0.065, +0.035] | +0.005 [-0.085, +0.134] | **+0.126** [+0.082, +0.179] |
| B-XAIC indole | head | +0.039 [-0.012, +0.087] | **-0.136** [-0.154, -0.104] | **+0.241** [+0.095, +0.372] | **+0.136** [+0.070, +0.230] | **+0.305** [+0.167, +0.424] |

Bold = outside +/-0.05. Full reset only (descriptive): 7 of 10 cells within; 90% CI inside the margin in 1 of 10.

GEA(trained) - GEA(full reset):

| Dataset | GNN | PG | SX fixed | IG | Saliency |
|---|---|---|---|---|---|
| Mutagenicity D-MPNN | +0.235 [+0.160, +0.310] | -0.039 [-0.127, +0.052] | +0.189 [+0.143, +0.233] | +0.347 [+0.235, +0.447] | +0.457 [+0.422, +0.493] |
| B-XAIC indole | +0.020 [-0.101, +0.116] | +0.336 [+0.320, +0.354] | +0.479 [+0.392, +0.560] | +0.413 [+0.283, +0.493] | +0.280 [+0.210, +0.352] |

So GNNExplainer on indole and PGExplainer on Mutagenicity D-MPNN cannot be told apart from their full-reset versions.

**PGExplainer sign flip on indole.** Trained: 67% of edges have sigmoid > 0.99, and 37% of atoms tie at exactly 1.0 in float64. After a reset, 100% of edges have sigmoid < 0.01. The Spearman with trained stays negative under mean-logit ranking (full -0.754 [-0.775, -0.737]; sigmoid-ranked -0.741). So rounding ties do not cause it, but the flip does occur together with the change of saturation direction; this analysis does not show that the direction causes it. The reset explainer's ranking is reversed relative to the ground truth: -0.159 [-0.161, -0.157] as ranked, +0.402 [+0.390, +0.415] reversed.

**Mean-logit ranking on Mutagenicity D-MPNN.** Full reset +0.167 [+0.142, +0.194], against -0.002 [-0.067, +0.063] for the trained explainer re-run. PGExplainer's mean-logit ranking is above random on a model with no learned signal, which suggests it follows molecular structure on this dataset.

## S. SubgraphX budget fix

| Group | Old | New | Counts old / new | New: size = k / < k |
|---|---|---|---|---|
| Mutagenicity D-MPNN | +0.152 [+0.133, +0.171] | +0.158 [+0.141, +0.176] | 5/0/0, 5/0/0 | 0.60 / 0.40 |
| Mutagenicity GIN | +0.096 [+0.045, +0.143] | +0.099 [+0.044, +0.148] | 3/0/2, 3/0/2 | 0.48 / 0.52 |
| Mutagenicity GINE | +0.114 [+0.068, +0.157] | +0.126 [+0.063, +0.182] | **4/0/1, 3/0/2** | 0.47 / 0.53 |
| Mutagenicity GCN | +0.052 [+0.032, +0.072] | +0.059 [+0.039, +0.080] | 1/0/4, 1/0/4 | 0.45 / 0.55 |
| B-XAIC indole | +0.459 [+0.416, +0.499] | +0.468 [+0.426, +0.507] | 5/0/0, 5/0/0 | 0.83 / 0.17 |
| B-XAIC PAINS | +0.307 [+0.291, +0.322] | +0.303 [+0.289, +0.316] | 5/0/0, 5/0/0 | 0.72 / 0.28 |
| B-XAIC X | +0.108 [+0.092, +0.125] | +0.107 [+0.092, +0.123] | 5/0/0, 5/0/0 | 0.15 / 0.85 |
| B-XAIC P | +0.128 [+0.124, +0.132] | +0.134 [+0.129, +0.138] | 5/0/0, 5/0/0 | 0.34 / 0.66 |
| MUTAG-GT (10 seeds) | -0.079 [-0.103, -0.054] | -0.077 [-0.101, -0.053] | 0/10/0, 0/10/0 | 0.84 / 0.16 |

The new budget removes every explanation larger than k, but many are still smaller than k (16% to 85%, depending on the group), because SubgraphX returns the best subgraph with at most k atoms. So "matched budget" means "never above k", not "exactly k". Old vs new Fid: 0 of 108 cells changed CI category (per-seed counts changed in a few, listed in `S_rest.txt`).

Sensitivity (100 fixed molecules; GEA minus base; share of chosen subgraphs that differ from base):

| Setting | Mutagenicity D-MPNN s0 | MUTAG-GT s0 |
|---|---|---|
| rollout 50 | +0.013, 22% | +0.003, 6% |
| rollout 100 | +0.027, 26% | +0.003, 8% |
| sample 100 | +0.006, 62% | +0.001, 61% |
| same settings, new seed | -0.015, 68% | +0.003, 60% |

The noise floor is about the same size as the setting effects, except rollout 100 on Mutagenicity, which is about twice the noise.

## M. Ten seeds

| Group | Method | 5 seeds | 10 seeds | Counts (10) | Sign test (10) |
|---|---|---|---|---|---|
| Mutagenicity D-MPNN | GNN | +0.255 [+0.176, +0.331] | +0.255 [+0.206, +0.300] | 10/0/0 | 10/10, p=0.002 |
| | PG | -0.004 [-0.091, +0.087] | +0.014 [-0.061, +0.098] | 4/6/0 | 4/10, p=0.754 |
| | SX fixed | +0.158 [+0.141, +0.176] | +0.141 [+0.114, +0.163] | 9/0/1 | 10/10, p=0.002 |
| | IG | +0.373 [+0.344, +0.401] | +0.334 [+0.301, +0.366] | 10/0/0 | 10/10, p=0.002 |
| | hetero | +0.230 [+0.212, +0.249] | +0.227 [+0.214, +0.242] | 10/0/0 | 10/10, p=0.002 |
| B-XAIC indole | GNN | +0.037 [-0.031, +0.107] | +0.056 [-0.014, +0.128] | 6/4/0 | 6/10, p=0.754 |
| | PG | +0.204 [+0.185, +0.222] | +0.265 [+0.204, +0.331] | 10/0/0 | 10/10, p=0.002 |
| | SX fixed | +0.468 [+0.426, +0.507] | +0.491 [+0.462, +0.513] | 10/0/0 | 10/10, p=0.002 |
| | IG | +0.418 [+0.394, +0.445] | +0.434 [+0.410, +0.459] | 10/0/0 | 10/10, p=0.002 |
| | hetero | -0.081 [-0.083, -0.079] | -0.082 [-0.083, -0.080] | 0/10/0 | 0/10, p=0.002 |

**Caveat on the 10-seed PG numbers.** They mix two tie rules, as the PLAN specifies: seeds 0-4 use the stored mask, where almost all indole tie-boundary molecules fall back to random ties; seeds 5-9 rank by their own float64 scores. On indole, seeds 5-9 average +0.325, close to the +0.321 of the logit-ranked re-run of seeds 0-4. So the rise from +0.204 to +0.265 may come partly from the rule rather than from the new seeds. P-M1 compares CI categories and is not affected.

## Contradictions with current results

Rule (`PLAN.md`): only a change of CI category counts. Compared with the T3 table. The paper's `.tex` files are not on this machine, so I compared with results files, not with the paper's wording.

**No CI category changed** in any explainer cell of T3 (GNN, PG, SX vs random-node).

These do not change a category, but they bear on current claims and should be checked against the text:

| Current result | New number | Why it matters |
|---|---|---|
| T3 Mutagenicity GIN PG -0.056 [-0.086, -0.024] | -0.093 [-0.107, -0.075] (H) | Same category, but the largest shift from T3 and not "slight"; the cause was not investigated |
| T3 indole PG +0.204 [+0.185, +0.221] | +0.321 [+0.266, +0.384] logit-ranked re-run | The stored value rests mostly on random tie-breaks; same category |
| Any claim that explainers beat a trivial baseline | MUTAG-GT hetero +0.503 vs GNN +0.213, PG +0.065, SX -0.079; Mutagenicity D-MPNN hetero +0.230 vs SX +0.152, PG -0.004 | A rule with no model scores about as well as, or better than, the explainers |
| Any claim that GEA above random shows an explainer read the model | P-R1 failed: Saliency +0.126 on fully reset indole; PG mean-logit +0.167 on fully reset Mutagenicity | Above-random GEA can occur with no learned signal |
| Any claim that indole PG / GNN or Mutagenicity PG reflect the model | Trained - full reset: GNN indole +0.020 [-0.101, +0.116]; PG Mutagenicity -0.039 [-0.127, +0.052] | These do not differ from their reset versions |

The MUTAG-GT claim for SubgraphX is confirmed (still below random with the new budget, and P-S1 held).

## What did not run or is limited

- Nothing registered was skipped. All jobs finished with 0 failures in their final runs.
- SubgraphX on randomised models covers the fixed 100 molecules per seed, as planned, not all molecules.
- Fid was not computed for randomised models (R) or for the new seeds (M). The PLAN did not list it for these items, but the original rule says "for every new method", so this is a gap if R or M are read as new methods. H, G and S have full Fid tables.
- IG stayed at 50 steps on Mutagenicity GINE s4 (user decision).
- Checkpoints, explainer MLPs and run_phase3 caches for R and M stay on ada, not in git.

## Things that broke

- Queue script dropped the first word of each command: all 50 H jobs failed; fixed and re-run (19e2922).
- S remainder job list crashed on an empty list, and the chain did not stop on it, so 30 scoring jobs failed with no output. Fixed (75f639c), dry-run checked, then re-run with user approval. Record kept in `CHAIN_STOPPED_20261031`.
- A stale label in the R reports' section [2] still says SX "pending until S priority is done". In `R_rescore.txt` the column is complete; `R.txt` holds the first, partial scoring.

## Commits

| Commit | Content |
|---|---|
| 012a801, ba20e16, f26adfe, 19e2922, c7f2185 | Budget option, jobs, PLAN, queue fix, report code |
| c3398ec | H |
| 3c76623, 0e2f64c, 50722cd | IG diagnostic, G, PLAN addendum |
| 3b99b03, cc92dcc, d88955c | R, S, M code, chain |
| 384478f | R first scoring |
| 01bf886 | R re-score additions |
| 916588d | S priority, R re-score |
| 8b4e7bf | S sensitivity |
| 8f1972d | M |
| 75f639c, e0bba43 | Chain fix, stage script |
| aa2457b | S remainder |
