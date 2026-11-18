# RQ1 Tier 5 summary table

Each cell is: **above random / above best rule / sanity / stable / seconds per molecule**.

- **Above random / above best rule.** `+` means the CI is entirely above 0, `-` entirely below 0, `0` includes 0. Old methods are on their registered molecules; new methods are N1, each at its scale level. The best rule is high degree for indole and PAINS and hetero for Mutagenicity.
- **Sanity.** GEA trained minus full reset: `+` / `-` / `0`. Old methods use Tier 2 R_rescore, where PG is the final-rule version; new methods use N2. n.t. = not tested; only the D-MPNN on Mutagenicity and indole was tested.
- **Stable.** `yes` means every seed has the same Holm verdict against random-node; the numbers in brackets are better/worse/n.s. counts. J is the top-k Jaccard across training seeds on shared molecules, with the random baseline in brackets; n.c. = fewer than 20 shared items.
- **Seconds per molecule.** Measured with 8-12 concurrent jobs on one shared V100.
  - SAME's time is the sum of its two budget runs. The own-size versions come from the same run.
  - SX fixed: its own registered runs (Tier 2 S / M, Tier 4 M1 / M2), under a different load.
  - n.r. = not recorded: GNNExplainer, PG and the gradient methods. Their cache and gradient files have no time field (checked).
- **Rows.** Rows are explainers only. The N4 combinations and the trivial rules are not rows.
- **Stable: yes** also covers methods that are worse than random in every seed, e.g. GNN on indole GINE (0/5/0).

## indole

| method | D-MPNN | GIN | GINE | GCN |
|---|---|---|---|---|
| GNN | 0 / 0 / 0 / no (6/4/0), J 0.23 (0.14) / n.r. | - / - / n.t. / no (1/4/0), J 0.22 (0.14) / n.r. | - / - / n.t. / yes (0/5/0), J 0.29 (0.14) / n.r. | - / - / n.t. / yes (0/5/0), J 0.47 (0.14) / n.r. |
| PG (logit) | + / + / + / yes (10/0/0), J 0.44 (0.14) / n.r. | 0 / 0 / n.t. / no (1/4/0), J 0.28 (0.14) / n.r. | - / - / n.t. / yes (0/5/0), J 0.58 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.64 (0.14) / n.r. |
| SX fixed | + / + / + / yes (10/0/0), J 0.72 (0.14) / 258.3 s | + / + / n.t. / no (3/0/2) / 249.7 s | + / + / n.t. / no (4/0/1) / 239.4 s | + / + / n.t. / yes (5/0/0) / 259.8 s |
| Saliency | + / + / + / yes (10/0/0), J 0.56 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.50 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.45 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.58 (0.14) / n.r. |
| IG | + / + / + / yes (10/0/0), J 0.65 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.68 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.40 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.58 (0.14) / n.r. |
| IxG | + / + / n.t. / yes (10/0/0), J 0.37 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.36 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.30 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.55 (0.14) / n.r. |
| EiG | + / + / + / yes (10/0/0), J 0.89 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.88 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.66 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.81 (0.14) / 0.2 s |
| SAME | + / + / 0 / yes (5/0/0) / 744.2 s | + / + / n.t. / yes (5/0/0) / 727.6 s | + / + / n.t. / no (4/0/1) / 742.2 s | + / + / n.t. / no (3/0/2) / 720.2 s |
| GStarX | + / 0 / - / yes (5/0/0) / 174.8 s | + / - / n.t. / yes (5/0/0) / 171.9 s | + / - / n.t. / yes (5/0/0) / 172.6 s | + / - / n.t. / no (4/0/1) / 173.6 s |
| GNNShap | + / + / + / yes (10/0/0), J 0.88 (0.14) / 1.3 s | + / + / n.t. / yes (5/0/0), J 0.89 (0.14) / 1.3 s | + / + / n.t. / yes (5/0/0), J 0.66 (0.14) / 1.3 s | + / + / n.t. / yes (5/0/0), J 0.84 (0.14) / 1.3 s |
| EiG (own size) | + / + / + / yes (10/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.2 s |
| SAME (own size) | + / + / 0 / yes (5/0/0) / 744.2 s | + / + / n.t. / yes (5/0/0) / 727.6 s | + / + / n.t. / yes (5/0/0) / 742.2 s | + / + / n.t. / no (3/0/2) / 720.2 s |

## PAINS

| method | D-MPNN | GIN | GINE | GCN |
|---|---|---|---|---|
| GNN | + / 0 / n.t. / yes (10/0/0), J 0.24 (0.14) / n.r. | 0 / - / n.t. / no (1/3/1), J 0.29 (0.14) / n.r. | 0 / - / n.t. / no (0/2/3), J 0.25 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.33 (0.14) / n.r. |
| PG (logit) | + / - / n.t. / no (8/0/2), J 0.18 (0.14) / n.r. | 0 / - / n.t. / no (1/1/3), J 0.34 (0.14) / n.r. | + / 0 / n.t. / no (3/0/2), J 0.39 (0.14) / n.r. | + / - / n.t. / no (2/0/3), J 0.63 (0.14) / n.r. |
| SX fixed | + / + / n.t. / yes (10/0/0), J 0.55 (0.14) / 238.3 s | + / + / n.t. / no (4/0/1) / 182.9 s | + / + / n.t. / yes (5/0/0) / 212.3 s | + / + / n.t. / yes (5/0/0) / 198.5 s |
| Saliency | + / + / n.t. / yes (10/0/0), J 0.59 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.42 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.50 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.63 (0.14) / n.r. |
| IG | + / + / n.t. / yes (10/0/0), J 0.43 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.51 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.41 (0.14) / n.r. | + / 0 / n.t. / no (4/0/1), J 0.48 (0.14) / n.r. |
| IxG | + / + / n.t. / yes (10/0/0), J 0.44 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.36 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.36 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.58 (0.14) / n.r. |
| EiG | + / + / n.t. / yes (10/0/0), J 0.68 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.63 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.60 (0.14) / 0.1 s | + / + / n.t. / yes (5/0/0), J 0.60 (0.14) / 0.2 s |
| SAME | + / + / n.t. / yes (5/0/0) / 630.7 s | + / + / n.t. / no (3/0/2) / 546.6 s | + / + / n.t. / yes (5/0/0) / 608.9 s | + / + / n.t. / no (1/0/4) / 555.6 s |
| GStarX | + / - / n.t. / no (3/0/2) / 159.9 s | + / - / n.t. / no (1/0/4) / 159.1 s | + / 0 / n.t. / no (4/0/1) / 159.5 s | 0 / - / n.t. / no (1/0/4) / 160.4 s |
| GNNShap | + / + / n.t. / yes (10/0/0), J 0.57 (0.14) / 1.2 s | + / + / n.t. / yes (5/0/0), J 0.52 (0.14) / 1.3 s | + / + / n.t. / yes (5/0/0), J 0.51 (0.14) / 1.3 s | + / + / n.t. / yes (5/0/0), J 0.59 (0.14) / 1.3 s |
| EiG (own size) | + / + / n.t. / yes (10/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.2 s |
| SAME (own size) | + / + / n.t. / yes (5/0/0) / 630.7 s | + / + / n.t. / no (3/0/2) / 546.6 s | + / + / n.t. / yes (5/0/0) / 608.9 s | + / 0 / n.t. / no (1/0/4) / 555.6 s |

## Mutagenicity

| method | D-MPNN | GIN | GINE | GCN |
|---|---|---|---|---|
| GNN | + / 0 / + / yes (10/0/0), J 0.37 (0.14) / n.r. | 0 / - / n.t. / no (2/1/2), J 0.30 (0.14) / n.r. | + / - / n.t. / no (3/1/1), J 0.18 (0.14) / n.r. | + / - / n.t. / no (4/0/1), J 0.47 (0.14) / n.r. |
| PG (logit) | 0 / - / 0 / no (4/6/0), J 0.29 (0.14) / n.r. | - / - / n.t. / no (0/4/1), J 0.29 (0.14) / n.r. | - / - / n.t. / no (0/4/1), J 0.25 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.57 (0.14) / n.r. |
| SX fixed | + / - / + / no (9/0/1), J 0.38 (0.14) / 164.4 s | + / - / n.t. / no (3/0/2), J 0.33 (0.14) / 133.8 s | + / - / n.t. / no (3/0/2), J 0.48 (0.14) / 232.9 s | + / - / n.t. / no (1/0/4), J 0.47 (0.14) / 250.4 s |
| Saliency | + / + / + / yes (10/0/0), J 0.72 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.74 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.69 (0.14) / n.r. | + / + / n.t. / yes (5/0/0), J 0.79 (0.14) / n.r. |
| IG | + / + / + / yes (10/0/0), J 0.54 (0.14) / n.r. | 0 / - / n.t. / no (3/1/1), J 0.44 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.49 (0.14) / n.r. | + / - / n.t. / yes (5/0/0), J 0.69 (0.14) / n.r. |
| IxG | + / + / n.t. / yes (10/0/0), J 0.58 (0.14) / n.r. | + / - / n.t. / yes (5/0/0), J 0.54 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.56 (0.14) / n.r. | + / 0 / n.t. / yes (5/0/0), J 0.74 (0.14) / n.r. |
| EiG | + / + / + / yes (10/0/0), J 0.67 (0.14) / 0.1 s | + / 0 / n.t. / yes (5/0/0), J 0.57 (0.14) / 0.1 s | + / 0 / n.t. / yes (5/0/0), J 0.56 (0.14) / 0.1 s | + / 0 / n.t. / yes (5/0/0), J 0.65 (0.14) / 0.2 s |
| SAME | + / 0 / + / yes (5/0/0) / 607.4 s | + / 0 / n.t. / yes (5/0/0) / 647.1 s | + / 0 / n.t. / no (4/0/1) / 585.1 s | + / - / n.t. / yes (5/0/0) / 531.8 s |
| GStarX | + / - / + / no (3/0/2), J 0.47 (0.14) / 131.1 s | + / - / n.t. / no (4/0/1), J 0.49 (0.14) / 134.1 s | + / 0 / n.t. / yes (5/0/0), J 0.33 (0.14) / 133.1 s | 0 / - / n.t. / yes (0/0/5), J 0.49 (0.14) / 134.5 s |
| GNNShap | + / + / + / yes (10/0/0), J 0.70 (0.14) / 1.1 s | + / + / n.t. / yes (5/0/0), J 0.68 (0.14) / 1.2 s | + / 0 / n.t. / yes (5/0/0), J 0.50 (0.14) / 1.2 s | + / + / n.t. / yes (5/0/0), J 0.72 (0.14) / 1.2 s |
| EiG (own size) | + / 0 / + / yes (10/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / + / n.t. / yes (5/0/0) / 0.1 s | + / 0 / n.t. / yes (5/0/0) / 0.2 s |
| SAME (own size) | + / + / + / yes (5/0/0) / 607.4 s | + / + / n.t. / yes (5/0/0) / 647.1 s | + / 0 / n.t. / no (4/0/1) / 585.1 s | + / 0 / n.t. / no (4/0/1) / 531.8 s |
