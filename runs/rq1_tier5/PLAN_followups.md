# RQ1 Tier 5 follow-ups (F1-F6; numbered as in the request: F2 = FINAL_REPORT.md + DONE): definitions, fixed before any follow-up number is computed

These were requested after N1 and N2 were finished. They are **exploratory**. No new predictions are added. The registered N3 check (EiG's two runs must be identical) is reported as passed or failed.

Output: runs/rq1_tier5/F/, F.txt / F.json, summary_table.csv / summary_table.md, fig1_v3.pdf / .png, FINAL_REPORT.md, DONE.

## F1. Stability

Tier 4 L6 covered SubgraphX only, and only run-to-run (3 explainer seeds on the seed-0 model). Tier 4 has no GNNExplainer stability report and no cross-training-seed report. So F1 reports three separate things:

- **(a) Run to run (registered N3), in the L6 format.**
  - Units: Mutagenicity D-MPNN s0, indole s0, MUTAG-GT s0, on the method level's molecules.
  - Runs: T5a/T5b/T5c (EiG: T5a/T5b).
  - Reported per method:
    - Jaccard of the top-k sets between runs (mean, median), and the share of molecules with identical sets in all pairs;
    - Jaccard against the N1 run (tag T5);
    - atoms picked in >= 2 of 3 runs, as a share of atoms picked at least once;
    - consensus set size against k;
    - GEA of single-run and consensus sets scored as sets, and consensus minus single;
    - minus random-node for single and consensus sets, and the single run under the final protocol.
- **(b) Determinism: same model, same seed (tag T5), twice.**
  - Rerun: the first 20 molecules of the method's level list on the same three seed-0 units, written to F/det/.
  - SAME is rerun at budget k only. Its seed is set before each budget, so this reproduces the N1 "node" scores.
  - Reported: the number of molecules with bitwise-identical scores, max |score difference|, and top-k Jaccard against N1.
  - EiG: N3's T5a and T5b give the same check.
- **(c) Across training seeds.**
  - For every cell (dataset x architecture) and every pair of seeds, use the molecules explained in both seeds and compute the Jaccard of the top-k sets of the two models. Random tie order per molecule follows the stream rule of N2/N3.
  - Methods: the six old methods (on their registered molecules) and the four new ones (on their level's molecules).
  - Reported: the mean over all (seed pair, molecule) items, the number of seed pairs and items, and a random baseline. The baseline is the exact expected Jaccard of two independent random k-sets on the same molecules (the overlap is hypergeometric).
  - Seeds' test splits differ, so shared molecules are few (B-XAIC about 8 per pair; Mutagenicity about 26; MUTAG-GT all 188). With fewer than 20 items in a cell, it is "not computable". This is the case for SAME (L4), and for GStarX (L2) except on MUTAG-GT.
  - Descriptive only: no CI, because pairs share seeds.

## F4. Summary table (rows = methods; columns = benchmark x model x 5 fields; CSV + markdown)

Cells: indole, PAINS and Mutagenicity, each x D-MPNN, GIN, GINE, GCN.

- **Above random:** CI(method - random-node) entirely above 0, using the P-N3 sources: old methods on their registered molecules (N4 all__ arrays), new methods from N1. Values: above / below / incl. 0. Code asserts it matches the P-N3 table.
- **Above best trivial rule:** CI(method - rule) entirely above 0. The rule is fixed per dataset in the PLAN (high degree for indole and PAINS, hetero for Mutagenicity). Same molecules as "above random"; for old methods the rule GEA is computed on their registered molecules.
- **Sanity:** CI(GEA trained - GEA full reset) entirely above 0.
  - New methods: N2.
  - Old methods: Tier 2 R_rescore (trained - full reset; PG there is the final-rule version, not logit).
  - Only Mutagenicity D-MPNN and indole D-MPNN were tested; other cells say "not tested". IxG was not in Tier 2 R.
- **Stable across seeds:**
  - "same verdict": every seed has the same Holm verdict against random-node (all better, or all worse, or all n.s.).
  - Plus the F1(c) cross-seed Jaccard and its random baseline where computable.
- **Seconds per molecule:** the mean of the N1 rows (new methods), measured with 8-12 concurrent jobs on one shared GPU.
  - SAME's time is the sum of both budgets.
  - Old methods: "not recorded" unless a logged time is found.

## F3. fig1_v3

The same layout and data as fig1_v2 (4 panels: MUTAG-GT, Mutagenicity D-MPNN, indole, PAINS; D-MPNN; all seeds), plus EiG and GNNShap from N1. Both are at L0, and N1 uses the same molecules as fig1_v2: checked per seed, the IG, SX and random-node arrays are equal.

- Random-node null band and best trivial rule band as in fig1_v2.
- Written to runs/rq1_tier5/; fig1_v2 is not touched.

## F5. EiG own size vs top-25%

For all 12 cells: EiG at the fixed top-25% budget and at its own chosen size, side by side (GEA, minus random-node).

- Own-size fraction = chosen atoms / atoms, mean per cell; also the bonds chosen against the cap floor(0.3 x bonds).
- Set GEA depends on set size, so the own-size version is also compared with a size-matched random-node null: 20 random atom sets of the same size, as a set.

## F6. EiG fidelity

- N1 Fid+/Fid- of EiG are reported, marked "not independent": EiG's search maximises Fid+ - Fid-.
- Next to them, GEA-only rankings per cell:
  1. on minus random-node, each method on its own molecules;
  2. on a common set of molecules: the SAME L4 slice where every method covers it (seeds 0-4).
