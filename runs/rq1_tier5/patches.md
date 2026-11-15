# Tier 5: changes around the authors' code

The authors' files in third_party/ are not edited; the commits and file hashes are in `third_party_manifest.json`. Everything we
replace, add or work around is listed here. Code: `src/explain/t5_common.py`, `src/explain/t5_explainers.py`.

## All repositories
- **Repository loading:** each repository is imported with its directory first on sys.path, and its module names are removed from sys.modules afterwards (`load_repo`). GStarX and SAME both have top-level `utils`, `shapley` and `gnnNets` modules.
- **Model calls:** they go through `IdAdapter`. An atom-index column is appended to x, so our model gets each remaining bond's real features; the model never sees the column. Target = predicted class.

## EiG-Search (b1e975d)
- The authors' code is **not imported**. The graph-classification branch of `main.py` (bond scores by removal, then the Fid+ - Fid- size search) is ported to our models in `EiG` (`t5_explainers.py`).
- The authors' per-model `fwd()` (edge weight = epsilon = 0) is replaced by removing the bond physically.
- The search runs over undirected bonds (2, 3, ..., floor(0.3 B) bonds) instead of slices of directed edges.
- **The two-graph batch:** the authors evaluate Fid- and Fid+ with one graph built in two batches (graph 0 is kept). Our port removes those atoms directly; N0 checks that this gives the same result.
- An empty graph gets probability 1/2.
- Deviations: target = predicted class instead of the label; misclassified molecules are not skipped.

## SAME (f0b4170)
- **Used unchanged:**
  - `initialization_mcts_GC.MCTS` and `reward_func`;
  - `shapley.GnnNets_GC2value_func` and `mc_l_shapley`;
  - `utils.find_explanations`.
- The `config` object is a SimpleNamespace with the authors' "mutag" or "bbbp" block (PLAN).
- `data.y` is set to the predicted class, so the authors' value functions (which read `data.y`) explain the predicted class.
- No self-loops are added (SAME.py adds them for the authors' own models).
- **Size:** min_atoms = max_nodes = ceil(0.5 N) at own size, or floor(0.25 N) at our budget (two runs).

## GStarX (770b286)
- `gstarx.GStarX` is used unchanged with the molecular config.
- `payoff_avg` = the unit's own mean test-split probabilities.
- **`more_itertools` shim** (`src/explain/t5_shims/more_itertools.py`): the package is not installed on ada, and ada has no internet. GStarX imports `set_partitions` only for the superadditive extension, which the config turns off. The shim raises if it is ever called.

## GNNShap (f967229)
- **Used:** the authors' `GNNShapSampler` (CUDA extension) and `WLSSolver`.
- **Our own graph-level wrapper** (`GNNShapExplainer`) replaces the authors' node-classification `explain()`:
  - players = undirected bonds;
  - value = predicted-class probability with the coalition's bonds kept;
  - fnull = no bonds; ffull = all bonds.
- **`colorlog` shim** (`src/explain/t5_shims/colorlog.py`): the package is not installed on ada, and ada has no internet. GNNShap uses it only for coloured logging; the shim maps it to the standard `logging` module.
- **Build environment for the CUDA extension** (torch.utils.cpp_extension.load, as the authors call it): ada has neither ninja on the default PATH nor a new enough default compiler (gcc 7.5; torch 2.6 headers need gcc >= 9). Every GNNShap job runs with:
  - `PATH=/opt/anaconda/pkgs/ninja-1.10.2-hff7bd54_1/bin:$PATH` (an existing conda package cache on ada; nothing installed);
  - `CC=gcc-9 CXX=g++-9` (system compilers);
  - `TORCH_CUDA_ARCH_LIST=7.0`;
  - nvcc from /usr/local/cuda (11.3) against torch built for CUDA 11.8.

  The .cu file is unchanged.
- Molecules with fewer than 2 bonds are not explained (scores 0) and are listed.
