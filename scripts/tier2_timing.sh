#!/usr/bin/env bash
# RQ1 Tier 2 timing test (timings only, no GEA/Fid): 12 concurrent single-thread SubgraphX workers with the fixed
# budget (4 each on Mutagenicity D-MPNN s0, B-XAIC indole s0, MUTAG-GT s0; 3 molecules per worker), then one
# gradient job (Saliency + IG + IxG) on 40 indole molecules. GPU 1 only.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=runs/rq1_tier2/timing
mkdir -p $T/logs
$PY - <<'EOF'
import json, torch
from src.analysis.revision_logits import units
from src.analysis.tier2_common import gt_mol_idx, dataset_of
U = units()
for u in ("mutagenicity_dmpnn_s0", "bxaic_indole_s0", "mutag_graphxai_s0"):
    t = torch.load(U[u]["cache"], weights_only=False)["key"]["test_idx"]
    json.dump(gt_mol_idx(u, t, dataset_of(u))[:12], open(f"runs/rq1_tier2/timing/{u}_mols.json", "w"))
EOF
date '+%F %T' > $T/START
for u in mutagenicity_dmpnn_s0 bxaic_indole_s0 mutag_graphxai_s0; do
  CK=$($PY -c "from src.analysis.revision_logits import units; print(units()['$u']['ckpt'])")
  for i in 0 1 2 3; do
    $PY -u -m src.analysis.tier2_explain sx --unit $u --ckpt $CK --mols-file $T/${u}_mols.json --shard $i/4 \
        --out $T/sx/$u --tag timing > $T/logs/sx_${u}_$i.log 2>&1 &
  done
done
wait
date '+%F %T' > $T/SX_DONE
CK=$($PY -c "from src.analysis.revision_logits import units; print(units()['bxaic_indole_s0']['ckpt'])")
$PY - <<EOF > $T/logs/grad.log 2>&1
import time, torch
from src.analysis import tier2_explain as te
from src.analysis.revision_logits import units
t = torch.load(units()["bxaic_indole_s0"]["cache"], weights_only=False)["key"]["test_idx"][:40]
t0 = time.time(); te.run_grad("bxaic_indole_s0", "$CK", t, "runs/rq1_tier2/timing/grad_indole_s0.npz", torch.device("cuda")); print(f"grad 40 mols {time.time()-t0:.1f}s")
EOF
date '+%F %T' > $T/FINISHED
