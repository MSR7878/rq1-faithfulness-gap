"""
RQ1 Tier 4, item L7: re-tune the 10 fine-tuned copies that drifted beyond 0.02 in Tier 3 F (runs/rq1_tier4/PLAN.md).
Same random-hiding rule as Tier 3 (tier3_finetune). Ladder of learning rates 5e-5, 2e-5, 1e-5, each from the original model and up
to 5 epochs; clean test accuracy and AUROC are checked after every epoch. Within a learning rate the last epoch within 0.02 is kept;
the ladder stops at the first learning rate with such an epoch. If none qualifies, the 1e-5 / 1-epoch copy is kept, marked not fixed.

    python -m src.analysis.tier4_L7 tune --unit U       -> runs/rq1_tier4/L7/ckpt/<unit>.pt
    python -m src.analysis.tier4_L7 score --unit U      -> runs/rq1_tier4/L7/score/<unit>.npz (tier3_F.run on the re-tuned copy)
"""
from __future__ import annotations

import argparse
import copy
import os
import time
import zlib

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from . import reliance_fidelity as rf
from . import tier2_common as C
from . import tier3_F
from . import tier4_common as T
from .tier3_finetune import random_mask_batch
from ..train.train import _collect, _metrics

DRIFTED = ["mutagenicity_gin_s3"] + [f"mutag_graphxai_s{s}" for s in (0, 1, 2, 3, 5, 6, 7, 8, 9)]
LADDER, MAX_EPOCHS, LIMIT = (5e-5, 2e-5, 1e-5), 5, 0.02


def tune(unit, device):
    out = f"{T.T4}/L7/ckpt/{unit}.pt"
    if os.path.exists(out):
        return
    sp = C.orig_spec(unit); base, ck = rf.load_model(sp["ckpt"], device); dl, _ = C.load_data(C.dataset_of(unit))
    bs = 512 if unit.startswith("bxaic") else 50; S = ck["split"]
    tl_data, te = [dl[int(i)] for i in S["train"]], [dl[int(i)] for i in S["test"]]; testl = DataLoader(te, batch_size=bs)
    m0 = _metrics(*_collect(base, testl, device)[:2]); log = []; kept = None; fallback = None; t0 = time.time()
    for lr in LADDER:
        model = copy.deepcopy(base).to(device); seed = zlib.crc32(f"retune|{unit}|{lr}".encode())
        torch.manual_seed(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
        tl = DataLoader(tl_data, batch_size=bs, shuffle=True); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss(); best = None
        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            for b in tl:
                b = b.to(device); x = b.x.float().clone(); x[random_mask_batch(b, gen)] = 0.0
                opt.zero_grad(); loss = lossf(model(x, b.edge_index, getattr(b, "edge_attr", None), b.batch), b.y.view(-1).long())
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            model.eval(); m = _metrics(*_collect(model, testl, device)[:2]); ok = abs(m[0] - m0[0]) <= LIMIT and abs(m[1] - m0[1]) <= LIMIT
            log.append({"lr": lr, "epoch": ep, "test_acc": m[0], "test_auroc": m[1], "within": bool(ok)})
            snap = (lr, ep, m, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if ok:
                best = snap
            if lr == LADDER[-1] and ep == 1:
                fallback = snap
        if best is not None:
            kept = best; break
    fixed = kept is not None; lr, ep, m, state = kept if fixed else fallback
    new = {k: v for k, v in ck.items() if k not in ("state_dict", "metrics")}
    new.update(state_dict=state, metrics=ck.get("metrics"),
               finetune=dict(unit=unit, orig_ckpt=sp["ckpt"], bound=0.5, lr=lr, epochs=ep, batch_size=bs, seconds=time.time() - t0, n_train=len(tl_data),
                             test_acc_before=m0[0], test_auroc_before=m0[1], test_acc_after=m[0], test_auroc_after=m[1], fixed=fixed, ladder=log))
    os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(new, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"{unit}: re-tuned, lr {lr} epoch {ep}, |d acc| {abs(m[0] - m0[0]):.3f} |d AUROC| {abs(m[1] - m0[1]):.3f}, {'fixed' if fixed else 'NOT fixed'}", flush=True)


def score(unit, device):
    tier3_F.run(unit, device, ckpt=f"{T.T4}/L7/ckpt/{unit}.pt", out_path=f"{T.T4}/L7/score/{unit}.npz")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["tune", "score"]); a.add_argument("--unit", required=True); a = a.parse_args()
    {"tune": tune, "score": score}[a.cmd](a.unit, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
