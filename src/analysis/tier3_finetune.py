"""
RQ1 Tier 3, item F: fine-tune a COPY of a trained model on inputs with random node masks (F-Fidelity style).
Per training molecule and epoch: fraction f ~ U(0, 0.5), floor(f N) uniformly random atoms zero-filled (as in R2), topology
kept. Adam, lr 1e-4, 5 epochs, original batch size, gradient clip 5.0 as in training, last epoch kept; seeded by the unit name.
The original checkpoint is only read. Clean test accuracy / AUROC before and after are stored in the copy.

    python -m src.analysis.tier3_finetune --unit U --out F.pt
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
from ..train.train import _collect, _metrics

BOUND, LR, EPOCHS = 0.5, 1e-4, 5


def random_mask_batch(batch, gen):
    """Boolean over atoms: within each graph, floor(f N) random atoms, f ~ U(0, BOUND)."""
    b = batch.batch; dev = b.device; g = b.max().item() + 1
    N = torch.bincount(b, minlength=g); f = torch.rand(g, device=dev, generator=gen) * BOUND; hide = (f * N).floor().long()
    r = torch.rand(b.size(0), device=dev, generator=gen); order = torch.argsort(r); order = order[torch.argsort(b[order], stable=True)]
    start = torch.cat([N.new_zeros(1), N.cumsum(0)[:-1]]); pos = torch.arange(b.size(0), device=dev) - start[b[order]]
    m = torch.zeros_like(b, dtype=torch.bool); m[order] = pos < hide[b[order]]
    return m


def finetune(unit, out, device):
    sp = C.orig_spec(unit); model, ck = rf.load_model(sp["ckpt"], device); model = copy.deepcopy(model).to(device); dl, _ = C.load_data(C.dataset_of(unit))
    bs = 512 if unit.startswith("bxaic") else 50
    seed = zlib.crc32(f"finetune|{unit}".encode()); torch.manual_seed(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    S = ck["split"]; tr = [dl[int(i)] for i in S["train"]]; te = [dl[int(i)] for i in S["test"]]
    tl, testl = DataLoader(tr, batch_size=bs, shuffle=True), DataLoader(te, batch_size=bs)
    m0 = _metrics(*_collect(model, testl, device)[:2])
    opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.CrossEntropyLoss(); t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device); x = b.x.float().clone(); x[random_mask_batch(b, gen)] = 0.0
            opt.zero_grad(); loss = lossf(model(x, b.edge_index, getattr(b, "edge_attr", None), b.batch), b.y.view(-1).long())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    sec = time.time() - t0; model.eval(); m1 = _metrics(*_collect(model, testl, device)[:2])
    new = {k: v for k, v in ck.items() if k not in ("state_dict", "metrics")}
    new.update(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, metrics=ck.get("metrics"),
               finetune=dict(unit=unit, orig_ckpt=sp["ckpt"], bound=BOUND, lr=LR, epochs=EPOCHS, batch_size=bs, seconds=sec, n_train=len(tr), test_acc_before=m0[0], test_auroc_before=m0[1], test_acc_after=m1[0], test_auroc_after=m1[1]))
    os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(new, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"{unit}: fine-tuned, {len(tr)} train molecules, {sec:.0f} s", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit", required=True); a.add_argument("--out", required=True); a = a.parse_args()
    finetune(a.unit, a.out, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
