"""
Smoke test for the GINE and GCN architecture checks (Mutagenicity) -- run
BEFORE any training or explanation.  Same checks as test_gin_explainers.py,
plus the bond-feature path for GINE:

  1. forward: single graph == the same graph inside a batch; parameter count
     close to the D-MPNN's.  GINE: the output DEPENDS on edge_attr (permuting
     the bond features changes it).  GCN: edge_attr is ignored.
  2. edge mask: PyG ``set_masks`` with an all-ones mask reproduces the
     unmasked output (with bond features passed; GCN's added self-loops
     handled) and a random mask CHANGES it.
  3. embeddings: ``get_embeddings`` (edge_attr passed through) returns one
     [N, hidden] tensor per conv.
  4. explainers, each called exactly as run_phase3 calls them:
     GNNExplainer -> finite, non-constant node AND edge masks;
     PGExplainer trains on 10 graphs -> finite [E] edge mask;
     SubgraphX (DIG stub): the DIG adapter's output equals the model called
     with the real bond features, single and tiled over a batch;
     R3 masking + Fidelity/GEF run.

    python -m src.train.test_arch_explainers            # gine and gcn
    python -m src.train.test_arch_explainers gine
"""

from __future__ import annotations

import sys

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_embeddings

from ..data import LOADERS
from ..explain.dig_subgraphx import SubgraphXWrapper, _DIGAdapter
from ..explain.pyg_explainers import GNNExplainerWrapper, PGExplainerWrapper
from ..metrics.fidelity import compute_fidelity
from ..metrics.gef import compute_gef
from ..metrics.masking import mask_r3_distribution_aware, training_fill_vector
from .baselines import build_model, make_cfg

HIDDEN = {"gine": 214, "gcn": 300}
USES_BONDS = {"gine": True, "gcn": False}


def _count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def check(arch: str, dl, meta, device) -> None:
    h = HIDDEN[arch]
    kw = dict(node_in=meta.num_node_features, edge_in=meta.num_edge_features, depth=3, dropout=0.0,
              num_classes=2, pool="sum")
    torch.manual_seed(0)
    model = build_model(arch, vars(make_cfg(arch, hidden=h, ffn_hidden=h, **kw))).to(device).eval()
    n_dm = _count(build_model("dmpnn", vars(make_cfg("dmpnn", hidden=300, ffn_hidden=300, **kw))))
    n = _count(model)
    print(f"[{arch}] params: {arch.upper()}({h}) {n:,}  D-MPNN(300) {n_dm:,}  ratio {n / n_dm:.3f}")
    assert 0.95 < n / n_dm < 1.05

    # 1. forward
    d = dl[0].clone().to(device)
    b = next(iter(DataLoader([g.clone() for g in dl[:16]], batch_size=16))).to(device)
    E = d.edge_index.size(1)
    with torch.no_grad():
        single = model(d.x, d.edge_index, d.edge_attr)
        batched = model(b.x, b.edge_index, b.edge_attr, b.batch)
        rolled = model(d.x, d.edge_index, torch.roll(d.edge_attr, 1, dims=1))
        no_attr = model(d.x, d.edge_index, None)
    assert single.shape == (1, 2) and batched.shape == (16, 2)
    assert torch.allclose(single[0], batched[0], atol=1e-4), "single graph != same graph in a batch"
    if USES_BONDS[arch]:
        assert not torch.allclose(single, rolled, atol=1e-4), "GINE output does not depend on the bond features"
        assert not torch.allclose(single, no_attr, atol=1e-4), "GINE output unchanged without bond features"
        print(f"[{arch}] 1. forward ok (single == batched; bond features change logits by {(single - rolled).abs().max().item():.3g})")
    else:
        assert torch.allclose(single, rolled) and torch.allclose(single, no_attr), "GCN must ignore edge_attr"
        print(f"[{arch}] 1. forward ok (single == batched, edge_attr ignored)")

    # 2. edge mask, with bond features passed through
    with torch.no_grad():
        set_masks(model, torch.ones(E, device=device), d.edge_index, apply_sigmoid=False)
        ones = model(d.x, d.edge_index, d.edge_attr)
        torch.manual_seed(1)
        set_masks(model, torch.rand(E, device=device), d.edge_index, apply_sigmoid=False)
        rnd = model(d.x, d.edge_index, d.edge_attr)
        clear_masks(model)
        after = model(d.x, d.edge_index, d.edge_attr)
    assert torch.allclose(ones, single, atol=1e-5), "all-ones edge mask must reproduce the unmasked output"
    assert not torch.allclose(rnd, single, atol=1e-4), "random edge mask did not change the output"
    assert torch.allclose(after, single, atol=1e-5), "clear_masks did not restore the model"
    print(f"[{arch}] 2. edge mask ok (ones == unmasked; random mask moves logits by {(rnd - single).abs().max().item():.3g})")

    # 3. embeddings, edge_attr passed as PGExplainer passes it
    with torch.no_grad():
        emb = get_embeddings(model, d.x, d.edge_index, edge_attr=d.edge_attr)
    assert len(emb) == 3 and all(e.shape == (d.num_nodes, h) for e in emb), [e.shape for e in emb]
    print(f"[{arch}] 3. embeddings ok: {len(emb)} x {tuple(emb[-1].shape)}")

    # 4a. PGExplainer
    pg = PGExplainerWrapper(model, device, epochs=3)
    losses = pg.train([g.clone() for g in dl[:10]])
    assert all(torch.isfinite(torch.tensor(losses))), losses
    r_pg = pg.explain(dl[0].clone())
    assert r_pg.edge_importance.shape == (E,) and torch.isfinite(r_pg.edge_importance).all()
    assert r_pg.node_importance.shape == (d.num_nodes,)
    print(f"[{arch}] 4a. PGExplainer trained on 10 graphs (loss {losses[0]:.3f} -> {losses[-1]:.3f}), edge mask [{E}] finite")

    # 4b. GNNExplainer: both learned masks must have moved (gradient reaches the edge mask through the conv)
    r_gnn = GNNExplainerWrapper(model, device, epochs=30).explain(dl[0].clone())
    assert r_gnn.node_importance.shape == (d.num_nodes,) and torch.isfinite(r_gnn.node_importance).all()
    assert float(r_gnn.node_importance.std()) > 0, "GNNExplainer node mask is constant"
    assert r_gnn.edge_importance is not None and r_gnn.edge_importance.shape == (E,)
    assert float(r_gnn.edge_importance.std()) > 0, "GNNExplainer edge mask is constant"
    assert r_gnn.target == int(single.argmax()), "GNNExplainer explained a class other than the model's prediction with bond features"
    print(f"[{arch}] 4b. GNNExplainer node mask [{d.num_nodes}] and edge mask [{E}] finite, non-constant")

    # 4c. SubgraphX: the DIG adapter must re-supply the real bond features
    adapter = _DIGAdapter(model, d.edge_attr, d.edge_index).to(device).eval()
    two = Batch.from_data_list([Data(x=d.x, edge_index=d.edge_index), Data(x=d.x * 0.0, edge_index=d.edge_index)])
    with torch.no_grad():
        a_single = adapter(d.x, d.edge_index)
        a_batch = adapter(data=two)
        zero_x = model(d.x * 0.0, d.edge_index, d.edge_attr)
    assert torch.allclose(a_single, single, atol=1e-5), "DIG adapter (positional) != model with bond features"
    assert torch.allclose(a_batch[0], single[0], atol=1e-4), "DIG adapter (batched) != model with bond features"
    assert torch.allclose(a_batch[1], zero_x[0], atol=1e-4), "DIG adapter: zero-filled graph lost its bond features"
    r_sx = SubgraphXWrapper(model, device, rollout=3, sample_num=5).explain(dl[0].clone())
    assert int(r_sx.node_importance.sum()) >= 1 and r_sx.target == int(single.argmax())
    print(f"[{arch}] 4c. SubgraphX adapter == model with bond features (single, batched, zero-filled); coalition "
          f"{int(r_sx.node_importance.sum())} atoms (budget {r_sx.meta['max_nodes']}, N={d.num_nodes})")

    # 4d. R3 scoring
    fill = training_fill_vector(torch.cat([g.x.float().cpu() for g in dl[:50]], dim=0), "mean")
    cpu = dl[0].clone().cpu()
    for r in (r_gnn, r_pg, r_sx):
        imp = r.node_importance.detach().cpu().float()
        E_ = mask_r3_distribution_aware(cpu, imp, fill, keep_top_k=0.25).to(device)
        G_ = mask_r3_distribution_aware(cpu, -imp, fill, keep_top_k=0.75).to(device)
        assert E_.edge_attr is not None and E_.edge_attr.shape == d.edge_attr.shape, "R3 masking dropped edge_attr"
        fid = compute_fidelity(model, d, r.target, explanation_removed=G_, explanation_only=E_, masking_reference="R3")
        gef = compute_gef(model, d, masked_data=E_, masking_reference="R3")
        assert all(map(lambda v: v == v, (fid.fid_plus, fid.fid_minus, gef.gef)))
    print(f"[{arch}] 4d. R3 masking keeps edge_attr; Fidelity/GEF ok")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archs = sys.argv[1:] or ["gine", "gcn"]
    dl, meta = LOADERS["mutagenicity_graphxai"]()
    for arch in archs:
        check(arch, dl, meta, device)
    print(f"\nPASS -- {', '.join(a.upper() for a in archs)} run through GNNExplainer, PGExplainer, SubgraphX and R3 scoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
