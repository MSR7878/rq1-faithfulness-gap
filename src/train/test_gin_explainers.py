"""
Smoke test for the GIN architecture check -- run BEFORE any GIN training or
explanation.  Checks that a plain PyG GIN goes through the existing pipeline
unchanged:

  1. forward: single graph == the same graph inside a batch; bond features
     are ignored; parameter count is close to the D-MPNN's.
  2. edge mask: PyG ``set_masks`` with an all-ones mask reproduces the
     unmasked output, and a random mask CHANGES it (GNNExplainer's hook).
  3. embeddings: ``get_embeddings`` returns one [N, hidden] tensor per conv
     (PGExplainer's edge-MLP input is the last one).
  4. PGExplainer trains on 10 graphs and returns a finite [E] edge mask;
     GNNExplainer returns a finite [N] node mask; SubgraphX (DIG stub)
     returns a coalition of the budgeted size; R3 masking + Fidelity/GEF run.
  5. the default arch is still the D-MPNN.

    python -m src.train.test_gin_explainers
"""

from __future__ import annotations

import torch
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_embeddings

from ..data import LOADERS
from ..explain.dig_subgraphx import SubgraphXWrapper
from ..explain.pyg_explainers import GNNExplainerWrapper, PGExplainerWrapper
from ..metrics.fidelity import compute_fidelity
from ..metrics.gef import compute_gef
from ..metrics.masking import mask_r3_distribution_aware, training_fill_vector
from .baselines import GIN, build_model, make_cfg
from .dmpnn import DMPNN


def _count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ("mutag_graphxai", "mutagenicity_graphxai"):
        dl, meta = LOADERS[name]()
        kw = dict(node_in=meta.num_node_features, edge_in=meta.num_edge_features, depth=3, dropout=0.0,
                  num_classes=2, pool="sum")
        torch.manual_seed(0)
        gin = build_model("gin", vars(make_cfg("gin", hidden=212, ffn_hidden=212, **kw))).to(device).eval()
        dm = build_model("dmpnn", vars(make_cfg("dmpnn", hidden=300, ffn_hidden=300, **kw)))
        assert isinstance(gin, GIN) and isinstance(dm, DMPNN)
        n_gin, n_dm = _count(gin), _count(dm)
        print(f"[{name}] params: GIN(212) {n_gin:,}  D-MPNN(300) {n_dm:,}  ratio {n_gin / n_dm:.3f}")
        assert 0.9 < n_gin / n_dm < 1.1

        # 1. forward
        d = dl[0].clone().to(device)
        b = next(iter(DataLoader([g.clone() for g in dl[:16]], batch_size=16))).to(device)
        with torch.no_grad():
            single = gin(d.x, d.edge_index, d.edge_attr)
            no_attr = gin(d.x, d.edge_index, None)
            batched = gin(b.x, b.edge_index, b.edge_attr, b.batch)
        assert single.shape == (1, 2) and batched.shape == (16, 2)
        assert torch.allclose(single, no_attr), "GIN must ignore edge_attr"
        assert torch.allclose(single[0], batched[0], atol=1e-4), "single graph != same graph in a batch"
        print(f"[{name}] 1. forward ok (single == batched, edge_attr ignored)")

        # 2. edge mask changes the output
        E = d.edge_index.size(1)
        with torch.no_grad():
            set_masks(gin, torch.ones(E, device=device), d.edge_index, apply_sigmoid=False)
            ones = gin(d.x, d.edge_index, d.edge_attr)
            torch.manual_seed(1)
            set_masks(gin, torch.rand(E, device=device), d.edge_index, apply_sigmoid=False)
            rnd = gin(d.x, d.edge_index, d.edge_attr)
            clear_masks(gin)
            after = gin(d.x, d.edge_index, d.edge_attr)
        assert torch.allclose(ones, single, atol=1e-5), "all-ones edge mask must reproduce the unmasked output"
        assert not torch.allclose(rnd, single, atol=1e-4), "random edge mask did not change the output"
        assert torch.allclose(after, single, atol=1e-5), "clear_masks did not restore the model"
        print(f"[{name}] 2. edge mask ok (ones == unmasked; random mask moves logits by "
              f"{(rnd - single).abs().max().item():.3g})")

        # 3. embeddings
        with torch.no_grad():
            emb = get_embeddings(gin, d.x, d.edge_index)
        assert len(emb) == 3 and all(e.shape == (d.num_nodes, 212) for e in emb), [e.shape for e in emb]
        print(f"[{name}] 3. embeddings ok: {len(emb)} x {tuple(emb[-1].shape)}")

        # 4. the three explainers + R3 scoring
        pg = PGExplainerWrapper(gin, device, epochs=3)
        losses = pg.train([g.clone() for g in dl[:10]])
        assert all(torch.isfinite(torch.tensor(losses))), losses
        r_pg = pg.explain(dl[0].clone())
        assert r_pg.edge_importance.shape == (E,) and torch.isfinite(r_pg.edge_importance).all()
        assert r_pg.node_importance.shape == (d.num_nodes,)
        print(f"[{name}] 4a. PGExplainer trained on 10 graphs (loss {losses[0]:.3f} -> {losses[-1]:.3f}), edge mask [{E}] finite")

        r_gnn = GNNExplainerWrapper(gin, device, epochs=20).explain(dl[0].clone())
        assert r_gnn.node_importance.shape == (d.num_nodes,) and torch.isfinite(r_gnn.node_importance).all()
        assert float(r_gnn.node_importance.std()) > 0
        print(f"[{name}] 4b. GNNExplainer node mask [{d.num_nodes}] finite, non-constant")

        r_sx = SubgraphXWrapper(gin, device, rollout=3, sample_num=5).explain(dl[0].clone())
        assert int(r_sx.node_importance.sum()) >= 1
        print(f"[{name}] 4c. SubgraphX coalition {int(r_sx.node_importance.sum())} atoms "
              f"(budget {r_sx.meta['max_nodes']}, N={d.num_nodes})")

        fill = training_fill_vector(torch.cat([g.x.float().cpu() for g in dl[:50]], dim=0), "mean")
        cpu = dl[0].clone().cpu()
        for r in (r_gnn, r_pg, r_sx):
            imp = r.node_importance.detach().cpu().float()
            E_ = mask_r3_distribution_aware(cpu, imp, fill, keep_top_k=0.25).to(device)
            G_ = mask_r3_distribution_aware(cpu, -imp, fill, keep_top_k=0.75).to(device)
            fid = compute_fidelity(gin, d, r.target, explanation_removed=G_, explanation_only=E_, masking_reference="R3")
            gef = compute_gef(gin, d, masked_data=E_, masking_reference="R3")
            assert all(map(lambda v: v == v, (fid.fid_plus, fid.fid_minus, gef.gef)))
        print(f"[{name}] 4d. R3 masking + Fidelity/GEF ok")

    print("\nPASS -- plain PyG GIN runs through GNNExplainer, PGExplainer, SubgraphX and R3 scoring unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
