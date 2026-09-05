"""
Phase 3 -- explainer wrappers over the trained D-MPNN.

Uniform interface: every wrapper's ``.explain(data, target)`` returns an
:class:`ExplanationResult` carrying a per-node importance vector (and a per-edge
one when the method produces edge masks).  The metrics harness
(``src/metrics/``) is node-importance based, so ``node_importance`` is the
common currency; ``mask_r3_distribution_aware`` consumes it directly.

Explainers:
* ``GNNExplainerWrapper``  -- torch_geometric.explain, node + edge object masks
* ``PGExplainerWrapper``   -- torch_geometric.explain, edge masks (trained once)
* ``SubgraphXWrapper``     -- DIG's SubgraphX (MCTS + Shapley), connected node subset
"""

from .common import ExplanationResult, binarize_mean, node_importance_from_edges

__all__ = [
    "ExplanationResult",
    "binarize_mean",
    "node_importance_from_edges",
    "GNNExplainerWrapper",
    "PGExplainerWrapper",
    "SubgraphXWrapper",
]


def __getattr__(name):  # lazy: avoid importing DIG / building explainers at package import
    if name in ("GNNExplainerWrapper", "PGExplainerWrapper"):
        from . import pyg_explainers
        return getattr(pyg_explainers, name)
    if name == "SubgraphXWrapper":
        from . import dig_subgraphx
        return dig_subgraphx.SubgraphXWrapper
    raise AttributeError(name)
