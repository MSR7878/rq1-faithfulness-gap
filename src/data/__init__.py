"""
Dataset loaders for RQ1 (Phase 2).

Every loader returns ``(data_list, meta)`` where

* ``data_list`` is a ``list[torch_geometric.data.Data]`` with, at minimum,
  ``x``, ``edge_index``, ``edge_attr`` (may be ``None``), and ``y``.
  Graph-classification label ``y`` has shape ``[1, 1]`` (float for binary
  MoleculeNet tasks, long for MUTAG) so a single tensor stacks cleanly.
* ``meta`` is a :class:`DatasetMeta` describing feature widths, the task,
  and whether node/edge ground-truth masks are attached
  (``data.node_gt_mask`` / ``data.edge_gt_mask``).

The five RQ1 datasets (see ``configs/rq1_metric_spec.md``):

    mutag            MUTAG, standard TUDataset            no ground truth
    mutag_graphxai   MUTAG + GraphXAI substructure GT     node + edge GT
    bbbp             BBBP, MoleculeNet                    no ground truth
    tox21_srp53      Tox21 SR-p53 task only, MoleculeNet  no ground truth
    bxaic            B-XAIC (mproszewska/B-XAIC)          node (+ edge) GT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetMeta:
    name: str
    num_graphs: int
    num_node_features: int
    num_edge_features: int
    num_classes: int
    task_type: str  # "binary-classification" for all Phase-1 datasets
    has_node_gt: bool = False
    has_edge_gt: bool = False
    notes: str = ""


from .standard import load_mutag, load_bbbp, load_tox21_srp53  # noqa: E402
from .graphxai_mutag import load_mutag_graphxai  # noqa: E402
from .bxaic import load_bxaic, BXAIC_TASKS  # noqa: E402

LOADERS = {
    "mutag": load_mutag,
    "mutag_graphxai": load_mutag_graphxai,
    "bbbp": load_bbbp,
    "tox21_srp53": load_tox21_srp53,
    "bxaic": load_bxaic,
}

__all__ = [
    "DatasetMeta",
    "LOADERS",
    "BXAIC_TASKS",
    "load_mutag",
    "load_mutag_graphxai",
    "load_bbbp",
    "load_tox21_srp53",
    "load_bxaic",
]
