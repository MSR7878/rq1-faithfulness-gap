"""Shim for GStarX's `from more_itertools import set_partitions` (more_itertools is not installed on ada, which has no
internet). set_partitions is used only by GStarX's superadditive extension, which the authors' config switches off
(superadditive_ext: False). Calling it raises, so it can never be used silently."""


def set_partitions(*a, **k):
    raise RuntimeError("more_itertools.set_partitions shim: superadditive extension is not available (and is off in the config)")
