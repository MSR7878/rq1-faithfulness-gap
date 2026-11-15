"""Shim for GNNShap's `import colorlog` (not installed on ada, which has no internet). GNNShap uses it only for coloured log
output in gnnshap.utils.get_logger; this maps it to the standard logging module (no colours)."""
import logging
from logging import StreamHandler, getLogger  # noqa: F401


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, *a, log_colors=None, **k):
        super().__init__((fmt or "%(message)s").replace("%(log_color)s", ""))
