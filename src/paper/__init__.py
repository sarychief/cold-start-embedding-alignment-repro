"""Paper-faithful reproduction and comparison runners."""

from .comparison import main as comparison_main
from .repro import main as repro_main

__all__ = ["comparison_main", "repro_main"]
