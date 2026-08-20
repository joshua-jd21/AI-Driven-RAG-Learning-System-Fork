"""Lightweight PageIndex package init.

Keep this module free of eager imports so callers can load the lightweight
artifact readers without pulling in the full PageIndex LLM stack.
"""

from .results_loader import DocumentArtifacts, results_dir_for_pdf

__all__ = ["DocumentArtifacts", "results_dir_for_pdf"]
