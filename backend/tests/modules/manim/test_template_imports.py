"""Regression tests for the reusable Manim scene package imports."""
from __future__ import annotations


def test_chalkboard_template_package_imports_cleanly() -> None:
    # Importing the package should not fail via equation/diagram scene type hints.
    from modules.manim import templates  # noqa: F401

    assert True
