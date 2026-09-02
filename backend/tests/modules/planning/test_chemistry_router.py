"""Regression tests for chemistry template routing."""
from __future__ import annotations

from modules.planning.chemistry_router import (
    CHEMISTRY_TEMPLATE_IDS,
    route_chemistry_template,
)


def test_physics_topic_does_not_route_on_atomic_structure_tag_alone() -> None:
    assert (
        route_chemistry_template(
            "Inertia in Motion",
            "visual_intuition",
            ["atomic-structure"],
            ["bus"],
        )
        is None
    )


def test_chemistry_topic_with_atomic_structure_tag_still_routes() -> None:
    routed = route_chemistry_template(
        "Atomic Structure",
        "formal_concept",
        ["atomic-structure"],
        ["nucleus"],
    )
    assert routed in CHEMISTRY_TEMPLATE_IDS
