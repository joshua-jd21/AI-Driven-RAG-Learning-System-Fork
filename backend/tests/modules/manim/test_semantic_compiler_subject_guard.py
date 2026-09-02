"""Regression tests for subject-aware template resolution."""
from __future__ import annotations

from modules.manim.semantic_compiler import _resolve_template
from modules.planning.chemistry_router import CHEMISTRY_TEMPLATE_IDS


def test_physics_plan_with_chemistry_metadata_stays_generic() -> None:
    template_cls, resolved_id, source = _resolve_template(
        {
            "scene_id": 2,
            "concept_template": "freeform",
            "scene_role": "visual_intuition",
            "title": "Inertia in Motion",
            "subject": "Physics",
            "semantic_tags": ["atomic-structure"],
            "visualizable_elements": ["nucleus"],
        }
    )

    assert resolved_id == "freeform"
    assert source == "registered"
    assert template_cls is not None


def test_chemistry_plan_can_still_resolve_to_chemistry_template() -> None:
    template_cls, resolved_id, source = _resolve_template(
        {
            "scene_id": 2,
            "concept_template": "freeform",
            "scene_role": "visual_intuition",
            "title": "Atomic Structure",
            "subject": "Chemistry",
            "semantic_tags": ["atomic-structure"],
            "visualizable_elements": ["nucleus"],
        }
    )

    assert resolved_id in CHEMISTRY_TEMPLATE_IDS
    assert source in {"router_upgrade", "registered_chemistry"}
    assert template_cls is not None
