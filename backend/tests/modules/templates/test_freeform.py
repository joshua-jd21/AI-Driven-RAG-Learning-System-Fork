"""Regression tests for the freeform Manim template."""
from __future__ import annotations

from modules.templates.freeform import FreeformTemplate
from modules.templates import freeform as freeform_module


def test_freeform_compile_accepts_synced_timeline() -> None:
    plan = {
        "scene_id": 3,
        "concept_template": "freeform",
        "scene_role": "formal_concept",
        "title": "Newton's First Law — Formal Statement",
        "learning_goal": "state the law using standard notation",
        "anchor_example": "An object at rest stays at rest",
        "visual_instruction": "Black background with ΣF = 0 and constant velocity.",
    }
    timeline = {
        "audio_duration": 12.0,
        "events": [
            {"id": "e0", "start": 0.5, "run_time": 0.8, "hold_after": 0.0},
            {"id": "e1", "start": 2.0, "run_time": 0.8, "hold_after": 0.0},
        ],
    }

    code = FreeformTemplate.compile(plan, timeline)

    assert "class GeneratedScene" in code
    assert "self.wait(" in code
    assert "self.play(Write(title)" in code
    assert "self.wait(" not in code.split("self.play(Write(title)")[0]


def test_freeform_timeline_event_prefers_segments() -> None:
    timeline = {
        "audio_duration": 7.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Intro text.", "actions": ["FadeIn"]},
            {"start": 2.0, "end": 5.0, "text": "Second idea.", "actions": ["MoveAlongPath"]},
        ],
        "events": [
            {"id": "e0", "start": 1.0, "run_time": 0.8, "hold_after": 0.2},
            {"id": "e1", "start": 4.0, "run_time": 0.8, "hold_after": 0.2},
        ],
    }

    event = freeform_module._timeline_event(timeline, 0, default_start=0.0, default_rt=0.8)

    assert event["start"] == 0.0
    assert 0.34 <= event["run_time"] <= 0.8
    assert event["hold_after"] > 0.0
