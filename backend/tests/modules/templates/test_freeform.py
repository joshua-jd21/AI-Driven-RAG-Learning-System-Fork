"""Regression tests for the freeform Manim template."""
from __future__ import annotations

import ast
from textwrap import dedent

import pytest

from modules.templates.freeform import FreeformTemplate
from modules.templates import freeform as freeform_module


def _scheduled_runtime(source: str) -> float:
    tree = ast.parse(source)
    total = 0.0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "play":
            runtime = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "run_time"),
                None,
            )
        elif node.func.attr == "wait" and node.args:
            runtime = node.args[0]
        else:
            continue
        total += float(ast.literal_eval(runtime))
    return total


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


def test_equation_elapsed_matches_declared_animation_durations() -> None:
    body, elapsed = freeform_module._equation_scene_body(
        "Relationship",
        "explain the relationship",
        "A concrete example",
        "The variables are related.",
        "Show the relationship.",
        {},
    )

    declared_animation_time = _scheduled_runtime(dedent(body))

    assert declared_animation_time == pytest.approx(4.7)
    assert elapsed == pytest.approx(declared_animation_time)


def test_scene_four_style_equation_schedule_matches_audio_duration() -> None:
    plan = {
        "scene_id": 4,
        "concept_template": "freeform",
        "title": "Sliding Box on Frictionless Track",
        "learning_goal": "apply v = u + at with a = 0",
        "anchor_example": "A 1000 kg box moving at 20 m/s",
        "visual_instruction": (
            "Illustrate a box on a frictionless track. Write equation v = u + at, "
            "set a = 0, and show constant speed."
        ),
    }
    timeline = {"audio_duration": 19.876281, "segments": []}

    code = FreeformTemplate.compile(plan, timeline)

    assert abs(_scheduled_runtime(code) - timeline["audio_duration"]) <= 0.20


def test_scene_four_style_concept_schedule_matches_audio_duration() -> None:
    plan = {
        "scene_id": 4,
        "concept_template": "freeform",
        "title": "Car on a Frictionless Road",
        "learning_goal": "apply first law quantitatively",
        "anchor_example": "A car continues moving at constant speed on a frictionless track",
        "visual_instruction": "Display a car on a gray track and a blue velocity arrow.",
    }
    timeline = {"audio_duration": 18.135, "segments": []}

    code = FreeformTemplate.compile(plan, timeline)

    assert abs(_scheduled_runtime(code) - timeline["audio_duration"]) <= 0.20


def test_concept_elapsed_matches_declared_animation_durations() -> None:
    body, elapsed = freeform_module._concept_scene_body(
        "Concept",
        "explain the concept",
        "A concrete example",
        "The state remains unchanged.",
        "Show the concept.",
        {},
    )

    declared_animation_time = _scheduled_runtime(dedent(body))

    assert declared_animation_time == pytest.approx(3.3)
    assert elapsed == pytest.approx(declared_animation_time)


def test_explicit_equation_reaches_generated_scene_without_unrelated_fallback() -> None:
    plan = {
        "scene_id": 4,
        "concept_template": "freeform",
        "title": "Constant Velocity",
        "learning_goal": "apply the kinematic relationship",
        "visual_instruction": "Write equation v = u + at and explain the result.",
    }

    code = FreeformTemplate.compile(plan, {"audio_duration": 10.0})

    assert "v = u + at" in code
    assert "V = I × R" not in code


def test_equation_fallback_remains_for_plans_without_explicit_equation() -> None:
    plan = {
        "scene_id": 9,
        "concept_template": "freeform",
        "title": "A Generic Relationship",
        "learning_goal": "explain a relationship",
        "visual_instruction": "Show the relationship and solve for the unknown.",
    }

    code = FreeformTemplate.compile(plan, {"audio_duration": 10.0})

    assert "V = I × R" in code
