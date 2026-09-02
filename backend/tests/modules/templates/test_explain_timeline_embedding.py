from __future__ import annotations

from modules.templates.explain.comparison import ComparisonTemplate
from modules.templates.explain.concept_card import ConceptCardTemplate
from modules.templates.explain.diagram import DiagramTemplate
from modules.templates.explain.equation import EquationTemplate
from modules.templates.explain.timeline import TimelineTemplate


def test_diagram_template_embeds_timeline_literal() -> None:
    plan = {
        "title": "Inertia in Motion — No Forces Needed",
        "visual_instruction": "Side view of a bus.",
        "content": {
            "title": "Inertia in Motion — No Forces Needed",
            "nodes": ["Bus moving right", "Sudden brake applied", "Passenger body tendency"],
        },
    }
    timeline = {
        "scene_id": 2,
        "audio_duration": 4.946,
        "timeline": {
            "total": 4.946,
            "events": [
                {"id": "e0", "start": 0.33, "run_time": 0.95, "hold_after": 0.0},
                {"id": "e1", "start": 3.627, "run_time": 0.819, "hold_after": 0.0},
                {"id": "e2", "start": 5.027, "run_time": 0.35, "hold_after": 0.0},
            ],
        },
    }

    code = DiagramTemplate.compile(plan, timeline)

    assert "timeline=timeline" not in code
    assert "timeline={" in code
    assert "audio_duration=4.946" in code
    assert "self.wait(" not in code
    compile(code, "generated_diagram.py", "exec")
    assert "true" not in code


def test_concept_card_template_embeds_timeline_literal() -> None:
    plan = {
        "title": "Inertia in Motion",
        "visual_instruction": "Show a book and a push arrow.",
        "content": {
            "main_title": "Inertia in Motion",
            "cards": [
                {"title": "Rest", "content": "The book stays put.", "color": "#7BA7C2"},
                {"title": "Force", "content": "A push changes motion.", "color": "#7AC2A0"},
            ],
        },
    }
    timeline = {
        "scene_id": 2,
        "audio_duration": 6.25,
        "segments": [
            {"start": 0.0, "end": 2.1, "text": "The book rests.", "visual_goal": "book rests", "actions": ["FadeIn"]},
            {"start": 2.1, "end": 4.4, "text": "A push changes motion.", "visual_goal": "push changes motion", "actions": ["FadeIn", "MoveAlongPath"]},
        ],
        "timeline": {
            "total": 6.25,
            "events": [
                {"id": "e0", "start": 0.0, "run_time": 0.8, "hold_after": 0.0},
                {"id": "e1", "start": 2.1, "run_time": 0.8, "hold_after": 0.0},
            ],
        },
    }

    code = ConceptCardTemplate.compile(plan, timeline)

    assert "timeline={" in code
    assert "self.wait(" not in code


def test_comparison_template_embeds_timeline_literal() -> None:
    plan = {
        "title": "Compare States",
        "visual_instruction": "Left side stays still; right side moves.",
        "content": {
            "left_title": "At rest",
            "left_content": "Nothing changes on the left.",
            "right_title": "In motion",
            "right_content": "The right side moves.",
        },
    }
    timeline = {
        "scene_id": 3,
        "audio_duration": 5.5,
        "segments": [
            {"start": 0.0, "end": 1.7, "text": "Start with rest.", "visual_goal": "rest", "actions": ["FadeIn"]},
            {"start": 1.7, "end": 3.9, "text": "Then motion.", "visual_goal": "motion", "actions": ["FadeIn", "Compare"]},
        ],
    }

    code = ComparisonTemplate.compile(plan, timeline)

    assert "timeline={" in code
    assert "self.wait(" not in code
    assert "true" not in code
    assert "false" not in code
    compile(code, "generated_comparison.py", "exec")


def test_timeline_template_embeds_timeline_literal() -> None:
    plan = {
        "title": "Sequence",
        "visual_instruction": "Show a three-step sequence.",
        "content": {
            "title": "Sequence",
            "events": ["First", "Second", "Third"],
        },
    }
    timeline = {
        "scene_id": 4,
        "audio_duration": 5.0,
        "segments": [
            {"start": 0.0, "end": 1.2, "text": "First idea.", "visual_goal": "first idea", "actions": ["FadeIn"]},
            {"start": 1.2, "end": 3.0, "text": "Second idea.", "visual_goal": "second idea", "actions": ["FadeIn"]},
        ],
    }

    code = TimelineTemplate.compile(plan, timeline)

    assert "timeline={" in code
    assert "self.wait(" not in code


def test_equation_template_embeds_timeline_literal() -> None:
    plan = {
        "title": "Newton's First Law - Formal Statement",
        "content": {
            "title": "Newton's First Law - Formal Statement",
            "equation": "\\Sigma F = 0 \\rightarrow a = 0",
            "explanation": "Net force zero means no acceleration.",
        },
    }
    timeline = {
        "scene_id": 3,
        "audio_duration": 65.132,
        "timeline": {
            "total": 65.132,
            "events": [
                {"id": "e0", "start": 0.065, "run_time": 2.0, "hold_after": 0.0},
                {"id": "e1", "start": 2.065, "run_time": 1.4, "hold_after": 0.0},
                {"id": "e2", "start": 3.465, "run_time": 1.4, "hold_after": 0.0},
                {"id": "e3", "start": 5.352, "run_time": 2.0, "hold_after": 0.0},
                {"id": "e4", "start": 8.185, "run_time": 0.35, "hold_after": 0.0},
            ],
        },
    }

    code = EquationTemplate.compile(plan, timeline)

    assert "timeline=timeline" not in code
    assert "timeline={" in code
    assert "audio_duration=65.132" in code
    assert "self.wait(" not in code
    compile(code, "generated_equation.py", "exec")
    assert "true" not in code
