from __future__ import annotations

from modules.templates.mechanics.intro import IntroTemplate


def test_intro_template_shows_title_before_any_wait() -> None:
    plan = {
        "title": "Newton's First Law — Overview",
        "subtitle": "Inertia and the absence of net force",
        "key_term": "Newton's First Law",
        "events": [
            {
                "id": "e0",
                "type": "place_title",
                "anchor_phrase": "Newton's First Law",
            },
            {
                "id": "e1",
                "type": "place_subtitle",
                "anchor_phrase": "absence of net force",
            },
            {
                "id": "e2",
                "type": "highlight_term",
                "anchor_phrase": "Newton's First Law",
            },
        ],
    }
    timeline = {
        "audio_duration": 8.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Intro sentence."},
            {"start": 2.0, "end": 5.0, "text": "Second sentence."},
        ],
        "timeline": {
            "total": 8.0,
            "events": [
                {"id": "e0", "start": 1.75, "run_time": 0.8, "hold_after": 0.3},
                {"id": "e1", "start": 2.7, "run_time": 0.7, "hold_after": 0.2},
                {"id": "e2", "start": 4.2, "run_time": 0.6, "hold_after": 0.5},
            ],
        },
    }

    code = IntroTemplate.compile(plan, timeline)

    assert "self.play(Write(title)" in code
    assert "self.wait(" not in code.split("self.play(Write(title)")[0]
    assert code.index("self.play(Write(title)") < code.index("self.wait(")
