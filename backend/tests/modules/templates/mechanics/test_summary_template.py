from __future__ import annotations

from modules.templates.mechanics.summary import SummaryTemplate


def test_summary_template_shows_title_before_any_wait() -> None:
    plan = {
        "title": "Key Takeaways",
        "summary_points": ["Objects at rest remain at rest", "External force changes motion"],
        "events": [
            {
                "id": "e0",
                "type": "place_title",
                "anchor_phrase": "Key Takeaways",
            },
            {
                "id": "e1",
                "type": "place_point",
                "anchor_phrase": "Objects at rest remain at rest",
            },
            {
                "id": "e2",
                "type": "hold",
                "anchor_phrase": "External force changes motion",
            },
        ],
    }
    timeline = {
        "audio_duration": 9.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "First sentence."},
            {"start": 2.5, "end": 5.0, "text": "Second sentence."},
        ],
        "timeline": {
            "total": 9.0,
            "events": [
                {"id": "e0", "start": 1.4, "run_time": 0.8, "hold_after": 0.2},
                {"id": "e1", "start": 2.6, "run_time": 0.7, "hold_after": 0.2},
                {"id": "e2", "start": 4.4, "run_time": 0.6, "hold_after": 0.4},
            ],
        },
    }

    code = SummaryTemplate.compile(plan, timeline)

    assert "self.play(Write(title)" in code
    assert "self.wait(" not in code.split("self.play(Write(title)")[0]
    assert code.index("self.play(Write(title)") < code.index("self.wait(")


def test_summary_segment_schedule_is_bounded_and_has_no_zero_waits() -> None:
    code = SummaryTemplate.compile(
        {
            "title": "Key Takeaways",
            "summary_points": ["Rest", "Constant velocity", "Net force changes motion"],
            "events": [],
        },
        {
            "audio_duration": 9.0,
            "segments": [
                {"start": 0.0, "end": 2.2},
                {"start": 2.2, "end": 4.8},
                {"start": 4.8, "end": 7.0},
                {"start": 7.0, "end": 9.0},
            ],
            "events": [],
        },
    )

    durations = [
        float(line.split("run_time=")[1].split(")")[0])
        for line in code.splitlines()
        if "self.play(" in line and "run_time=" in line
    ]
    waits = [
        float(line.split("self.wait(")[1].split(")")[0])
        for line in code.splitlines()
        if "self.wait(" in line
    ]
    assert "self.play(FadeIn(pt_0" in code
    assert "self.play(FadeIn(pt_1" in code
    assert "self.play(FadeIn(pt_2" in code
    assert all(duration > 0.0 for duration in waits)
    assert sum(durations + waits) <= 9.0 + 1e-6
