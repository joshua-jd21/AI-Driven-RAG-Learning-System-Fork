from __future__ import annotations

from modules.manim.templates.segment_timing import (
    bounded_segment_budget,
    segment_stage_groups,
)
from modules.templates.mechanics._base import event_hold, event_rt, event_start


def test_mechanics_event_helpers_prefer_narration_segments() -> None:
    timeline = {
        "audio_duration": 8.0,
        "segments": [
            {"start": 0.0, "end": 2.4, "text": "The object stays still.", "actions": ["FadeIn"]},
            {"start": 2.4, "end": 5.0, "text": "A force acts.", "actions": ["GrowArrow", "MoveAlongPath"]},
        ],
        "events": [
            {"id": "e0", "start": 1.8, "run_time": 0.8, "hold_after": 0.2},
            {"id": "e1", "start": 4.0, "run_time": 0.8, "hold_after": 0.2},
        ],
    }

    assert event_start(timeline, "e0") == 0.0
    assert event_start(timeline, "e1") == 2.4
    rt0 = event_rt(timeline, "e0", default=0.8)
    rt1 = event_rt(timeline, "e1", default=0.8)
    assert 0.34 <= rt0 <= 0.8
    assert 0.34 <= rt1 <= 0.8
    assert event_hold(timeline, "e0") > 0.0
    assert event_hold(timeline, "e1") > 0.0


def test_bounded_segment_budget_uses_only_remaining_interval() -> None:
    segment = {"start": 2.0, "end": 5.0}

    first = bounded_segment_budget(segment, cursor=3.0, requested_runtime=0.8)
    second = bounded_segment_budget(segment, cursor=4.5, requested_runtime=0.8)

    assert first == {"start": 3.0, "end": 5.0, "runtime": 0.8, "hold": 1.2}
    assert second == {"start": 4.5, "end": 5.0, "runtime": 0.5, "hold": 0.0}


def test_segment_stage_groups_distribute_stages_without_new_intervals() -> None:
    groups = segment_stage_groups(segment_count=2, stage_count=5)

    assert groups == [[0, 1, 2], [3, 4]]
    assert [stage for group in groups for stage in group] == list(range(5))
