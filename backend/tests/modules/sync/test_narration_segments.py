from __future__ import annotations

from modules.sync.narration_segments import build_narration_segments


def test_build_narration_segments_covers_audio_and_keeps_visual_state() -> None:
    narration = (
        "A book stays still on the table. "
        "A push changes its motion. "
        "That is why F = ma is useful."
    )
    word_timestamps = [
        {"word": "A", "start": 0.0, "end": 0.08},
        {"word": "book", "start": 0.08, "end": 0.24},
        {"word": "stays", "start": 0.24, "end": 0.42},
        {"word": "still", "start": 0.42, "end": 0.60},
        {"word": "on", "start": 0.60, "end": 0.70},
        {"word": "the", "start": 0.70, "end": 0.78},
        {"word": "table", "start": 0.78, "end": 1.02},
        {"word": "A", "start": 1.10, "end": 1.18},
        {"word": "push", "start": 1.18, "end": 1.34},
        {"word": "changes", "start": 1.34, "end": 1.58},
        {"word": "its", "start": 1.58, "end": 1.68},
        {"word": "motion", "start": 1.68, "end": 1.94},
        {"word": "That", "start": 2.02, "end": 2.12},
        {"word": "is", "start": 2.12, "end": 2.18},
        {"word": "why", "start": 2.18, "end": 2.28},
        {"word": "F", "start": 2.28, "end": 2.34},
        {"word": "=", "start": 2.34, "end": 2.40},
        {"word": "ma", "start": 2.40, "end": 2.58},
        {"word": "is", "start": 2.58, "end": 2.66},
        {"word": "useful", "start": 2.66, "end": 2.90},
    ]

    segments = build_narration_segments(narration, word_timestamps, 3.0)

    assert len(segments) == 3
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 3.0
    assert all(seg["end"] >= seg["start"] for seg in segments)
    assert all(seg["visual_goal"] for seg in segments)
    assert all(seg["visual_state"] for seg in segments)
    assert all(seg["actions"] for seg in segments)
    assert "Transform" in segments[-1]["actions"]
    assert "Circumscribe" in segments[-1]["actions"]


def test_build_narration_segments_falls_back_to_single_segment_without_words() -> None:
    segments = build_narration_segments(
        "Narration without aligned words.",
        [],
        4.25,
    )

    assert len(segments) == 1
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 4.25
    assert "Narration" in segments[0]["text"]


def test_segments_attach_grounded_event_to_matching_narration_sentence() -> None:
    events = [
        {
            "id": "e0",
            "anchor_phrase": "push changes motion",
            "visual_goal": "show cause and effect",
            "visual_state": "object moving after the push",
            "action": "transform",
            "action_reason": "the push changes the object's state",
            "visible_objects": ["object", "force_arrow"],
            "emphasis_targets": ["force_arrow"],
            "persistence_after_action": True,
        }
    ]
    narration = "The object is still. A push changes motion."
    words = [
        {"word": "The", "start": 0.0, "end": 0.1},
        {"word": "object", "start": 0.1, "end": 0.25},
        {"word": "is", "start": 0.25, "end": 0.3},
        {"word": "still", "start": 0.3, "end": 0.5},
        {"word": "A", "start": 0.6, "end": 0.7},
        {"word": "push", "start": 0.7, "end": 0.85},
        {"word": "changes", "start": 0.85, "end": 1.05},
        {"word": "motion", "start": 1.05, "end": 1.25},
    ]

    segments = build_narration_segments(narration, words, 1.5, events=events)

    assert segments[1]["narration_reference"] == "push changes motion"
    assert segments[1]["visual_goal"] == "show cause and effect"
    assert segments[1]["action"] == "transform"
    assert "transform" in segments[1]["actions"]
    assert segments[1]["action_reason"] == "the push changes the object's state"
