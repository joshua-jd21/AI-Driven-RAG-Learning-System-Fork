from __future__ import annotations

from pathlib import Path

import modules.sync.sync_engine as sync_engine


def test_synchronize_scene_includes_segments(monkeypatch, tmp_path) -> None:
    plan = {
        "scene_id": 1,
        "narration": "A book stays still. A push changes motion.",
        "events": [
            {
                "id": "e0",
                "type": "place_title",
                "anchor_phrase": "book stays still",
            },
            {
                "id": "e1",
                "type": "highlight",
                "anchor_phrase": "push changes motion",
            },
        ],
    }

    monkeypatch.setattr(sync_engine, "get_audio_duration", lambda _path: 4.0)
    monkeypatch.setattr(
        sync_engine,
        "align",
        lambda _wav, _narration: [
            {"word": "A", "start": 0.0, "end": 0.1},
            {"word": "book", "start": 0.1, "end": 0.3},
            {"word": "stays", "start": 0.3, "end": 0.5},
            {"word": "still", "start": 0.5, "end": 0.7},
            {"word": "A", "start": 0.9, "end": 1.0},
            {"word": "push", "start": 1.0, "end": 1.2},
            {"word": "changes", "start": 1.2, "end": 1.4},
            {"word": "motion", "start": 1.4, "end": 1.7},
        ],
    )
    monkeypatch.setattr(
        sync_engine,
        "build_event_timeline",
        lambda events, word_timestamps, audio_duration: {
            "total": audio_duration,
            "events": [
                {"id": "e0", "start": 0.0, "run_time": 0.8, "hold_after": 0.2},
                {"id": "e1", "start": 0.9, "run_time": 0.8, "hold_after": 0.0},
            ],
        },
    )
    monkeypatch.setattr(sync_engine, "PATHS", {**sync_engine.PATHS, "timelines": tmp_path})

    result = sync_engine.synchronize_scene(plan, Path("/tmp/fake.wav"))

    assert result["timeline"]["segments"] == result["segments"]
    assert len(result["segments"]) == 2
    assert result["timeline"]["segments"][0]["start"] == 0.0
    assert result["timeline"]["segments"][-1]["end"] == 4.0
    assert result["visual_audit"]["segment_count"] == 2
    assert result["visual_audit"]["visual_gap_count"] == 0
