from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main


def _install_pipeline_fakes(monkeypatch, tmp_path: Path, proof_verdicts: list[str]) -> list[bool]:
    plans = [
        {"scene_id": 1, "concept_template": "intro", "narration": "one"},
        {"scene_id": 2, "concept_template": "summary", "narration": "two"},
    ]
    timelines = [
        {"scene_id": 1, "audio_duration": 4.0},
        {"scene_id": 2, "audio_duration": 5.0},
    ]
    merge_called: list[bool] = []
    monkeypatch.setattr(main, "ensure_api_keys", lambda: None)
    monkeypatch.setattr(main, "retrieve_curriculum", lambda *args, **kwargs: {"context_text": "", "sections": []})
    monkeypatch.setattr(main, "reset_registry", lambda: None)
    monkeypatch.setattr(main, "build_storyboard", lambda *args, **kwargs: [])
    monkeypatch.setattr(main, "build_all_semantic_plans", lambda *args, **kwargs: plans)
    monkeypatch.setattr(main, "write_all_narrations", lambda plans, **kwargs: plans)
    monkeypatch.setattr(main, "synthesize", lambda text, path: (path, 1.0))
    monkeypatch.setattr(main, "synchronize_all", lambda *args, **kwargs: timelines)
    monkeypatch.setattr(
        main,
        "semantic_compile_all",
        lambda *args, **kwargs: [(tmp_path / "scene_1.py", "", "GeneratedScene1"), (tmp_path / "scene_2.py", "", "GeneratedScene2")],
    )
    monkeypatch.setattr(
        main,
        "render",
        lambda *args, **kwargs: SimpleNamespace(artifact_path=tmp_path / "scene.mp4"),
    )

    reports = iter(proof_verdicts)
    monkeypatch.setattr(
        main,
        "validate_render_result",
        lambda *args, **kwargs: {
            "scene_id": kwargs["scene_id"],
            "template": kwargs["template"],
            "expected_duration": kwargs["expected_duration"],
            "actual_raw_duration": kwargs["expected_duration"],
            "failure_classifications": [] if (verdict := next(reports)) == "PASS" else ["DURATION_MISMATCH"],
            "verdict": verdict,
        },
    )
    monkeypatch.setattr(main, "merge", lambda *args, **kwargs: merge_called.append(True) or tmp_path / "final.mp4")
    return merge_called


def test_failed_scene_blocks_merge_and_completion(monkeypatch, tmp_path: Path) -> None:
    merge_called = _install_pipeline_fakes(monkeypatch, tmp_path, ["PASS", "FAIL"])

    with pytest.raises(RuntimeError, match="Phase 1 render proof failed"):
        main.run("test topic")

    assert merge_called == []


def test_all_phase1_scene_proofs_allow_normal_merge(monkeypatch, tmp_path: Path) -> None:
    merge_called = _install_pipeline_fakes(monkeypatch, tmp_path, ["PASS", "PASS"])

    result = main.run("test topic")

    assert result == tmp_path / "final.mp4"
    assert merge_called == [True]

