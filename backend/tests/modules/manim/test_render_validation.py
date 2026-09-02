from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.manim import render_validation


def _raw_result(
    path: Path,
    *,
    status: str = "success",
    fallback: bool = False,
    semantic_success: bool = True,
):
    return SimpleNamespace(
        artifact_path=path,
        render_status=status,
        fallback_used=fallback,
        fallback_type="minimal_safe" if fallback else None,
        semantic_render_succeeded=semantic_success,
        static_validation={
            "passed": True,
            "checks": [],
            "failure_classifications": [],
        },
        as_dict=lambda: {},
    )


def test_invalid_generated_python_fails_static_validation() -> None:
    report = render_validation.validate_static_source(
        "class GeneratedScene(Scene):\n    def construct(self):\n        if\n",
        "GeneratedScene",
    )

    assert report["passed"] is False
    assert "STATIC_VALIDATION_FAILED" in report["failure_classifications"]


def test_missing_expected_scene_class_fails_static_validation() -> None:
    report = render_validation.validate_static_source(
        "class OtherScene:\n    pass\n",
        "GeneratedScene",
    )

    assert report["passed"] is False
    assert "expected_scene_class" in {check["check"] for check in report["checks"]}


def test_registered_template_direct_scene_stub_fails_static_validation() -> None:
    report = render_validation.validate_static_source(
        "class GeneratedScene(Scene):\n    def construct(self):\n        pass\n",
        "GeneratedScene",
        expected_template="inertia",
    )

    assert report["passed"] is False
    assert "COMPILE_FALLBACK" in report["failure_classifications"]


def test_missing_mp4_fails_artifact_validation(tmp_path: Path) -> None:
    report = render_validation.validate_raw_artifact(tmp_path / "missing.mp4", 17.61)

    assert report["passed"] is False
    assert "RAW_MP4_INVALID" in report["failure_classifications"]


def test_invalid_mp4_container_fails_artifact_validation(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "invalid.mp4"
    path.write_bytes(b"not an mp4")
    monkeypatch.setattr(
        render_validation,
        "_probe_raw_mp4",
        lambda _path: (_ for _ in ()).throw(RuntimeError("moov atom not found")),
    )

    report = render_validation.validate_raw_artifact(path, 17.61)

    assert report["passed"] is False
    assert "RAW_MP4_INVALID" in report["failure_classifications"]


@pytest.mark.parametrize(
    "media,expected_failure",
    [
        ({"has_video_stream": False}, "RAW_MP4_INVALID"),
        ({"has_video_stream": True, "width": None, "height": 1080, "frame_count": 30, "duration": 17.61}, "RAW_MP4_INVALID"),
        ({"has_video_stream": True, "width": 1920, "height": 1080, "frame_count": 0, "duration": 17.61}, "RAW_MP4_INVALID"),
        ({"has_video_stream": True, "width": 1920, "height": 1080, "frame_count": 30, "duration": None}, "RAW_MP4_INVALID"),
    ],
)
def test_invalid_video_metadata_fails_artifact_validation(
    monkeypatch, tmp_path: Path, media: dict, expected_failure: str
) -> None:
    path = tmp_path / "scene.mp4"
    path.touch()
    monkeypatch.setattr(render_validation, "_probe_raw_mp4", lambda _path: media)

    report = render_validation.validate_raw_artifact(path, 17.61)

    assert report["passed"] is False
    assert expected_failure in report["failure_classifications"]


@pytest.mark.parametrize("expected", [17.61, 18.22])
def test_short_raw_render_fails_duration_validation(monkeypatch, tmp_path: Path, expected: float) -> None:
    path = tmp_path / "scene.mp4"
    path.touch()
    monkeypatch.setattr(
        render_validation,
        "_probe_raw_mp4",
        lambda _path: {
            "has_video_stream": True,
            "width": 1920,
            "height": 1080,
            "frame_count": 30,
            "duration": 2.0,
        },
    )

    report = render_validation.validate_raw_artifact(path, expected)

    assert report["passed"] is False
    assert "DURATION_MISMATCH" in report["failure_classifications"]


def test_duration_within_explicit_tolerance_passes(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "scene.mp4"
    path.touch()
    monkeypatch.setattr(
        render_validation,
        "_probe_raw_mp4",
        lambda _path: {
            "has_video_stream": True,
            "width": 1920,
            "height": 1080,
            "frame_count": 30,
            "duration": 17.75,
        },
    )

    report = render_validation.validate_raw_artifact(path, 17.61)

    assert report["passed"] is True
    assert report["failure_classifications"] == []


def test_fallback_never_passes_even_with_valid_padded_artifact(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "padded.mp4"
    path.touch()
    monkeypatch.setattr(
        render_validation,
        "_probe_raw_mp4",
        lambda _path: {
            "has_video_stream": True,
            "width": 1920,
            "height": 1080,
            "frame_count": 300,
            "duration": 17.61,
        },
    )

    report = render_validation.validate_render_result(
        _raw_result(path, status="fallback", fallback=True, semantic_success=False),
        scene_id=3,
        template="equation",
        expected_duration=17.61,
    )

    assert report["verdict"] == "FAIL"
    assert "RENDER_FALLBACK" in report["failure_classifications"]
    assert "RENDER_FAILED" in report["failure_classifications"]


def test_valid_render_result_passes(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "scene.mp4"
    path.touch()
    monkeypatch.setattr(
        render_validation,
        "_probe_raw_mp4",
        lambda _path: {
            "has_video_stream": True,
            "width": 1920,
            "height": 1080,
            "frame_count": 300,
            "duration": 17.61,
        },
    )

    report = render_validation.validate_render_result(
        _raw_result(path),
        scene_id=3,
        template="equation",
        expected_duration=17.61,
    )

    assert report["verdict"] == "PASS"
