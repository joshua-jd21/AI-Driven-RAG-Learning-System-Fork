"""Deterministic Phase 1 validation for generated Manim renders.

This module validates execution and artifact integrity only. It does not claim
that a rendered scene is educationally or semantically correct.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

from modules.config import get_logger

logger = get_logger(__name__)

# Raw Manim output is checked before FFmpeg can pad or trim it.
RAW_DURATION_TOLERANCE = 0.20


def validate_static_source(
    source: str,
    expected_scene_class: str,
    expected_template: str | None = None,
) -> dict[str, Any]:
    """Validate Python syntax and the expected generated scene class."""
    checks: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
        checks.append({
            "check": "python_syntax",
            "passed": True,
            "details": "source parsed successfully",
        })
    except SyntaxError as exc:
        checks.append({
            "check": "python_syntax",
            "passed": False,
            "details": f"{exc.msg} at line {exc.lineno}",
        })
        return {
            "passed": False,
            "checks": checks,
            "failure_classifications": ["STATIC_VALIDATION_FAILED"],
        }

    class_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    class_present = expected_scene_class in class_names
    checks.append({
        "check": "expected_scene_class",
        "passed": class_present,
        "details": (
            f"found {expected_scene_class}"
            if class_present
            else f"missing {expected_scene_class}; found {sorted(class_names)}"
        ),
    })
    failures = [] if class_present else ["STATIC_VALIDATION_FAILED"]

    if class_present and expected_template and expected_template != "freeform":
        scene_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == expected_scene_class
        )
        base_names = {
            base.id
            for base in scene_node.bases
            if isinstance(base, ast.Name)
        }
        # Registered templates inherit a project scene base. A direct Scene
        # subclass here is the deterministic compiler stub shape.
        template_shape_valid = "Scene" not in base_names or "ChalkboardScene" in base_names
        checks.append({
            "check": "registered_template_shape",
            "passed": template_shape_valid,
            "details": {
                "template": expected_template,
                "bases": sorted(base_names),
            },
        })
        if not template_shape_valid:
            failures.append("COMPILE_FALLBACK")

    return {
        "passed": not failures,
        "checks": checks,
        "failure_classifications": failures,
    }


def _probe_raw_mp4(path: Path) -> dict[str, Any]:
    """Probe a video stream, tolerating containers without nb_frames metadata."""
    command = [
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_type,width,height,duration,nb_frames,nb_read_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        return {"has_video_stream": False}

    frame_count = _positive_int(video.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _positive_int(video.get("nb_frames"))
    if frame_count is None:
        frame_count = _probe_packet_count(path)

    duration = _positive_float(video.get("duration"))
    if duration is None:
        duration = _positive_float((data.get("format") or {}).get("duration"))

    return {
        "has_video_stream": True,
        "width": _positive_int(video.get("width")),
        "height": _positive_int(video.get("height")),
        "frame_count": frame_count,
        "duration": duration,
    }


def _probe_packet_count(path: Path) -> int | None:
    """Use packet count when a container omits decoded-frame metadata."""
    command = [
        "ffprobe", "-v", "error",
        "-count_packets",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_packets,nb_packets",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout or "{}").get("streams") or []
        stream = streams[0] if streams else {}
        return _positive_int(stream.get("nb_read_packets")) or _positive_int(stream.get("nb_packets"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def validate_raw_artifact(
    artifact_path: Path,
    expected_duration: float,
    tolerance: float = RAW_DURATION_TOLERANCE,
) -> dict[str, Any]:
    """Validate a raw Manim MP4 and compare its duration to the scene audio."""
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    exists = artifact_path.is_file()
    checks.append({
        "check": "file_exists",
        "passed": exists,
        "details": str(artifact_path),
    })
    if not exists:
        return {
            "passed": False,
            "actual_duration": None,
            "checks": checks,
            "failure_classifications": ["RAW_MP4_INVALID"],
        }

    try:
        media = _probe_raw_mp4(artifact_path)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        checks.append({
            "check": "valid_container",
            "passed": False,
            "details": str(exc),
        })
        return {
            "passed": False,
            "actual_duration": None,
            "checks": checks,
            "failure_classifications": ["RAW_MP4_INVALID"],
        }

    has_stream = bool(media.get("has_video_stream"))
    checks.append({
        "check": "video_stream",
        "passed": has_stream,
        "details": "video stream found" if has_stream else "no video stream found",
    })
    if not has_stream:
        failures.append("RAW_MP4_INVALID")

    width = media.get("width")
    height = media.get("height")
    dimensions_valid = bool(width and height)
    checks.append({
        "check": "dimensions",
        "passed": dimensions_valid,
        "details": f"{width}x{height}",
    })
    if not dimensions_valid:
        failures.append("RAW_MP4_INVALID")

    frame_count = media.get("frame_count")
    frames_valid = isinstance(frame_count, (int, float)) and frame_count > 0
    checks.append({
        "check": "frame_count",
        "passed": frames_valid,
        "details": frame_count,
    })
    if not frames_valid:
        failures.append("RAW_MP4_INVALID")

    actual_duration = media.get("duration")
    duration_available = isinstance(actual_duration, (int, float)) and actual_duration > 0
    checks.append({
        "check": "duration_available",
        "passed": duration_available,
        "details": actual_duration,
    })
    if not duration_available:
        failures.append("RAW_MP4_INVALID")

    duration_matches = False
    if duration_available:
        duration_matches = abs(float(actual_duration) - float(expected_duration)) <= tolerance
    checks.append({
        "check": "duration",
        "passed": duration_matches,
        "details": {
            "expected": expected_duration,
            "actual": actual_duration,
            "tolerance": tolerance,
        },
    })
    if not duration_matches:
        failures.append("DURATION_MISMATCH")

    return {
        "passed": not failures,
        "actual_duration": actual_duration,
        "checks": checks,
        "failure_classifications": list(dict.fromkeys(failures)),
    }


def validate_render_result(
    render_result: Any,
    scene_id: int,
    template: str,
    expected_duration: float,
    tolerance: float = RAW_DURATION_TOLERANCE,
) -> dict[str, Any]:
    """Combine renderer provenance and raw artifact checks into a scene verdict."""
    static = getattr(render_result, "static_validation", {})
    artifact_path = Path(getattr(render_result, "artifact_path", ""))
    artifact = validate_raw_artifact(artifact_path, expected_duration, tolerance)

    failures = list(static.get("failure_classifications", []))
    if not getattr(render_result, "semantic_render_succeeded", False):
        if getattr(render_result, "render_status", "") != "static_validation_failed":
            failures.append("RENDER_FAILED")
        if getattr(render_result, "fallback_used", False):
            failures.append("RENDER_FALLBACK")
    failures.extend(artifact.get("failure_classifications", []))
    failures = list(dict.fromkeys(failures))

    checks = list(static.get("checks", []))
    checks.append({
        "check": "manim_execution",
        "passed": bool(getattr(render_result, "semantic_render_succeeded", False)),
        "details": getattr(render_result, "render_status", "unknown"),
    })
    checks.append({
        "check": "fallback_provenance",
        "passed": not bool(getattr(render_result, "fallback_used", False)),
        "details": getattr(render_result, "fallback_type", None),
    })
    checks.extend(artifact.get("checks", []))

    return {
        "scene_id": scene_id,
        "template": template,
        "expected_duration": expected_duration,
        "actual_raw_duration": artifact.get("actual_duration"),
        "fallback_used": bool(getattr(render_result, "fallback_used", False)),
        "render_status": getattr(render_result, "render_status", "unknown"),
        "checks": checks,
        "failure_classifications": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "render_result": render_result.as_dict() if hasattr(render_result, "as_dict") else render_result,
    }
