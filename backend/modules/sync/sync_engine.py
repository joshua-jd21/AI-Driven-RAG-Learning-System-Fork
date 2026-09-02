"""Synchronization engine: audio + WhisperX alignment → event timeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.config import PATHS, RenderWorkspace, get_logger
from modules.sync.narration_segments import build_narration_segments
from modules.sync.timeline_builder import build_event_timeline
from modules.sync.whisper_align import align
from modules.tts.piper_tts import get_audio_duration

logger = get_logger(__name__)


def synchronize_scene(
    plan: dict[str, Any],
    wav_path: Path,
    workspace: RenderWorkspace | None = None,
) -> dict[str, Any]:
    """Build a timed event timeline for one scene."""
    scene_id = plan["scene_id"]
    narration = plan.get("narration", "")
    events = plan.get("events", [])
    logger.info("Synchronizing scene %d (%d events)", scene_id, len(events))

    audio_duration = get_audio_duration(wav_path)
    word_timestamps = align(wav_path, narration)
    timeline = build_event_timeline(events, word_timestamps, audio_duration)
    segments = build_narration_segments(
        narration,
        word_timestamps,
        audio_duration,
        events=events,
    )
    timeline["segments"] = segments
    gap_count, gap_total = _segment_gap_stats(segments)
    visual_duration = float(segments[-1]["end"]) if segments else 0.0
    first_visual = float(segments[0]["start"]) if segments else 0.0
    last_visual = float(segments[-1]["end"]) if segments else 0.0
    drift = round(visual_duration - audio_duration, 3)
    animation_count = sum(len(seg.get("actions", [])) for seg in segments)

    result: dict[str, Any] = {
        "scene_id": scene_id,
        "audio_duration": audio_duration,
        "word_timestamps": word_timestamps,
        "timeline": timeline,
        "segments": segments,
        "visual_audit": {
            "scene_id": scene_id,
            "audio_duration": round(audio_duration, 3),
            "visual_duration": round(visual_duration, 3),
            "segment_count": len(segments),
            "animation_count": animation_count,
            "visual_gap_count": gap_count,
            "visual_gap_duration": gap_total,
            "first_visual_time": round(first_visual, 3),
            "last_visual_time": round(last_visual, 3),
            "timing_drift": drift,
        },
    }

    timelines_root = (
        workspace.timelines_dir if workspace is not None else PATHS["timelines"]
    )
    out_path = timelines_root / f"scene_{scene_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Timeline saved: %s (audio=%.2fs, %d events, %d segments, gap=%.2fs)",
        out_path, audio_duration, len(timeline["events"]), len(segments), gap_total,
    )
    logger.info(
        "[VISUAL AUDIT] scene=%d audio=%.2fs video=%.2fs segments=%d animations=%d "
        "gaps=%d gap_duration=%.2fs first_visual=%.2fs last_visual=%.2fs drift=%.2fs",
        scene_id,
        audio_duration,
        visual_duration,
        len(segments),
        animation_count,
        gap_count,
        gap_total,
        first_visual,
        last_visual,
        drift,
    )
    if gap_total > 0.35:
        logger.warning(
            "[VISUAL GAP] scene=%d start=%.2fs duration=%.2fs",
            scene_id,
            first_visual,
            gap_total,
        )
    return result


def synchronize_all(
    plans: list[dict[str, Any]],
    audio_paths: dict[int, Path],
    workspace: RenderWorkspace | None = None,
) -> list[dict[str, Any]]:
    """Build timelines for all scenes and write master timeline."""
    timelines = [
        synchronize_scene(plan, audio_paths[plan["scene_id"]], workspace=workspace)
        for plan in plans
    ]

    timelines_root = (
        workspace.timelines_dir if workspace is not None else PATHS["timelines"]
    )
    master_path = timelines_root / "master_timeline.json"
    master_path.write_text(json.dumps(timelines, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Master timeline saved: %s", master_path)
    return timelines


def _segment_gap_stats(segments: list[dict[str, Any]]) -> tuple[int, float]:
    if len(segments) < 2:
        return 0, 0.0
    count = 0
    total = 0.0
    prev_end = float(segments[0].get("end", 0.0))
    for seg in segments[1:]:
        start = float(seg.get("start", prev_end))
        if start > prev_end:
            count += 1
            total += start - prev_end
        prev_end = max(prev_end, float(seg.get("end", prev_end)))
    return count, round(total, 3)
