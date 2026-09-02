"""Shared narration-segment timing helpers for Manim templates."""
from __future__ import annotations

from typing import Any


def segments_from_timeline(timeline: dict[str, Any] | None) -> list[dict[str, Any]]:
    segments = list((timeline or {}).get("segments", []))
    return sorted(
        [seg for seg in segments if isinstance(seg, dict)],
        key=lambda seg: float(seg.get("start", 0.0)),
    )


def segment_count(timeline: dict[str, Any] | None) -> int:
    return len(segments_from_timeline(timeline))


def segment_at(timeline: dict[str, Any] | None, index: int) -> dict[str, Any] | None:
    segments = segments_from_timeline(timeline)
    if 0 <= index < len(segments):
        return segments[index]
    return None


def segment_start(segment: dict[str, Any] | None, default: float = 0.0) -> float:
    if not segment:
        return default
    return float(segment.get("start", default))


def segment_end(segment: dict[str, Any] | None, default: float = 0.0) -> float:
    if not segment:
        return default
    return float(segment.get("end", default))


def segment_duration(segment: dict[str, Any] | None, default: float = 0.0) -> float:
    if not segment:
        return default
    return max(0.0, segment_end(segment, default) - segment_start(segment, default))


def segment_actions(segment: dict[str, Any] | None) -> list[str]:
    if not segment:
        return []
    actions = segment.get("actions", [])
    return [str(action) for action in actions if action]


def segment_text(segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return str(segment.get("text", "")).strip()


def segment_visual_goal(segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return str(segment.get("visual_goal", "")).strip()


def segment_visual_state(segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return str(segment.get("visual_state", "")).strip()


def segment_rt(
    segment: dict[str, Any] | None,
    default: float = 0.7,
    floor: float = 0.35,
    cap: float | None = None,
    fraction: float = 0.40,
) -> float:
    """Return a segment-aware run time that leaves persistent state visible."""
    if not segment:
        return max(default, floor)

    dur = segment_duration(segment, default)
    if dur <= 0:
        return max(default, floor)

    rt = max(floor, dur * fraction)
    rt = min(rt, default) if default > 0 else rt
    if cap is not None:
        rt = min(rt, cap)
    return max(0.1, round(rt, 3))


def segment_hold(
    segment: dict[str, Any] | None,
    run_time: float,
    minimum: float = 0.0,
) -> float:
    if not segment:
        return minimum
    hold = max(0.0, segment_duration(segment) - run_time)
    return round(hold, 3)


def bounded_segment_budget(
    segment: dict[str, Any] | None,
    cursor: float,
    requested_runtime: float,
    scene_end: float | None = None,
) -> dict[str, float]:
    """Allocate one action inside the unused portion of a segment.

    ``cursor`` is authoritative: a segment may be revisited for another small
    action, but only its remaining interval is available. This keeps grouped
    actions bounded without inventing another timing source.
    """
    if not segment:
        return {
            "start": float(cursor),
            "end": float(cursor),
            "runtime": 0.0,
            "hold": 0.0,
        }

    start = max(float(cursor), segment_start(segment, cursor))
    end = max(start, segment_end(segment, start))
    if scene_end is not None:
        end = min(end, max(start, float(scene_end)))
    available = max(0.0, end - start)
    runtime = min(max(0.0, float(requested_runtime)), available)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "runtime": round(runtime, 3),
        "hold": round(max(0.0, available - runtime), 3),
    }


def segment_stage_groups(segment_count: int, stage_count: int) -> list[list[int]]:
    """Distribute ordered visual stages across ordered segments once."""
    if segment_count <= 0 or stage_count <= 0:
        return []
    groups: list[list[int]] = [[] for _ in range(segment_count)]
    for stage_index in range(stage_count):
        group_index = min(
            segment_count - 1,
            (stage_index * segment_count) // stage_count,
        )
        groups[group_index].append(stage_index)
    return groups


def segment_gap_stats(timeline: dict[str, Any] | None) -> tuple[int, float]:
    segments = segments_from_timeline(timeline)
    if len(segments) < 2:
        return 0, 0.0

    gap_count = 0
    gap_total = 0.0
    prev_end = segment_end(segments[0], 0.0)
    for seg in segments[1:]:
        start = segment_start(seg, prev_end)
        if start > prev_end:
            gap_count += 1
            gap_total += start - prev_end
        prev_end = max(prev_end, segment_end(seg, prev_end))
    return gap_count, round(gap_total, 3)
