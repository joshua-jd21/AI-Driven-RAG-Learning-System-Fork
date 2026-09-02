#!/usr/bin/env python3
"""Topic2Manim CLI: educational video generation pipeline.

New semantic architecture:
  Topic
    → Storyboard (concept arc)
    → Semantic Plans (template slots + event anchor phrases)
    → Narration (phrases embedded verbatim)
    → TTS (Piper)
    → WhisperX alignment (word-level timestamps)
    → Event Timelines (phrase-anchored start/run_time per event)
    → Semantic Compiler (template.compile → scene_N.py)
    → Manim Render
    → FFmpeg merge
    → final_video.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.config import FINAL_VIDEO, ensure_api_keys, get_logger
from modules.manim.renderer import render
from modules.manim.render_validation import validate_render_result
from modules.manim.semantic_compiler import semantic_compile_all
from modules.planning.asset_registry import reset_registry
from modules.planning.narration_writer import write_all_narrations
from modules.planning.semantic_plan import build_all_semantic_plans
from modules.planning.storyboard import build_storyboard
from modules.retrieval.pageindex_retriever import retrieve_curriculum
from modules.sync.sync_engine import synchronize_all
from modules.tts.piper_tts import synthesize
from modules.video.ffmpeg_merge import merge

logger = get_logger("main")

DEFAULT_TOPIC = "Explain Newton's First Law"


def run(topic: str, document_id: str | None = None, subject: str = "Physics") -> Path:
    """Execute the full semantic video generation pipeline."""
    logger.info("=" * 60)
    logger.info("Topic2Manim  |  Semantic Architecture")
    logger.info("Topic: %s", topic)
    if document_id:
        logger.info("Document: %s", document_id)
    logger.info("=" * 60)

    ensure_api_keys()

    curriculum = retrieve_curriculum(topic, document_id=document_id, subject=subject)
    curriculum_context = curriculum.get("context_text", "")
    curriculum_sections = curriculum.get("sections", [])
    if curriculum.get("matched"):
        logger.info(
            "Retrieved %d curriculum sections from %s",
            len(curriculum_sections),
            curriculum.get("document_id"),
        )
    else:
        logger.warning("No curriculum match for topic=%r document_id=%r", topic, document_id)

    # Fresh asset registry for this run
    reset_registry()

    # ── Step 1: Storyboard ────────────────────────────────────────────
    logger.info("[1/8] Building storyboard (5-scene concept arc)")
    storyboard = build_storyboard(
        topic,
        curriculum_context=curriculum_context,
        curriculum_sections=curriculum_sections,
        subject=subject,
    )

    # ── Step 2: Semantic plans ────────────────────────────────────────
    logger.info("[2/8] Building semantic plans (template slots + event anchors)")
    plans = build_all_semantic_plans(
        storyboard,
        curriculum_context=curriculum_context,
        curriculum_sections=curriculum_sections,
        topic=topic,
        subject=subject,
    )

    # ── Step 3: Narration ─────────────────────────────────────────────
    logger.info("[3/8] Writing narrations (anchor phrases embedded verbatim)")
    plans = write_all_narrations(
        plans,
        curriculum_context=curriculum_context,
        curriculum_sections=curriculum_sections,
        topic=topic,
        subject=subject,
    )

    # ── Step 4: TTS ───────────────────────────────────────────────────
    logger.info("[4/8] Synthesizing narration audio (Piper TTS)")
    audio_paths: dict[int, Path] = {}
    for plan in plans:
        sid = plan["scene_id"]
        wav_path = ROOT / "data" / "audio" / f"scene_{sid}.wav"
        wav, _duration = synthesize(plan["narration"], wav_path)
        audio_paths[sid] = wav

    # ── Step 5: Sync (WhisperX + event timeline) ──────────────────────
    logger.info("[5/8] Aligning audio (WhisperX) + building event timelines")
    timelines = synchronize_all(plans, audio_paths)

    # ── Step 6: Semantic compile ──────────────────────────────────────
    logger.info("[6/8] Compiling semantic Manim code")
    manim_files = semantic_compile_all(plans, timelines)

    # ── Step 7: Render ────────────────────────────────────────────────
    logger.info("[7/8] Rendering Manim scenes")
    scene_mp4s: list[Path] = []
    proof_reports: list[dict] = []
    timeline_by_scene = {item["scene_id"]: item for item in timelines}
    for plan, (manim_py, fallback_code, scene_class) in zip(plans, manim_files):
        render_result = render(
            manim_py,
            scene_class=scene_class,
            fallback_code=fallback_code,
            return_report=True,
            expected_template=plan.get("concept_template"),
        )
        expected_duration = float(
            timeline_by_scene[plan["scene_id"]].get("audio_duration", 0.0)
        )
        proof = validate_render_result(
            render_result,
            scene_id=plan["scene_id"],
            template=plan.get("concept_template", "unknown"),
            expected_duration=expected_duration,
        )
        proof_reports.append(proof)
        if proof["verdict"] == "PASS":
            scene_mp4s.append(render_result.artifact_path)
        else:
            logger.error(
                "[PHASE1][FAIL] scene=%s template=%s failures=%s expected=%.3fs actual=%s",
                proof["scene_id"],
                proof["template"],
                proof["failure_classifications"],
                proof["expected_duration"],
                proof["actual_raw_duration"],
            )

    failed_proofs = [report for report in proof_reports if report["verdict"] != "PASS"]
    if failed_proofs:
        failed_scenes = [report["scene_id"] for report in failed_proofs]
        raise RuntimeError(
            "Phase 1 render proof failed for scene(s): "
            + ", ".join(str(scene_id) for scene_id in failed_scenes)
        )

    # ── Step 8: Merge ─────────────────────────────────────────────────
    logger.info("[8/8] Merging audio + video (FFmpeg)")
    scene_wavs = [audio_paths[p["scene_id"]] for p in plans]
    final = merge(scene_mp4s, scene_wavs)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Final video: %s", final)
    logger.info("=" * 60)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate semantic educational videos from a topic."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f'Topic to explain (default: "{DEFAULT_TOPIC}")',
    )
    parser.add_argument(
        "--document-id",
        dest="document_id",
        default=None,
        help="PageIndex results folder name (e.g. Chemistry.pdf)",
    )
    parser.add_argument(
        "--subject",
        default="Physics",
        help="Subject label for planners (default: Physics)",
    )
    args = parser.parse_args()
    try:
        run(args.topic, document_id=args.document_id, subject=args.subject)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
