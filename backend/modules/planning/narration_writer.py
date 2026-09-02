"""Narration writer: generates and validates narration for each scene.

Given a semantic plan with events that have anchor_phrases, this module
generates a narration script that:
  1. Contains every anchor_phrase verbatim (as a contiguous substring)
  2. Preserves the anchor phrases in order
  3. Sounds natural and educational (35-60 words per scene)

Validation: verifies each anchor_phrase is a case-insensitive substring
of the final narration. Retries up to 3 times with stricter prompts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.config import (
    NVIDIA_NARRATION_MAX_TOKENS,
    NVIDIA_PLANNER_MODEL,
    PATHS,
    get_logger,
)
from modules.llm.nvidia_client import NvidiaClient, NvidiaEmptyResponseError
from modules.planning.profile_context import (
    format_learner_context,
    normalize_profile,
    pace_word_budget,
)

logger = get_logger(__name__)

NARRATION_SYSTEM = """You are an expert educational video narrator.
Write clear, engaging narration for short animated physics scenes.
CRITICAL: You MUST include every required phrase EXACTLY as given — verbatim, as-is.
Respond with ONLY the narration text. No reasoning, no analysis, no planning commentary,
and no JSON."""

NARRATION_PROMPT = """Write narration for this scene. It will be read aloud over an animation.

CURRICULUM CONTEXT:
{curriculum_context}
Use the curriculum context as the primary source of truth.
Use textbook terminology whenever available.
Do not contradict the curriculum context.

LESSON SUBJECT: {subject}
LESSON TOPIC: {topic}

{learner_context}

SCENE TITLE: {title}
ANCHOR EXAMPLE: {anchor_example}
LEARNING GOAL: {learning_goal}
VISUAL INSTRUCTION: {visual_instruction}

NARRATION-GROUNDED VISUAL STAGES (keep these ideas in this order):
{visual_stages}

REQUIRED PHRASES — USE IN THIS EXACT ORDER:
{phrases}

RULES:
- Length: {word_lo}-{word_hi} words (calibrated to the learner's pace).
- Conversational, clear, educational tone matching the learner's level.
- Include every required phrase verbatim and preserve the exact order shown above.
- Do not paraphrase phrases or introduce one phrase before an earlier phrase.
- Surround each required phrase with natural connective language.
- Output narration text only: no JSON, bullets, headings, or commentary.
- For low-confidence learners, include one micro-analogy or relatable example inside the scene.
- If the curriculum context lists Prerequisites for this topic, open with a one-sentence verbal
  recap of the most relevant prerequisite before introducing the new concept (stay within word budget).

Return ONLY the narration text:"""

NARRATION_REPAIR_PROMPT = """Your previous narration failed validation.

CURRICULUM CONTEXT:
{curriculum_context}

ACTIONABLE FAILURE FEEDBACK:
{failure_feedback}

SCENE: {title}
REQUIRED PHRASES — USE IN THIS EXACT ORDER:
{phrases}

Rewrite the narration in {word_lo}-{word_hi} words. Include EVERY phrase verbatim,
in the exact order listed above, while preserving natural educational narration.
Output narration text only: no JSON, bullets, headings, or commentary."""


def write_narration(
    plan: dict[str, Any],
    curriculum_context: str = "",
    curriculum_sections: list | None = None,
    learner_profile: dict[str, Any] | None = None,
    topic: str = "",
    subject: str = "Physics",
    learner_context: str | None = None,
) -> str:
    """Generate narration for one scene and validate anchor phrases."""
    scene_id = plan["scene_id"]
    title = plan.get("title", f"Scene {scene_id}")
    anchor_example = plan.get("anchor_example", "")
    learning_goal = plan.get("learning_goal", "")
    visual_instruction = plan.get("visual_instruction", "")
    events = plan.get("events", [])
    phrases = [ev["anchor_phrase"] for ev in events if ev.get("anchor_phrase", "").strip()]
    seen: set[str] = set()
    unique_phrases: list[str] = []
    for p in phrases:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique_phrases.append(p)

    p_norm = normalize_profile(learner_profile, subject)
    word_lo, word_hi = pace_word_budget(p_norm)
    learner_context_text = plan.get("_learner_context") or learner_context or format_learner_context(
        learner_profile, topic or title, subject
    )

    if not unique_phrases:
        logger.warning("Scene %d has no anchor phrases; generating free narration", scene_id)
        return _generate_free(
            title, anchor_example, learning_goal, scene_id,
            visual_instruction=visual_instruction,
            learner_context=learner_context_text, word_lo=word_lo, word_hi=word_hi,
        )

    client = NvidiaClient()
    phrases_display = _format_required_phrases(unique_phrases)
    visual_stages = _format_visual_stages(events)

    # Enrich context with visual metadata from curriculum sections
    enriched_context = curriculum_context
    if curriculum_sections:
        from modules.retrieval.pageindex_retriever import format_sections_for_prompt
        vis_block = format_sections_for_prompt(curriculum_sections)
        if vis_block:
            enriched_context = f"{vis_block}\n\n{curriculum_context}"

    prompt = NARRATION_PROMPT.format(
        curriculum_context=enriched_context,
        subject=subject,
        topic=topic or title,
        learner_context=learner_context_text,
        title=title,
        anchor_example=anchor_example,
        learning_goal=learning_goal,
        visual_instruction=visual_instruction,
        visual_stages=visual_stages,
        phrases=phrases_display,
        word_lo=word_lo,
        word_hi=word_hi,
    )
    messages = [
        {"role": "system", "content": NARRATION_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    narration = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            text = client.chat(
                NVIDIA_PLANNER_MODEL,
                messages,
                temperature=0.35,
                max_tokens=NVIDIA_NARRATION_MAX_TOKENS,
            )
        except NvidiaEmptyResponseError as exc:
            last_error = exc
            logger.warning(
                "Scene %d narration attempt %d returned empty content (%s); retrying",
                scene_id, attempt + 1, exc,
            )
            continue
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Scene %d narration attempt %d failed: %s; retrying",
                scene_id, attempt + 1, exc,
            )
            continue
        text = text.strip()
        errors = _narration_validation_errors(text, unique_phrases, word_lo, word_hi)
        if not errors:
            narration = text
            break
        last_error = ValueError("; ".join(errors))
        logger.warning(
            "Scene %d narration attempt %d failed validation: %s",
            scene_id, attempt + 1, "; ".join(errors),
        )
        repair_prompt = NARRATION_REPAIR_PROMPT.format(
            curriculum_context=curriculum_context,
            failure_feedback=_format_retry_feedback(errors),
            title=title,
            phrases=phrases_display,
            word_lo=word_lo,
            word_hi=word_hi,
        )
        messages = [
            {"role": "system", "content": NARRATION_SYSTEM},
            {"role": "user", "content": repair_prompt},
        ]

    if not narration:
        raise RuntimeError(
            f"Scene {scene_id} narration generation failed after 3 attempts: {last_error}"
        ) from last_error

    _save_narration(scene_id, narration)
    return narration


def _format_visual_stages(events: list[dict[str, Any]]) -> str:
    """Expose the semantic contract to narration without adding timing guesses."""
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        reference = str(event.get("narration_reference", event.get("anchor_phrase", ""))).strip()
        goal = str(event.get("visual_goal", "")).strip()
        state = str(event.get("visual_state", "")).strip()
        action = str(event.get("action", "hold")).strip()
        reason = str(event.get("action_reason", "")).strip()
        if reference or goal or state:
            lines.append(
                f"{index}. phrase={reference!r}; goal={goal!r}; state={state!r}; "
                f"action={action!r}; reason={reason!r}"
            )
    return "\n".join(lines) or "(No explicit stages; keep every sentence visually specific.)"


def _format_required_phrases(phrases: list[str]) -> str:
    """Render the phrase contract in a stable, unambiguous order."""
    return "\n".join(f'{index}. "{phrase}"' for index, phrase in enumerate(phrases, start=1))


def _format_retry_feedback(errors: list[str]) -> str:
    """Turn validator output into concrete rewrite instructions."""
    feedback: list[str] = []
    for error in errors:
        if error == "required phrases are out of order":
            feedback.append(
                "The previous attempt failed because the required phrases occurred in the wrong order."
            )
        elif error.startswith("word count "):
            feedback.append(
                f"The previous attempt failed its length check: {error}. "
                "Rewrite within the requested range while preserving every required phrase verbatim "
                "and in the exact required order."
            )
        else:
            feedback.append(error)
    return "\n".join(f"- {item}" for item in feedback)


def write_all_narrations(
    plans: list[dict[str, Any]],
    curriculum_context: str = "",
    curriculum_sections: list | None = None,
    learner_profile: dict[str, Any] | None = None,
    topic: str = "",
    subject: str = "Physics",
    learner_context: str | None = None,
) -> list[dict[str, Any]]:
    """Write narrations for all plans and attach them in-place."""
    for plan in plans:
        narration = write_narration(
            plan,
            curriculum_context=curriculum_context,
            curriculum_sections=curriculum_sections,
            learner_profile=learner_profile,
            topic=topic,
            subject=subject,
            learner_context=learner_context,
        )
        plan["narration"] = narration
        logger.info(
            "Narration for scene %d (%d words): %s…",
            plan["scene_id"],
            len(narration.split()),
            narration[:60],
        )
    return plans


def _find_missing(narration: str, phrases: list[str]) -> list[str]:
    """Return any phrases that are NOT verbatim substrings of the narration."""
    lower = narration.lower()
    return [p for p in phrases if p.lower() not in lower]


def _narration_validation_errors(
    narration: str,
    phrases: list[str],
    word_lo: int,
    word_hi: int,
) -> list[str]:
    errors = [f"missing required phrase {phrase!r}" for phrase in _find_missing(narration, phrases)]
    word_count = len(narration.split())
    if word_count < word_lo or word_count > word_hi:
        errors.append(f"word count {word_count} outside requested range {word_lo}-{word_hi}")
    positions = [narration.lower().find(phrase.lower()) for phrase in phrases]
    if any(position < 0 for position in positions):
        return errors
    if positions != sorted(positions):
        errors.append("required phrases are out of order")
    return errors


def _generate_free(
    title: str,
    anchor: str,
    goal: str,
    scene_id: int,
    visual_instruction: str = "",
    learner_context: str = "",
    word_lo: int = 40,
    word_hi: int = 60,
) -> str:
    """Fallback: generate narration without phrase constraints."""
    client = NvidiaClient()
    prompt = (
        f"{learner_context}\n\n"
        f"Write a {word_lo}-{word_hi} word educational narration for: {title}. "
        f"Anchor example: {anchor}. Goal: {goal}. "
        f"Visual instruction: {visual_instruction}. "
        "Return narration text only: no reasoning, analysis, planning commentary, or JSON. "
        "Make it unique to this scene; do not repeat phrasing from other scenes."
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            text = client.chat(
                NVIDIA_PLANNER_MODEL,
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=NVIDIA_NARRATION_MAX_TOKENS,
            ).strip()
        except Exception as exc:
            last_error = exc
            logger.warning("Scene %d free narration attempt %d failed: %s", scene_id, attempt + 1, exc)
            continue
        errors = _narration_validation_errors(text, [], word_lo, word_hi)
        if not errors:
            _save_narration(scene_id, text)
            return text
        last_error = ValueError("; ".join(errors))
        logger.warning("Scene %d free narration attempt %d failed validation: %s", scene_id, attempt + 1, last_error)
    raise RuntimeError(
        f"Scene {scene_id} narration generation failed after 3 attempts: {last_error}"
    ) from last_error


def _save_narration(scene_id: int, narration: str) -> None:
    txt_path = PATHS["audio"] / f"scene_{scene_id}.txt"
    txt_path.write_text(narration, encoding="utf-8")
    logger.debug("Narration saved to %s", txt_path)
