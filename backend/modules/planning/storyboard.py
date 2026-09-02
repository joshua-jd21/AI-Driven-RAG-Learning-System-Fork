"""Storyboard planner: generates the conceptual arc for a topic.

Outputs an ordered list of 5 scenes, each with a concept_template and
anchor_example, forming a progressive educational narrative:
  Scene 1: intro (overview + key term)
  Scenes 2-4: core concept templates (mechanics)
  Scene 5: summary

Scene 2-4 templates and anchor examples MUST all differ — enforced both via
prompt rules and post-validation that rewrites duplicates to "freeform".
"""
from __future__ import annotations

import json
from typing import Any

from modules.config import NVIDIA_PLANNER_MODEL, PATHS, get_logger
from modules.llm.nvidia_client import NvidiaClient
from modules.planning.profile_context import format_learner_context
from modules.templates import (
    EXPLAIN_TEMPLATE_IDS,
    MECHANICS_TEMPLATE_IDS,
    VALID_TEMPLATE_IDS,
)
from modules.planning.chemistry_router import (
    CHEMISTRY_TEMPLATE_IDS,
    route_chemistry_template,
)
from modules.retrieval.pageindex_retriever import (
    format_prerequisites_for_prompt,
    format_sections_for_prompt,
)

logger = get_logger(__name__)

STORYBOARD_SYSTEM = """You are an educational video director.
Return one valid JSON array of 5 scenes.
No timing fields, markdown fences, or commentary."""

STORYBOARD_PROMPT = """Design a 5-scene educational arc for topic: {topic}

CURRICULUM CONTEXT:
{curriculum_context}

{prerequisite_block}

Rules:
- Prefer curriculum evidence when present.
- If prerequisites exist, teach them before dependent ideas.
- If curriculum is absent, use only broadly established, non-advanced knowledge.
- Adapt the arc to the learner context.

LESSON SUBJECT: {subject}
LESSON TOPIC: {topic}

{learner_context}

concept_template must always be one of the canonical registered template IDs listed below. Use only the provided canonical template names. Never prefix, suffix, categorize, or invent template names. Family names below are guidance only and must never be output as values. Prefer the most specific registered ID for the scene; use freeform only when no registered ID fits.

Registered template IDs:
- Mechanics: {mechanics_ids}
- Explain: {explain_ids}
- Chemistry: {chemistry_ids}
- freeform

Arc: hook → visual_intuition → formal_concept → worked_example → summary.

Every scene must include a `visual_instruction` field that names concrete
animated objects, labels, and transitions. The instruction should be specific
enough for a Manim generator to draw the scene without inventing unrelated
content.

Return a JSON array of exactly 5 objects:
[
  {{
    "scene_id": 1,
    "concept_template": "intro",
    "scene_role": "hook",
    "title": "<topic> — Overview",
    "anchor_example": "<short hook>",
    "learning_goal": "introduce the concept",
    "subtitle": "<one-line tagline>",
    "key_term": "<central term>",
    "visual_instruction": "<concrete on-screen objects and animation beat>"
  }},
  {{
    "scene_id": 2,
    "concept_template": "<registered_id>",
    "scene_role": "visual_intuition",
    "title": "...",
    "anchor_example": "<example>",
    "learning_goal": "<goal>",
    "visual_instruction": "<concrete on-screen objects and animation beat>"
  }},
  {{
    "scene_id": 3,
    "concept_template": "<registered_id>",
    "scene_role": "formal_concept",
    "title": "...",
    "anchor_example": "<example>",
    "learning_goal": "<goal>",
    "visual_instruction": "<concrete on-screen objects and animation beat>"
  }},
  {{
    "scene_id": 4,
    "concept_template": "<registered_id>",
    "scene_role": "worked_example",
    "title": "...",
    "anchor_example": "<example>",
    "learning_goal": "<goal>",
    "visual_instruction": "<concrete on-screen objects and animation beat>"
  }},
  {{
    "scene_id": 5,
    "concept_template": "summary",
    "scene_role": "summary",
    "title": "Key Takeaways",
    "anchor_example": "all scenarios",
    "learning_goal": "consolidate learning",
    "summary_points": ["...", "...", "..."],
    "visual_instruction": "<relationship diagram or summary visual>"
  }}
]
"""

STORYBOARD_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 5,
    "maxItems": 5,
    "items": {
        "type": "object",
        "required": [
            "scene_id",
            "concept_template",
            "scene_role",
            "title",
            "anchor_example",
            "learning_goal",
            "visual_instruction",
        ],
        "properties": {
            "scene_id": {"type": "integer"},
            "concept_template": {"type": "string", "enum": VALID_TEMPLATE_IDS},
            "scene_role": {"type": "string"},
            "title": {"type": "string"},
            "anchor_example": {"type": "string"},
            "learning_goal": {"type": "string"},
            "subtitle": {"type": "string"},
            "key_term": {"type": "string"},
            "summary_points": {
                "type": "array",
                "items": {"type": "string"},
            },
            "visual_instruction": {"type": "string"},
        },
        "additionalProperties": False,
    },
}


def _build_curriculum_anchor(curriculum_context: str, curriculum_sections: list | None) -> str:
    """Prepend a rich visual metadata block for tighter LLM alignment.

    Uses format_sections_for_prompt to include semantic_tags,
    visualizable_elements, and prerequisites so the LLM can select
    chemistry-appropriate templates and order scenes pedagogically.
    """
    if not curriculum_sections:
        return curriculum_context
    visual_block = format_sections_for_prompt(curriculum_sections)
    prereq_block = format_prerequisites_for_prompt(curriculum_sections)
    parts = [visual_block]
    if prereq_block:
        parts.append(prereq_block)
    header = "\n\n".join(parts)
    if curriculum_context:
        return f"{header}\n\nDETAILED CONTEXT:\n{curriculum_context}"
    return header


def build_storyboard(
    topic: str,
    curriculum_context: str = "",
    curriculum_sections: list | None = None,
    learner_profile: dict[str, Any] | None = None,
    subject: str = "Physics",
    learner_context: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a 5-scene storyboard arc for the given topic."""
    logger.info("Building storyboard for topic: %s", topic)
    client = NvidiaClient()

    enriched_context = _build_curriculum_anchor(curriculum_context, curriculum_sections)
    prerequisite_block = format_prerequisites_for_prompt(curriculum_sections or [])
    if not prerequisite_block:
        prerequisite_block = "(No prerequisite ordering data available.)"

    mechanics_middle = [
        t for t in MECHANICS_TEMPLATE_IDS if t not in ("intro", "summary")
    ]
    mechanics_ids = ", ".join(mechanics_middle)
    explain_ids = ", ".join(EXPLAIN_TEMPLATE_IDS)
    chemistry_ids = ", ".join(CHEMISTRY_TEMPLATE_IDS)
    learner_context_text = learner_context or format_learner_context(learner_profile, topic, subject)
    prompt = STORYBOARD_PROMPT.format(
        topic=topic,
        curriculum_context=enriched_context,
        prerequisite_block=prerequisite_block,
        subject=subject,
        learner_context=learner_context_text,
        mechanics_ids=mechanics_ids,
        explain_ids=explain_ids,
        chemistry_ids=chemistry_ids,
    )
    messages = [
        {"role": "system", "content": STORYBOARD_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    request_overrides = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "StoryboardScenes",
                "schema": STORYBOARD_RESPONSE_SCHEMA,
            },
        },
    }
    raw = client.chat_json(
        NVIDIA_PLANNER_MODEL,
        messages,
        temperature=0.55,
        max_tokens=4096,
        extra_body=request_overrides,
    )
    logger.info("Storyboard parsed payload: %s", _summarize_storyboard_payload(raw))

    raw_scenes = _unwrap_storyboard_scenes(raw)

    validated = [
        _validate_entry(
            entry,
            idx,
            topic=topic,
            curriculum_sections=curriculum_sections,
            subject=subject,
        )
        for idx, entry in enumerate(raw_scenes, start=1)
    ]
    validated[0]["concept_template"] = "intro"
    validated[0]["scene_role"] = "hook"
    validated[-1]["concept_template"] = "summary"
    validated[-1]["scene_role"] = "summary"
    validated = _enforce_scene_roles(validated)
    validated = _enforce_distinct_middle(validated)

    out = PATHS["json"] / "storyboard.json"
    out.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Storyboard saved: %s (%d scenes)", out, len(validated))
    return validated


def _unwrap_storyboard_scenes(raw: Any) -> list[dict[str, Any]]:
    """Accept the documented raw list or the known `{\"scenes\": [...]}` envelope."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "scenes" in raw:
        scenes = raw["scenes"]
        if isinstance(scenes, list):
            return scenes
        raise ValueError(
            f"Storyboard LLM returned 'scenes' as {type(scenes)}, expected list"
        )
    raise ValueError(f"Storyboard LLM returned {type(raw)}, expected list")


def _summarize_storyboard_payload(raw: Any, preview_limit: int = 1000) -> str:
    """Create a safe, compact summary for storyboard parse diagnostics."""
    if isinstance(raw, dict):
        preview_source = json.dumps(raw, ensure_ascii=False, default=str)
        preview = preview_source[:preview_limit]
        if len(preview_source) > preview_limit:
            preview += "..."
        return f"type=dict keys={list(raw.keys())[:10]} preview={preview}"
    if isinstance(raw, list):
        preview_source = json.dumps(raw, ensure_ascii=False, default=str)
        preview = preview_source[:preview_limit]
        if len(preview_source) > preview_limit:
            preview += "..."
        return f"type=list len={len(raw)} preview={preview}"

    preview = repr(raw)
    if len(preview) > preview_limit:
        preview = preview[:preview_limit] + "..."
    return f"type={type(raw).__name__} preview={preview}"


_VALID_SCENE_ROLES = frozenset({
    "hook", "visual_intuition", "formal_concept", "worked_example", "summary",
})

_MIDDLE_ROLE_ORDER = ["visual_intuition", "formal_concept", "worked_example"]


def _validate_entry(
    entry: dict[str, Any],
    default_id: int,
    topic: str = "",
    curriculum_sections: list | None = None,
    subject: str = "Physics",
) -> dict[str, Any]:
    scene_id = int(entry.get("scene_id", default_id))
    template = str(entry.get("concept_template", "intro"))

    # Accept chemistry templates even if not in VALID_TEMPLATE_IDS yet (registered later)
    all_known = set(VALID_TEMPLATE_IDS) | set(CHEMISTRY_TEMPLATE_IDS)
    if template not in all_known:
        logger.warning(
            "Scene %d has unknown template '%s'; falling back to 'freeform'",
            scene_id, template,
        )
        template = "freeform"

    # Chemistry router override: if topic/tags suggest chemistry and template is generic,
    # replace with the most appropriate chemistry template.
    scene_role = str(entry.get("scene_role", ""))
    subject_lower = subject.strip().lower()
    chemistry_allowed = subject_lower == "chemistry"

    if subject_lower != "chemistry" and template in CHEMISTRY_TEMPLATE_IDS:
        logger.warning(
            "Scene %d requested chemistry template '%s' under subject=%r; downgrading to 'freeform'",
            scene_id, template, subject,
        )
        template = "freeform"

    if chemistry_allowed and template not in ("intro", "summary") and template not in CHEMISTRY_TEMPLATE_IDS:
        top_section = (curriculum_sections or [{}])[0] if curriculum_sections else {}
        chem_override = route_chemistry_template(
            topic=topic,
            scene_role=scene_role,
            semantic_tags=top_section.get("semantic_tags", []),
            visualizable_elements=top_section.get("visualizable_elements", []),
        )
        if chem_override:
            logger.info(
                "Scene %d: chemistry router overrides template '%s' → '%s'",
                scene_id, template, chem_override,
            )
            template = chem_override

    result: dict[str, Any] = {
        "scene_id": scene_id,
        "concept_template": template,
        "subject": subject,
        "title": str(entry.get("title", entry.get("anchor_example", f"Scene {scene_id}"))),
        "anchor_example": str(entry.get("anchor_example", "")),
        "learning_goal": str(entry.get("learning_goal", "")),
    }

    # Preserve scene_role if LLM provided a valid one
    if scene_role in _VALID_SCENE_ROLES:
        result["scene_role"] = scene_role

    if "subtitle" in entry:
        result["subtitle"] = str(entry["subtitle"])
    if "key_term" in entry:
        result["key_term"] = str(entry["key_term"])
    if "visual_instruction" in entry:
        result["visual_instruction"] = str(entry["visual_instruction"])
    if "summary_points" in entry and isinstance(entry["summary_points"], list):
        result["summary_points"] = [str(p) for p in entry["summary_points"][:4]]
    return result


def _enforce_scene_roles(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enforce pedagogical role ordering.

    Scene 1 → hook, scene 5 → summary.
    Scenes 2–4 must be visual_intuition, formal_concept, worked_example in that order.
    """
    if not scenes:
        return scenes
    scenes[0]["scene_role"] = "hook"
    if len(scenes) >= 5:
        scenes[-1]["scene_role"] = "summary"
    for i, scene in enumerate(scenes[1:-1], start=0):
        assigned = scene.get("scene_role", "")
        if assigned not in _VALID_SCENE_ROLES or assigned in ("hook", "summary"):
            scene["scene_role"] = _MIDDLE_ROLE_ORDER[i] if i < len(_MIDDLE_ROLE_ORDER) else "formal_concept"
    return scenes


def _enforce_distinct_middle(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure scenes 2-4 have distinct templates AND distinct anchor examples.

    Duplicates are rewritten to use the freeform template so the LLM generates
    a unique visual for that scene instead of replaying the same animation.
    """
    seen_templates: set[str] = set()
    seen_anchors: set[str] = set()
    for s in scenes[1:-1]:
        tpl = s.get("concept_template", "freeform")
        anchor_key = s.get("anchor_example", "").strip().lower()
        if tpl in seen_templates or tpl in ("intro", "summary"):
            logger.warning(
                "Scene %d duplicated template '%s'; rewriting to 'freeform'",
                s["scene_id"], tpl,
            )
            s["concept_template"] = "freeform"
            tpl = "freeform"
        seen_templates.add(tpl)
        if anchor_key and anchor_key in seen_anchors:
            logger.warning(
                "Scene %d duplicated anchor_example '%s'; rewriting to freeform",
                s["scene_id"], anchor_key,
            )
            s["concept_template"] = "freeform"
            s["anchor_example"] = f"{s['anchor_example']} (alternate framing)"
        seen_anchors.add(s.get("anchor_example", "").strip().lower())
    return scenes
