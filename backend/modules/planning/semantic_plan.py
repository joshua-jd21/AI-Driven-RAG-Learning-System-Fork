"""Semantic plan generator.

For each storyboard entry, the LLM fills the template's slot schema:
  - which assets to use (from ASSET_REGISTRY)
  - which event types (from the template's ALLOWED_EVENTS)
  - anchor_phrases that MUST later appear verbatim in the narration

Strict validation:
  - asset_ids must be in ASSET_REGISTRY
  - event types must be in the template's ALLOWED_EVENTS
  - anchor_phrases are recorded (the narration writer will embed them)
"""
from __future__ import annotations

import json
import re
from typing import Any

from modules.assets import ASSET_REGISTRY
from modules.assets.mechanics import ASSET_PARAM_SCHEMAS, validate_params
from modules.config import (
    NVIDIA_PLANNER_MODEL,
    NVIDIA_SEMANTIC_PLAN_MAX_TOKENS,
    PATHS,
    get_logger,
)
from modules.llm.nvidia_client import NvidiaClient
from modules.planning.asset_registry import get_registry
from modules.planning.profile_context import format_learner_context
from modules.templates import TEMPLATES
from modules.templates.explain import EXPLAIN_TEMPLATE_IDS
from modules.templates.explain._base import merge_content
from modules.templates.chemistry import CHEMISTRY_TEMPLATE_IDS

logger = get_logger(__name__)
_VISUAL_ACTIONS = frozenset({"reveal", "highlight", "transform", "motion", "compare", "hold"})
_GENERIC_ROLE_NAMES = frozenset({
    "object", "item", "thing", "subject", "main", "visual", "element", "target",
})

SEMANTIC_PLAN_SYSTEM = """You are an educational animation director.
You assign visual representations to the exact ideas spoken in each scene.
NEVER include run_time, duration, seconds, or timing fields.
Return ONLY the single JSON object requested by the schema.
Do not provide reasoning, analysis, commentary, multiple candidate plans, or markdown fences.
Never return a top-level JSON array."""

NARRATION_GROUNDED_EVENT_RULES = """NARRATION-GROUNDED VISUAL CONTRACT:
Each event is a narration-grounded visual stage, not a decorative animation.
For every event, first identify the idea expressed by its anchor_phrase, then specify:
- narration_reference: the exact anchor phrase this stage supports
- visual_goal: what the learner should understand from the spoken idea
- visible_objects: concrete objects or terms that must be visible
- visual_state: the persistent state that should remain after this stage
- action: exactly one primitive from {reveal, highlight, transform, motion, compare, hold}
- action_reason: why that action teaches the narrated idea
- emphasis_targets: visible_objects or terms to emphasize during the action
- persistence_after_action: true unless the narration explicitly removes the concept
Use the curriculum and storyboard content as the source of meaning. Do not invent
unrelated motion, camera movement, arrows, or formulas. A generic action is valid only
when its fields explain the specific narrated concept. Events must be ordered by the
order their anchor phrases should occur in the narration.
"""

SEMANTIC_PLAN_EXPLAIN_PROMPT = """Fill the semantic plan for this CHALKBOARD EXPLANATION scene.

CURRICULUM CONTEXT:
{curriculum_context}
Use the curriculum context as the primary source of truth.
Use terminology, formulas, labels, and concepts from the curriculum context whenever possible.
Do not invent unsupported facts.

LESSON SUBJECT: {subject}
LESSON TOPIC: {topic}

{learner_context}

STORYBOARD ENTRY:
{storyboard_entry}

The storyboard entry includes a `visual_instruction` field. Preserve it and use
it to choose concrete mobjects, labels, and transitions that match the lesson.

TEMPLATE: {template_id}
TEMPLATE ALLOWED EVENTS: {allowed_events}

LEGAL ASSET ROLES FOR THIS TEMPLATE:
{supported_roles}

ASSET PARAMETER CONTRACT:
{asset_parameters}

ASSET ROLE CONTRACT:
- `role` MUST be exactly one of the legal asset roles listed above.
- `role` is a renderer/template-supported semantic slot, not the object's name.
- `asset_id` is the actual visual object identity and must not be used as `role`.
- `params` is scoped to the matching `asset_id`; use only the parameter names
  listed on that asset's contract row. Never borrow a field from another asset.
- Never invent roles such as `block`, `arrow_force`, or `velocity_indicator`.
- If the list says there are no legal roles, `assets` MUST be [] and describe
  template-native visuals through `content`, `visible_objects`, and events.

{grounded_event_rules}

CONTENT SCHEMA (fill the "content" object exactly in this shape):
{content_schema}

For "equation" fields use simple LaTeX with DOUBLE backslashes in JSON (e.g. "W = F \\\\cdot d").

RULES:
1. Return a "content" object matching CONTENT SCHEMA — all strings must be specific to this
   topic and anchor_example (not generic placeholders).
2. "assets" must be an empty array [] because this template uses native visuals.
3. Provide 2-4 "events" using only ALLOWED EVENTS; each needs an anchor_phrase (3-7 words)
   that will appear VERBATIM in the narration later.
4. If Prerequisites are listed in the curriculum context, include a brief recap phrase in one
   event's anchor_phrase when the scene builds on a prior concept.
5. "phase" is "before", "on", or "after"; "importance" is 1-5.

Return ONLY this JSON shape:
{{
  "scene_id": {scene_id},
  "concept_template": "{template_id}",
  "title": "<short scene title>",
  "anchor_example": "{anchor_example}",
  "content": <object matching CONTENT SCHEMA>,
  "assets": [],
  "events": [
    {{
      "id": "e0",
      "type": "<from ALLOWED EVENTS>",
      "targets": [],
      "anchor_phrase": "<3-7 verbatim words for narration>",
      "narration_reference": "<same anchor phrase>",
      "visual_goal": "<specific learner-facing meaning>",
      "visible_objects": ["<object or term>"],
      "visual_state": "<state that persists after this stage>",
      "action": "<reveal|highlight|transform|motion|compare|hold>",
      "action_reason": "<why this action represents the narrated idea>",
      "emphasis_targets": ["<object or term>"],
      "persistence_after_action": true,
      "phase": "on",
      "importance": 3
    }}
  ]
}}"""

SEMANTIC_PLAN_CHEMISTRY_PROMPT = """Fill the semantic plan for this CHEMISTRY ANIMATION scene.

CURRICULUM CONTEXT:
{curriculum_context}
Use the curriculum context as the PRIMARY source of truth.
Use textbook terminology, element symbols, formulas, and model names from the curriculum context.
Do not invent unsupported chemical facts.

LESSON SUBJECT: {subject}
LESSON TOPIC: {topic}

{learner_context}

STORYBOARD ENTRY:
{storyboard_entry}

The storyboard entry includes a `visual_instruction` field. Preserve it and use
it to choose concrete chemistry visuals, labels, and transitions that match the lesson.

TEMPLATE: {template_id}
TEMPLATE ALLOWED EVENTS: {allowed_events}

LEGAL ASSET ROLES FOR THIS TEMPLATE:
{supported_roles}

ASSET PARAMETER CONTRACT:
{asset_parameters}

ASSET ROLE CONTRACT:
- `role` MUST be exactly one of the legal asset roles listed above.
- `role` is a renderer/template-supported semantic slot, not the object's name.
- `asset_id` is the actual visual object identity and must not be used as `role`.
- `params` is scoped to the matching `asset_id`; use only the parameter names
  listed on that asset's contract row. Never borrow a field from another asset.
- Never invent roles such as `block`, `arrow_force`, or `velocity_indicator`.
- If the list says there are no legal roles, `assets` MUST be [] and describe
  template-native visuals through `content`, `visible_objects`, and events.

{grounded_event_rules}

CONTENT SCHEMA — fill the "content" object exactly in this shape:
{content_schema}

CRITICAL RULES:
1. "content" must match the CONTENT SCHEMA exactly — use real element symbols, correct atomic numbers, and accurate electron shell counts from the curriculum context.
2. "assets" must be an empty array [] because this template uses native visuals.
3. Provide 3-5 "events" using only ALLOWED EVENTS; each event needs an anchor_phrase (3-7 words) that will appear VERBATIM in the narration.
4. For atomic_structure: shells must be a list of integers (e.g. [2, 8, 1] for sodium).
5. "phase" is "before", "on", or "after"; "importance" is 1-5 (5 = most critical visual beat).
6. Do NOT use physics assets like block, hockey_puck, or car — these are chemistry scenes.
7. If Prerequisites are listed in the curriculum context, weave a brief recap into one event's
   anchor_phrase when the scene depends on prior knowledge (e.g. "recall Rutherford's nucleus").

Return ONLY this JSON shape:
{{
  "scene_id": {scene_id},
  "concept_template": "{template_id}",
  "title": "<specific scene title from curriculum>",
  "anchor_example": "{anchor_example}",
  "content": <object matching CONTENT SCHEMA>,
  "assets": [],
  "events": [
    {{
      "id": "e0",
      "type": "<from ALLOWED EVENTS>",
      "targets": [],
      "anchor_phrase": "<3-7 verbatim words that will appear in narration>",
      "narration_reference": "<same anchor phrase>",
      "visual_goal": "<specific chemical meaning being taught>",
      "visible_objects": ["<atom, shell, bond, or term>"],
      "visual_state": "<persistent chemical visual state>",
      "action": "<reveal|highlight|transform|motion|compare|hold>",
      "action_reason": "<why this action represents the narrated idea>",
      "emphasis_targets": ["<object or term>"],
      "persistence_after_action": true,
      "phase": "on",
      "importance": 4
    }}
  ]
}}"""

SEMANTIC_PLAN_PROMPT = """Fill the semantic plan for this scene.

CURRICULUM CONTEXT:
{curriculum_context}
Use the curriculum context as the primary source of truth.
Use textbook terminology whenever available.

LESSON SUBJECT: {subject}
LESSON TOPIC: {topic}

{learner_context}

STORYBOARD ENTRY:
{storyboard_entry}

The storyboard entry includes a `visual_instruction` field. Preserve it and use
it to choose concrete objects, equations, and transitions that match the lesson.

TEMPLATE: {template_id}
TEMPLATE ALLOWED EVENTS: {allowed_events}

LEGAL ASSET ROLES FOR THIS TEMPLATE:
{supported_roles}

ASSET PARAMETER CONTRACT:
{asset_parameters}

ASSET ROLE CONTRACT:
- `role` MUST be exactly one of the legal asset roles listed above.
- `role` is a renderer/template-supported semantic slot, not the object's name.
- `asset_id` is the actual visual object identity and must not be used as `role`.
- `params` is scoped to the matching `asset_id`; use only the parameter names
  listed on that asset's contract row. Never borrow a field from another asset.
- Never invent roles such as `block`, `arrow_force`, or `velocity_indicator`.
- If the list says there are no legal roles, `assets` MUST be [] and describe
  template-native visuals through `content`, `visible_objects`, and events.

{grounded_event_rules}

AVAILABLE ASSET IDs: {asset_ids}

RULES:
1. Return exactly one JSON object, never a top-level array, candidate list, or commentary.
2. Fill the "assets" array using only asset_ids from the AVAILABLE list and roles from LEGAL ASSET ROLES.
3. Each event "type" must be from ALLOWED EVENTS.
4. Each event "anchor_phrase" must be 3-7 words that will appear VERBATIM
   in the narration script later. Choose clear, descriptive phrases.
5. "phase" must be "before", "on", or "after" (when relative to the phrase).
6. "importance" is 1-5 (5 = most critical, gets longest animation time).
7. instance_id must be unique snake_case (e.g. "puck_a", "ice_surface").
8. Do NOT invent asset_ids — only use those listed.

Return ONLY this JSON shape:
{{
  "scene_id": {scene_id},
  "concept_template": "{template_id}",
  "title": "<short scene title>",
  "anchor_example": "{anchor_example}",
  "assets": [
    {{
      "role": "<slot role matching template>",
      "asset_id": "<from AVAILABLE list>",
      "instance_id": "<unique snake_case>",
      "params": {{<asset-specific params like label, color, direction>}}
    }}
  ],
  "events": [
    {{
      "id": "e0",
      "type": "<from ALLOWED EVENTS>",
      "targets": ["<instance_id>"],
      "anchor_phrase": "<3-7 verbatim words for narration>",
      "narration_reference": "<same anchor phrase>",
      "visual_goal": "<specific meaning taught by this phrase>",
      "visible_objects": ["<concrete object or term>"],
      "visual_state": "<persistent visual state>",
      "action": "<reveal|highlight|transform|motion|compare|hold>",
      "action_reason": "<why this action represents the narrated idea>",
      "emphasis_targets": ["<object or term>"],
      "persistence_after_action": true,
      "phase": "on",
      "importance": 3
    }}
  ]
}}"""


def build_semantic_plan(
    storyboard_entry: dict[str, Any],
    curriculum_context: str = "",
    curriculum_sections: list | None = None,
    learner_profile: dict[str, Any] | None = None,
    topic: str = "",
    subject: str = "Physics",
    learner_context: str | None = None,
) -> dict[str, Any]:
    """Generate and validate a semantic plan for one storyboard entry."""
    scene_id = storyboard_entry["scene_id"]
    template_id = storyboard_entry["concept_template"]
    logger.info("Building semantic plan for scene %d (template=%s)", scene_id, template_id)

    template_cls = TEMPLATES.get(template_id)
    if template_cls is None:
        raise ValueError(f"Unknown template '{template_id}'")

    allowed_events = sorted(getattr(template_cls, "ALLOWED_EVENTS", set()))
    asset_ids = sorted(ASSET_REGISTRY.keys())
    anchor_example = storyboard_entry.get("anchor_example", "")
    learner_context_text = learner_context or format_learner_context(
        learner_profile, topic or storyboard_entry.get("title", ""), subject
    )
    content_schema = getattr(template_cls, "CONTENT_SCHEMA", None)
    supported_roles = _format_supported_roles(getattr(template_cls, "SLOTS", {}))
    supported_slots = getattr(template_cls, "SLOTS", {})
    supported_asset_ids = sorted({
        str(asset_id)
        for asset_ids_for_role in supported_slots.values()
        for asset_id in _string_list(asset_ids_for_role)
    })
    asset_parameters = _format_asset_parameters(
        supported_asset_ids if supported_slots else []
    )
    is_explain = template_id in EXPLAIN_TEMPLATE_IDS
    is_chemistry = template_id in CHEMISTRY_TEMPLATE_IDS

    # Build visual metadata block from curriculum sections
    sections_visual_metadata = ""
    if curriculum_sections:
        from modules.retrieval.pageindex_retriever import format_sections_for_prompt
        sections_visual_metadata = format_sections_for_prompt(curriculum_sections)

    # Enrich curriculum context with visual metadata for planners
    enriched_context = curriculum_context
    if sections_visual_metadata:
        enriched_context = f"{sections_visual_metadata}\n\n{curriculum_context}"

    client = NvidiaClient()
    if is_chemistry and content_schema:
        prompt = SEMANTIC_PLAN_CHEMISTRY_PROMPT.format(
            curriculum_context=enriched_context,
            storyboard_entry=json.dumps(storyboard_entry, indent=2),
            template_id=template_id,
            allowed_events=", ".join(allowed_events),
            content_schema=content_schema,
            grounded_event_rules=NARRATION_GROUNDED_EVENT_RULES,
            supported_roles=supported_roles,
            asset_parameters=asset_parameters,
            scene_id=scene_id,
            anchor_example=anchor_example,
            subject=subject,
            topic=topic or storyboard_entry.get("title", ""),
            learner_context=learner_context_text,
        )
    elif is_explain and content_schema:
        prompt = SEMANTIC_PLAN_EXPLAIN_PROMPT.format(
            curriculum_context=enriched_context,
            storyboard_entry=json.dumps(storyboard_entry, indent=2),
            template_id=template_id,
            allowed_events=", ".join(allowed_events),
            content_schema=content_schema,
            grounded_event_rules=NARRATION_GROUNDED_EVENT_RULES,
            supported_roles=supported_roles,
            asset_parameters=asset_parameters,
            scene_id=scene_id,
            anchor_example=anchor_example,
            subject=subject,
            topic=topic or storyboard_entry.get("title", ""),
            learner_context=learner_context_text,
        )
    else:
        prompt = SEMANTIC_PLAN_PROMPT.format(
            curriculum_context=enriched_context,
            storyboard_entry=json.dumps(storyboard_entry, indent=2),
            template_id=template_id,
            allowed_events=", ".join(allowed_events),
            asset_ids=", ".join(asset_ids),
            grounded_event_rules=NARRATION_GROUNDED_EVENT_RULES,
            supported_roles=supported_roles,
            asset_parameters=_format_asset_parameters(supported_asset_ids or asset_ids),
            scene_id=scene_id,
            anchor_example=anchor_example,
            subject=subject,
            topic=topic or storyboard_entry.get("title", ""),
            learner_context=learner_context_text,
        )

    request_overrides = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "SemanticPlan",
                "schema": _semantic_plan_response_schema(
                    template_id=template_id,
                    allowed_events=allowed_events,
                    supported_slots=supported_slots,
                ),
            },
        },
    }
    messages = [
        {"role": "system", "content": SEMANTIC_PLAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    request_messages = messages
    plan = None
    last_error: ValueError | json.JSONDecodeError | None = None
    for attempt in range(2):
        try:
            raw = client.chat_json(
                NVIDIA_PLANNER_MODEL,
                request_messages,
                temperature=0.3,
                max_tokens=NVIDIA_SEMANTIC_PLAN_MAX_TOKENS,
                extra_body=request_overrides,
            )
            plan = _validate_plan(
                raw,
                scene_id,
                template_id,
                allowed_events,
                storyboard_entry=storyboard_entry,
                supported_slots=supported_slots,
            )
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                retry_contract = _format_asset_parameters(
                    supported_asset_ids if supported_slots else []
                )
                request_messages = messages + [{
                    "role": "user",
                    "content": (
                        "The previous semantic plan was rejected by strict validation.\n"
                        f"VALIDATION ERROR: {exc}\n\n"
                        "Regenerate the complete JSON object. For every asset, use only "
                        "the parameters listed for that asset below; remove any "
                        "unsupported parameter name. Do not use semantic synonyms "
                        "such as magnitude, strength, force, or size unless that exact "
                        "name appears in the contract.\n\n"
                        "PER-ASSET PARAMETER CONTRACT:\n"
                        f"{retry_contract}"
                    ),
                }]
                logger.warning(
                    "Scene %d semantic plan attempt was invalid or truncated (%s); retrying",
                    scene_id, exc,
                )

    if plan is None:
        if is_explain and last_error is not None:
            logger.warning(
                "Scene %d explain plan JSON failed after retry (%s); using fallback content",
                scene_id, last_error,
            )
            plan = _validate_plan(
                _fallback_explain_raw(storyboard_entry, template_id, scene_id),
                scene_id,
                template_id,
                allowed_events,
                storyboard_entry=storyboard_entry,
                supported_slots=supported_slots,
            )
        else:
            assert last_error is not None
            raise last_error
    plan["subject"] = subject or storyboard_entry.get("subject", "")

    for field in ("subtitle", "key_term", "summary_points", "learning_goal"):
        if field in storyboard_entry and field not in plan:
            plan[field] = storyboard_entry[field]
    if "visual_instruction" in storyboard_entry and "visual_instruction" not in plan:
        plan["visual_instruction"] = storyboard_entry["visual_instruction"]

    plan["_learner_context"] = learner_context_text

    # Register all assets in the global registry
    registry = get_registry()
    for asset in plan.get("assets", []):
        registry.register(
            instance_id=asset["instance_id"],
            asset_id=asset["asset_id"],
            params=asset.get("params", {}),
            scene_id=scene_id,
        )

    out = PATHS["json"] / f"semantic_plan_{scene_id}.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Semantic plan saved: %s (%d assets, %d events)",
        out, len(plan.get("assets", [])), len(plan.get("events", []))
    )
    return plan


def build_all_semantic_plans(
    storyboard: list[dict[str, Any]],
    curriculum_context: str = "",
    curriculum_sections: list | None = None,
    learner_profile: dict[str, Any] | None = None,
    topic: str = "",
    subject: str = "Physics",
    learner_context: str | None = None,
) -> list[dict[str, Any]]:
    return [
        build_semantic_plan(
            entry,
            curriculum_context=curriculum_context,
            curriculum_sections=curriculum_sections,
            learner_profile=learner_profile,
            topic=topic,
            subject=subject,
            learner_context=learner_context,
        )
        for entry in storyboard
    ]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def _fallback_explain_raw(
    storyboard_entry: dict[str, Any],
    template_id: str,
    scene_id: int,
) -> dict[str, Any]:
    """Deterministic plan when LLM JSON is invalid (e.g. LaTeX backslashes)."""
    base = {
        "scene_id": scene_id,
        "concept_template": template_id,
        "title": storyboard_entry.get("title", f"Scene {scene_id}"),
        "anchor_example": storyboard_entry.get("anchor_example", ""),
        "learning_goal": storyboard_entry.get("learning_goal", ""),
        "visual_instruction": storyboard_entry.get("visual_instruction", ""),
        "assets": [],
        "content": merge_content(storyboard_entry, template_id),
        "events": [
            {
                "id": "e0",
                "type": "place_title",
                "targets": [],
                "anchor_phrase": "let us begin",
                "phase": "on",
                "importance": 3,
            },
            {
                "id": "e1",
                "type": "reveal",
                "targets": [],
                "anchor_phrase": "the key idea",
                "phase": "on",
                "importance": 4,
            },
            {
                "id": "e2",
                "type": "hold",
                "targets": [],
                "anchor_phrase": "in summary",
                "phase": "on",
                "importance": 2,
            },
        ],
    }
    return base


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _format_supported_roles(slots: dict[str, Any]) -> str:
    if not slots:
        return "(none; use template-native visuals and keep objects in visible_objects)"
    return "\n".join(
        f"- {role}: {', '.join(str(asset) for asset in assets)}"
        for role, assets in slots.items()
    )


def _format_asset_parameters(asset_ids: list[str]) -> str:
    if not asset_ids:
        return "(none; assets must be [] for this template)"
    lines: list[str] = []
    for asset_id in asset_ids:
        schema = ASSET_PARAM_SCHEMAS.get(asset_id, {})
        if not schema:
            lines.append(f"- {asset_id}: no parameters")
            continue
        fields = []
        for name, rule in schema.items():
            detail = rule["type"]
            if "enum" in rule:
                detail += f" in {rule['enum']}"
            if "minimum" in rule or "maximum" in rule:
                detail += f" [{rule.get('minimum', '-inf')}, {rule.get('maximum', 'inf')}]"
            fields.append(f"{name} ({detail})")
        lines.append(f"- {asset_id}: {', '.join(fields)}")
    return "\n".join(lines)


def _validate_text_controls(value: Any, path: str) -> None:
    """Reject control characters that cannot safely enter generated source."""
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_text_controls(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_text_controls(item, f"{path}[{index}]")
    elif isinstance(value, str):
        for char in value:
            if ord(char) < 0x20 and char not in {"\t", "\n", "\r"} or ord(char) == 0x7F:
                raise ValueError(
                    f"Semantic plan contains invalid control character "
                    f"U+{ord(char):04X} at {path}"
                )


def _semantic_plan_response_schema(
    template_id: str,
    allowed_events: list[str],
    supported_slots: dict[str, Any],
) -> dict[str, Any]:
    """Build the response contract from the selected template's canonical slots."""
    # Bind each parameter object to exactly one asset ID. A flat union of
    # parameter properties would allow a field belonging to one asset (for
    # example, velocity_indicator.magnitude) on another asset such as
    # arrow_force. Keep the variants shallow so the provider only has to
    # process a single discriminated array-item object.
    asset_variants: list[dict[str, Any]] = []
    for role, role_asset_ids in sorted(supported_slots.items()):
        for asset_id in sorted(_string_list(role_asset_ids)):
            asset_variants.append({
                "type": "object",
                "required": ["role", "asset_id", "instance_id", "params"],
                "properties": {
                    "role": {"type": "string", "enum": [role]},
                    "asset_id": {"type": "string", "enum": [asset_id]},
                    "instance_id": {"type": "string"},
                    "params": {
                        "type": "object",
                        "properties": ASSET_PARAM_SCHEMAS.get(asset_id, {}),
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            })

    if asset_variants:
        asset_item: dict[str, Any] = {"oneOf": asset_variants}
    else:
        # Keep a provider-safe item shape for native templates, whose array is
        # separately constrained to zero items below.
        asset_item = {
            "type": "object",
            "required": ["role", "asset_id", "instance_id", "params"],
            "properties": {
                "role": {"type": "string", "enum": []},
                "asset_id": {"type": "string", "enum": []},
                "instance_id": {"type": "string"},
                "params": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }
    assets_schema: dict[str, Any] = {
        "type": "array",
        "items": asset_item,
    }
    if not supported_slots:
        assets_schema["maxItems"] = 0

    event_schema = {
        "type": "object",
        "required": ["id", "type", "anchor_phrase", "action"],
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": allowed_events or ["hold"]},
            "targets": {"type": "array", "items": {"type": "string"}},
            "anchor_phrase": {"type": "string"},
            "narration_reference": {"type": "string"},
            "visual_goal": {"type": "string"},
            "visible_objects": {"type": "array", "items": {"type": "string"}},
            "visual_state": {"type": "string"},
            "action": {"type": "string", "enum": sorted(_VISUAL_ACTIONS)},
            "action_reason": {"type": "string"},
            "emphasis_targets": {"type": "array", "items": {"type": "string"}},
            "persistence_after_action": {"type": "boolean"},
            "phase": {"type": "string", "enum": ["before", "on", "after"]},
            "importance": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": [
            "scene_id", "concept_template", "title", "anchor_example", "assets", "events",
        ],
        "properties": {
            "scene_id": {"type": "integer"},
            "concept_template": {"type": "string", "enum": [template_id]},
            "title": {"type": "string"},
            "anchor_example": {"type": "string"},
            "assets": assets_schema,
            "events": {"type": "array", "items": event_schema},
            "content": {"type": "object"},
            "subtitle": {"type": "string"},
            "key_term": {"type": "string"},
            "learning_goal": {"type": "string"},
            "visual_instruction": {"type": "string"},
            "summary_points": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def _compatible_event_type(action: str, allowed_events: list[str]) -> str:
    """Keep legacy template lookups valid without losing the generic action."""
    if action in allowed_events:
        return action
    aliases = {
        "reveal": ("reveal", "show", "place", "place_title", "introduce"),
        "highlight": ("highlight", "highlight_term", "show_labels", "label"),
        "transform": ("transform", "change", "update"),
        "motion": ("motion", "move", "slide", "travel", "accelerate"),
        "compare": ("compare", "contrast"),
        "hold": ("hold",),
    }
    for candidate in aliases.get(action, ()):
        if candidate in allowed_events:
            return candidate
    if "hold" in allowed_events:
        return "hold"
    return allowed_events[0] if allowed_events else action


def _normalize_asset_role(
    role: str,
    asset_id: str,
    supported_slots: dict[str, Any],
) -> str | None:
    """Return a supported slot, mapping only when the asset has one clear slot."""
    if role in supported_slots:
        allowed_assets = _string_list(supported_slots[role])
        return role if asset_id in allowed_assets else None

    candidates = [
        slot
        for slot, allowed_assets in supported_slots.items()
        if asset_id in _string_list(allowed_assets)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _asset_role_is_compatible(role: str, asset_id: str) -> bool:
    """Reject obvious role/asset contradictions without a topic-specific map."""
    role_tokens = set(re.findall(r"[a-z0-9]+", role.lower()))
    if not role_tokens or role_tokens & _GENERIC_ROLE_NAMES:
        return True
    description = f"{asset_id} {ASSET_REGISTRY.get(asset_id, '')}".lower()
    asset_tokens = set(re.findall(r"[a-z0-9]+", description))
    return bool(role_tokens & asset_tokens)


def _validate_plan(
    raw: Any,
    scene_id: int,
    template_id: str,
    allowed_events: list[str],
    storyboard_entry: dict[str, Any] | None = None,
    supported_slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Semantic plan must be a dict, got {type(raw)}")

    _validate_text_controls(raw, "plan")

    # Normalise scene_id
    plan = dict(raw)
    plan["scene_id"] = scene_id
    plan["concept_template"] = template_id

    # Validate assets
    assets = plan.get("assets", [])
    if not isinstance(assets, list):
        plan["assets"] = []
        assets = []

    seen_instances: set[str] = set()
    valid_assets = []
    for asset in assets:
        if not isinstance(asset, dict):
            logger.warning("Scene %d: dropping malformed non-object asset", scene_id)
            continue
        asset_id = asset.get("asset_id", "")
        if asset_id not in ASSET_REGISTRY:
            message = f"Scene {scene_id}: unknown asset_id '{asset_id}'"
            logger.error(message)
            raise ValueError(message)
        role = str(asset.get("role", asset_id)).strip()
        if supported_slots is not None:
            normalized_role = _normalize_asset_role(role, asset_id, supported_slots)
            if normalized_role is None:
                message = (
                    f"Scene {scene_id}: asset '{asset_id}' role '{role}' is "
                    "unsupported by the selected template"
                )
                logger.error(message)
                raise ValueError(message)
            if normalized_role != role:
                logger.info(
                    "Scene %d: normalized asset role '%s' -> '%s' for asset '%s'",
                    scene_id, role, normalized_role, asset_id,
                )
                role = normalized_role
        if (supported_slots is None or role not in supported_slots) and not _asset_role_is_compatible(role, asset_id):
            message = f"Scene {scene_id}: asset_id '{asset_id}' is incompatible with role '{role}'"
            logger.error(message)
            raise ValueError(message)
        params = validate_params(asset_id, asset.get("params", {}))
        iid = str(asset.get("instance_id", f"{asset_id}_{scene_id}"))
        if iid in seen_instances:
            iid = f"{iid}_{len(seen_instances)}"
        seen_instances.add(iid)
        valid_assets.append({
            "role": role,
            "asset_id": asset_id,
            "instance_id": iid,
            "params": params,
        })
    plan["assets"] = valid_assets

    # Validate events
    events = plan.get("events", [])
    if not isinstance(events, list):
        plan["events"] = []
        events = []

    valid_events = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            logger.warning("Scene %d: dropping malformed non-object event", scene_id)
            continue
        raw_type = str(ev.get("type", "")).lower().strip()
        action = str(ev.get("action", raw_type if raw_type in _VISUAL_ACTIONS else "hold")).lower().strip()
        if action not in _VISUAL_ACTIONS:
            logger.warning(
                "Scene %d event '%s' has unknown semantic action '%s'; dropping",
                scene_id, ev.get("id", i), action,
            )
            continue
        etype = raw_type
        if allowed_events and etype not in allowed_events:
            compatible_type = _compatible_event_type(action, allowed_events)
            logger.info(
                "Scene %d event '%s': preserving semantic action '%s' and normalizing "
                "type '%s' to '%s' for template compatibility",
                scene_id, ev.get("id", i), action, raw_type, compatible_type,
            )
            etype = compatible_type
        valid_events.append({
            "id": str(ev.get("id", f"e{i}")),
            "type": etype,
            "semantic_type": raw_type,
            "targets": list(ev.get("targets", [])),
            "anchor_phrase": str(ev.get("anchor_phrase", "")),
            "phase": str(ev.get("phase", "on")),
            "importance": max(1, min(5, int(ev.get("importance", 3)))),
            "narration_reference": str(ev.get("narration_reference", ev.get("anchor_phrase", ""))),
            "visual_goal": str(ev.get("visual_goal", ev.get("anchor_phrase", ""))),
            "visible_objects": _string_list(ev.get("visible_objects", ev.get("targets", []))),
            "visual_state": str(ev.get("visual_state", "Keep the represented concept visible while it is explained.")),
            "action": action,
            "action_reason": str(ev.get("action_reason", "Keep the visual state connected to the narrated idea.")),
            "emphasis_targets": _string_list(ev.get("emphasis_targets", ev.get("targets", []))),
            "persistence_after_action": bool(ev.get("persistence_after_action", True)),
        })
    plan["events"] = valid_events

    # Ensure title
    if "title" not in plan or not plan["title"]:
        plan["title"] = str(raw.get("anchor_example", f"Scene {scene_id}"))

    plan["visual_objects"] = list(dict.fromkeys(
        item
        for event in valid_events
        for item in event.get("visible_objects", [])
    ))

    if template_id in EXPLAIN_TEMPLATE_IDS:
        # Explain templates render their visual objects directly, but valid
        # registry assets still belong in the normalized plan for consumers
        # that support them.
        base_entry = dict(storyboard_entry or {})
        base_entry.update({
            "title": plan.get("title", base_entry.get("title", "")),
            "learning_goal": plan.get("learning_goal", base_entry.get("learning_goal", "")),
            "anchor_example": plan.get("anchor_example", base_entry.get("anchor_example", "")),
        })
        if isinstance(plan.get("content"), dict):
            base_entry["content"] = plan["content"]
        plan["content"] = merge_content(base_entry, template_id)
        if "visual_instruction" in base_entry and "visual_instruction" not in plan:
            plan["visual_instruction"] = base_entry["visual_instruction"]

    elif template_id in CHEMISTRY_TEMPLATE_IDS:
        # Chemistry templates never use physics assets; preserve LLM-filled content dict.
        plan["assets"] = []
        if not isinstance(plan.get("content"), dict):
            plan["content"] = {}

    return plan
