from __future__ import annotations

import json

import pytest

import modules.planning.semantic_plan as semantic_plan


@pytest.mark.parametrize("action", ["reveal", "highlight", "transform", "motion", "compare", "hold"])
def test_all_generic_visual_actions_survive_normalization(action: str) -> None:
    result = semantic_plan._validate_plan(
        {
            "title": "Generic stage",
            "assets": [],
            "events": [{
                "id": "e0",
                "type": action,
                "anchor_phrase": "the key idea",
                "action": action,
                "visual_goal": "teach the key idea",
                "visible_objects": ["concept"],
                "visual_state": "concept remains visible",
                "action_reason": "the action represents the spoken idea",
                "emphasis_targets": ["concept"],
                "persistence_after_action": True,
            }],
        },
        scene_id=1,
        template_id="intro",
        allowed_events=["place_title", "highlight_term", "hold"],
    )

    assert result["events"][0]["action"] == action
    assert result["events"][0]["visual_goal"] == "teach the key idea"


def test_semantic_event_normalizes_grounded_visual_contract() -> None:
    result = semantic_plan._validate_plan(
        {
            "title": "State change",
            "assets": [],
            "events": [{
                "id": "e0",
                "type": "reveal",
                "anchor_phrase": "state changes",
                "visual_goal": "show the before and after state",
                "visible_objects": ["object"],
                "visual_state": "object in the new state",
                "action": "apply_force_then_move_object",
                "action_reason": "the narration describes a caused transition",
                "emphasis_targets": ["object"],
            }],
        },
        scene_id=2,
        template_id="intro",
        allowed_events=["reveal"],
    )

    assert result["events"] == []

    valid = semantic_plan._validate_plan(
        {
            "title": "State change",
            "assets": [],
            "events": [{
                "id": "e0",
                "type": "motion",
                "anchor_phrase": "state changes",
                "action": "motion",
                "visible_objects": ["object"],
            }],
        },
        scene_id=2,
        template_id="intro",
        allowed_events=["hold"],
    )
    event = valid["events"][0]
    assert event["action"] == "motion"
    assert event["type"] == "hold"
    assert event["visual_goal"] == "state changes"
    assert event["visible_objects"] == ["object"]
    assert event["persistence_after_action"] is True


def test_semantic_plan_rejects_obvious_role_asset_mismatch_and_keeps_valid_asset() -> None:
    with pytest.raises(ValueError, match="incompatible with role"):
        semantic_plan._validate_plan(
            {
                "title": "Objects",
                "assets": [
                    {"role": "hand", "asset_id": "car", "instance_id": "hand_a"},
                ],
                "events": [],
            },
            scene_id=1,
            template_id="intro",
            allowed_events=[],
        )

    result = semantic_plan._validate_plan(
        {
            "title": "Objects",
            "assets": [
                {"role": "object", "asset_id": "block", "instance_id": "object_a"},
            ],
            "events": [{
                "id": "e0",
                "type": "reveal",
                "anchor_phrase": "show the object",
                "action": "reveal",
                "visible_objects": ["object", "force_arrow"],
            }],
        },
        scene_id=1,
        template_id="intro",
        allowed_events=["place_title"],
    )

    assert [asset["instance_id"] for asset in result["assets"]] == ["object_a"]
    assert result["visual_objects"] == ["object", "force_arrow"]


def test_template_asset_role_is_mapped_only_when_unambiguous() -> None:
    result = semantic_plan._validate_plan(
        {
            "title": "Inertia",
            "assets": [
                {"role": "block", "asset_id": "block", "instance_id": "book_a"},
                {"role": "arrow_force", "asset_id": "arrow_force", "instance_id": "force_a"},
            ],
            "events": [],
        },
        scene_id=2,
        template_id="inertia",
        allowed_events=[],
        supported_slots={
            "stationary_object": ["block"],
            "external_force": ["arrow_force"],
        },
    )

    assert [asset["role"] for asset in result["assets"]] == [
        "stationary_object",
        "external_force",
    ]


def test_summary_semantic_plan_uses_capped_thinking_budget(monkeypatch, tmp_path):
    captured = []

    class FakeClient:
        def chat_json(self, model, messages, temperature=0.0, max_tokens=0, extra_body=None):
            captured.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": extra_body,
                }
            )
            return {
                "scene_id": 5,
                "concept_template": "summary",
                "title": "Key Takeaways",
                "anchor_example": "all scenarios",
                "summary_points": ["a", "b", "c"],
                "events": [
                    {
                        "id": "e0",
                        "type": "place_title",
                        "targets": [],
                        "anchor_phrase": "in summary",
                        "phase": "on",
                        "importance": 3,
                    }
                ],
                "assets": [],
            }

    monkeypatch.setattr(semantic_plan, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(semantic_plan, "PATHS", {**semantic_plan.PATHS, "json": tmp_path})

    result = semantic_plan.build_semantic_plan(
        {
            "scene_id": 5,
            "concept_template": "summary",
            "scene_role": "summary",
            "title": "Key Takeaways",
            "anchor_example": "all scenarios",
            "learning_goal": "consolidate learning",
            "summary_points": ["a", "b", "c"],
        },
        curriculum_context="",
        curriculum_sections=[],
        learner_profile=None,
        topic="Newton's First Law",
        subject="Physics",
        learner_context="",
    )

    assert result["scene_id"] == 5
    assert captured[0]["max_tokens"] == semantic_plan.NVIDIA_SEMANTIC_PLAN_MAX_TOKENS
    request_schema = captured[0]["extra_body"]["response_format"]
    assert captured[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert request_schema["type"] == "json_schema"
    assert request_schema["json_schema"]["name"] == "SemanticPlan"
    assert request_schema["json_schema"]["schema"]["type"] == "object"
    assert request_schema["json_schema"]["schema"]["properties"]["assets"]["maxItems"] == 0
    assert "LEGAL ASSET ROLES FOR THIS TEMPLATE" in captured[0]["messages"][1]["content"]

    semantic_plan.build_semantic_plan(
        {
            "scene_id": 1,
            "concept_template": "intro",
            "scene_role": "hook",
            "title": "Newton's First Law — Overview",
            "anchor_example": "A book sliding on a table eventually stops.",
            "learning_goal": "introduce the concept",
            "subtitle": "Inertia and the absence of net force",
            "key_term": "Newton's First Law",
        },
        curriculum_context="",
        curriculum_sections=[],
        learner_profile=None,
        topic="Newton's First Law",
        subject="Physics",
        learner_context="",
    )

    assert captured[1]["extra_body"]["response_format"]["type"] == "json_schema"


def test_intro_prompt_rejects_registry_asset_roles_and_explains_native_visuals(
    monkeypatch, tmp_path
) -> None:
    captured = []

    class FakeClient:
        def chat_json(self, model, messages, temperature=0.0, max_tokens=0, extra_body=None):
            captured.append(messages[1]["content"])
            return {
                "scene_id": 1,
                "concept_template": "intro",
                "title": "Overview",
                "anchor_example": "a car moves",
                "assets": [],
                "events": [{"id": "e0", "type": "hold", "action": "hold", "anchor_phrase": "a car moves"}],
            }

    monkeypatch.setattr(semantic_plan, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(semantic_plan, "PATHS", {**semantic_plan.PATHS, "json": tmp_path})

    semantic_plan.build_semantic_plan(
        {
            "scene_id": 1,
            "concept_template": "intro",
            "title": "Overview",
            "anchor_example": "a car moves",
        },
        subject="Physics",
        learner_context="",
    )

    prompt = captured[0]
    assert "LEGAL ASSET ROLES FOR THIS TEMPLATE" in prompt
    assert "(none; use template-native visuals" in prompt
    assert "`role` is a renderer/template-supported semantic slot" in prompt
    assert "`asset_id` is the actual visual object identity" in prompt
    assert "`assets` MUST be []" in prompt


def test_template_slots_reject_invalid_intro_role_but_allow_car_in_object_slot() -> None:
    with pytest.raises(ValueError, match="unsupported by the selected template"):
        semantic_plan._validate_plan(
            {
                "assets": [{"role": "block", "asset_id": "car", "instance_id": "car_a"}],
                "events": [],
            },
            scene_id=1,
            template_id="intro",
            allowed_events=[],
            supported_slots={},
        )

    result = semantic_plan._validate_plan(
        {
            "assets": [{"role": "object", "asset_id": "car", "instance_id": "car_a"}],
            "events": [],
        },
        scene_id=2,
        template_id="force",
        allowed_events=[],
        supported_slots={
            "object": ["block", "car", "hockey_puck"],
            "surface": ["ground"],
            "primary_force": ["arrow_force"],
        },
    )

    assert result["assets"][0]["role"] == "object"
    assert result["assets"][0]["asset_id"] == "car"


def test_semantic_plan_top_level_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="Semantic plan must be a dict"):
        semantic_plan._validate_plan(
            [{"scene_id": 4}],
            scene_id=4,
            template_id="work_energy",
            allowed_events=[],
            supported_slots={"object": ["block", "car"]},
        )


def test_semantic_plan_rejects_literal_control_character() -> None:
    with pytest.raises(ValueError, match=r"U\+0000.*plan\.title"):
        semantic_plan._validate_plan(
            {"title": "Bad\x00Title", "assets": [], "events": []},
            scene_id=1,
            template_id="intro",
            allowed_events=[],
        )


def test_escaped_json_control_character_is_rejected_after_decoding() -> None:
    raw = json.loads(r'{"title":"Bad\u0000Title","assets":[],"events":[]}')
    with pytest.raises(ValueError, match=r"U\+0000.*plan\.title"):
        semantic_plan._validate_plan(
            raw,
            scene_id=1,
            template_id="intro",
            allowed_events=[],
        )


def test_normal_unicode_and_whitespace_text_is_preserved() -> None:
    title = "Newton's First Law —\tOverview\n"
    result = semantic_plan._validate_plan(
        {"title": title, "assets": [], "events": []},
        scene_id=1,
        template_id="intro",
        allowed_events=[],
    )
    assert result["title"] == title


def test_semantic_plan_schema_uses_canonical_roles_and_flexible_asset_ids() -> None:
    work_energy = semantic_plan._semantic_plan_response_schema(
        "work_energy", ["place", "hold"], {"object": ["block", "car"]}
    )
    assert work_energy["type"] == "object"
    work_energy_variants = work_energy["properties"]["assets"]["items"]["oneOf"]
    assert {
        variant["properties"]["role"]["enum"][0]
        for variant in work_energy_variants
    } == {"object"}
    assert any(
        variant["properties"]["asset_id"]["enum"] == ["car"]
        for variant in work_energy_variants
    )

    inertia = semantic_plan._semantic_plan_response_schema(
        "inertia",
        ["place", "hold"],
        {
            "stationary_object": ["block", "car"],
            "surface": ["ground"],
            "external_force": ["arrow_force"],
        },
    )
    variants = inertia["properties"]["assets"]["items"]["oneOf"]
    assert {
        variant["properties"]["role"]["enum"][0]
        for variant in variants
    } == {"external_force", "stationary_object", "surface"}

    intro = semantic_plan._semantic_plan_response_schema("intro", [], {})
    assert intro["properties"]["assets"]["maxItems"] == 0
    assert intro["properties"]["assets"]["items"]["properties"]["role"]["enum"] == []


def test_mechanics_schema_constrains_asset_parameters() -> None:
    schema = semantic_plan._semantic_plan_response_schema(
        "inertia",
        ["place", "hold"],
        {
            "stationary_object": ["block"],
            "external_force": ["arrow_force"],
        },
    )
    arrow_variant = next(
        variant for variant in schema["properties"]["assets"]["items"]["oneOf"]
        if variant["properties"]["asset_id"]["enum"] == ["arrow_force"]
    )
    params = arrow_variant["properties"]["params"]
    assert params["properties"]["length"]["type"] == "number"
    assert "brown" not in params["properties"]["color"]["enum"]
    assert "magnitude" not in params["properties"]


def test_asset_parameter_schema_is_scoped_to_selected_asset_ids() -> None:
    schema = semantic_plan._semantic_plan_response_schema(
        "inertia",
        ["place", "hold"],
        {"external_force": ["arrow_force"]},
    )
    asset = schema["properties"]["assets"]["items"]["oneOf"][0]
    params = asset["properties"]["params"]["properties"]

    assert asset["properties"]["asset_id"]["enum"] == ["arrow_force"]
    assert set(params) == {"label", "direction", "color", "length"}
    assert "magnitude" not in params


def test_planner_schema_keeps_disjoint_asset_parameters_isolated() -> None:
    schema = semantic_plan._semantic_plan_response_schema(
        "mixed",
        ["place"],
        {
            "force": ["arrow_force"],
            "velocity": ["velocity_indicator"],
        },
    )
    variants = schema["properties"]["assets"]["items"]["oneOf"]
    by_asset = {
        variant["properties"]["asset_id"]["enum"][0]: variant["properties"]["params"]["properties"]
        for variant in variants
    }

    assert set(by_asset["arrow_force"]) == {"label", "direction", "color", "length"}
    assert set(by_asset["velocity_indicator"]) == {"color", "magnitude"}
    assert "magnitude" not in by_asset["arrow_force"]
    assert "length" not in by_asset["velocity_indicator"]


def test_planner_schema_reads_authoritative_asset_schema(monkeypatch) -> None:
    monkeypatch.setitem(
        semantic_plan.ASSET_PARAM_SCHEMAS["arrow_force"],
        "calibration",
        {"type": "number", "minimum": 0.0},
    )

    schema = semantic_plan._semantic_plan_response_schema(
        "inertia",
        ["place", "hold"],
        {"external_force": ["arrow_force"]},
    )

    arrow_variant = next(
        variant for variant in schema["properties"]["assets"]["items"]["oneOf"]
        if variant["properties"]["asset_id"]["enum"] == ["arrow_force"]
    )
    assert arrow_variant["properties"]["params"]["properties"]["calibration"] == {
        "type": "number",
        "minimum": 0.0,
    }


def test_invalid_asset_parameter_retry_includes_exact_contract(monkeypatch, tmp_path) -> None:
    captured: list[list[dict[str, str]]] = []

    class FakeClient:
        def chat_json(self, model, messages, temperature=0.0, max_tokens=0, extra_body=None):
            captured.append(messages)
            if len(captured) == 1:
                return {
                    "scene_id": 2,
                    "concept_template": "inertia",
                    "title": "Inertia",
                    "anchor_example": "a block moves",
                    "assets": [{
                        "role": "external_force",
                        "asset_id": "arrow_force",
                        "instance_id": "force_a",
                        "params": {"magnitude": 1.0},
                    }],
                    "events": [],
                }
            return {
                "scene_id": 2,
                "concept_template": "inertia",
                "title": "Inertia",
                "anchor_example": "a block moves",
                "assets": [{
                    "role": "external_force",
                    "asset_id": "arrow_force",
                    "instance_id": "force_a",
                    "params": {"length": 1.0, "direction": "RIGHT"},
                }],
                "events": [],
            }

    monkeypatch.setattr(semantic_plan, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(semantic_plan, "PATHS", {**semantic_plan.PATHS, "json": tmp_path})

    semantic_plan.build_semantic_plan(
        {
            "scene_id": 2,
            "concept_template": "inertia",
            "title": "Inertia",
            "anchor_example": "a block moves",
        },
        subject="Physics",
        learner_context="",
    )

    retry_prompt = captured[1][-1]["content"]
    contract = retry_prompt.split("PER-ASSET PARAMETER CONTRACT:", 1)[1]
    assert "magnitude" in retry_prompt
    assert "arrow_force" in retry_prompt
    assert "length (number" in contract
    assert "magnitude" not in contract


def test_invalid_semantic_plan_retries_with_configured_budget(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeClient:
        def chat_json(self, model, messages, temperature=0.0, max_tokens=0, extra_body=None):
            calls.append({"max_tokens": max_tokens, "extra_body": extra_body})
            if len(calls) == 1:
                return [{"scene_id": 4}]
            return {
                "scene_id": 4,
                "concept_template": "work_energy",
                "title": "Work",
                "anchor_example": "a car moves",
                "assets": [{"role": "object", "asset_id": "car", "instance_id": "car_a", "params": {}}],
                "events": [{"id": "e0", "type": "place", "action": "reveal", "anchor_phrase": "a car moves"}],
            }

    monkeypatch.setattr(semantic_plan, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(semantic_plan, "NVIDIA_SEMANTIC_PLAN_MAX_TOKENS", 8192)
    monkeypatch.setattr(semantic_plan, "PATHS", {**semantic_plan.PATHS, "json": tmp_path})

    result = semantic_plan.build_semantic_plan(
        {"scene_id": 4, "concept_template": "work_energy", "title": "Work", "anchor_example": "a car moves"},
        subject="Physics",
        learner_context="",
    )

    assert result["assets"][0]["asset_id"] == "car"
    assert len(calls) == 2
    assert all(call["max_tokens"] == 8192 for call in calls)
    schema = calls[0]["extra_body"]["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert {
        variant["properties"]["role"]["enum"][0]
        for variant in schema["properties"]["assets"]["items"]["oneOf"]
    } == {"object"}
