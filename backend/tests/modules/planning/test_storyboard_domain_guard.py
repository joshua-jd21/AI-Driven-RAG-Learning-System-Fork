"""Regression tests for Physics-first storyboard validation."""
from __future__ import annotations

import json

import pytest

import modules.llm.nvidia_client as nvidia_client
from modules.planning import storyboard
from modules.planning.storyboard import STORYBOARD_RESPONSE_SCHEMA, _validate_entry
from modules.planning.chemistry_router import CHEMISTRY_TEMPLATE_IDS
from modules.templates import TEMPLATES


def test_physics_storyboard_does_not_upgrade_to_chemistry_from_generic_tags() -> None:
    entry = {
        "scene_id": 2,
        "concept_template": "concept_card",
        "scene_role": "visual_intuition",
        "title": "Inertia in Motion",
        "anchor_example": "A passenger lurching forward when a bus brakes",
        "learning_goal": "visualize inertia as resistance to change in motion",
        "visual_instruction": "Bus scene with passenger silhouette.",
    }
    result = _validate_entry(
        entry,
        2,
        topic="Explain Newton's First Law",
        curriculum_sections=[
            {
                "semantic_tags": ["atomic-structure"],
                "visualizable_elements": ["nucleus", "electron shell"],
            }
        ],
        subject="Physics",
    )

    assert result["concept_template"] == "concept_card"
    assert result["scene_role"] == "visual_intuition"
    assert result["subject"] == "Physics"


def test_concept_card_is_already_the_canonical_renderer_template() -> None:
    result = _validate_entry(
        {
            "scene_id": 2,
            "concept_template": "concept_card",
            "scene_role": "visual_intuition",
            "title": "Concept",
            "anchor_example": "a concrete example",
            "learning_goal": "explain the idea",
        },
        2,
        subject="Physics",
    )

    assert result["concept_template"] == "concept_card"
    assert TEMPLATES[result["concept_template"]].__name__ == "ConceptCardTemplate"


def test_storyboard_schema_uses_only_canonical_template_ids() -> None:
    enum = STORYBOARD_RESPONSE_SCHEMA["items"]["properties"]["concept_template"]["enum"]

    assert "concept_card" in enum
    assert "Explain: concept_card" not in enum
    assert "Mechanics: inertia" not in enum


def test_category_prefixed_template_falls_back_before_rendering() -> None:
    result = _validate_entry(
        {
            "scene_id": 2,
            "concept_template": "Mechanics: inertia",
            "scene_role": "visual_intuition",
            "title": "Inertia",
            "anchor_example": "A passenger on a bus",
            "learning_goal": "show inertia",
        },
        2,
        subject="Physics",
    )

    assert result["concept_template"] == "freeform"


def test_chemistry_storyboard_can_still_route_to_chemistry_template() -> None:
    entry = {
        "scene_id": 2,
        "concept_template": "concept_card",
        "scene_role": "visual_intuition",
        "title": "Atomic Structure",
        "anchor_example": "Electron shells around a nucleus",
        "learning_goal": "visualize the atom",
        "visual_instruction": "Atom diagram with shells.",
    }
    result = _validate_entry(
        entry,
        2,
        topic="Explain Atomic Structure",
        curriculum_sections=[
            {
                "semantic_tags": ["atomic-structure"],
                "visualizable_elements": ["nucleus", "electron shell"],
            }
        ],
        subject="Chemistry",
    )

    assert result["concept_template"] in CHEMISTRY_TEMPLATE_IDS
    assert result["subject"] == "Chemistry"


def test_build_storyboard_accepts_scenes_envelope(
    monkeypatch,
    tmp_path,
) -> None:
    scenes = [
        {
            "scene_id": 1,
            "concept_template": "intro",
            "scene_role": "hook",
            "title": "Newton's First Law — Overview",
            "anchor_example": "A book sliding on a table eventually stops.",
            "learning_goal": "introduce the concept",
            "subtitle": "Inertia and the absence of net force",
            "key_term": "Newton's First Law",
            "visual_instruction": "A book on a flat surface.",
        },
        {
            "scene_id": 2,
            "concept_template": "concept_card",
            "scene_role": "visual_intuition",
            "title": "Inertia in Motion",
            "anchor_example": "A passenger lurching forward when a bus brakes",
            "learning_goal": "visualize inertia as resistance to change in motion",
            "visual_instruction": "Bus scene with passenger silhouette.",
        },
        {
            "scene_id": 3,
            "concept_template": "diagram",
            "scene_role": "formal_concept",
            "title": "Force and Motion",
            "anchor_example": "Balanced and unbalanced forces",
            "learning_goal": "connect net force to changes in motion",
            "visual_instruction": "Force arrows on a block.",
        },
        {
            "scene_id": 4,
            "concept_template": "equation",
            "scene_role": "worked_example",
            "title": "Worked Example",
            "anchor_example": "A moving cart on a smooth track",
            "learning_goal": "apply the law to a concrete case",
            "visual_instruction": "Cart motion with force labels.",
        },
        {
            "scene_id": 5,
            "concept_template": "summary",
            "scene_role": "summary",
            "title": "Key Takeaways",
            "anchor_example": "all scenarios",
            "learning_goal": "consolidate learning",
            "summary_points": ["a", "b", "c"],
            "visual_instruction": "Summary diagram.",
        },
    ]

    class FakeClient:
        def chat_json(
            self,
            model,
            messages,
            temperature=0.0,
            max_tokens=0,
            extra_body=None,
        ):
            return {"scenes": scenes}

    monkeypatch.setattr(storyboard, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(storyboard, "PATHS", {**storyboard.PATHS, "json": tmp_path})

    result = storyboard.build_storyboard(
        topic="Explain Newton's First Law",
        curriculum_context="",
        curriculum_sections=[],
        learner_profile=None,
        subject="Physics",
        learner_context="",
    )

    assert len(result) == 5
    assert result[0]["scene_role"] == "hook"
    assert result[-1]["scene_role"] == "summary"
    assert all(scene["subject"] == "Physics" for scene in result)


def test_storyboard_request_uses_structured_output_override(
    monkeypatch,
    tmp_path,
) -> None:
    payloads = []

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self._content,
                        }
                    }
                ]
            }

    storyboard_payload = json.dumps(
        [
            {
                "scene_id": 1,
                "concept_template": "intro",
                "scene_role": "hook",
                "title": "Newton's First Law — Overview",
                "anchor_example": "A book sliding on a table eventually stops.",
                "learning_goal": "introduce the concept",
                "subtitle": "Inertia and the absence of net force",
                "key_term": "Newton's First Law",
                "visual_instruction": "A book on a flat surface.",
            },
            {
                "scene_id": 2,
                "concept_template": "concept_card",
                "scene_role": "visual_intuition",
                "title": "Inertia in Motion",
                "anchor_example": "A passenger lurching forward when a bus brakes",
                "learning_goal": "visualize inertia as resistance to change in motion",
                "visual_instruction": "Bus scene with passenger silhouette.",
            },
            {
                "scene_id": 3,
                "concept_template": "diagram",
                "scene_role": "formal_concept",
                "title": "Force and Motion",
                "anchor_example": "Balanced and unbalanced forces",
                "learning_goal": "connect net force to changes in motion",
                "visual_instruction": "Force arrows on a block.",
            },
            {
                "scene_id": 4,
                "concept_template": "equation",
                "scene_role": "worked_example",
                "title": "Worked Example",
                "anchor_example": "A moving cart on a smooth track",
                "learning_goal": "apply the law to a concrete case",
                "visual_instruction": "Cart motion with force labels.",
            },
            {
                "scene_id": 5,
                "concept_template": "summary",
                "scene_role": "summary",
                "title": "Key Takeaways",
                "anchor_example": "all scenarios",
                "learning_goal": "consolidate learning",
                "summary_points": ["a", "b", "c"],
                "visual_instruction": "Summary diagram.",
            },
        ],
        ensure_ascii=False,
    )

    def fake_post(url, headers=None, json=None, timeout=None):
        payloads.append(json)
        return FakeResponse(storyboard_payload)

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(nvidia_client.requests, "post", fake_post)
    monkeypatch.setattr(storyboard, "PATHS", {**storyboard.PATHS, "json": tmp_path})

    result = storyboard.build_storyboard(
        topic="Explain Newton's First Law",
        curriculum_context="",
        curriculum_sections=[],
        learner_profile=None,
        subject="Physics",
        learner_context="",
    )

    assert len(result) == 5
    assert payloads[0]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert payloads[0]["response_format"]["type"] == "json_schema"
    assert payloads[0]["response_format"]["json_schema"]["name"] == "StoryboardScenes"
    schema = payloads[0]["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "array"
    assert schema["minItems"] == 5
    assert schema["maxItems"] == 5
    assert schema["items"]["type"] == "object"
    assert payloads[0]["response_format"]["type"] != "json_object"

    from modules.llm.nvidia_client import NvidiaClient

    client = NvidiaClient()
    client.chat_json(
        "test-model",
        [{"role": "user", "content": "prompt"}],
        temperature=0.1,
        max_tokens=64,
    )

    assert "response_format" not in payloads[1]
    assert payloads[1]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }


def test_unwrap_storyboard_scenes_accepts_raw_list() -> None:
    scenes = [{"scene_id": 1}, {"scene_id": 2}]

    result = storyboard._unwrap_storyboard_scenes(scenes)

    assert result is scenes


def test_unwrap_storyboard_scenes_accepts_known_scenes_envelope() -> None:
    scenes = [{"scene_id": 1}, {"scene_id": 2}]

    result = storyboard._unwrap_storyboard_scenes({"scenes": scenes})

    assert result is scenes


def test_unwrap_storyboard_scenes_rejects_unknown_dict_envelope() -> None:
    with pytest.raises(ValueError, match="Storyboard LLM returned <class 'dict'>, expected list"):
        storyboard._unwrap_storyboard_scenes({"storyboard": [{"scene_id": 1}]})
