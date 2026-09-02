from __future__ import annotations

from modules.llm.nvidia_client import NvidiaClient


def test_extract_json_preserves_top_level_array() -> None:
    text = """
    [
      {"scene_id": 1, "title": "Scene One"},
      {"scene_id": 2, "title": "Scene Two"}
    ]
    """

    parsed = NvidiaClient._extract_json(text)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["scene_id"] == 1
    assert parsed[1]["scene_id"] == 2


def test_extract_json_returns_top_level_object_as_dict() -> None:
    text = """
    {
      "scene_id": 1,
      "title": "Scene One"
    }
    """

    parsed = NvidiaClient._extract_json(text)

    assert isinstance(parsed, dict)
    assert parsed["scene_id"] == 1
    assert parsed["title"] == "Scene One"
