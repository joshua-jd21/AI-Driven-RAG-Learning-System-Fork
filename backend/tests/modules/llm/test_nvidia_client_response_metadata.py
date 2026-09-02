from __future__ import annotations

import modules.llm.nvidia_client as nvidia_client


def test_summarize_nim_response_includes_finish_reason_and_usage() -> None:
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "[{\"scene_id\": 1}]",
                    "reasoning_content": "internal reasoning trace",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 512,
            "total_tokens": 1712,
        },
    }

    summary = nvidia_client.NvidiaClient._summarize_nim_response(data)

    assert summary["type"] == "dict"
    assert summary["finish_reason"] == "length"
    assert summary["usage"] == {
        "prompt_tokens": 1200,
        "completion_tokens": 512,
        "total_tokens": 1712,
    }
    assert summary["content_chars"] == len('[{"scene_id": 1}]')
    assert summary["reasoning_chars"] == len("internal reasoning trace")


def test_chat_returns_full_content_without_truncation(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "[{\"scene_id\": 1}, {\"scene_id\": 2}]",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 512,
                    "total_tokens": 1712,
                },
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(nvidia_client.requests, "post", fake_post)

    client = nvidia_client.NvidiaClient()
    content = client.chat(
        "test-model",
        [{"role": "user", "content": "prompt"}],
        temperature=0.1,
        max_tokens=4096,
    )

    assert content == '[{"scene_id": 1}, {"scene_id": 2}]'
    assert captured["json"]["max_tokens"] == 4096


def test_chat_json_passes_request_overrides(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "{\"ok\": true}",
                        }
                    }
                ]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(nvidia_client.requests, "post", fake_post)

    client = nvidia_client.NvidiaClient()
    parsed = client.chat_json(
        "test-model",
        [{"role": "user", "content": "prompt"}],
        temperature=0.1,
        max_tokens=256,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "thinking_token_budget": 512,
            }
        },
    )

    assert parsed == {"ok": True}
    assert captured["json"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking_token_budget": 512,
    }
