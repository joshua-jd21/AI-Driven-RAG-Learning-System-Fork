from __future__ import annotations

import pytest

from modules.llm import nvidia_client


def test_empty_content_preserves_length_finish_reason(monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

    monkeypatch.setattr(nvidia_client.requests, "post", lambda *args, **kwargs: Response())
    client = nvidia_client.NvidiaClient(max_retries=1)

    with pytest.raises(nvidia_client.NvidiaEmptyResponseError) as exc_info:
        client.chat("openai/gpt-oss-20b", [{"role": "user", "content": "Write text."}])

    assert exc_info.value.finish_reason == "length"
