from __future__ import annotations

from types import SimpleNamespace

from scripts import check_nvidia


def test_mask_key_never_prints_full_key() -> None:
    assert check_nvidia.mask_key("abcdefgh12345678") == "abcd...5678"
    assert check_nvidia.mask_key("short") == "*****"


def test_check_reports_configured_model(monkeypatch, capsys) -> None:
    response = SimpleNamespace(status_code=200, text='{"choices":[{"message":{"content":"OK"}}]}', ok=True)
    monkeypatch.setattr(check_nvidia.requests, "post", lambda *args, **kwargs: response)

    assert check_nvidia.check(check_nvidia.MODELS[0]) is True
    output = capsys.readouterr().out
    assert f"model={check_nvidia.MODELS[0]}" in output
    assert "status=200" in output
    assert "body=" in output
