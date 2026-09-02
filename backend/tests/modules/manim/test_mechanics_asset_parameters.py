from __future__ import annotations

import pytest

from modules.assets.mechanics import get_code


def test_non_numeric_force_length_is_rejected_before_code_generation() -> None:
    with pytest.raises(ValueError, match=r"length.*arrow_force.*short"):
        get_code("arrow_force", "force", {"length": "short"})


def test_numeric_force_length_is_accepted() -> None:
    code = get_code("arrow_force", "force", {"length": 1.25, "color": "red"})
    assert "RIGHT*1.25" in code


def test_supported_block_color_is_accepted() -> None:
    code = get_code("block", "block", {"color": "#f7c948"})
    assert 'color="#f7c948"' in code


def test_unsupported_block_color_is_rejected_before_code_generation() -> None:
    with pytest.raises(ValueError, match=r"color.*block.*brown"):
        get_code("block", "block", {"color": "brown"})
