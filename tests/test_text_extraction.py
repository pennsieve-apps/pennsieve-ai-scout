"""Reading text back out of a model response.

Kept separate from the request-shaping tests because `first_text` needs no
SDK stub — it only walks content blocks.
"""
from __future__ import annotations

import pytest

from scout.agents._common import first_text


class _Block:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class _Msg:
    def __init__(self, *blocks):
        self.content = list(blocks)


def test_first_text_skips_a_leading_thinking_block():
    """Opus 5 runs adaptive thinking even when the request omits the
    `thinking` param, so the first block is not the answer."""
    msg = _Msg(_Block("thinking", "hmm"), _Block("text", '{"a": 1}'))
    assert first_text(msg) == '{"a": 1}'


def test_first_text_plain_response():
    assert first_text(_Msg(_Block("text", "hi"))) == "hi"


def test_first_text_raises_when_no_text_block():
    with pytest.raises(ValueError, match="No text block"):
        first_text(_Msg(_Block("thinking", "only thought")))
