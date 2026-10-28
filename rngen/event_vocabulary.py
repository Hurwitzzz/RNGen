"""Shared event vocabulary and accidental tokens for music encoders."""

import json
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_SPELLING_TOKENS = (
    "<8-#>", "<8-##>", "<8-h>", "<8-b>", "<8-bb>", "<8-###>", "<8-bbb>",
)


def load_event_vocab(spelling_tokens: Optional[Sequence[str]] = None):
    path = Path(__file__).parent / "data" / "events_incl_spelling_stoi.json"
    with path.open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    next_id = max(vocab.values()) + 1
    for token in spelling_tokens or ():
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    return vocab


def event_vocab_size(spelling_tokens: Optional[Sequence[str]] = None):
    return max(load_event_vocab(spelling_tokens).values()) + 1


def get_spelling_event_token(spelling: str) -> str:
    """Match longer accidentals first so triple and double signs stay distinct."""
    if not isinstance(spelling, str):
        raise TypeError(f"Invalid spelling_str: Expected a string, got {type(spelling)}")
    for accidental in ("bbb", "bb", "b", "###", "##", "#"):
        if spelling.endswith(accidental):
            return f"<8-{accidental}>"
    return "<8-h>"
