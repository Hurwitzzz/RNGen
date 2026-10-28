"""Legacy dataset and collation helpers for spelled-pitch warm-up.

These helpers are retained for backward compatibility and are not used by the
default RNGen training pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset
from rngen.event_vocabulary import (
    DEFAULT_SPELLING_TOKENS,
    load_event_vocab,
    get_spelling_event_token as _get_spelling_event_token,
)

CANONICAL_SPELLED_PITCH_CLASSES: List[str] = [
    "Cbbb",
    "Cbb",
    "Cb",
    "C",
    "C#",
    "C##",
    "C###",
    "Dbbb",
    "Dbb",
    "Db",
    "D",
    "D#",
    "D##",
    "D###",
    "Ebbb",
    "Ebb",
    "Eb",
    "E",
    "E#",
    "E##",
    "E###",
    "Fbbb",
    "Fbb",
    "Fb",
    "F",
    "F#",
    "F##",
    "F###",
    "Gbbb",
    "Gbb",
    "Gb",
    "G",
    "G#",
    "G##",
    "G###",
    "Abbb",
    "Abb",
    "Ab",
    "A",
    "A#",
    "A##",
    "A###",
    "Bbbb",
    "Bbb",
    "Bb",
    "B",
    "B#",
    "B##",
    "B###",
]

SPELLED_PITCH_CLASS_TO_ID: Dict[str, int] = {
    label: idx for idx, label in enumerate(CANONICAL_SPELLED_PITCH_CLASSES)
}

MUSICBERT_PAD_TOKEN_ID = 1

SPECIAL_SEQUENCE_TOKENS = {"<s>", "</s>"}
_SPELLING_REGEX = re.compile(r"^([A-Ga-g])((?:bbb|###|bb|##|x|b|#)?)(-?\d+)?$")


def canonicalize_spelled_pitch_class(spelling: str) -> str:
    spelling = str(spelling).strip()
    if not spelling:
        raise ValueError("Spelling label cannot be empty.")
    if spelling in SPECIAL_SEQUENCE_TOKENS:
        raise ValueError("Special sequence tokens are not valid spelled pitch labels.")

    match = _SPELLING_REGEX.fullmatch(spelling)
    if match is None:
        raise ValueError(f"Unsupported spelled pitch label: {spelling}")

    letter, accidental, _octave = match.groups()
    accidental = accidental.replace("x", "##")
    canonical = f"{letter.upper()}{accidental}"
    if canonical not in SPELLED_PITCH_CLASS_TO_ID:
        raise ValueError(f"Unsupported canonical spelled pitch label: {canonical}")
    return canonical


def _split_compound_groups(
    events_tokens: Sequence[str], compound_ratio: int
) -> List[List[str]]:
    if len(events_tokens) % compound_ratio != 0:
        raise ValueError(
            f"Events token length {len(events_tokens)} is not a multiple of {compound_ratio}."
        )
    return [
        list(events_tokens[i : i + compound_ratio])
        for i in range(0, len(events_tokens), compound_ratio)
    ]


def _is_special_group(group: Sequence[str]) -> bool:
    return bool(group) and len(set(group)) == 1 and group[0] in SPECIAL_SEQUENCE_TOKENS


def parse_spelling_sequence(spelling_str: str) -> List[str]:
    raw_tokens = [token for token in str(spelling_str).split() if token]
    filtered = [token for token in raw_tokens if token not in SPECIAL_SEQUENCE_TOKENS]
    return [canonicalize_spelled_pitch_class(token) for token in filtered]


def get_spelling_event_token(spelling: str) -> str:
    canonical = canonicalize_spelled_pitch_class(spelling)
    return _get_spelling_event_token(canonical)


def build_warmup_event_tokens(
    events_str: str,
    spelling_str: str,
    *,
    compound_ratio: int = 9,
) -> List[str]:
    event_tokens = str(events_str).split()
    groups = _split_compound_groups(event_tokens, compound_ratio)
    spelling_labels = parse_spelling_sequence(spelling_str)

    rebuilt_groups: List[List[str]] = []
    label_cursor = 0
    for group in groups:
        rebuilt_group = list(group)
        if _is_special_group(group):
            rebuilt_groups.append(rebuilt_group)
            continue

        if label_cursor >= len(spelling_labels):
            raise ValueError(
                "More non-special event groups than spelled-pitch labels. "
                f"Groups={len(groups)} labels={len(spelling_labels)}"
            )
        rebuilt_group[-1] = get_spelling_event_token(spelling_labels[label_cursor])
        label_cursor += 1
        rebuilt_groups.append(rebuilt_group)

    if label_cursor != len(spelling_labels):
        raise ValueError(
            "More spelled-pitch labels than non-special event groups. "
            f"Consumed={label_cursor} available={len(spelling_labels)}"
        )

    return [token for group in rebuilt_groups for token in group]


def build_note_level_labels(
    events_str: str,
    spelling_str: str,
    *,
    compound_ratio: int = 9,
) -> List[int]:
    event_tokens = str(events_str).split()
    groups = _split_compound_groups(event_tokens, compound_ratio)
    spelling_labels = parse_spelling_sequence(spelling_str)

    labels = [-100] * len(groups)
    label_cursor = 0
    for group_idx, group in enumerate(groups):
        if _is_special_group(group):
            continue
        if label_cursor >= len(spelling_labels):
            raise ValueError(
                "More non-special event groups than spelled-pitch labels. "
                f"Groups={len(groups)} labels={len(spelling_labels)}"
            )
        labels[group_idx] = SPELLED_PITCH_CLASS_TO_ID[spelling_labels[label_cursor]]
        label_cursor += 1

    if label_cursor != len(spelling_labels):
        raise ValueError(
            "More spelled-pitch labels than non-special event groups. "
            f"Consumed={label_cursor} available={len(spelling_labels)}"
        )
    return labels


@dataclass
class SpellWarmupSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sample_metadata: dict


class SpellWarmupDataset(Dataset):
    """Note-level spelled-pitch classification dataset over Octuple+ sequences."""

    def __init__(
        self,
        json_path: str,
        *,
        compound_ratio: int = 9,
        max_notes: int | None = None,
        event_vocab: Dict[str, int] | None = None,
        is_eval: bool = False,
    ):
        self.json_path = json_path
        self.compound_ratio = compound_ratio
        self.max_notes = max_notes
        self.event_vocab = event_vocab or load_event_vocab(DEFAULT_SPELLING_TOKENS)
        self.is_eval = is_eval

        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        self.sample_ids = sorted(key for key in raw_samples.keys() if key != "template")
        self.samples = [raw_samples[sample_id] for sample_id in self.sample_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_events(self, events: str | Sequence[str]) -> List[int]:
        tokens = str(events).split() if isinstance(events, str) else list(events)
        token_ids = []
        for token in tokens:
            if token not in self.event_vocab:
                raise ValueError(f"Token '{token}' not found in events vocabulary.")
            token_ids.append(self.event_vocab[token])
        return token_ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = self.sample_ids[idx]

        event_tokens = build_warmup_event_tokens(
            sample["events"],
            sample.get("spelling", ""),
            compound_ratio=self.compound_ratio,
        )
        music_token_ids = self._tokenize_events(event_tokens)
        labels = build_note_level_labels(
            sample["events"],
            sample.get("spelling", ""),
            compound_ratio=self.compound_ratio,
        )

        note_len = len(labels)
        if self.max_notes is not None:
            note_len = min(note_len, self.max_notes)
            music_token_ids = music_token_ids[: note_len * self.compound_ratio]
            labels = labels[:note_len]

        input_ids = torch.tensor(music_token_ids, dtype=torch.long)
        attention_mask = torch.ones(len(labels), dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }

        if self.is_eval:
            result["sample_metadata"] = {
                "sample_id": sample_id,
                "events": sample["events"],
                "spelling": sample.get("spelling", ""),
            }

        return result


class DataCollatorForSpellWarmup:
    """Pad Octuple+ note-classification features for the spell warm-up stage."""

    def __init__(self, musicbert_pad_token_id: int = MUSICBERT_PAD_TOKEN_ID):
        if musicbert_pad_token_id < 0:
            raise ValueError(
                "musicbert_pad_token_id must be a valid encoder token id, not "
                f"{musicbert_pad_token_id}. Use -100 only for ignored labels."
            )
        self.musicbert_pad_token_id = musicbert_pad_token_id

    def __call__(
        self, features: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [feature["input_ids"] for feature in features],
            batch_first=True,
            padding_value=self.musicbert_pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [feature["attention_mask"] for feature in features],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=-100,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "sample_metadata" in features[0]:
            batch["batch_metadata"] = [
                feature["sample_metadata"] for feature in features
            ]

        return batch
