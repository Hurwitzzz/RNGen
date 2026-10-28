"""Convert P.DCML score CSVs into masked AugmentedNet/Micchi2020 tensors.

The converter intentionally lives in RNGen rather than a vendored baseline.
It reproduces the AugmentedNet v1.0 input grids while enforcing three controls
needed by the ISMIR 2026 comparison:

* fixed core vocabularies cover the complete unaugmented P.DCML corpus
  ontology, while the auxiliary CommonRNs head retains its official 76 classes;
* every sequence carries valid-frame, core-label, and head-specific masks;
* Micchi2020 labels and masks are downsampled together by a factor of four.

The processed P.DCML CSV root is preferred because it restores canonical
special-chord labels (for example, ``Ger6``) and ``gt_rnbert``. The older raw
root collapses augmented-six functions to ``x`` and cannot support a faithful
unified RNA evaluation without the original harmony tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from rngen.baselines.augnet_mi20.model import PCSETS as OFFICIAL_PCSETS
from rngen.utils.rna import parse_gt_rnbert
from rngen.paths import PROCESSED_SCORES_DIR


FIXED_OFFSET = 0.125
SEQUENCE_LENGTH = 640
MICCHI_DOWNSAMPLE = 4
NONE_LABEL = "None"
NOTE_NAMES = ("C", "D", "E", "F", "G", "A", "B")
SPELLINGS = tuple(
    f"{letter}{accidental}"
    for letter in NOTE_NAMES
    for accidental in ("--", "-", "", "#", "##")
)
CORE_COMPONENTS = (
    "local_key",
    "primary_degree",
    "secondary_degree",
    "inversion",
    "quality",
    "chord_root",
)
AUXILIARY_COMPONENTS = (
    "bass",
    "roman_numeral_76",
    "tonicized_key",
    "pitch_class_set",
    "harmonic_rhythm",
)
COMPONENTS = CORE_COMPONENTS + AUXILIARY_COMPONENTS
AUGMENTED_PATH_RE = re.compile(r"^(.*)_aug_(\d+)\.csv$")
ROMAN_TO_DEGREE = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
}
SPECIAL_AUGMENTED_SIX_DEGREES = {"Fr": "2", "Ger": "#4", "It": "#4"}
AUGMENTED_SIX_QUALITIES = {
    "Fr": "French augmented sixth chord",
    "Ger": "German augmented sixth chord",
    "It": "Italian augmented sixth chord",
}
AUGMENTED_SIX_PREFIX_BY_QUALITY = {
    quality: prefix for prefix, quality in AUGMENTED_SIX_QUALITIES.items()
}
AUGMENTED_SIX_CANONICAL_FIGURES = {
    "Fr": {"0": "Fr7", "1": "Fr65", "2": "Fr43", "3": "Fr2"},
    "Ger": {"0": "Ger7", "1": "Ger6", "2": "Ger43", "3": "Ger2"},
    "It": {"1": "It6", "2": "It64"},
}
AUGMENTED_SIX_INVERSION_BY_FIGURE = {
    figure: inversion
    for figures in AUGMENTED_SIX_CANONICAL_FIGURES.values()
    for inversion, figure in figures.items()
}
AUGMENTED_SIX_INVERSION_BY_FIGURE.update({"Fr6": "2", "It": "1"})
REQUIRED_COLUMNS = {
    "type",
    "pitch",
    "spelling",
    "onset",
    "release",
    "harmony_onset",
    "primary_degree",
    "primary_alteration",
    "secondary_degree",
    "secondary_alteration",
    "quality",
    "inversion",
    "key",
    "mode",
}
OPTIONAL_COLUMNS = {
    "gt_rnbert",
    "chord_tones",
    "chord_tone_factors",
    "chord_factors",
}

# Names are the official AugmentedNet v1.0 ChordQuality15 class names. Only
# classes observed in P.DCML are mapped here; unknown values remain explicit in
# the audit rather than silently collapsing into another quality.
PDCML_TO_AUGNET_QUALITY = {
    "M": "major triad",
    "m": "minor triad",
    "o": "diminished triad",
    "+": "augmented triad",
    "m7": "minor seventh chord",
    "M7": "major seventh chord",
    "Mm7": "dominant seventh chord",
    "o7": "diminished seventh chord",
    "ø7": "half-diminished seventh chord",
}
AUGNET_TO_PDCML_QUALITY = {value: key for key, value in PDCML_TO_AUGNET_QUALITY.items()}
AUGNET_TO_PDCML_QUALITY.update(
    {quality: "aug6" for quality in AUGMENTED_SIX_QUALITIES.values()}
)

# Official AugmentedNet v1.0 class order, converted to the P.DCML spelling
# convention only where the converter intentionally retains that convention
# (local keys). Legal corpus labels absent from these tuples are appended in
# stable sorted order when the frozen ontology is built.
OFFICIAL_LOCAL_KEYS = (
    "Fb",
    "Cb",
    "Gb",
    "Db",
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "db",
    "ab",
    "eb",
    "bb",
    "f",
    "c",
    "g",
    "d",
    "a",
    "e",
    "b",
    "f#",
    "c#",
    "g#",
    "d#",
    "a#",
    "e#",
    NONE_LABEL,
)
OFFICIAL_DEGREES = (
    "-1",
    "-2",
    "-3",
    "-4",
    "-5",
    "-6",
    "-7",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "#1",
    "#2",
    "#3",
    "#4",
    "#5",
    "#6",
    "#7",
    NONE_LABEL,
)
OFFICIAL_QUALITIES = (
    "major triad",
    "minor triad",
    "diminished triad",
    "augmented triad",
    "minor seventh chord",
    "major seventh chord",
    "dominant seventh chord",
    "incomplete dominant-seventh chord",
    "diminished seventh chord",
    "half-diminished seventh chord",
    "augmented sixth",
    "German augmented sixth chord",
    "French augmented sixth chord",
    "Italian augmented sixth chord",
    "minor-augmented tetrachord",
    NONE_LABEL,
)
OFFICIAL_ROMAN_NUMERALS = (
    "I",
    "V7",
    "V",
    "i",
    "IV",
    "ii",
    "vi",
    "iv",
    "viio7",
    "viio",
    "V7/V",
    "V7/IV",
    "viio7/V",
    "VI",
    "ii7",
    "V/V",
    "v",
    "V7/ii",
    "III",
    "iiø7",
    "iii",
    "iio",
    "viio/V",
    "V7/vi",
    "VII",
    "viio7/ii",
    "I/V",
    "V7/iv",
    "V/vi",
    "vi7",
    "Ger7",
    "N",
    "viio7/vi",
    "V/ii",
    "viiø7",
    "V9",
    "viio/ii",
    "V/iv",
    "Cad/V",
    "iv7",
    "viio7/iv",
    "IV7",
    "V7/III",
    "viiø7/V",
    "It",
    "viio7/v",
    "viio7/iii",
    "IV/V",
    "I+",
    "I7",
    "viio/IV",
    "V/III",
    "V7/iii",
    "viio/iv",
    "iio7",
    "VI7",
    "I/III",
    "V7/VI",
    "bVII",
    "bVI",
    "V+",
    "viio/vi",
    "III+",
    "V/iii",
    "ii/V",
    "I/-VI",
    "viio7/IV",
    "V7/v",
    "i7",
    "iii7",
    "Fr7",
    "V/IV",
    "vii",
    "V/v",
    "II",
    NONE_LABEL,
)
OFFICIAL_PITCH_CLASS_SETS = tuple(
    NONE_LABEL if value == NONE_LABEL else ",".join(str(pc) for pc in value)
    for value in OFFICIAL_PCSETS
)
OFFICIAL_CLASS_ORDER = {
    "local_key": OFFICIAL_LOCAL_KEYS,
    "primary_degree": OFFICIAL_DEGREES,
    "secondary_degree": OFFICIAL_DEGREES,
    "inversion": ("0", "1", "2", "3"),
    "quality": OFFICIAL_QUALITIES,
    "chord_root": SPELLINGS,
    "bass": SPELLINGS,
    "roman_numeral_76": OFFICIAL_ROMAN_NUMERALS,
    "tonicized_key": OFFICIAL_LOCAL_KEYS,
    "pitch_class_set": OFFICIAL_PITCH_CLASS_SETS,
    "harmonic_rhythm": ("True", "False"),
}


@dataclass(frozen=True)
class HarmonyEvent:
    onset: float
    labels: Mapping[str, str]
    gt_rnbert: str
    canonical_rn: str
    chord_tones: str
    chord_factors: str
    source: str


@dataclass
class PieceTensors:
    relative_path: str
    frame_offsets: np.ndarray
    reference_note_onsets: np.ndarray
    harmony_events: tuple[HarmonyEvent, ...]
    bass19: np.ndarray
    chromagram19: np.ndarray
    bass_chromagram70: np.ndarray
    label_strings: dict[str, np.ndarray]
    label_mask: np.ndarray
    diagnostics: dict[str, Any]


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "_", "na", "nan", "none", "<na>"}:
        return ""
    return text


def read_split_paths(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def assert_output_outside_source_roots(
    output_path: Path, source_roots: Iterable[Path]
) -> None:
    """Prevent generated artifacts from being written into source datasets."""
    for source_root in source_roots:
        if _is_relative_to(output_path, source_root):
            raise ValueError(
                f"Refusing to write output {output_path} inside read-only source "
                f"dataset root {source_root}"
            )


def validate_augmented_manifest(
    augmented_paths: Sequence[str],
    base_train_paths: Sequence[str],
    *,
    multiplier: int = 12,
) -> dict[str, list[tuple[int, str]]]:
    """Validate and group an exact base-piece x transposition manifest."""
    if len(augmented_paths) != len(set(augmented_paths)):
        raise ValueError("Augmented training manifest contains duplicate paths")
    base_set = set(base_train_paths)
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for relative_path in augmented_paths:
        match = AUGMENTED_PATH_RE.fullmatch(relative_path)
        if match is None:
            raise ValueError(f"Invalid augmented path: {relative_path!r}")
        base_path = f"{match.group(1)}.csv"
        groups[base_path].append((int(match.group(2)), relative_path))

    if set(groups) != base_set:
        missing = sorted(base_set - set(groups))
        unexpected = sorted(set(groups) - base_set)
        raise ValueError(
            "Augmented/base manifest group mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    expected_indices = list(range(multiplier))
    for base_path, variants in groups.items():
        variants.sort()
        indices = [index for index, _ in variants]
        if indices != expected_indices:
            raise ValueError(
                f"Expected augmentation indices {expected_indices} for {base_path}, "
                f"got {indices}"
            )
    if len(augmented_paths) != len(base_train_paths) * multiplier:
        raise ValueError(
            f"Expected {len(base_train_paths) * multiplier} augmented paths, "
            f"got {len(augmented_paths)}"
        )
    return dict(groups)


def canonicalize_enharmonic_to_classes(
    value: Any, classes: Sequence[str], *, preserve_mode_case: bool
) -> str:
    """Map a tonal spelling to the least-altered equivalent fixed class."""
    raw = _norm(value)
    normalized = _normalize_tonal_spelling(raw)
    pitch_class = _pitch_class(normalized)
    if not normalized or pitch_class is None:
        return ""
    is_minor = raw[:1].islower()
    direction = "#" if "#" in normalized else "-" if "-" in normalized else ""
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for class_index, candidate in enumerate(classes):
        if candidate == NONE_LABEL or not candidate:
            continue
        if preserve_mode_case and candidate[:1].islower() != is_minor:
            continue
        candidate_normalized = _normalize_tonal_spelling(candidate)
        if _pitch_class(candidate_normalized) != pitch_class:
            continue
        accidental = candidate_normalized[1:]
        direction_penalty = int(
            bool(direction) and direction not in accidental and bool(accidental)
        )
        candidates.append(
            ((len(accidental), direction_penalty, class_index), candidate)
        )
    return min(candidates)[1] if candidates else ""


def pdcml_to_augnet_spelling(value: Any) -> str:
    """Convert representable P.DCML spellings to AugNet notation."""
    spelling = _norm(value)
    normalized = _normalize_tonal_spelling(spelling)
    if not normalized:
        return ""
    normalized = normalized[0].upper() + normalized[1:]
    return normalized if normalized in SPELLINGS else ""


def _unbounded_augnet_spelling(value: Any) -> str:
    """Convert P.DCML accidentals without forcing the Spelling35 range."""
    normalized = _normalize_tonal_spelling(_norm(value))
    return normalized[0].upper() + normalized[1:] if normalized else ""


def _canonical_key(row: Mapping[str, Any]) -> str:
    key = _norm(row.get("key"))
    mode = _norm(row.get("mode"))
    if not key:
        return ""
    is_minor = mode == "m" or mode.lower() == "minor"
    first = key[0].lower() if is_minor else key[0].upper()
    return first + key[1:]


def _main_rn_text(gt_rnbert: Any) -> str:
    return _norm(gt_rnbert).split(": ", 1)[0].strip()


def _augnet_alteration(value: Any) -> str:
    """Convert P.DCML degree accidentals to AugNet/music21 notation."""
    alteration = _norm(value)
    if not alteration:
        return ""
    converted = alteration.replace("x", "##").replace("b", "-")
    if set(converted) <= {"-", "#"}:
        return converted
    return f"<UNMAPPED:alteration={alteration}>"


def _roman_figure_to_degree(figure: Any, *, allow_none: bool) -> str:
    text = _norm(figure)
    if not text:
        return NONE_LABEL if allow_none else "<UNMAPPED:missing-degree>"
    if text.startswith("N"):
        return "-2"
    match = re.match(r"^([#bx-]*)([ivIV]+)", text)
    if not match:
        return f"<UNMAPPED:degree={text}>"
    alteration, numeral = match.groups()
    degree = ROMAN_TO_DEGREE.get(numeral.upper())
    if degree is None:
        return f"<UNMAPPED:degree={text}>"
    converted_alteration = _augnet_alteration(alteration)
    if converted_alteration.startswith("<UNMAPPED:"):
        return converted_alteration
    return f"{converted_alteration}{degree}"


def _augmented_six_prefix(primary_figure: str) -> str:
    match = re.match(r"^(Fr|Ger|It)", primary_figure)
    return match.group(1) if match else ""


def canonicalize_special_rn(rn: str) -> str:
    """Canonicalize known P.DCML augmented-six aliases symmetrically."""
    main = _main_rn_text(rn)
    if not main:
        return ""
    primary, *secondary = main.split("/", 1)
    primary = {"Fr6": "Fr43", "It": "It6"}.get(primary, primary)
    return "/".join([primary, *secondary])


def decode_augmented_six_rn(quality: str, inversion: str) -> str:
    """Decode one canonical augmented-six primary figure from Q and I."""
    prefix = AUGMENTED_SIX_PREFIX_BY_QUALITY.get(quality)
    if prefix is None:
        raise ValueError(f"Not an augmented-six subtype quality: {quality!r}")
    try:
        return AUGMENTED_SIX_CANONICAL_FIGURES[prefix][str(inversion)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {prefix} augmented-six inversion: {inversion!r}"
        ) from exc


def roman_numeral_76_label(rn: Any) -> str:
    """Apply AugmentedNet v1 RomanNumeral76 preprocessing to one RN label.

    The official implementation performs global string replacement. This
    token-aware equivalent preserves P.DCML labels such as ``bIII`` while
    matching the intended synonym, bracket, and inversion normalization.
    """
    figure = _main_rn_text(rn)
    if not figure:
        return "<UNMAPPED:missing-roman-numeral>"
    figure = re.sub(r"\[.*?\]", "", figure)
    for source, target in (("6/4", "64"), ("6/5", "65"), ("4/3", "43"), ("4/2", "2")):
        figure = figure.replace(source, target)

    tokens = figure.split("/")
    normalized: list[str] = []
    for token in tokens:
        if re.fullmatch(r"bII(?:64|65|43|6|7|2)?", token):
            token = "N" + token[3:]
        if token.endswith(("65", "43")):
            token = token[:-2] + "7"
        elif token.endswith("64"):
            token = token[:-2]
        elif token.endswith("6"):
            token = token[:-1]
        elif token.endswith("2"):
            token = token[:-1] + "7"
        normalized.append(token)
    return "/".join(normalized)


_NATURAL_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}
_FACTOR_INTERVALS = {
    "major triad": {"R": (0, 0), "T": (2, 4), "F": (4, 7)},
    "minor triad": {"R": (0, 0), "T": (2, 3), "F": (4, 7)},
    "diminished triad": {"R": (0, 0), "T": (2, 3), "F": (4, 6)},
    "augmented triad": {"R": (0, 0), "T": (2, 4), "F": (4, 8)},
    "minor seventh chord": {"R": (0, 0), "T": (2, 3), "F": (4, 7), "S": (6, 10)},
    "major seventh chord": {"R": (0, 0), "T": (2, 4), "F": (4, 7), "S": (6, 11)},
    "dominant seventh chord": {"R": (0, 0), "T": (2, 4), "F": (4, 7), "S": (6, 10)},
    "incomplete dominant-seventh chord": {
        "R": (0, 0),
        "T": (2, 4),
        "F": (4, 7),
        "S": (6, 10),
    },
    "diminished seventh chord": {"R": (0, 0), "T": (2, 3), "F": (4, 6), "S": (6, 9)},
    "half-diminished seventh chord": {
        "R": (0, 0),
        "T": (2, 3),
        "F": (4, 6),
        "S": (6, 10),
    },
    "Italian augmented sixth chord": {"R": (0, 0), "T": (2, 2), "F": (4, 6)},
    "German augmented sixth chord": {
        "R": (0, 0),
        "T": (2, 2),
        "F": (4, 6),
        "S": (6, 9),
    },
    "French augmented sixth chord": {
        "R": (0, 0),
        "T": (2, 4),
        "F": (4, 6),
        "S": (6, 10),
    },
}
_INVERSION_FACTORS = ("R", "T", "F", "S")


def _normalize_tonal_spelling(spelling: str) -> str:
    match = re.fullmatch(r"([A-Ga-g])([b#-]*|x)", _norm(spelling))
    if not match:
        return ""
    letter, accidental = match.groups()
    return f"{letter.upper()}{accidental.replace('x', '##').replace('b', '-')}"


def _pitch_class(spelling: str) -> int | None:
    converted = _normalize_tonal_spelling(spelling)
    if not converted:
        return None
    accidental = converted[1:]
    return (
        _NATURAL_PITCH_CLASSES[converted[0]]
        + accidental.count("#")
        - accidental.count("-")
    ) % 12


def _spell_pitch(letter: str, pitch_class: int) -> str:
    difference = (pitch_class - _NATURAL_PITCH_CLASSES[letter]) % 12
    if difference > 6:
        difference -= 12
    accidental = "#" * difference if difference >= 0 else "-" * (-difference)
    return f"{letter}{accidental}"


def _transpose_diatonic(tonic: str, degree: str, mode: str) -> str:
    tonic_spelling = _normalize_tonal_spelling(tonic)
    match = re.fullmatch(r"([#-]*)([1-7])", degree)
    if not tonic_spelling or not match or mode not in _SCALE_INTERVALS:
        return f"<UNMAPPED:degree-root={tonic}/{mode}/{degree}>"
    alteration, number_text = match.groups()
    number = int(number_text)
    tonic_pc = _pitch_class(tonic_spelling)
    assert tonic_pc is not None
    target_pc = (
        tonic_pc
        + _SCALE_INTERVALS[mode][number - 1]
        + alteration.count("#")
        - alteration.count("-")
    ) % 12
    tonic_letter_index = NOTE_NAMES.index(tonic_spelling[0])
    target_letter = NOTE_NAMES[(tonic_letter_index + number - 1) % 7]
    return _spell_pitch(target_letter, target_pc)


def _figure_degree_and_mode(token: str) -> tuple[str, str]:
    token = re.sub(r"\[.*?\]", "", token)
    if token.startswith("N"):
        return "-2", "major"
    special = _augmented_six_prefix(token)
    if special:
        return SPECIAL_AUGMENTED_SIX_DEGREES[special], "major"
    match = re.match(r"^([#bx-]*)([ivIV]+)", token)
    if not match:
        return f"<UNMAPPED:degree={token}>", "major"
    alteration, numeral = match.groups()
    degree = _roman_figure_to_degree(f"{alteration}{numeral}", allow_none=False)
    mode = "major" if numeral.isupper() else "minor"
    return degree, mode


def _key_label(tonic: str, mode: str) -> str:
    if tonic.startswith("<UNMAPPED:"):
        return tonic
    pdcml = tonic.replace("-", "b")
    return pdcml if mode == "major" else pdcml[0].lower() + pdcml[1:]


def _tonicized_state(
    row: Mapping[str, Any], secondary_expression: str
) -> tuple[str, str]:
    tonic = _norm(row.get("key"))
    mode_text = _norm(row.get("mode"))
    mode = "minor" if mode_text == "m" or mode_text.lower() == "minor" else "major"
    if not secondary_expression:
        return pdcml_to_augnet_spelling(tonic), mode
    for token in reversed(secondary_expression.split("/")):
        degree, next_mode = _figure_degree_and_mode(token)
        tonic = _transpose_diatonic(tonic, degree, mode)
        mode = next_mode
    return tonic, mode


def _chord_tones_and_factors(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    tones_text = _norm(row.get("chord_tones"))
    factors_text = _norm(row.get("chord_tone_factors", row.get("chord_factors")))
    tones = tones_text.split("-") if tones_text else []
    factors = factors_text.split("-") if factors_text else []
    return tones, factors


def _spell_factor(root: str, quality: str, factor: str) -> str:
    intervals = _FACTOR_INTERVALS.get(quality, {})
    if root.startswith("<UNMAPPED:") or factor not in intervals:
        return f"<UNMAPPED:factor={quality}/{factor}>"
    generic_steps, semitones = intervals[factor]
    root_pc = _pitch_class(root)
    if root_pc is None:
        return f"<UNMAPPED:factor-root={root}>"
    root_letter_index = NOTE_NAMES.index(_normalize_tonal_spelling(root)[0])
    target_letter = NOTE_NAMES[(root_letter_index + generic_steps) % 7]
    return _spell_pitch(target_letter, (root_pc + semitones) % 12)


def _annotation_bass(
    row: Mapping[str, Any], root: str, quality: str, inversion: str
) -> str:
    try:
        factor = _INVERSION_FACTORS[int(inversion)]
    except (ValueError, IndexError):
        return f"<UNMAPPED:bass-inversion={inversion}>"
    tones, factors = _chord_tones_and_factors(row)
    if factor in factors:
        return _unbounded_augnet_spelling(tones[factors.index(factor)])
    return _spell_factor(root, quality, factor)


def _pitch_class_set_label(row: Mapping[str, Any], root: str, quality: str) -> str:
    tones, _ = _chord_tones_and_factors(row)
    if not tones:
        tones = [
            _spell_factor(root, quality, factor)
            for factor in _FACTOR_INTERVALS.get(quality, {})
        ]
    pitch_classes = [_pitch_class(tone) for tone in tones]
    if not pitch_classes or any(pc is None for pc in pitch_classes):
        return "<UNMAPPED:pitch-class-set>"
    return ",".join(str(pc) for pc in sorted(set(pitch_classes)))


def _figure_inversion(primary_figure: str) -> str:
    if _augmented_six_prefix(primary_figure):
        return AUGMENTED_SIX_INVERSION_BY_FIGURE.get(
            primary_figure, f"<UNMAPPED:augmented-six-figure={primary_figure}>"
        )
    figure = re.sub(r"\[.*?\]", "", primary_figure)
    for suffix, inversion in (
        ("65", "1"),
        ("43", "2"),
        ("42", "3"),
        ("64", "2"),
        ("2", "3"),
        ("6", "1"),
    ):
        if figure.endswith(suffix):
            return inversion
    return "0"


def _labels_from_row(row: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """Extract fixed-schema AugNet component and auxiliary labels."""
    gt_rnbert = _norm(row.get("gt_rnbert"))
    rn_main = _main_rn_text(gt_rnbert)
    primary_figure, *secondary_parts = rn_main.split("/", 1) if rn_main else [""]
    secondary_figure = secondary_parts[0] if secondary_parts else ""
    special_prefix = _augmented_six_prefix(primary_figure)

    parsed: dict[str, str] | None = None
    if gt_rnbert:
        try:
            parsed = parse_gt_rnbert(gt_rnbert)
        except ValueError:
            parsed = None

    if special_prefix:
        primary = SPECIAL_AUGMENTED_SIX_DEGREES[special_prefix]
        secondary = _roman_figure_to_degree(secondary_figure, allow_none=True)
        source = "gt_rnbert"
    elif primary_figure.startswith("N"):
        primary = "-2"
        secondary = _roman_figure_to_degree(secondary_figure, allow_none=True)
        source = "gt_rnbert"
    elif parsed is not None:
        primary = _roman_figure_to_degree(
            f"{parsed.get('primary_alteration', '')}{parsed.get('primary_numeral', '')}",
            allow_none=False,
        )
        secondary = _roman_figure_to_degree(
            f"{parsed.get('secondary_alteration', '')}{parsed.get('secondary_numeral', '')}",
            allow_none=True,
        )
        source = "gt_rnbert"
    else:
        raw_primary = _norm(row.get("primary_degree")).split("/", 1)[0]
        primary = _roman_figure_to_degree(
            f"{_norm(row.get('primary_alteration'))}{raw_primary}",
            allow_none=False,
        )
        raw_secondary = _norm(row.get("secondary_degree"))
        secondary = (
            NONE_LABEL
            if raw_secondary.upper() == "I"
            and not _norm(row.get("secondary_alteration"))
            else _roman_figure_to_degree(
                f"{_norm(row.get('secondary_alteration'))}{raw_secondary}",
                allow_none=True,
            )
        )
        source = "raw_columns"

    raw_quality = _norm(row.get("quality"))
    if special_prefix:
        quality = (
            AUGMENTED_SIX_QUALITIES[special_prefix]
            if raw_quality == "aug6"
            else f"<UNMAPPED:{raw_quality or 'missing'}-{special_prefix}>"
        )
    else:
        quality = PDCML_TO_AUGNET_QUALITY.get(
            raw_quality, f"<UNMAPPED:{raw_quality or 'missing'}>"
        )
    raw_inversion = pd.to_numeric(row.get("inversion"), errors="coerce")
    inversion = (
        _figure_inversion(primary_figure)
        if primary_figure
        else ("" if pd.isna(raw_inversion) else str(int(raw_inversion)))
    )

    tonicized_tonic, tonicized_mode = _tonicized_state(row, secondary_figure)
    tones, factors = _chord_tones_and_factors(row)
    if "R" in factors:
        chord_root = _unbounded_augnet_spelling(tones[factors.index("R")])
    else:
        chord_root = _transpose_diatonic(tonicized_tonic, primary, tonicized_mode)
    tonicized_key = (
        _canonical_key(row)
        if not secondary_figure
        else _key_label(tonicized_tonic, tonicized_mode)
    )
    bass = _annotation_bass(row, chord_root, quality, inversion)
    pitch_class_set = _pitch_class_set_label(row, chord_root, quality)
    labels = {
        "local_key": _canonical_key(row),
        "primary_degree": primary,
        "secondary_degree": secondary,
        "inversion": inversion,
        "quality": quality,
        "chord_root": chord_root,
        "bass": bass,
        "roman_numeral_76": roman_numeral_76_label(gt_rnbert),
        "tonicized_key": tonicized_key,
        "pitch_class_set": pitch_class_set,
        "harmonic_rhythm": "True",
    }
    return labels, source


def _read_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    available = set(header)
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    selected = sorted((REQUIRED_COLUMNS | OPTIONAL_COLUMNS) & available)
    return pd.read_csv(path, usecols=selected, low_memory=False)


def extract_harmony_events(
    df: pd.DataFrame, source_path: str = ""
) -> tuple[list[HarmonyEvent], dict[str, Any]]:
    harmony_rows = df[
        df["harmony_onset"].astype(str).str.strip().str.lower().eq("yes")
    ].copy()
    harmony_rows["onset"] = pd.to_numeric(harmony_rows["onset"], errors="coerce")
    harmony_rows = harmony_rows.dropna(subset=["onset"])
    events: list[HarmonyEvent] = []
    source_counts: Counter[str] = Counter()
    augmented_six_subtypes: Counter[str] = Counter()
    unresolved_special = 0
    unmapped_quality = 0
    canonical_inversion_overrides = 0
    augmented_six_inversion_overrides = 0
    augmented_six_roundtrip_mismatches = 0

    for onset, group in harmony_rows.groupby("onset", sort=True):
        candidates: list[tuple[dict[str, str], str, str, str, str]] = []
        for _, row in group.iterrows():
            labels, label_source = _labels_from_row(row)
            gt = _norm(row.get("gt_rnbert"))
            tones = _norm(row.get("chord_tones"))
            factors = _norm(row.get("chord_tone_factors", row.get("chord_factors")))
            candidates.append((labels, label_source, gt, tones, factors))

        unique_labels = {tuple(candidate[0].items()) for candidate in candidates}
        if len(unique_labels) != 1:
            raise ValueError(
                f"Conflicting harmony labels at onset {onset} in {source_path}: "
                f"{sorted(unique_labels)}"
            )
        labels, label_source, gt, tones, factors = candidates[0]
        source_counts[label_source] += 1
        primary_figure = _main_rn_text(gt).split("/", 1)[0]
        special_prefix = _augmented_six_prefix(primary_figure)
        canonical_rn = canonicalize_special_rn(gt)
        unresolved_special += int(
            labels["primary_degree"].startswith("<UNMAPPED:")
            or labels["inversion"].startswith("<UNMAPPED:")
            or (
                labels["quality"].startswith("<UNMAPPED:")
                and "aug6" in labels["quality"]
            )
        )
        unmapped_quality += int(labels["quality"].startswith("<UNMAPPED:"))
        raw_inversion = pd.to_numeric(group.iloc[0].get("inversion"), errors="coerce")
        raw_inversion_label = "" if pd.isna(raw_inversion) else str(int(raw_inversion))
        inversion_overridden = raw_inversion_label != labels["inversion"]
        canonical_inversion_overrides += int(inversion_overridden)
        if special_prefix:
            augmented_six_subtypes[special_prefix] += 1
            augmented_six_inversion_overrides += int(inversion_overridden)
            decoded_primary = decode_augmented_six_rn(
                labels["quality"], labels["inversion"]
            )
            canonical_primary = canonical_rn.split("/", 1)[0]
            augmented_six_roundtrip_mismatches += int(
                decoded_primary != canonical_primary
            )
        events.append(
            HarmonyEvent(
                onset=float(onset),
                labels=labels,
                gt_rnbert=gt,
                canonical_rn=canonical_rn,
                chord_tones=tones,
                chord_factors=factors,
                source=label_source,
            )
        )

    return events, {
        "harmony_rows": int(len(harmony_rows)),
        "harmony_events": len(events),
        "label_source_counts": dict(source_counts),
        "augmented_six_subtype_counts": dict(augmented_six_subtypes),
        "unresolved_special_events": unresolved_special,
        "unmapped_quality_events": unmapped_quality,
        "canonical_inversion_overrides": canonical_inversion_overrides,
        "augmented_six_inversion_overrides": augmented_six_inversion_overrides,
        "augmented_six_roundtrip_mismatches": augmented_six_roundtrip_mismatches,
    }


def _frame_index_at_or_after(offset: float, fixed_offset: float) -> int:
    return max(0, int(math.ceil(offset / fixed_offset - 1e-9)))


def _extract_notes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    notes = df[df["type"].astype(str).str.lower().eq("note")].copy()
    for column in ("pitch", "onset", "release"):
        notes[column] = pd.to_numeric(notes[column], errors="coerce")
    invalid = notes[["pitch", "onset", "release"]].isna().any(axis=1)
    invalid |= notes["release"] <= notes["onset"]
    invalid_count = int(invalid.sum())
    return notes[~invalid].sort_values(["onset", "pitch"]), invalid_count


def convert_piece(
    csv_path: Path,
    *,
    relative_path: str | None = None,
    fixed_offset: float = FIXED_OFFSET,
    canonicalize_extreme_input_spellings: bool = False,
) -> PieceTensors:
    """Convert one score-level CSV without padding or vocabulary encoding."""
    df = _read_csv(csv_path)
    notes, invalid_note_rows = _extract_notes(df)
    if notes.empty:
        raise ValueError(f"No valid note rows in {csv_path}")
    events, event_diagnostics = extract_harmony_events(df, str(csv_path))
    if not events:
        raise ValueError(f"No harmony-onset rows in {csv_path}")

    end_offset = float(notes["release"].max())
    n_frames = max(1, _frame_index_at_or_after(end_offset, fixed_offset))
    frame_offsets = np.arange(n_frames, dtype=np.float64) * fixed_offset
    bass19 = np.zeros((n_frames, 19), dtype=np.uint8)
    chromagram19 = np.zeros((n_frames, 19), dtype=np.uint8)
    bass_chromagram70 = np.zeros((n_frames, 70), dtype=np.uint8)
    bass_midi = np.full(n_frames, np.inf, dtype=np.float64)
    bass_spellings = np.full(n_frames, "", dtype=object)
    unsupported_spelling_rows = 0
    unsupported_spelling_frame_assignments = 0
    enharmonic_input_spelling_rows = 0
    enharmonic_input_spelling_frame_assignments = 0
    spelling_pitch_class_mismatches = 0

    letter_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    for row in notes.itertuples(index=False):
        start = min(n_frames, _frame_index_at_or_after(float(row.onset), fixed_offset))
        stop = min(n_frames, _frame_index_at_or_after(float(row.release), fixed_offset))
        if stop <= start:
            continue
        midi = float(row.pitch)
        pitch_class = int(round(midi)) % 12
        raw_spelling = _norm(row.spelling)
        letter = raw_spelling[:1].upper()
        augnet_spelling = pdcml_to_augnet_spelling(raw_spelling)
        if not augnet_spelling and canonicalize_extreme_input_spellings:
            augnet_spelling = canonicalize_enharmonic_to_classes(
                raw_spelling, SPELLINGS, preserve_mode_case=False
            )
            if augnet_spelling:
                enharmonic_input_spelling_rows += 1
                enharmonic_input_spelling_frame_assignments += stop - start

        if letter in NOTE_NAMES:
            letter_index = NOTE_NAMES.index(letter)
            chromagram19[start:stop, letter_index] = 1
            chromagram19[start:stop, 7 + pitch_class] = 1
        if augnet_spelling:
            spelling_index = SPELLINGS.index(augnet_spelling)
            bass_chromagram70[start:stop, 35 + spelling_index] = 1
            accidental = augnet_spelling[1:]
            expected_pc = (
                letter_to_pc[augnet_spelling[0]]
                + accidental.count("#")
                - accidental.count("-")
            ) % 12
            spelling_pitch_class_mismatches += int(expected_pc != pitch_class)
        else:
            unsupported_spelling_rows += 1
            unsupported_spelling_frame_assignments += stop - start

        replace = midi < bass_midi[start:stop]
        if np.any(replace):
            target = np.flatnonzero(replace) + start
            bass_midi[target] = midi
            bass_spellings[target] = augnet_spelling

    sounding = np.isfinite(bass_midi)
    for frame in np.flatnonzero(sounding):
        midi = int(round(bass_midi[frame]))
        spelling = str(bass_spellings[frame])
        if spelling:
            letter_index = NOTE_NAMES.index(spelling[0])
            spelling_index = SPELLINGS.index(spelling)
            bass19[frame, letter_index] = 1
            bass19[frame, 7 + midi % 12] = 1
            bass_chromagram70[frame, spelling_index] = 1

    empty_score_frames_before_fill = int((~sounding).sum())
    if empty_score_frames_before_fill:
        # AugmentedNet v1.0 score_parser._reindexDataFrame applies ffill then
        # bfill to s_notes. Reproduce that protocol explicitly for rests.
        sounding_indices = np.flatnonzero(sounding)
        frame_indices = np.arange(n_frames)
        source_positions = (
            np.searchsorted(sounding_indices, frame_indices, side="right") - 1
        )
        source_positions[source_positions < 0] = 0
        source_frames = sounding_indices[source_positions]
        empty = ~sounding
        bass19[empty] = bass19[source_frames[empty]]
        chromagram19[empty] = chromagram19[source_frames[empty]]
        bass_chromagram70[empty] = bass_chromagram70[source_frames[empty]]

    label_strings = {
        component: np.full(n_frames, "", dtype=object) for component in COMPONENTS
    }
    label_mask = np.zeros(n_frames, dtype=bool)
    off_grid_harmony_events = 0
    for index, event in enumerate(events):
        start = min(n_frames, _frame_index_at_or_after(event.onset, fixed_offset))
        stop = (
            n_frames
            if index + 1 == len(events)
            else min(
                n_frames,
                _frame_index_at_or_after(events[index + 1].onset, fixed_offset),
            )
        )
        off_grid_harmony_events += int(
            not math.isclose(
                event.onset / fixed_offset,
                round(event.onset / fixed_offset),
                abs_tol=1e-7,
            )
        )
        if stop <= start:
            continue
        for component in COMPONENTS:
            if component == "harmonic_rhythm":
                label_strings[component][start:stop] = "False"
            else:
                label_strings[component][start:stop] = event.labels[component]
        if math.isclose(
            event.onset / fixed_offset,
            round(event.onset / fixed_offset),
            abs_tol=1e-7,
        ):
            label_strings["harmonic_rhythm"][start] = "True"
        supervised = all(bool(event.labels[component]) for component in CORE_COMPONENTS)
        supervised &= not any(
            event.labels[component].startswith("<UNMAPPED:")
            for component in CORE_COMPONENTS
        )
        label_mask[start:stop] = supervised

    reference_note_onsets = np.sort(notes["onset"].unique().astype(np.float64))
    reference_note_onset_frames = np.asarray(
        [
            min(n_frames - 1, _frame_index_at_or_after(onset, fixed_offset))
            for onset in reference_note_onsets
        ],
        dtype=np.int64,
    )
    diagnostics = {
        **event_diagnostics,
        "frames": n_frames,
        "reference_note_onsets": int(len(reference_note_onsets)),
        "reference_note_onset_grid_frames": int(
            len(np.unique(reference_note_onset_frames))
        ),
        "reference_note_onset_grid_collisions": int(
            len(reference_note_onsets) - len(np.unique(reference_note_onset_frames))
        ),
        "duration_quarter_notes": end_offset,
        "fixed_offset_quarter_notes": fixed_offset,
        "invalid_note_rows": invalid_note_rows,
        "empty_score_frames_before_fill": empty_score_frames_before_fill,
        "empty_score_frames_after_fill": int((~np.any(bass19, axis=1)).sum()),
        "rest_fill_policy": "forward-fill then backward-fill (AugmentedNet v1.0)",
        "unsupervised_score_frames": int((~label_mask).sum()),
        "unsupported_spelling_rows": unsupported_spelling_rows,
        "unsupported_spelling_frame_assignments": unsupported_spelling_frame_assignments,
        "enharmonic_input_spelling_rows": enharmonic_input_spelling_rows,
        "enharmonic_input_spelling_frame_assignments": (
            enharmonic_input_spelling_frame_assignments
        ),
        "spelling_pitch_class_mismatches": spelling_pitch_class_mismatches,
        "off_grid_note_onsets": int(
            (
                ~np.isclose(
                    notes["onset"] / fixed_offset,
                    np.round(notes["onset"] / fixed_offset),
                )
            ).sum()
        ),
        "off_grid_harmony_events": off_grid_harmony_events,
    }
    return PieceTensors(
        relative_path=relative_path or csv_path.name,
        frame_offsets=frame_offsets,
        reference_note_onsets=reference_note_onsets,
        harmony_events=tuple(events),
        bass19=bass19,
        chromagram19=chromagram19,
        bass_chromagram70=bass_chromagram70,
        label_strings=label_strings,
        label_mask=label_mask,
        diagnostics=diagnostics,
    )


def _is_legal_ontology_label(label: str) -> bool:
    return bool(label) and not label.startswith("<UNMAPPED:")


def build_vocabularies(
    dataset_root: Path, split_dir: Path
) -> tuple[dict[str, list[str]], dict[str, Counter[str]], dict[str, Any]]:
    """Build frozen core ontologies and official fixed auxiliary schemas."""
    counters = {component: Counter() for component in COMPONENTS}
    aggregate: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    manifest_metadata: dict[str, Any] = {}
    total_files = 0

    for split in ("train", "valid", "test"):
        manifest = split_dir / f"{split}_paths.txt"
        files = read_split_paths(manifest)
        total_files += len(files)
        manifest_metadata[split] = {
            "path": str(manifest),
            "sha256": manifest_sha256(manifest),
            "files": len(files),
        }
        for relative_path in files:
            df = _read_csv(dataset_root / relative_path)
            events, diagnostics = extract_harmony_events(df, relative_path)
            for event in events:
                for component in COMPONENTS:
                    counters[component][event.labels[component]] += 1
            aggregate.update(diagnostics["label_source_counts"])
            subtype_counts.update(diagnostics["augmented_six_subtype_counts"])
            for key in (
                "harmony_events",
                "unresolved_special_events",
                "unmapped_quality_events",
                "canonical_inversion_overrides",
                "augmented_six_inversion_overrides",
                "augmented_six_roundtrip_mismatches",
            ):
                aggregate[key] += diagnostics[key]

    vocabularies: dict[str, list[str]] = {}
    schema_extensions: dict[str, list[str]] = {}
    masked_auxiliary_labels: dict[str, list[str]] = {}
    for component in COMPONENTS:
        official = list(OFFICIAL_CLASS_ORDER[component])
        observed = {
            label for label in counters[component] if _is_legal_ontology_label(label)
        }
        extensions = sorted(observed - set(official))
        if component in AUXILIARY_COMPONENTS:
            vocabularies[component] = official
            schema_extensions[component] = []
            masked_auxiliary_labels[component] = extensions
        else:
            vocabularies[component] = official + extensions
            schema_extensions[component] = extensions

    return (
        vocabularies,
        counters,
        {
            "files": total_files,
            "manifests": manifest_metadata,
            "vocabulary_source": (
                "unaugmented train+valid+test corpus ontology for core heads; "
                "official AugmentedNet v1.0 classes for auxiliary heads"
            ),
            "uses_label_frequencies": False,
            "contains_unknown_class": False,
            "auxiliary_out_of_schema_policy": (
                "mask loss and metrics for the affected auxiliary head only"
            ),
            "official_class_counts": {
                component: len(OFFICIAL_CLASS_ORDER[component])
                for component in COMPONENTS
            },
            "schema_extensions": schema_extensions,
            "masked_auxiliary_labels": masked_auxiliary_labels,
            "augmented_six_subtype_counts": dict(sorted(subtype_counts.items())),
            **dict(aggregate),
        },
    )


def _pad_first_axis(array: np.ndarray, target: int) -> np.ndarray:
    if len(array) >= target:
        return array
    return np.pad(array, ((0, target - len(array)),) + ((0, 0),) * (array.ndim - 1))


def _segment(array: np.ndarray, sequence_length: int) -> np.ndarray:
    n_sequences = max(1, math.ceil(len(array) / sequence_length))
    padded = _pad_first_axis(array, n_sequences * sequence_length)
    return padded.reshape((n_sequences, sequence_length) + array.shape[1:])


def encode_and_segment(
    piece: PieceTensors,
    vocabularies: Mapping[str, Sequence[str]],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict[str, np.ndarray]:
    """Encode labels and return padding-safe arrays with per-head masks."""
    n_frames = len(piece.frame_offsets)
    valid_mask = np.ones(n_frames, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "X_Bass19": _segment(piece.bass19, sequence_length),
        "X_Chromagram19": _segment(piece.chromagram19, sequence_length),
        "X_BassChromagram70": _segment(piece.bass_chromagram70, sequence_length),
        "valid_mask": _segment(valid_mask, sequence_length),
        "label_mask": _segment(piece.label_mask.astype(np.float32), sequence_length),
        "frame_offsets": _segment(piece.frame_offsets, sequence_length),
    }
    for component in COMPONENTS:
        classes = list(vocabularies[component])
        if len(classes) != len(set(classes)):
            raise ValueError(f"Duplicate classes in {component} ontology")
        label_to_index = {label: index for index, label in enumerate(classes)}
        encoded = np.zeros(n_frames, dtype=np.int32)
        component_mask = piece.label_mask.copy()
        for frame, raw_label in enumerate(piece.label_strings[component]):
            label = str(raw_label)
            if not label:
                component_mask[frame] = False
                continue
            if label not in label_to_index:
                if (
                    component in AUXILIARY_COMPONENTS
                    and piece.label_mask[frame]
                    and _is_legal_ontology_label(label)
                ):
                    component_mask[frame] = False
                    continue
                offset = float(piece.frame_offsets[frame])
                raise ValueError(
                    f"Unknown label for {component} in {piece.relative_path} "
                    f"at frame {frame} (offset {offset}): {label!r}"
                )
            encoded[frame] = label_to_index[label]
        segmented = _segment(encoded, sequence_length)
        segmented_mask = _segment(component_mask.astype(np.float32), sequence_length)
        arrays[f"y_{component}"] = segmented[..., np.newaxis]
        arrays[f"label_mask_{component}"] = segmented_mask
        arrays[f"mi20_y_{component}"] = segmented[:, ::MICCHI_DOWNSAMPLE, np.newaxis]
        arrays[f"mi20_label_mask_{component}"] = segmented_mask[:, ::MICCHI_DOWNSAMPLE]
    arrays["mi20_label_mask"] = arrays["label_mask"][:, ::MICCHI_DOWNSAMPLE]
    return arrays


def save_piece_npz(
    output_path: Path,
    piece: PieceTensors,
    vocabularies: Mapping[str, Sequence[str]],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = encode_and_segment(piece, vocabularies, sequence_length=sequence_length)
    np.savez_compressed(output_path, **arrays)


def _scan_split(
    dataset_root: Path,
    manifest: Path,
    vocabularies: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    paths = read_split_paths(manifest)
    label_counts = {component: Counter() for component in COMPONENTS}
    oov_counts = {component: Counter() for component in COMPONENTS}
    spelling_counts: Counter[str] = Counter()
    aggregate: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    auxiliary_coverage = {component: Counter() for component in AUXILIARY_COMPONENTS}
    examples: dict[str, list[str]] = defaultdict(list)

    for relative_path in paths:
        df = _read_csv(dataset_root / relative_path)
        events, diagnostics = extract_harmony_events(df, relative_path)
        for event in events:
            for component in COMPONENTS:
                value = event.labels[component]
                label_counts[component][value] += 1
                if value not in vocabularies[component]:
                    if component in AUXILIARY_COMPONENTS and _is_legal_ontology_label(
                        value
                    ):
                        auxiliary_coverage[component]["masked"] += 1
                        if len(examples[f"masked_{component}"]) < 10:
                            examples[f"masked_{component}"].append(
                                f"{relative_path}@{event.onset}:{value}"
                            )
                    else:
                        oov_counts[component][value] += 1
                        if len(examples[f"oov_{component}"]) < 10:
                            examples[f"oov_{component}"].append(
                                f"{relative_path}@{event.onset}:{value}"
                            )
                elif component in AUXILIARY_COMPONENTS:
                    auxiliary_coverage[component]["covered"] += 1
        notes, invalid_note_rows = _extract_notes(df)
        aggregate["invalid_note_rows"] += invalid_note_rows
        for spelling, count in notes["spelling"].map(_norm).value_counts().items():
            spelling_counts[str(spelling)] += int(count)
        aggregate.update(diagnostics["label_source_counts"])
        subtype_counts.update(diagnostics["augmented_six_subtype_counts"])
        for key in (
            "harmony_rows",
            "harmony_events",
            "unresolved_special_events",
            "unmapped_quality_events",
            "canonical_inversion_overrides",
            "augmented_six_inversion_overrides",
            "augmented_six_roundtrip_mismatches",
        ):
            aggregate[key] += diagnostics[key]

    unsupported_spellings = {
        spelling: count
        for spelling, count in spelling_counts.items()
        if not pdcml_to_augnet_spelling(spelling)
    }
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "files": len(paths),
        "counts": dict(aggregate),
        "augmented_six_subtype_counts": dict(sorted(subtype_counts.items())),
        "label_event_counts": {
            component: dict(sorted(counter.items()))
            for component, counter in label_counts.items()
        },
        "oov_event_counts": {
            component: {
                "total": int(sum(counter.values())),
                "labels": dict(sorted(counter.items())),
            }
            for component, counter in oov_counts.items()
        },
        "auxiliary_coverage": {
            component: {
                "covered_events": int(counter["covered"]),
                "masked_events": int(counter["masked"]),
                "total_events": int(counter["covered"] + counter["masked"]),
                "coverage": (
                    counter["covered"] / (counter["covered"] + counter["masked"])
                    if counter["covered"] + counter["masked"]
                    else 0.0
                ),
            }
            for component, counter in auxiliary_coverage.items()
        },
        "unsupported_input_spelling_rows": int(sum(unsupported_spellings.values())),
        "unsupported_input_spellings": dict(sorted(unsupported_spellings.items())),
        "examples": dict(examples),
    }


def canonicalize_piece_enharmonic_labels(
    piece: PieceTensors, vocabularies: Mapping[str, Sequence[str]]
) -> Counter[str]:
    """Canonicalize augmentation-only tonal OOVs without editing source CSVs."""
    normalizations: Counter[str] = Counter()
    for component, preserve_mode_case in (
        ("local_key", True),
        ("chord_root", False),
    ):
        classes = vocabularies[component]
        values = piece.label_strings[component]
        for label in sorted(set(map(str, values.tolist())) - {""}):
            if label in classes:
                continue
            replacement = canonicalize_enharmonic_to_classes(
                label, classes, preserve_mode_case=preserve_mode_case
            )
            if not replacement:
                raise ValueError(
                    f"Cannot enharmonically canonicalize {component} label "
                    f"{label!r} in {piece.relative_path}"
                )
            affected = values == label
            frame_count = int(affected.sum())
            values[affected] = replacement
            normalizations[f"{component}:{label}->{replacement}"] += frame_count
    return normalizations


def _validate_encoded_arrays(
    piece: PieceTensors, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Assert AugN/Mi20 shape, padding, mask, and downsampling invariants."""
    n_frames = len(piece.frame_offsets)
    n_sequences = max(1, math.ceil(n_frames / SEQUENCE_LENGTH))
    padded_frames = n_sequences * SEQUENCE_LENGTH
    expected_inputs = {
        "X_Bass19": (n_sequences, SEQUENCE_LENGTH, 19),
        "X_Chromagram19": (n_sequences, SEQUENCE_LENGTH, 19),
        "X_BassChromagram70": (n_sequences, SEQUENCE_LENGTH, 70),
    }
    for name, shape in expected_inputs.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Unexpected {name} shape for {piece.relative_path}: "
                f"{arrays[name].shape}"
            )
        if not np.isin(arrays[name], (0, 1)).all():
            raise ValueError(
                f"Non-binary input values in {name} for {piece.relative_path}"
            )

    valid_mask = arrays["valid_mask"]
    label_mask = arrays["label_mask"]
    expected_valid = np.zeros(padded_frames, dtype=np.float32)
    expected_valid[:n_frames] = 1
    if not np.array_equal(valid_mask.reshape(-1), expected_valid):
        raise ValueError(f"Invalid padding mask for {piece.relative_path}")
    if np.any(label_mask > valid_mask):
        raise ValueError(f"Label mask enters padding for {piece.relative_path}")
    if not np.array_equal(
        arrays["mi20_label_mask"], label_mask[:, ::MICCHI_DOWNSAMPLE]
    ):
        raise ValueError(
            f"Mi20 shared-mask downsampling mismatch for {piece.relative_path}"
        )

    head_frames: dict[str, int] = {}
    mi20_head_frames: dict[str, int] = {}
    for component in COMPONENTS:
        y = arrays[f"y_{component}"]
        head_mask = arrays[f"label_mask_{component}"]
        mi20_y = arrays[f"mi20_y_{component}"]
        mi20_mask = arrays[f"mi20_label_mask_{component}"]
        if y.shape != (n_sequences, SEQUENCE_LENGTH, 1):
            raise ValueError(
                f"Unexpected y_{component} shape for {piece.relative_path}: "
                f"{y.shape}"
            )
        if head_mask.shape != (n_sequences, SEQUENCE_LENGTH):
            raise ValueError(
                f"Unexpected {component} mask shape for {piece.relative_path}"
            )
        if np.any(head_mask > label_mask):
            raise ValueError(
                f"{component} mask exceeds core mask for {piece.relative_path}"
            )
        if component in CORE_COMPONENTS and not np.array_equal(head_mask, label_mask):
            raise ValueError(
                f"Core {component} mask differs from shared mask for "
                f"{piece.relative_path}"
            )
        if not np.array_equal(mi20_y, y[:, ::MICCHI_DOWNSAMPLE]):
            raise ValueError(
                f"Mi20 {component} labels are misaligned for {piece.relative_path}"
            )
        if not np.array_equal(mi20_mask, head_mask[:, ::MICCHI_DOWNSAMPLE]):
            raise ValueError(
                f"Mi20 {component} mask is misaligned for {piece.relative_path}"
            )
        head_frames[component] = int(head_mask.sum())
        mi20_head_frames[component] = int(mi20_mask.sum())

    return {
        "frames": n_frames,
        "sequences": n_sequences,
        "padded_frames": padded_frames,
        "padding_frames": padded_frames - n_frames,
        "valid_frames": int(valid_mask.sum()),
        "label_frames": int(label_mask.sum()),
        "mi20_label_frames": int(arrays["mi20_label_mask"].sum()),
        "head_label_frames": head_frames,
        "mi20_head_label_frames": mi20_head_frames,
    }


def _string_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype("string").fillna("<NA>")


def _validate_transposition_pair(
    base_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    *,
    base_path: str,
    augmented_path: str,
) -> int:
    """Return the constant MIDI shift after checking annotation invariants."""
    if len(base_df) != len(augmented_df):
        raise ValueError(f"Row-count mismatch: {base_path} vs {augmented_path}")
    stable_string_columns = (
        "type",
        "harmony_onset",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "quality",
        "inversion",
        "gt_rnbert",
        "chord_factors",
        "chord_tone_factors",
    )
    for column in stable_string_columns:
        if column not in base_df or column not in augmented_df:
            continue
        if not _string_series(base_df, column).equals(
            _string_series(augmented_df, column)
        ):
            raise ValueError(
                f"Transposition changed {column} in {base_path} -> {augmented_path}"
            )
    for column in ("onset", "release"):
        base_values = pd.to_numeric(base_df[column], errors="coerce").to_numpy()
        augmented_values = pd.to_numeric(
            augmented_df[column], errors="coerce"
        ).to_numpy()
        if not np.allclose(base_values, augmented_values, equal_nan=True):
            raise ValueError(
                f"Transposition changed {column} in {base_path} -> {augmented_path}"
            )

    base_notes = base_df["type"].astype(str).str.lower().eq("note")
    augmented_notes = augmented_df["type"].astype(str).str.lower().eq("note")
    if not np.array_equal(base_notes.to_numpy(), augmented_notes.to_numpy()):
        raise ValueError(f"Note-row mismatch: {base_path} vs {augmented_path}")
    base_pitch = pd.to_numeric(
        base_df.loc[base_notes, "pitch"], errors="coerce"
    ).to_numpy()
    augmented_pitch = pd.to_numeric(
        augmented_df.loc[augmented_notes, "pitch"], errors="coerce"
    ).to_numpy()
    deltas = augmented_pitch - base_pitch
    if len(deltas) == 0 or np.isnan(deltas).any():
        raise ValueError(f"Missing note pitches: {base_path} vs {augmented_path}")
    rounded = np.rint(deltas).astype(int)
    if not np.allclose(deltas, rounded) or len(set(rounded.tolist())) != 1:
        raise ValueError(
            "Pitch shift is not one constant integer in "
            f"{base_path} -> {augmented_path}"
        )
    return int(rounded[0])


def _audit_evaluation_copy_policy(
    base_root: Path,
    augmented_root: Path,
    base_split_dir: Path,
    augmented_split_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "selected_root": str(base_root),
        "policy": "validation and test are read only from the unaugmented base root",
        "splits": {},
    }
    for split in ("valid", "test"):
        base_manifest = base_split_dir / f"{split}_paths.txt"
        augmented_manifest = augmented_split_dir / f"{split}_paths.txt"
        base_paths = read_split_paths(base_manifest)
        augmented_paths = read_split_paths(augmented_manifest)
        if base_paths != augmented_paths:
            raise ValueError(f"Base and augmented {split} manifests differ")
        identical = 0
        differing_files: list[dict[str, Any]] = []
        for relative_path in base_paths:
            base_file = base_root / relative_path
            augmented_file = augmented_root / relative_path
            if base_file.read_bytes() == augmented_file.read_bytes():
                identical += 1
                continue
            base_df = _read_csv(base_file)
            augmented_df = _read_csv(augmented_file)
            if len(base_df) != len(augmented_df):
                raise ValueError(
                    f"Evaluation copy row-count mismatch for {relative_path}"
                )
            common_columns = sorted(
                (set(base_df.columns) & set(augmented_df.columns)) - {"Unnamed: 0"}
            )
            changed_columns: Counter[str] = Counter()
            examples: list[str] = []
            for column in common_columns:
                left = _string_series(base_df, column)
                right = _string_series(augmented_df, column)
                changed = left.ne(right)
                count = int(changed.sum())
                if count:
                    changed_columns[column] = count
                    for row_index in changed[changed].index[: 3 - len(examples)]:
                        examples.append(
                            f"row {row_index} {column}: "
                            f"{left.loc[row_index]!r} -> {right.loc[row_index]!r}"
                        )
                if len(examples) >= 3 and changed_columns:
                    continue
            differing_files.append(
                {
                    "relative_path": relative_path,
                    "changed_cells_by_column": dict(changed_columns),
                    "examples": examples,
                }
            )
        report["splits"][split] = {
            "files": len(base_paths),
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": manifest_sha256(base_manifest),
            "augmented_manifest": str(augmented_manifest),
            "augmented_manifest_sha256": manifest_sha256(augmented_manifest),
            "path_lists_identical": True,
            "byte_identical_files": identical,
            "differing_files": differing_files,
        }
    return report


def audit_augmented_training(
    base_root: Path,
    augmented_root: Path,
    base_split_dir: Path,
    augmented_split_dir: Path,
    augmentation_settings: Path,
    *,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Fully convert and audit the controlled 355 x 12 training corpus."""
    started = time.monotonic()
    vocabularies, _, vocabulary_metadata = build_vocabularies(base_root, base_split_dir)
    base_train_manifest = base_split_dir / "train_paths.txt"
    augmented_train_manifest = augmented_split_dir / "train_paths.txt"
    base_train_paths = read_split_paths(base_train_manifest)
    augmented_train_paths = read_split_paths(augmented_train_manifest)
    groups = validate_augmented_manifest(augmented_train_paths, base_train_paths)

    valid_paths = set(read_split_paths(base_split_dir / "valid_paths.txt"))
    test_paths = set(read_split_paths(base_split_dir / "test_paths.txt"))
    if set(base_train_paths) & (valid_paths | test_paths) or valid_paths & test_paths:
        raise ValueError("Base train/valid/test manifests are not disjoint")

    settings = json.loads(augmentation_settings.read_text())
    required_settings = {
        "aug_by_key": True,
        "aug_by_key_n_keys": 12,
        "aug_within_range": False,
        "aug_rhythms": False,
    }
    settings_mismatches = {
        key: {"expected": expected, "actual": settings.get(key)}
        for key, expected in required_settings.items()
        if settings.get(key) != expected
    }
    if settings_mismatches:
        raise ValueError(
            f"Unexpected saved augmentation settings: {settings_mismatches}"
        )

    evaluation_policy = _audit_evaluation_copy_policy(
        base_root, augmented_root, base_split_dir, augmented_split_dir
    )
    totals: Counter[str] = Counter()
    diagnostic_totals: Counter[str] = Counter()
    head_frames: Counter[str] = Counter()
    mi20_head_frames: Counter[str] = Counter()
    label_frame_counts = {component: Counter() for component in COMPONENTS}
    by_index = {
        index: {
            "files": 0,
            "frames": 0,
            "sequences": 0,
            "shift_counts": Counter(),
        }
        for index in range(12)
    }
    per_piece: list[dict[str, Any]] = []
    enharmonic_normalization_frames: Counter[str] = Counter()
    completed = 0
    for base_path in sorted(groups):
        base_df = _read_csv(base_root / base_path)
        group_shifts: list[int] = []
        for augmentation_index, relative_path in groups[base_path]:
            augmented_df = _read_csv(augmented_root / relative_path)
            shift = _validate_transposition_pair(
                base_df,
                augmented_df,
                base_path=base_path,
                augmented_path=relative_path,
            )
            group_shifts.append(shift)
            piece = convert_piece(
                augmented_root / relative_path,
                relative_path=relative_path,
                canonicalize_extreme_input_spellings=True,
            )
            piece_normalizations = canonicalize_piece_enharmonic_labels(
                piece, vocabularies
            )
            enharmonic_normalization_frames.update(piece_normalizations)
            arrays = encode_and_segment(piece, vocabularies)
            encoded = _validate_encoded_arrays(piece, arrays)

            totals["files"] += 1
            for key in (
                "frames",
                "sequences",
                "padded_frames",
                "padding_frames",
                "valid_frames",
                "label_frames",
                "mi20_label_frames",
            ):
                totals[key] += encoded[key]
            for component in COMPONENTS:
                head_frames[component] += encoded["head_label_frames"][component]
                mi20_head_frames[component] += encoded["mi20_head_label_frames"][
                    component
                ]
                values = piece.label_strings[component][piece.label_mask]
                label_frame_counts[component].update(map(str, values.tolist()))
            for key, value in piece.diagnostics.items():
                if isinstance(value, (int, np.integer)):
                    diagnostic_totals[key] += int(value)
            index_stats = by_index[augmentation_index]
            index_stats["files"] += 1
            index_stats["frames"] += encoded["frames"]
            index_stats["sequences"] += encoded["sequences"]
            index_stats["shift_counts"][str(shift)] += 1
            per_piece.append(
                {
                    "relative_path": relative_path,
                    "base_relative_path": base_path,
                    "augmentation_index": augmentation_index,
                    "midi_shift": shift,
                    "frames": encoded["frames"],
                    "sequences": encoded["sequences"],
                    "padding_frames": encoded["padding_frames"],
                    "label_frames": encoded["label_frames"],
                    "empty_score_frames_before_fill": piece.diagnostics[
                        "empty_score_frames_before_fill"
                    ],
                    "enharmonic_normalization_frames": int(
                        sum(piece_normalizations.values())
                    ),
                }
            )
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"converted={completed}/{len(augmented_train_paths)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
        if len(set(group_shifts)) != 12 or 0 not in group_shifts:
            raise ValueError(
                f"Expected 12 distinct pitch shifts including zero for {base_path}, "
                f"got {sorted(group_shifts)}"
            )

    failure_diagnostics = (
        "unresolved_special_events",
        "unmapped_quality_events",
        "augmented_six_roundtrip_mismatches",
        "empty_score_frames_after_fill",
        "unsupported_spelling_rows",
        "spelling_pitch_class_mismatches",
    )
    nonzero_failures = {
        key: int(diagnostic_totals[key])
        for key in failure_diagnostics
        if diagnostic_totals[key]
    }
    if nonzero_failures:
        raise ValueError(
            f"Full augmented conversion audit failures: {nonzero_failures}"
        )

    vocabulary_json = json.dumps(
        vocabularies, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return {
        "status": "pass",
        "input_access_mode": "read_only; converter has no source-CSV write path",
        "artifacts_persisted": (
            "audit JSON only; converted tensors were validated in memory"
        ),
        "dataset": {
            "base_root": str(base_root),
            "augmented_root": str(augmented_root),
            "base_train_manifest": str(base_train_manifest),
            "base_train_manifest_sha256": manifest_sha256(base_train_manifest),
            "augmented_train_manifest": str(augmented_train_manifest),
            "augmented_train_manifest_sha256": manifest_sha256(
                augmented_train_manifest
            ),
            "augmentation_settings": str(augmentation_settings),
            "augmentation_settings_sha256": manifest_sha256(augmentation_settings),
            "augmentation_settings_verified": required_settings,
        },
        "structure": {
            "base_train_pieces": len(base_train_paths),
            "transposition_multiplier": 12,
            "augmented_train_paths": len(augmented_train_paths),
            "groups_complete": True,
            "augmentation_indices": list(range(12)),
            "split_disjointness_verified": True,
            "suffix_semantics": (
                "enumeration index over the saved 12 target key signatures; "
                "not a semitone-shift value"
            ),
        },
        "ontology": {
            "source": vocabulary_metadata["vocabulary_source"],
            "sha256": hashlib.sha256(vocabulary_json).hexdigest(),
            "class_counts": {
                component: len(classes) for component, classes in vocabularies.items()
            },
            "classes": vocabularies,
            "core_unknown_labels": 0,
            "contains_unknown_class": False,
        },
        "evaluation_copy_policy": evaluation_policy,
        "totals": dict(totals),
        "diagnostic_totals": dict(sorted(diagnostic_totals.items())),
        "enharmonic_canonicalization": {
            "policy": (
                "in-memory only: augmentation-only local-key and chord-root "
                "spellings outside the fixed ontology are mapped to the "
                "least-altered enharmonic class; source CSVs remain unchanged"
            ),
            "difference_from_official_augnet": (
                "AugmentedNet v1 filters out transposition intervals when any "
                "local key leaves Key35; this controlled protocol retains the "
                "precommitted 12 variants and canonicalizes their spellings"
            ),
            "normalized_frames": int(sum(enharmonic_normalization_frames.values())),
            "frames_by_mapping": dict(sorted(enharmonic_normalization_frames.items())),
        },
        "head_label_frames": dict(head_frames),
        "mi20_head_label_frames": dict(mi20_head_frames),
        "label_frame_counts": {
            component: dict(sorted(counter.items()))
            for component, counter in label_frame_counts.items()
        },
        "by_augmentation_index": {
            str(index): {
                **{key: value for key, value in stats.items() if key != "shift_counts"},
                "shift_counts": dict(
                    sorted(
                        stats["shift_counts"].items(),
                        key=lambda item: int(item[0]),
                    )
                ),
            }
            for index, stats in by_index.items()
        },
        "per_piece": per_piece,
    }


def audit_dataset(dataset_root: Path, split_dir: Path) -> dict[str, Any]:
    vocabularies, corpus_counters, vocabulary_metadata = build_vocabularies(
        dataset_root, split_dir
    )
    report = {
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "source_annotation_requirement": (
            "processed base CSVs with gt_rnbert and restored special chords"
        ),
        "augmentednet_source": {
            "version": "v1.0.0",
            "commit": "709dc59470412453fd9046ef4fb53e5663141144",
            "fixed_offset_quarter_notes": FIXED_OFFSET,
            "sequence_length": SEQUENCE_LENGTH,
            "micchi_label_downsample": MICCHI_DOWNSAMPLE,
        },
        "vocabulary_metadata": vocabulary_metadata,
        "vocabularies": vocabularies,
        "corpus_label_event_counts": {
            component: dict(sorted(counter.items()))
            for component, counter in corpus_counters.items()
        },
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        report["splits"][split] = _scan_split(
            dataset_root,
            split_dir / f"{split}_paths.txt",
            vocabularies,
        )
    return report


def _print_audit_summary(report: Mapping[str, Any]) -> None:
    print(f"dataset_root={report['dataset_root']}")
    print("vocabulary_source=" f"{report['vocabulary_metadata']['vocabulary_source']}")
    for component, classes in report["vocabularies"].items():
        print(f"vocabulary[{component}]={len(classes)} classes")
    for split, split_report in report["splits"].items():
        print(
            f"[{split}] files={split_report['files']} "
            f"events={split_report['counts'].get('harmony_events', 0)} "
            f"unresolved_special={split_report['counts'].get('unresolved_special_events', 0)} "
            f"unmapped_quality={split_report['counts'].get('unmapped_quality_events', 0)} "
            f"canonical_inversion_overrides="
            f"{split_report['counts'].get('canonical_inversion_overrides', 0)} "
            f"aug6_inversion_overrides="
            f"{split_report['counts'].get('augmented_six_inversion_overrides', 0)} "
            f"aug6_roundtrip_mismatches="
            f"{split_report['counts'].get('augmented_six_roundtrip_mismatches', 0)} "
            f"unsupported_spelling_rows={split_report['unsupported_input_spelling_rows']}"
        )
        for component in COMPONENTS:
            oov = split_report["oov_event_counts"][component]
            print(f"  oov[{component}]={oov['total']} {oov['labels']}")
        for component in AUXILIARY_COMPONENTS:
            coverage = split_report["auxiliary_coverage"][component]
            print(
                f"  coverage[{component}]="
                f"{coverage['covered_events']}/{coverage['total_events']} "
                f"({coverage['coverage']:.4%}); "
                f"masked={coverage['masked_events']}"
            )


def _default_smoke_pieces(split_dir: Path) -> dict[str, str]:
    return {
        split: read_split_paths(split_dir / f"{split}_paths.txt")[0]
        for split in ("train", "valid", "test")
    }


def _parse_piece_overrides(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SPLIT=RELATIVE_PATH, got {value!r}")
        split, relative_path = value.split("=", 1)
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        parsed[split] = relative_path
    return parsed


def run_smoke(
    dataset_root: Path,
    split_dir: Path,
    output_dir: Path,
    piece_overrides: Iterable[str] = (),
) -> dict[str, Any]:
    vocabularies, _, vocabulary_metadata = build_vocabularies(dataset_root, split_dir)
    pieces = _default_smoke_pieces(split_dir)
    pieces.update(_parse_piece_overrides(piece_overrides))
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "vocabulary_metadata": vocabulary_metadata,
        "vocabularies": vocabularies,
        "pieces": {},
    }
    for split, relative_path in pieces.items():
        piece = convert_piece(dataset_root / relative_path, relative_path=relative_path)
        output_path = output_dir / f"{split}.npz"
        save_piece_npz(output_path, piece, vocabularies)
        arrays = encode_and_segment(piece, vocabularies)
        report["pieces"][split] = {
            "relative_path": relative_path,
            "output_npz": str(output_path),
            "diagnostics": piece.diagnostics,
            "shapes": {name: list(array.shape) for name, array in arrays.items()},
            "valid_frames": int(arrays["valid_mask"].sum()),
            "label_frames": int(arrays["label_mask"].sum()),
            "head_label_frames": {
                component: int(arrays[f"label_mask_{component}"].sum())
                for component in COMPONENTS
            },
            "padding_frames": int(
                arrays["valid_mask"].size - arrays["valid_mask"].sum()
            ),
            "mi20_label_frames": int(arrays["mi20_label_mask"].sum()),
        }
    metadata_path = output_dir / "smoke_report.json"
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {metadata_path}")
    return report


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROCESSED_SCORES_DIR,
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data_splits/p_dcml"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    _common_arguments(audit_parser)
    audit_parser.add_argument("--output-json", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    _common_arguments(smoke_parser)
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument(
        "--piece",
        action="append",
        default=[],
        help="Override a smoke piece as SPLIT=RELATIVE_PATH (repeatable).",
    )
    augmented_parser = subparsers.add_parser("audit-augmented-training")
    _common_arguments(augmented_parser)
    augmented_parser.add_argument(
        "--augmented-root",
        type=Path,
        required=True,
    )
    augmented_parser.add_argument(
        "--augmented-split-dir",
        type=Path,
        default=Path("data_splits/p_dcml/augmented_splits"),
    )
    augmented_parser.add_argument(
        "--augmentation-settings",
        type=Path,
        required=True,
    )
    augmented_parser.add_argument("--output-json", type=Path, required=True)
    augmented_parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.command == "audit":
        assert_output_outside_source_roots(args.output_json, [args.dataset_root])
        report = audit_dataset(args.dataset_root, args.split_dir)
        _print_audit_summary(report)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Wrote {args.output_json}")
    elif args.command == "smoke":
        assert_output_outside_source_roots(args.output_dir, [args.dataset_root])
        run_smoke(
            args.dataset_root,
            args.split_dir,
            args.output_dir,
            args.piece,
        )
    else:
        assert_output_outside_source_roots(
            args.output_json, [args.dataset_root, args.augmented_root]
        )
        report = audit_augmented_training(
            args.dataset_root,
            args.augmented_root,
            args.split_dir,
            args.augmented_split_dir,
            args.augmentation_settings,
            progress_every=args.progress_every,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            f"status={report['status']} files={report['totals']['files']} "
            f"frames={report['totals']['frames']}"
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
