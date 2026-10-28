"""Helper utilities for RNA component-wise metrics."""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from rngen.utils.rna import parse_gt_rnbert

MISSING_FACTOR = "<missing-factor>"
MISSING_TONE = "<missing-tone>"

AnnotationLike = Union[Mapping[str, Any], str]

_LETTER_TO_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_MAJOR_SCALE_DEGREE_TO_PC = {
    "I": 0,
    "II": 2,
    "III": 4,
    "IV": 5,
    "V": 7,
    "VI": 9,
    "VII": 11,
}

_MINOR_SCALE_DEGREE_TO_PC = {
    "I": 0,
    "II": 2,
    "III": 3,
    "IV": 5,
    "V": 7,
    "VI": 8,
    "VII": 10,
}


def _normalize_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_annotation_fields(annotation: AnnotationLike) -> Optional[Dict[str, str]]:
    """Extract structured RNA fields from a parsed annotation or RN string."""
    if isinstance(annotation, str):
        try:
            return parse_gt_rnbert(annotation)
        except ValueError:
            return None

    if not isinstance(annotation, Mapping):
        return None

    if "primary_numeral" in annotation:
        return {
            "primary_numeral": _normalize_str(annotation.get("primary_numeral")),
            "primary_alteration": _normalize_str(annotation.get("primary_alteration")),
            "secondary_numeral": _normalize_str(annotation.get("secondary_numeral")),
            "secondary_alteration": _normalize_str(
                annotation.get("secondary_alteration")
            ),
            "quality": _normalize_str(annotation.get("quality")),
            "inversion": _normalize_str(annotation.get("inversion")),
            "chord_tones": _normalize_str(annotation.get("chord_tones")),
            "chord_factors": _normalize_str(annotation.get("chord_factors")),
            "local_key": _normalize_str(annotation.get("local_key")),
        }

    # `rn` is not expected in the structured `rna_info` examples. This fallback
    # exists to support the current scorer's internal harmony dict format
    # produced in metrics_rna_shared.py, which stores one parsed RN string in
    # `rn` alongside `tones` / `factors` / `local_key`.
    rn = _normalize_str(annotation.get("rn"))
    if not rn:
        return None

    try:
        parsed = parse_gt_rnbert(rn)
    except ValueError:
        return None

    parsed["chord_tones"] = _normalize_str(
        annotation.get("chord_tones", annotation.get("tones"))
    )
    parsed["chord_factors"] = _normalize_str(
        annotation.get("chord_factors", annotation.get("factors"))
    )
    parsed["local_key"] = _normalize_str(annotation.get("local_key"))
    return parsed


def _canonicalize_function(
    alteration: str,
    numeral: str,
    *,
    case_sensitive: bool,
    allow_empty: bool,
) -> Optional[str]:
    alteration = _normalize_str(alteration)
    numeral = _normalize_str(numeral)

    if not numeral:
        if alteration:
            return None
        return "" if allow_empty else None

    numeral = numeral if case_sensitive else numeral.upper()
    return f"{alteration}{numeral}"


def canonicalize_local_key(annotation: AnnotationLike) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    local_key = _normalize_str(fields.get("local_key"))
    return local_key or None


def canonicalize_primary_function(annotation: AnnotationLike) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    return _canonicalize_function(
        fields.get("primary_alteration", ""),
        fields.get("primary_numeral", ""),
        case_sensitive=False,
        allow_empty=False,
    )


def canonicalize_secondary_function(annotation: AnnotationLike) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    return _canonicalize_function(
        fields.get("secondary_alteration", ""),
        fields.get("secondary_numeral", ""),
        case_sensitive=False,
        allow_empty=True,
    )


def canonicalize_secondary_function_case_sensitive(
    annotation: AnnotationLike,
) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    return _canonicalize_function(
        fields.get("secondary_alteration", ""),
        fields.get("secondary_numeral", ""),
        case_sensitive=True,
        allow_empty=True,
    )


def canonicalize_degree(annotation: AnnotationLike) -> Optional[Tuple[str, str]]:
    primary = canonicalize_primary_function(annotation)
    secondary = canonicalize_secondary_function(annotation)
    if primary is None or secondary is None:
        return None
    return (primary, secondary)


def canonicalize_quality(annotation: AnnotationLike) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    quality = _normalize_str(fields.get("quality"))
    return quality or None


def canonicalize_inversion(annotation: AnnotationLike) -> Optional[str]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None
    return _normalize_str(fields.get("inversion"))


def extract_secondary_presence(annotation: AnnotationLike) -> Optional[bool]:
    secondary = canonicalize_secondary_function(annotation)
    if secondary is None:
        return None
    return bool(secondary)


def parse_local_key_tonic_and_mode(local_key: str) -> Optional[Tuple[int, str]]:
    """Parse a local-key label into tonic pitch class and mode."""
    local_key = _normalize_str(local_key)
    if not local_key:
        return None

    match = re.match(
        r"^([A-Ga-g])([#b]*)(?:\s+(major|minor))?$",
        local_key,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    letter, accidental_str, explicit_mode = match.groups()
    tonic_pc = spelled_tone_to_pitch_class(f"{letter.upper()}{accidental_str}")
    if explicit_mode:
        mode = explicit_mode.lower()
    else:
        mode = "minor" if letter.islower() else "major"

    return tonic_pc, mode


def _alteration_to_offset(alteration: str) -> Optional[int]:
    alteration = _normalize_str(alteration)
    offset = 0
    for char in alteration:
        if char == "#":
            offset += 1
        elif char == "b":
            offset -= 1
        else:
            return None
    return offset


def _function_label_to_relative_pc(
    function_label: str,
    *,
    mode: str,
    allow_empty: bool,
) -> Optional[Union[int, str]]:
    function_label = _normalize_str(function_label)
    if not function_label:
        return "" if allow_empty else None

    match = re.match(r"^([#b]*)([A-Z]+)$", function_label)
    if not match:
        return None

    alteration, numeral = match.groups()
    accidental_offset = _alteration_to_offset(alteration)
    if accidental_offset is None:
        return None

    scale_degree_to_pc = (
        _MAJOR_SCALE_DEGREE_TO_PC if mode == "major" else _MINOR_SCALE_DEGREE_TO_PC
    )

    if numeral == "N":
        base_pc = scale_degree_to_pc["II"] - 1
    else:
        base_pc = scale_degree_to_pc.get(numeral)

    if base_pc is None:
        return None

    return (base_pc + accidental_offset) % 12


def _secondary_numeral_to_mode(numeral: str) -> Optional[str]:
    numeral = _normalize_str(numeral)
    if not numeral:
        return None
    if any(char.islower() for char in numeral):
        return "minor"
    if any(char.isupper() for char in numeral):
        return "major"
    return None


def _resolve_primary_reference_frame(
    annotation: AnnotationLike,
) -> Optional[Tuple[int, str]]:
    fields = _coerce_annotation_fields(annotation)
    if fields is None:
        return None

    local_key = _normalize_str(fields.get("local_key"))
    tonic_info = parse_local_key_tonic_and_mode(local_key)
    if tonic_info is None:
        return None

    tonic_pc, mode = tonic_info

    secondary_function = canonicalize_secondary_function(annotation)
    if secondary_function is None:
        return None
    if secondary_function == "":
        return tonic_pc, mode

    secondary_relative_pc = _function_label_to_relative_pc(
        secondary_function,
        mode=mode,
        allow_empty=False,
    )
    if secondary_relative_pc is None or isinstance(secondary_relative_pc, str):
        return None

    secondary_mode = _secondary_numeral_to_mode(fields.get("secondary_numeral", ""))
    if secondary_mode is None:
        return None

    return (tonic_pc + secondary_relative_pc) % 12, secondary_mode


def canonicalize_primary_absolute_pc(annotation: AnnotationLike) -> Optional[int]:
    primary_function = canonicalize_primary_function(annotation)
    if primary_function is None:
        return None

    reference_frame = _resolve_primary_reference_frame(annotation)
    if reference_frame is None:
        return None

    tonic_pc, mode = reference_frame
    relative_pc = _function_label_to_relative_pc(
        primary_function,
        mode=mode,
        allow_empty=False,
    )
    if relative_pc is None or isinstance(relative_pc, str):
        return None

    return (tonic_pc + relative_pc) % 12


def canonicalize_secondary_absolute_pc(
    annotation: AnnotationLike,
) -> Optional[Union[int, str]]:
    local_key = canonicalize_local_key(annotation)
    secondary_function = canonicalize_secondary_function(annotation)
    if local_key is None or secondary_function is None:
        return None

    tonic_info = parse_local_key_tonic_and_mode(local_key)
    if tonic_info is None:
        return None

    tonic_pc, mode = tonic_info
    relative_pc = _function_label_to_relative_pc(
        secondary_function,
        mode=mode,
        allow_empty=True,
    )
    if relative_pc is None:
        return None
    if relative_pc == "":
        return ""

    return (tonic_pc + relative_pc) % 12


def canonicalize_degree_absolute_pc(
    annotation: AnnotationLike,
) -> Optional[Tuple[int, Union[int, str]]]:
    primary_abs_pc = canonicalize_primary_absolute_pc(annotation)
    secondary_abs_pc = canonicalize_secondary_absolute_pc(annotation)
    if primary_abs_pc is None or secondary_abs_pc is None:
        return None
    return (primary_abs_pc, secondary_abs_pc)


def _coerce_field_value(
    value: AnnotationLike,
    structured_key: str,
    harmony_key: str,
) -> str:
    if isinstance(value, str):
        return _normalize_str(value)
    if isinstance(value, Mapping):
        return _normalize_str(value.get(structured_key, value.get(harmony_key)))
    return ""


def parse_chord_tones(value: AnnotationLike) -> Tuple[str, ...]:
    raw_value = _coerce_field_value(value, "chord_tones", "tones")
    if not raw_value:
        return ()
    return tuple(token.strip() for token in raw_value.split("-") if token.strip())


def parse_chord_factors(value: AnnotationLike) -> Tuple[str, ...]:
    raw_value = _coerce_field_value(value, "chord_factors", "factors")
    if not raw_value:
        return ()
    return tuple(token.strip() for token in raw_value.split("-") if token.strip())


def _invalid_metric_token(side_label: str, index: int, raw_value: str) -> str:
    return f"<invalid-{side_label}-{index}:{raw_value}>"


def _normalize_tones_for_metric(
    tones: Sequence[str],
    *,
    spelling_sensitive: bool,
    side_label: str,
) -> Tuple[Union[str, int], ...]:
    if spelling_sensitive:
        return tuple(tones)

    normalized = []
    for index, tone in enumerate(tones):
        try:
            normalized.append(spelled_tone_to_pitch_class(tone))
        except ValueError:
            normalized.append(_invalid_metric_token(side_label, index, tone))
    return tuple(normalized)


def spelled_tone_to_pitch_class(tone: str) -> int:
    tone = _normalize_str(tone)
    if not tone:
        raise ValueError("Empty tone cannot be converted to pitch class")

    letter = tone[0].upper()
    if letter not in _LETTER_TO_PC:
        raise ValueError(f"Unsupported spelled tone: {tone}")

    accidental_str = tone[1:]
    accidental_offset = 0
    i = 0
    while i < len(accidental_str):
        char = accidental_str[i]
        if char == "#":
            accidental_offset += 1
        elif char == "b":
            accidental_offset -= 1
        elif char in {"x", "X"}:
            accidental_offset += 2
        else:
            raise ValueError(f"Unsupported accidental sequence in tone: {tone}")
        i += 1

    return (_LETTER_TO_PC[letter] + accidental_offset) % 12


def build_tone_factor_pairs(
    chord_tones: AnnotationLike,
    chord_factors: AnnotationLike,
    *,
    spelling_sensitive: bool = True,
    side_label: str = "tone-factor",
) -> Tuple[Tuple[Union[str, int], str], ...]:
    tones = _normalize_tones_for_metric(
        parse_chord_tones(chord_tones),
        spelling_sensitive=spelling_sensitive,
        side_label=side_label,
    )
    factors = parse_chord_factors(chord_factors)

    pair_count = max(len(tones), len(factors))
    pairs = []
    for idx in range(pair_count):
        tone: Union[str, int] = tones[idx] if idx < len(tones) else MISSING_TONE
        factor = factors[idx] if idx < len(factors) else MISSING_FACTOR

        pairs.append((tone, factor))

    return tuple(pairs)


def compute_set_metrics(
    predicted_items: Iterable[Any], reference_items: Iterable[Any]
) -> Dict[str, float]:
    predicted_set = set(predicted_items)
    reference_set = set(reference_items)

    true_positive = len(predicted_set & reference_set)
    false_positive = len(predicted_set - reference_set)
    false_negative = len(reference_set - predicted_set)

    if not predicted_set and not reference_set:
        precision = recall = f1 = 1.0
    else:
        precision = true_positive / len(predicted_set) if predicted_set else 0.0
        recall = true_positive / len(reference_set) if reference_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(predicted_set == reference_set),
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def compute_chord_tone_metrics(
    predicted: AnnotationLike,
    reference: AnnotationLike,
    *,
    spelling_sensitive: bool = True,
) -> Dict[str, float]:
    predicted_tones: Sequence[Union[str, int]] = _normalize_tones_for_metric(
        parse_chord_tones(predicted),
        spelling_sensitive=spelling_sensitive,
        side_label="pred",
    )
    reference_tones: Sequence[Union[str, int]] = _normalize_tones_for_metric(
        parse_chord_tones(reference),
        spelling_sensitive=spelling_sensitive,
        side_label="ref",
    )

    return compute_set_metrics(predicted_tones, reference_tones)


def compute_tone_factor_pair_metrics(
    predicted: AnnotationLike,
    reference: AnnotationLike,
    *,
    spelling_sensitive: bool = True,
) -> Dict[str, float]:
    predicted_pairs = build_tone_factor_pairs(
        predicted,
        predicted,
        spelling_sensitive=spelling_sensitive,
        side_label="pred",
    )
    reference_pairs = build_tone_factor_pairs(
        reference,
        reference,
        spelling_sensitive=spelling_sensitive,
        side_label="ref",
    )
    return compute_set_metrics(predicted_pairs, reference_pairs)


def _empty_classwise_stats() -> Dict[str, int]:
    return {
        "support_count": 0,
        "predicted_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def initialize_classwise_counts(
    families: Sequence[str],
) -> Dict[str, Dict[Any, Dict[str, int]]]:
    return {
        family: defaultdict(lambda: _empty_classwise_stats()) for family in families
    }


def update_classwise_counts(
    classwise_counts: Dict[str, Dict[Any, Dict[str, int]]],
    family: str,
    reference_label: Any,
    predicted_label: Any,
) -> None:
    if reference_label is None:
        return

    family_counts = classwise_counts[family]
    family_counts[reference_label]["support_count"] += 1

    if predicted_label is not None:
        family_counts[predicted_label]["predicted_count"] += 1

    if predicted_label == reference_label:
        family_counts[reference_label]["tp"] += 1
        return

    family_counts[reference_label]["fn"] += 1
    if predicted_label is not None:
        family_counts[predicted_label]["fp"] += 1


def serialize_classwise_label(family: str, label: Any) -> Dict[str, Any]:
    if family == "D" and isinstance(label, tuple) and len(label) == 2:
        primary_function, secondary_function = label
        secondary_display = secondary_function if secondary_function else "<none>"
        return {
            "label": f"P={primary_function} | S={secondary_display}",
            "primary_function": primary_function,
            "secondary_function": secondary_function,
        }

    return {"label": str(label)}


def build_classwise_metric_report(
    classwise_counts: Dict[str, Dict[Any, Dict[str, int]]],
) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}

    for family, family_counts in classwise_counts.items():
        rows: List[Dict[str, Any]] = []
        total_support = sum(
            stats["support_count"]
            for stats in family_counts.values()
            if stats["support_count"] > 0
        )

        for label, stats in family_counts.items():
            support_count = stats["support_count"]
            predicted_count = stats["predicted_count"]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            precision = tp / predicted_count if predicted_count > 0 else 0.0
            recall = tp / support_count if support_count > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            row = {
                **serialize_classwise_label(family, label),
                "support_count": support_count,
                "support_ratio": support_count / total_support
                if total_support > 0
                else 0.0,
                "predicted_count": predicted_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            rows.append(row)

        rows.sort(key=lambda row: (-row["support_count"], row["label"]))

        support_rows = [row for row in rows if row["support_count"] > 0]
        macro_f1 = (
            sum(row["f1"] for row in support_rows) / len(support_rows)
            if support_rows
            else 0.0
        )
        weighted_f1 = (
            sum(row["f1"] * row["support_count"] for row in support_rows)
            / total_support
            if total_support > 0
            else 0.0
        )

        report[family] = {
            "total_onsets": total_support,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "rows": rows,
        }

    return report
