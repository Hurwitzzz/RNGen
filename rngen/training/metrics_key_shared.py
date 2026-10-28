"""Shared parsing and scoring helpers for key-generation metrics."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from rngen.training.metrics_event_ranges import (
    count_notes_and_onsets_in_bar_position_range,
    identify_notes_and_onsets_in_range,
)

logger = logging.getLogger(__name__)

KeySegment = Tuple[str, str, str]
KeyPoint = Tuple[int, int]
KeySpan = Tuple[int, int, int, int]

_PATTERN_WITH_MODE = re.compile(
    r"bar\s+(\d+)\s+position\s+(\d+),\s*bar\s+(\d+)\s+position\s+(\d+):\s*([A-Ga-g][#b]*\s+(?:Major|minor))"
)
_PATTERN_WITHOUT_MODE = re.compile(
    r"bar\s+(\d+)\s+position\s+(\d+),\s*bar\s+(\d+)\s+position\s+(\d+):\s*([A-Ga-g][#b]*(?!\s+(?:Major|minor)))"
)


def _format_key_point(bar: str, pos: str) -> str:
    return f"bar {bar} position {pos}"


def _parse_key_point(point: str) -> Optional[KeyPoint]:
    match = re.match(r"bar\s+(\d+)\s+position\s+(\d+)", point)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _compute_overlap_span(ref_span: KeySpan, pred_span: KeySpan) -> Optional[KeySpan]:
    ref_start_bar, ref_start_pos, ref_end_bar, ref_end_pos = ref_span
    pred_start_bar, pred_start_pos, pred_end_bar, pred_end_pos = pred_span

    overlap_start_bar = max(pred_start_bar, ref_start_bar)
    overlap_end_bar = min(pred_end_bar, ref_end_bar)

    if overlap_start_bar > overlap_end_bar:
        return None

    overlap_start_pos = (
        max(pred_start_pos, ref_start_pos)
        if overlap_start_bar == ref_start_bar and overlap_start_bar == pred_start_bar
        else ref_start_pos
        if overlap_start_bar == ref_start_bar
        else pred_start_pos
    )
    overlap_end_pos = (
        min(pred_end_pos, ref_end_pos)
        if overlap_end_bar == ref_end_bar and overlap_end_bar == pred_end_bar
        else ref_end_pos
        if overlap_end_bar == ref_end_bar
        else pred_end_pos
    )

    if overlap_start_bar == overlap_end_bar and overlap_start_pos > overlap_end_pos:
        return None

    return (
        overlap_start_bar,
        overlap_start_pos,
        overlap_end_bar,
        overlap_end_pos,
    )


def extract_key_segments(text: str) -> List[KeySegment]:
    """Extract `(start, end, key)` segments from a key-generation string."""
    segments = []

    for match in _PATTERN_WITH_MODE.finditer(text):
        start = _format_key_point(match.group(1), match.group(2))
        end = _format_key_point(match.group(3), match.group(4))
        segments.append((start, end, match.group(5).strip()))

    if segments:
        return segments

    for match in _PATTERN_WITHOUT_MODE.finditer(text):
        start = _format_key_point(match.group(1), match.group(2))
        end = _format_key_point(match.group(3), match.group(4))
        segments.append((start, end, match.group(5).strip()))

    return segments


def compute_key_detection_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute region-level key metrics from decoded prediction/reference strings."""
    total_samples = len(predictions)
    correct_segments = 0
    total_segments = 0
    samples_with_key_changes = 0
    correct_samples_with_key_changes = 0
    correct_key_only = 0
    correct_boundaries_only = 0

    for pred, ref in zip(predictions, references):
        pred_segments = extract_key_segments(pred)
        ref_segments = extract_key_segments(ref)

        if len(ref_segments) > 1:
            samples_with_key_changes += 1
            if pred == ref:
                correct_samples_with_key_changes += 1

        total_segments += len(ref_segments)

        for ref_start, ref_end, ref_key in ref_segments:
            key_match_found = False
            boundary_match_found = False

            for pred_start, pred_end, pred_key in pred_segments:
                if (
                    ref_start == pred_start
                    and ref_end == pred_end
                    and ref_key == pred_key
                ):
                    correct_segments += 1
                    key_match_found = True
                    boundary_match_found = True
                    break

                if ref_key == pred_key:
                    key_match_found = True

                if ref_start == pred_start and ref_end == pred_end:
                    boundary_match_found = True

            if key_match_found:
                correct_key_only += 1

            if boundary_match_found:
                correct_boundaries_only += 1

    return {
        "sample_level_accuracy": sum(p == r for p, r in zip(predictions, references))
        / max(total_samples, 1),
        "segment_accuracy": correct_segments / max(total_segments, 1),
        "key_changes_accuracy": correct_samples_with_key_changes
        / max(samples_with_key_changes, 1)
        if samples_with_key_changes > 0
        else 0.0,
        "has_key_changes_ratio": samples_with_key_changes / max(total_samples, 1),
        "key_only_accuracy": correct_key_only / max(total_segments, 1),
        "boundary_only_accuracy": correct_boundaries_only / max(total_segments, 1),
    }


def compute_note_and_onset_level_accuracy_from_events(
    predictions: List[str],
    references: List[str],
    metadata_list: List[dict],
    compound_ratio: int,
) -> Dict[str, float]:
    """Compute note/onset overlap metrics for key-generation outputs."""
    if metadata_list is None:
        raise ValueError(
            "metadata_list is required for key note/onset metrics. "
            "Make sure TrainingArguments.include_for_metrics contains 'metadata'."
        )

    if len(metadata_list) != len(predictions):
        raise ValueError(
            "metadata_list length does not match decoded sample count: "
            f"{len(metadata_list)} metadata entries for {len(predictions)} predictions."
        )

    total_notes = 0
    correct_notes = 0
    total_onsets = 0
    correct_onsets = 0

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_segments = extract_key_segments(pred)
        ref_segments = extract_key_segments(ref)
        events = metadata_list[i]["events"]

        notes_already_counted: Set[int] = set()
        onsets_already_counted: Set[Tuple[int, int]] = set()
        segment_total_notes = 0

        for ref_start_point, ref_end_point, ref_key in ref_segments:
            ref_start = _parse_key_point(ref_start_point)
            ref_end = _parse_key_point(ref_end_point)
            if ref_start is None or ref_end is None:
                raise ValueError(
                    f"Invalid reference segment format: {ref_start_point} to {ref_end_point}"
                )

            ref_span = (*ref_start, *ref_end)
            if not events:
                raise ValueError("Events string is empty or invalid for note counting.")

            (
                current_segment_total_notes,
                current_segment_total_onsets,
            ) = count_notes_and_onsets_in_bar_position_range(
                events,
                compound_ratio,
                ref_span[0],
                ref_span[1],
                ref_span[2],
                ref_span[3],
            )
            total_notes += current_segment_total_notes
            segment_total_notes += current_segment_total_notes
            total_onsets += current_segment_total_onsets

            for pred_start_point, pred_end_point, pred_key in pred_segments:
                if pred_key != ref_key:
                    continue

                pred_start = _parse_key_point(pred_start_point)
                pred_end = _parse_key_point(pred_end_point)
                if pred_start is None or pred_end is None:
                    continue

                overlap_span = _compute_overlap_span(ref_span, (*pred_start, *pred_end))
                if overlap_span is None:
                    continue

                (
                    overlap_note_indices,
                    overlap_onsets,
                ) = identify_notes_and_onsets_in_range(
                    events,
                    compound_ratio,
                    overlap_span[0],
                    overlap_span[1],
                    overlap_span[2],
                    overlap_span[3],
                )
                onsets_already_counted.update(overlap_onsets)

                for note_token_idx in overlap_note_indices:
                    if note_token_idx not in notes_already_counted:
                        notes_already_counted.add(note_token_idx)
                        correct_notes += 1

        if segment_total_notes != metadata_list[i]["note_count"]:
            logger.warning(
                "Segment total notes %s does not match expected count %s for sample %s",
                segment_total_notes,
                metadata_list[i]["note_count"],
                metadata_list[i]["sample_id"],
            )

        correct_onsets += len(onsets_already_counted)

    return {
        "note_level_accuracy": correct_notes / max(total_notes, 1)
        if total_notes > 0
        else 0.0,
        "note_level_correct": correct_notes,
        "note_level_total": total_notes,
        "onset_level_accuracy": correct_onsets / max(total_onsets, 1)
        if total_onsets > 0
        else 0.0,
        "onset_level_correct": correct_onsets,
        "onset_level_total": total_onsets,
    }
