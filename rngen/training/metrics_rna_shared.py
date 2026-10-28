"""Shared RNA note/onset scoring helpers reused across metric entrypoints."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from rngen.training.metrics_event_ranges import (
    identify_notes_and_onsets_in_range,
)
from rngen.training.metrics_rna_components import (
    build_classwise_metric_report,
    canonicalize_degree,
    canonicalize_degree_absolute_pc,
    canonicalize_inversion,
    canonicalize_local_key,
    canonicalize_primary_absolute_pc,
    canonicalize_primary_function,
    canonicalize_quality,
    canonicalize_secondary_absolute_pc,
    canonicalize_secondary_function,
    canonicalize_secondary_function_case_sensitive,
    compute_chord_tone_metrics,
    compute_tone_factor_pair_metrics,
    extract_secondary_presence,
    initialize_classwise_counts,
    update_classwise_counts,
)
from rngen.utils import repair_and_decode_json

logger = logging.getLogger(__name__)

_ONSET_COMPONENT_EXTRACTORS = {
    "K": canonicalize_local_key,
    "P": canonicalize_primary_function,
    "S": canonicalize_secondary_function,
    "S_case": canonicalize_secondary_function_case_sensitive,
    "D": canonicalize_degree,
    "Q": canonicalize_quality,
    "I": canonicalize_inversion,
}

_ABSOLUTE_PC_EXTRACTORS = {
    "P_abs_pc": canonicalize_primary_absolute_pc,
    "S_abs_pc": canonicalize_secondary_absolute_pc,
    "D_abs_pc": canonicalize_degree_absolute_pc,
}

_SET_F1_METRICS = {
    "chord_tone_f1_spelling": lambda pred, ref: compute_chord_tone_metrics(
        pred, ref, spelling_sensitive=True
    )["f1"],
    "chord_tone_f1_pc": lambda pred, ref: compute_chord_tone_metrics(
        pred, ref, spelling_sensitive=False
    )["f1"],
    "tone_factor_pair_f1_spelling": lambda pred, ref: compute_tone_factor_pair_metrics(
        pred, ref, spelling_sensitive=True
    )["f1"],
    "tone_factor_pair_f1_pc": lambda pred, ref: compute_tone_factor_pair_metrics(
        pred, ref, spelling_sensitive=False
    )["f1"],
}

_CLASSWISE_FAMILIES = ("K", "P", "S", "Q", "D", "I")


def _parse_harmony_string(harmony_str: str, local_key: str) -> Optional[Dict[str, Any]]:
    """Parse a single harmony annotation string."""
    # Format: "bar 35 quarterbeats 1.00: i: C-Eb-G, R-T-F"
    # Or: "bar 66 quarterbeats 1.00: V/IV: Eb, R"
    # Or: "bar 66 quarterbeats 1.00: V/IV"
    use_quarterbeats = "quarterbeats" in harmony_str
    match = (
        re.match(r"bar\s+(\d+)\s+quarterbeats\s+([\d.]+)", harmony_str)
        if use_quarterbeats
        else re.match(r"bar\s+(\d+)\s+position\s+(\d+)", harmony_str)
    )
    if not match:
        logger.warning(f"Failed to parse harmony string: {harmony_str}")
        return None

    bar = int(match.group(1))
    quarterbeats = float(match.group(2))
    # Position is 1-based in quarterbeats and 0-based in MusicBERT ticks.
    # quarterbeats = position / 16 + 1  => position = (quarterbeats - 1) * 16
    pos = round((quarterbeats - 1) * 16) if use_quarterbeats else int(match.group(2))

    parts = harmony_str.split(":", 1)
    if len(parts) < 2:
        logger.warning(
            f"Invalid harmony string format has less then 2 parts: {harmony_str}"
        )

    annotation_part = parts[1].strip()
    sub_parts = annotation_part.split(",")

    rn_and_tones = sub_parts[0].split(":")
    rn = rn_and_tones[0].strip()
    tones = rn_and_tones[1].strip() if len(rn_and_tones) > 1 else None

    factors = sub_parts[1].strip() if len(sub_parts) > 1 else None

    return {
        "bar": bar,
        "pos": pos,
        "local_key": local_key,
        "rn": rn,
        "tones": tones,
        "factors": factors,
    }


def _get_sorted_harmonies(
    annotation_data: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Parse and sort all harmonies from an annotation dictionary."""
    all_harmonies = []
    for key, harmony_list in annotation_data.items():
        if not isinstance(harmony_list, list):
            logger.warning(
                f"Expected list for key '{key}', got {type(harmony_list)}. Skipping."
            )
            continue
        for harmony_str in harmony_list:
            parsed = _parse_harmony_string(harmony_str, key)
            if parsed:
                all_harmonies.append(parsed)

    return sorted(all_harmonies, key=lambda x: (x["bar"], x["pos"]))


def _build_harmony_segments(
    sorted_harmonies: List[Dict[str, Any]],
    last_target_bar: int,
) -> List[Dict[str, Any]]:
    segments = []
    for idx, harmony in enumerate(sorted_harmonies):
        if idx + 1 < len(sorted_harmonies):
            next_harmony = sorted_harmonies[idx + 1]
            end_bar, end_pos = next_harmony["bar"], next_harmony["pos"]
        else:
            end_bar, end_pos = last_target_bar, float("inf")

        segments.append(
            {
                "harmony": harmony,
                "start_bar": harmony["bar"],
                "start_pos": harmony["pos"],
                "end_bar": end_bar,
                "end_pos": end_pos,
            }
        )
    return segments


def _expand_harmony_assignments(
    sorted_harmonies: List[Dict[str, Any]],
    events: str,
    compound_ratio: int,
    last_target_bar: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    note_assignments: Dict[int, Dict[str, Any]] = {}
    onset_assignments: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for segment in _build_harmony_segments(sorted_harmonies, last_target_bar):
        note_indices, onsets = identify_notes_and_onsets_in_range(
            events,
            compound_ratio,
            segment["start_bar"],
            segment["start_pos"],
            segment["end_bar"],
            segment["end_pos"],
            left_close=True,
            right_close=False,
        )
        for note_index in note_indices:
            note_assignments[note_index] = segment["harmony"]
        for onset in onsets:
            onset_assignments[onset] = segment["harmony"]

    return note_assignments, onset_assignments


def _harmonies_match_exact(
    predicted: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    incl_chord_tones_acc: bool,
    incl_chord_factors_acc: bool,
) -> bool:
    if predicted["local_key"] != reference["local_key"]:
        return False
    if predicted["rn"] != reference["rn"]:
        return False
    if incl_chord_tones_acc and predicted.get("tones") != reference.get("tones"):
        return False
    if incl_chord_factors_acc and predicted.get("factors") != reference.get("factors"):
        return False
    return True


def compute_rna_note_and_onset_level_accuracy(
    predictions: List[str],
    references: List[str],
    metadata_list: List[Dict[str, Any]],
    compound_ratio: int,
    incl_chord_tones_acc: bool = False,
    incl_chord_factors_acc: bool = False,
    return_classwise: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """Compute note-level and onset-level accuracy for the RNA task."""
    total_notes = 0
    correct_notes = 0

    total_onsets = 0
    correct_onsets = 0

    component_correct = {name: 0 for name in _ONSET_COMPONENT_EXTRACTORS}
    component_total = {name: 0 for name in _ONSET_COMPONENT_EXTRACTORS}

    secondary_presence_tp = 0
    secondary_presence_fp = 0
    secondary_presence_fn = 0

    set_f1_sums = {name: 0.0 for name in _SET_F1_METRICS}

    abs_pc_correct = {name: 0 for name in _ABSOLUTE_PC_EXTRACTORS}
    abs_pc_total = {name: 0 for name in _ABSOLUTE_PC_EXTRACTORS}

    classwise_counts = initialize_classwise_counts(_CLASSWISE_FAMILIES)

    malformed_prediction_count = 0

    for i, (pred_str, ref_str) in enumerate(zip(predictions, references)):
        ref_data = json.loads(ref_str)
        if not isinstance(ref_data, dict):
            logger.warning(f"Reference JSON for sample {i} is not a dict. Skipping.")
            continue

        sorted_refs = _get_sorted_harmonies(ref_data)
        events = metadata_list[i].get("events")
        last_target_bar = metadata_list[i].get("last_target_bar")

        if not events or not isinstance(events, str):
            logger.warning(
                f"Events or last_target_bar not found or invalid for sample {i}. Skipping."
            )
            continue
        if last_target_bar is None:
            logger.warning(f"last_target_bar is missing for sample {i}. Skipping.")
            continue

        ref_note_map, ref_onset_map = _expand_harmony_assignments(
            sorted_refs,
            events,
            compound_ratio,
            last_target_bar,
        )

        current_seg_total_notes = len(ref_note_map)
        current_seg_total_onsets = len(ref_onset_map)
        total_notes += current_seg_total_notes
        total_onsets += current_seg_total_onsets

        if current_seg_total_notes != metadata_list[i]["note_count"]:
            logger.warning(
                f"Segment total notes {current_seg_total_notes} does not match expected count {metadata_list[i]['note_count']} for sample {metadata_list[i]['sample_id']}"
            )

        pred_data = None
        try:
            pred_data = json.loads(pred_str)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to decode JSON for sample {i}. Attempting to repair."
            )
            pred_data = repair_and_decode_json(pred_str)
            if pred_data is None:
                logger.error(
                    f"Could not repair JSON for sample {i}. Counting as fully incorrect."
                )

        pred_note_map: Dict[int, Dict[str, Any]] = {}
        pred_onset_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if pred_data is None:
            malformed_prediction_count += 1
        elif not isinstance(pred_data, dict):
            logger.warning(
                f"Prediction JSON for sample {i} is not a dict. Counting as fully incorrect."
            )
            malformed_prediction_count += 1
        else:
            sorted_preds = _get_sorted_harmonies(pred_data)
            pred_note_map, pred_onset_map = _expand_harmony_assignments(
                sorted_preds,
                events,
                compound_ratio,
                last_target_bar,
            )

        for note_index, ref_harmony in ref_note_map.items():
            pred_harmony = pred_note_map.get(note_index)
            if pred_harmony is not None and _harmonies_match_exact(
                pred_harmony,
                ref_harmony,
                incl_chord_tones_acc=incl_chord_tones_acc,
                incl_chord_factors_acc=incl_chord_factors_acc,
            ):
                correct_notes += 1

        for onset, ref_harmony in ref_onset_map.items():
            pred_harmony = pred_onset_map.get(onset)
            if pred_harmony is not None and _harmonies_match_exact(
                pred_harmony,
                ref_harmony,
                incl_chord_tones_acc=incl_chord_tones_acc,
                incl_chord_factors_acc=incl_chord_factors_acc,
            ):
                correct_onsets += 1

            for metric_name, extractor in _ONSET_COMPONENT_EXTRACTORS.items():
                ref_value = extractor(ref_harmony)
                if ref_value is None:
                    continue
                component_total[metric_name] += 1
                pred_value = (
                    extractor(pred_harmony) if pred_harmony is not None else None
                )
                if pred_value == ref_value:
                    component_correct[metric_name] += 1

                if metric_name in _CLASSWISE_FAMILIES:
                    update_classwise_counts(
                        classwise_counts,
                        metric_name,
                        ref_value,
                        pred_value,
                    )

            ref_secondary_presence = extract_secondary_presence(ref_harmony)
            if ref_secondary_presence is not None:
                pred_secondary_presence = (
                    extract_secondary_presence(pred_harmony)
                    if pred_harmony is not None
                    else False
                )
                if pred_secondary_presence and ref_secondary_presence:
                    secondary_presence_tp += 1
                elif pred_secondary_presence and not ref_secondary_presence:
                    secondary_presence_fp += 1
                elif (not pred_secondary_presence) and ref_secondary_presence:
                    secondary_presence_fn += 1

            if pred_harmony is not None:
                for metric_name, metric_fn in _SET_F1_METRICS.items():
                    set_f1_sums[metric_name] += metric_fn(pred_harmony, ref_harmony)

            for metric_name, extractor in _ABSOLUTE_PC_EXTRACTORS.items():
                ref_value = extractor(ref_harmony)
                if ref_value is None:
                    continue
                abs_pc_total[metric_name] += 1
                if pred_harmony is not None and extractor(pred_harmony) == ref_value:
                    abs_pc_correct[metric_name] += 1

    note_level_accuracy = (
        correct_notes / max(total_notes, 1) if total_notes > 0 else 0.0
    )
    onset_level_accuracy = (
        correct_onsets / max(total_onsets, 1) if total_onsets > 0 else 0.0
    )

    secondary_presence_precision = (
        secondary_presence_tp / (secondary_presence_tp + secondary_presence_fp)
        if secondary_presence_tp + secondary_presence_fp > 0
        else 0.0
    )
    secondary_presence_recall = (
        secondary_presence_tp / (secondary_presence_tp + secondary_presence_fn)
        if secondary_presence_tp + secondary_presence_fn > 0
        else 0.0
    )
    secondary_presence_f1 = (
        2
        * secondary_presence_precision
        * secondary_presence_recall
        / (secondary_presence_precision + secondary_presence_recall)
        if secondary_presence_precision + secondary_presence_recall > 0
        else 0.0
    )

    metrics = {
        "note_level_accuracy": note_level_accuracy,
        "note_level_correct": correct_notes,
        "note_level_total": total_notes,
        "onset_level_accuracy": onset_level_accuracy,
        "onset_level_correct": correct_onsets,
        "onset_level_total": total_onsets,
    }
    for metric_name in _ONSET_COMPONENT_EXTRACTORS:
        metric_total = component_total[metric_name]
        metrics[f"{metric_name}_acc"] = (
            component_correct[metric_name] / metric_total if metric_total > 0 else 0.0
        )

    metrics["secondary_presence_precision"] = secondary_presence_precision
    metrics["secondary_presence_recall"] = secondary_presence_recall
    metrics["secondary_presence_f1"] = secondary_presence_f1

    for metric_name, f1_sum in set_f1_sums.items():
        metrics[metric_name] = f1_sum / total_onsets if total_onsets > 0 else 0.0

    for metric_name in _ABSOLUTE_PC_EXTRACTORS:
        metric_total = abs_pc_total[metric_name]
        metrics[f"{metric_name}_acc"] = (
            abs_pc_correct[metric_name] / metric_total if metric_total > 0 else 0.0
        )
        metrics[f"{metric_name}_supported_onsets"] = metric_total

    metrics["malformed_prediction_count"] = malformed_prediction_count
    metrics["malformed_prediction_rate"] = (
        malformed_prediction_count / len(predictions) if predictions else 0.0
    )

    if return_classwise:
        return metrics, build_classwise_metric_report(classwise_counts)

    return metrics
