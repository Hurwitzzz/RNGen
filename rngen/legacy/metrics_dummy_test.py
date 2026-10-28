"""Legacy metrics helpers for the historical dummy-test task.

This module is currently kept for reproducibility of earlier experiments and
is not wired into the active task registry.
"""

import json
import logging
import os
from datetime import datetime
import re
from typing import Dict, List, Tuple, Union

from rngen.training.metrics_common import (
    build_prediction_snapshot,
    decode_predictions_and_labels,
    save_prediction_snapshot,
)

logger = logging.getLogger(__name__)
GroupEntry = Tuple[int, Union[int, str], List[str]]


def create_compute_metrics_fn(tokenizer, output_dir=None, format_type="simplified"):
    """Create legacy dummy-test metrics with group-based evaluation."""

    # Ensure we have a place to save predictions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Variables to track best performance
    best_loss = float("inf")  # Initialize with infinity

    def compute_metrics(eval_pred):
        """Compute metrics for simultaneous note detection task"""
        nonlocal best_loss
        predictions, labels, losses = eval_pred

        # Extract loss from the eval_pred
        current_loss = float(losses.mean()) if losses is not None else None

        # Decode generated tokens and reference tokens
        try:
            decoded_preds, decoded_labels = decode_predictions_and_labels(
                tokenizer,
                predictions,
                labels,
            )
        except Exception as e:
            logger.error(f"Error decoding predictions: {e}")
            if output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_dir = os.path.join(output_dir, "error_tensors")
                os.makedirs(error_dir, exist_ok=True)
                error_path = os.path.join(error_dir, f"predictions_{timestamp}.txt")

                try:
                    with open(error_path, "w") as f:
                        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
                            for pred_seq in predictions:
                                f.write(" ".join(map(str, pred_seq.tolist())))
                                f.write("\n")
                        else:
                            f.write(" ".join(map(str, predictions.tolist())))

                    error_log_path = os.path.join(
                        error_dir,
                        f"error_log_{timestamp}.txt",
                    )
                    with open(error_log_path, "w") as f:
                        f.write(f"Error: {type(e).__name__}: {str(e)}\n")
                        f.write(f"Shape: {predictions.shape}\n")
                        f.write(f"Dtype: {predictions.dtype}\n")
                        f.write(f"Min: {predictions.min()}, Max: {predictions.max()}\n")
                        f.write(f"Sample: {predictions[:]}\n")
                except Exception as save_error:
                    logger.error(f"Error saving tensor: {save_error}")

            # Use placeholder for predictions to continue execution
            decoded_preds = ["ERROR_DECODING"] * len(labels)
            decoded_labels = ["ERROR_DECODING"] * len(labels)

        # Calculate traditional exact match
        exact_match = sum(p == r for p, r in zip(decoded_preds, decoded_labels)) / max(
            len(decoded_preds), 1
        )

        # Calculate enhanced metrics
        all_metrics = compute_group_metrics(
            decoded_preds, decoded_labels, format_type=format_type
        )
        metrics = {"exact_match": exact_match}
        metrics.update(all_metrics)

        wrong_preds = []
        correct_preds = []
        for pred, ref in zip(decoded_preds, decoded_labels):
            if pred != ref:
                wrong_preds.append({"prediction": pred, "reference": ref})
            else:
                correct_preds.append({"prediction": pred, "reference": ref})

        cur_predictions = build_prediction_snapshot(
            decoded_preds,
            decoded_labels,
            wrong_preds,
            correct_preds,
            metrics,
            current_loss,
            extra_fields={"format": format_type},
        )
        best_loss = save_prediction_snapshot(
            output_dir,
            current_loss,
            best_loss,
            cur_predictions,
        )

        return metrics

    return compute_metrics


def compute_group_metrics(
    predictions: List[str], references: List[str], format_type: str = "simplified"
) -> Dict[str, float]:
    """Compute note-group overlap metrics for legacy dummy-test outputs."""

    # Initialize counters
    correct_groups = 0
    correct_pos_and_groups = 0
    total_groups = 0
    correct_positions = 0
    total_positions = 0
    correct_multi_note_groups = 0
    correct_pos_and_multi_note_groups = 0
    total_multi_note_groups = 0

    # Process each sample
    for pred, ref in zip(predictions, references):
        # Extract groups from prediction and reference
        if format_type == "simplified":
            pred_groups = extract_simplified_groups(pred)
            ref_groups = extract_simplified_groups(ref)
        else:
            pred_groups = extract_note_groups(pred)
            ref_groups = extract_note_groups(ref)

        # Track unique positions
        pred_positions = set(pos for bar, pos, _ in pred_groups)
        ref_positions = set(pos for bar, pos, _ in ref_groups)

        # Count correct positions
        correct_positions += len(pred_positions.intersection(ref_positions))
        total_positions += len(ref_positions)

        # Create position-grouped lists instead of dictionaries
        pred_by_pos = {}
        ref_by_pos = {}

        # Group notes by position while preserving all entries
        for bar, pos, notes in pred_groups:
            if pos not in pred_by_pos:
                pred_by_pos[pos] = []
            pred_by_pos[pos].append(set(notes))

        for bar, pos, notes in ref_groups:
            if pos not in ref_by_pos:
                ref_by_pos[pos] = []
            ref_by_pos[pos].append(set(notes))

        # Count total reference groups
        total_groups += len(ref_groups)

        # Count correct groups without considering position
        for pos, ref_note_sets in ref_by_pos.items():
            for ref_note_set in ref_note_sets:
                # Check if this exact group appears anywhere in prediction
                found_matching_group = False
                for pred_pos, pred_note_sets in pred_by_pos.items():
                    for pred_note_set in pred_note_sets:
                        if ref_note_set == pred_note_set:
                            found_matching_group = True
                            break
                    if found_matching_group:
                        break

                if found_matching_group:
                    correct_groups += 1

                # Track multi-note groups
                if len(ref_note_set) > 1:
                    total_multi_note_groups += 1
                    if found_matching_group:
                        correct_multi_note_groups += 1

        # Count position-specific correct groups
        for pos, ref_note_sets in ref_by_pos.items():
            if pos in pred_by_pos:
                for ref_note_set in ref_note_sets:
                    # Check if this exact group appears at this position
                    for pred_note_set in pred_by_pos[pos]:
                        if ref_note_set == pred_note_set:
                            correct_pos_and_groups += 1
                            # For multi-note groups
                            if len(ref_note_set) > 1:
                                correct_pos_and_multi_note_groups += 1
                            break

    # Calculate metrics
    metrics = {
        "group_accuracy": correct_groups / max(total_groups, 1),
        "position_accuracy": correct_positions / max(total_positions, 1),
        "pos_and_group_accuracy": correct_pos_and_groups / max(total_groups, 1),
    }

    # Add multi-note metrics if applicable
    if total_multi_note_groups > 0:
        metrics["multi_note_accuracy"] = (
            correct_multi_note_groups / total_multi_note_groups
        )
        metrics["pos_and_multi_note_accuracy"] = (
            correct_pos_and_multi_note_groups / total_multi_note_groups
        )
    else:
        metrics["multi_note_accuracy"] = 0.0
        metrics["pos_and_multi_note_accuracy"] = 0.0

    return metrics


def extract_note_groups(text: str) -> List[GroupEntry]:
    """Extract JSON note groups as normalized ``(bar, position, notes)`` triples.

    Historical JSON dummy-test outputs store only ``position`` and ``notes``.
    We normalize them to the same three-field shape used by the simplified
    formatter so downstream group-metric code can handle both formats.
    """
    # Find all position and notes groups using regex
    try:
        # Handle valid JSON format
        if text.strip().startswith("[") and text.strip().endswith("]"):
            try:
                groups = json.loads(text)
                result = []
                for group in groups:
                    if (
                        isinstance(group, dict)
                        and "position" in group
                        and "notes" in group
                    ):
                        position = group["position"]
                        notes = group["notes"]
                        if isinstance(notes, list):
                            result.append((0, position, notes))
                return result
            except json.JSONDecodeError:
                pass

        # Fallback to regex extraction
        pattern = r'"position":\s*"([^"]+)",\s*"notes":\s*\[((?:"[^"]+",?\s*)+)\]'
        matches = re.finditer(pattern, text)

        result = []
        for match in matches:
            position = match.group(1)
            notes_str = match.group(2)
            notes = re.findall(r'"([^"]+)"', notes_str)
            result.append((0, position, notes))

        return result
    except Exception:
        # Return empty list in case of parsing errors
        return []


def extract_simplified_groups(text: str) -> List[GroupEntry]:
    """Extract ``(bar, position, notes)`` triples from simplified text output."""
    groups = []

    # # Pattern to match "Position X: note1, note2, ..." with improved boundary detection
    # # This pattern ensures we don't capture the next position in the notes
    # pattern = r"Position\s+(\d+):\s+((?:[\w#]+(?:,\s*)?)+)"
    # matches = re.finditer(pattern, text, re.IGNORECASE)

    # for match in matches:
    #     position = match.group(1)
    #     notes_str = match.group(2)
    #     notes = [note.strip() for note in notes_str.split(",")]
    #     # Filter empty notes
    #     notes = [note for note in notes if note]
    #     groups.append((position, notes))

    # Pattern to match "Bar X, Position Y: note1, note2, ..." with improved boundary detection
    pattern = r"Bar\s+(\d+),\s+Position\s+(\d+):\s+((?:[\w#]+(?:,\s*)?)+)"
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        bar = int(match.group(1))
        position = int(match.group(2))
        notes_str = match.group(3)
        notes = [note.strip() for note in notes_str.split(",")]
        # Filter empty notes
        notes = [note for note in notes if note]
        groups.append((bar, position, notes))

    return groups
