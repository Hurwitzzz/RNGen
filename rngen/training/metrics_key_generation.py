"""Metrics entrypoint for key-generation training tasks.

This module keeps the historical import surface used across the codebase while
delegating reusable key parsing/scoring helpers to ``metrics_key_shared.py``.
"""

import logging
import os
from datetime import datetime

from rngen.training.metrics_common import (
    build_prediction_match_records,
    build_prediction_snapshot,
    decode_predictions_and_labels,
    save_prediction_snapshot,
)
from rngen.training.metrics_key_shared import (
    compute_key_detection_metrics,
    compute_note_and_onset_level_accuracy_from_events,
    extract_key_segments,
)

logger = logging.getLogger(__name__)


def _dump_decode_error(output_dir, predictions, error):
    """Persist decode-failure tensors for manual debugging when requested."""
    if not output_dir:
        return

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

        error_log_path = os.path.join(error_dir, f"error_log_{timestamp}.txt")
        with open(error_log_path, "w") as f:
            f.write(f"Error: {type(error).__name__}: {str(error)}\n")
            f.write(f"Shape: {predictions.shape}\n")
            f.write(f"Dtype: {predictions.dtype}\n")
            f.write(f"Min: {predictions.min()}, Max: {predictions.max()}\n")
            f.write(f"Sample: {predictions[0]}\n")
    except Exception as save_error:
        logger.error("Error saving tensor: %s", save_error)


def create_compute_key_metrics_fn(
    tokenizer, compound_ratio, output_dir=None, format_type="region"
):
    """Create the training-time compute_metrics callback for key generation.

    Args:
        tokenizer: The tokenizer used to decode predictions and labels.
        compound_ratio: The compound ratio used by the MusicBERT event encoding.
        output_dir: Directory used to save best-prediction snapshots and decode
            error dumps. If ``None``, metrics are computed without writing files.
        format_type: Label for the key-output format stored in saved prediction
            snapshots. The active task registry currently uses ``"region"``.

    Returns:
        A ``compute_metrics(eval_pred)`` callable compatible with the Hugging Face
        trainer API.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")

    def compute_metrics(eval_pred):
        """Compute metrics for key generation task."""
        nonlocal best_loss
        predictions, labels, losses, metadata_list = eval_pred
        current_loss = float(losses.mean()) if losses is not None else None

        try:
            decoded_preds, decoded_labels = decode_predictions_and_labels(
                tokenizer,
                predictions,
                labels,
            )
        except Exception as e:
            logger.error("Error decoding predictions: %s", e)
            _dump_decode_error(output_dir, predictions, e)
            decoded_preds = ["ERROR_DECODING"] * len(labels)
            decoded_labels = ["ERROR_DECODING"] * len(labels)

        metrics = {
            "exact_match": sum(p == r for p, r in zip(decoded_preds, decoded_labels))
            / max(len(decoded_preds), 1)
        }
        metrics.update(compute_key_detection_metrics(decoded_preds, decoded_labels))
        metrics.update(
            compute_note_and_onset_level_accuracy_from_events(
                decoded_preds,
                decoded_labels,
                metadata_list,
                compound_ratio,
            )
        )

        wrong_preds, correct_preds = build_prediction_match_records(
            decoded_preds,
            decoded_labels,
            metadata_list,
            metadata_id_key="sample_id",
            record_id_key="idx",
        )

        cur_predictions = build_prediction_snapshot(
            decoded_preds,
            decoded_labels,
            wrong_preds,
            correct_preds,
            metrics,
            current_loss,
            extra_fields={"format_type": format_type},
        )
        best_loss = save_prediction_snapshot(
            output_dir,
            current_loss,
            best_loss,
            cur_predictions,
        )

        return metrics

    return compute_metrics
