"""Metrics entrypoints for standard RNA SFT evaluation tasks."""

import json
import math
import os
from datetime import datetime

from rngen.training.metrics_common import (
    build_prediction_match_records,
    build_prediction_snapshot,
    decode_predictions_and_labels,
    save_prediction_snapshot,
)
from rngen.training.metrics_rna_shared import (
    compute_rna_note_and_onset_level_accuracy,
)


def _save_rna_classwise_report(output_dir, current_loss, best_loss, report):
    """Persist class-wise RNA metric artifacts alongside prediction snapshots."""
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)
    loss_label = "none" if current_loss is None else f"{current_loss:.4f}"

    if (
        current_loss is not None
        and math.isfinite(current_loss)
        and current_loss < best_loss
    ):
        best_path = os.path.join(output_dir, "best_rna_classwise_metrics.json")
        with open(best_path, "w") as f:
            json.dump(report, f, indent=2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        output_dir,
        f"{timestamp}_rna_classwise_metrics_loss_{loss_label}.json",
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def create_compute_rna_metrics_fn(tokenizer, compound_ratio, output_dir=None):
    """
    Create compute_metrics function for RNA task

    Args:
        tokenizer: The tokenizer to decode predictions
        compound_ratio: The compound ratio used in the MusicBERT model
        output_dir: Directory to save best predictions

    Returns:
        compute_metrics: Function to compute metrics for RNA task
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")

    def compute_metrics(eval_pred):
        """Compute metrics for RNA task"""
        nonlocal best_loss
        predictions, labels, losses, metadata_list = eval_pred

        current_loss = float(losses.mean()) if losses is not None else None
        decoded_preds, decoded_labels = decode_predictions_and_labels(
            tokenizer,
            predictions,
            labels,
        )

        metrics = {}

        # Calculate note-level accuracy using events
        (
            note_level_metrics,
            classwise_report,
        ) = compute_rna_note_and_onset_level_accuracy(
            decoded_preds,
            decoded_labels,
            metadata_list,
            compound_ratio,
            return_classwise=True,
        )
        metrics.update(note_level_metrics)

        _save_rna_classwise_report(
            output_dir,
            current_loss,
            best_loss,
            classwise_report,
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
        )
        best_loss = save_prediction_snapshot(
            output_dir,
            current_loss,
            best_loss,
            cur_predictions,
        )

        return metrics

    return compute_metrics
