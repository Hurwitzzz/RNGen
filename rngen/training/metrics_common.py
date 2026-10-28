"""Shared decode and prediction-snapshot helpers for training metrics."""

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def decode_predictions_and_labels(
    tokenizer, predictions, labels
) -> Tuple[List[str], List[str]]:
    """Decode model predictions and labels into stripped text."""
    predictions_clean = predictions.copy()
    mask = predictions_clean == -100
    predictions_clean[mask] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(
        predictions_clean,
        skip_special_tokens=True,
    )

    filtered_labels = [
        [label for label in label_row if label != -100] for label_row in labels
    ]
    decoded_labels = tokenizer.batch_decode(
        filtered_labels,
        skip_special_tokens=True,
    )

    return (
        [pred.strip() for pred in decoded_preds],
        [label.strip() for label in decoded_labels],
    )


def build_prediction_match_records(
    predictions: List[str],
    references: List[str],
    metadata_list: List[Dict[str, Any]],
    metadata_id_key: str,
    record_id_key: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split matched and mismatched predictions into JSON-serializable records."""
    wrong_preds = []
    correct_preds = []

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        record = {
            record_id_key: metadata_list[idx][metadata_id_key],
            "prediction": pred,
        }
        if pred != ref:
            wrong_preds.append(
                {
                    **record,
                    "reference": ref,
                }
            )
        else:
            correct_preds.append(record)

    return wrong_preds, correct_preds


def build_prediction_snapshot(
    predictions: List[str],
    references: List[str],
    wrong_predictions: List[Dict[str, Any]],
    correct_predictions: List[Dict[str, Any]],
    metrics: Dict[str, float],
    current_loss,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a prediction snapshot payload for debugging and evaluation review."""
    payload = {
        "predictions": predictions,
        "references": references,
        "wrong_predictions": wrong_predictions,
        "correct_predictions": correct_predictions,
        "matches": [pred == ref for pred, ref in zip(predictions, references)],
        "metrics": metrics,
        "loss": current_loss,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def save_prediction_snapshot(
    output_dir: Optional[str],
    current_loss,
    best_loss: float,
    snapshot: Dict[str, Any],
) -> float:
    """Persist best and timestamped prediction snapshots when an output dir is set."""
    if not output_dir:
        return best_loss

    os.makedirs(output_dir, exist_ok=True)
    loss_label = "none" if current_loss is None else f"{current_loss:.4f}"

    if (
        current_loss is not None
        and math.isfinite(current_loss)
        and current_loss < best_loss
    ):
        best_loss = current_loss
        best_path = os.path.join(output_dir, "best_predictions.json")
        with open(best_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"New best model with loss={current_loss:.4f}! Saved to {best_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_path = os.path.join(output_dir, f"{timestamp}_best_loss_{loss_label}.json")
    with open(loss_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return best_loss
