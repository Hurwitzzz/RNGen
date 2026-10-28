"""Masked multi-head training for the controlled AugN and Mi20 baselines.

TensorFlow is imported lazily so the data-contract helpers remain usable from
the regular RNGen (PyTorch) environment. The trainer keeps the official
eleven-head equal-weight loss sum, but each head's cross-entropy and accuracy
are normalized over that head's own converter-provided supervision mask.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.augnet_mi20.model import (
    OFFICIAL_HEAD_ORDER,
    OFFICIAL_MONITORED_HEADS,
    build_augmentednet,
    build_micchi2020,
)


COMPONENT_TO_HEAD = {
    "local_key": "LocalKey35",
    "primary_degree": "PrimaryDegree22",
    "secondary_degree": "SecondaryDegree22",
    "inversion": "Inversion4",
    "quality": "ChordQuality15",
    "chord_root": "ChordRoot35",
    "bass": "Bass35",
    "roman_numeral_76": "RomanNumeral76",
    "tonicized_key": "TonicizedKey35",
    "pitch_class_set": "PitchClassSet94",
    "harmonic_rhythm": "HarmonicRhythm2",
}
HEAD_TO_COMPONENT = {head: component for component, head in COMPONENT_TO_HEAD.items()}
SUPPORTED_ARCHITECTURES = ("AugmentedNet", "Micchi2020")
DEFAULT_READ_ONLY_SOURCE_ROOTS = (PROCESSED_SCORES_DIR,)


@dataclass(frozen=True)
class PreparedBaselineArrays:
    inputs: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    architecture: str
    sequences: int
    input_sequence_length: int
    output_sequence_length: int


@dataclass(frozen=True)
class MaskedTrainingConfig:
    architecture: str
    output_dir: Path
    epochs: int = 100
    batch_size: int = 16
    seed: int = 42
    learning_rate_boundaries: Tuple[int, ...] = (40,)
    learning_rate_values: Tuple[float, ...] = (0.001, 0.001)
    shuffle: bool = True
    max_validation_examples_per_head: int = 16
    verbose: int = 1
    run_eagerly: bool = False
    read_only_source_roots: Tuple[Path, ...] = DEFAULT_READ_ONLY_SOURCE_ROOTS
    data_provenance: Optional[Mapping[str, Any]] = None

    def validate(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_validation_examples_per_head < 0:
            raise ValueError("max_validation_examples_per_head cannot be negative")
        boundaries = tuple(self.learning_rate_boundaries)
        if any(value <= 0 for value in boundaries):
            raise ValueError("learning-rate epoch boundaries must be positive")
        if tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError(
                "learning-rate epoch boundaries must be strictly increasing"
            )
        if len(self.learning_rate_values) != len(boundaries) + 1:
            raise ValueError(
                "learning_rate_values must have one more item than boundaries"
            )
        if any(value <= 0 for value in self.learning_rate_values):
            raise ValueError("learning rates must be positive")


def _tensorflow_modules() -> Tuple[Any, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The controlled AugN/Mi20 trainer requires the isolated "
            "tensorflow==2.4.1 runtime."
        ) from exc
    return tf, keras


def head_sizes_from_vocabularies(
    vocabularies: Mapping[str, Sequence[str]],
) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for head in OFFICIAL_HEAD_ORDER:
        component = HEAD_TO_COMPONENT[head]
        if component not in vocabularies:
            raise ValueError(f"Missing vocabulary for {component}")
        classes = tuple(vocabularies[component])
        if not classes:
            raise ValueError(f"Empty vocabulary for {component}")
        if len(classes) != len(set(classes)):
            raise ValueError(f"Duplicate classes in {component} vocabulary")
        sizes[head] = len(classes)
    return sizes


def _require_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Missing converted array: {key}")
    return np.asarray(arrays[key])


def _validate_binary_mask(name: str, mask: np.ndarray) -> None:
    if not np.isfinite(mask).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} must be binary")


def prepare_baseline_arrays(
    arrays: Mapping[str, np.ndarray],
    vocabularies: Mapping[str, Sequence[str]],
    architecture: str,
) -> PreparedBaselineArrays:
    """Map converter tensors to official model inputs, targets, and masks."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    head_sizes = head_sizes_from_vocabularies(vocabularies)
    if architecture == "AugmentedNet":
        input_keys = ("X_Bass19", "X_Chromagram19")
        target_prefix = ""
        expected_features = (19, 19)
    else:
        input_keys = ("X_BassChromagram70",)
        target_prefix = "mi20_"
        expected_features = (70,)

    inputs: Dict[str, np.ndarray] = {}
    sequence_count: Optional[int] = None
    input_sequence_length: Optional[int] = None
    for input_key, feature_count in zip(input_keys, expected_features):
        value = _require_array(arrays, input_key)
        if value.ndim != 3 or value.shape[-1] != feature_count:
            raise ValueError(
                f"Unexpected {input_key} shape for {architecture}: {value.shape}"
            )
        if sequence_count is None:
            sequence_count = value.shape[0]
            input_sequence_length = value.shape[1]
        elif value.shape[:2] != (sequence_count, input_sequence_length):
            raise ValueError(f"Input sequence mismatch for {input_key}")
        inputs[input_key[2:]] = value

    assert sequence_count is not None
    assert input_sequence_length is not None
    expected_output_length = (
        input_sequence_length
        if architecture == "AugmentedNet"
        else math.ceil(input_sequence_length / 4)
    )
    shared_mask_key = f"{target_prefix}label_mask"
    shared_mask = _require_array(arrays, shared_mask_key)
    if shared_mask.shape != (sequence_count, expected_output_length):
        raise ValueError(
            f"Unexpected {shared_mask_key} shape: {shared_mask.shape}; expected "
            f"{(sequence_count, expected_output_length)}"
        )
    _validate_binary_mask(shared_mask_key, shared_mask)

    targets: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for head in OFFICIAL_HEAD_ORDER:
        component = HEAD_TO_COMPONENT[head]
        target_key = f"{target_prefix}y_{component}"
        mask_key = f"{target_prefix}label_mask_{component}"
        target = _require_array(arrays, target_key)
        mask = _require_array(arrays, mask_key)
        expected_target_shape = (sequence_count, expected_output_length, 1)
        expected_mask_shape = (sequence_count, expected_output_length)
        if target.shape != expected_target_shape:
            raise ValueError(
                f"Unexpected {target_key} shape: {target.shape}; expected "
                f"{expected_target_shape}"
            )
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"Unexpected {mask_key} shape: {mask.shape}; expected "
                f"{expected_mask_shape}"
            )
        _validate_binary_mask(mask_key, mask)
        if np.any(mask > shared_mask):
            raise ValueError(f"{mask_key} exceeds {shared_mask_key}")
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(f"{target_key} must contain integer class indices")
        if target.size and (target.min() < 0 or target.max() >= head_sizes[head]):
            raise ValueError(
                f"{target_key} contains an index outside [0, {head_sizes[head]})"
            )
        targets[head] = target.astype(np.int32, copy=False)
        masks[head] = mask.astype(np.float32, copy=False)

    return PreparedBaselineArrays(
        inputs=inputs,
        targets=targets,
        masks=masks,
        architecture=architecture,
        sequences=sequence_count,
        input_sequence_length=input_sequence_length,
        output_sequence_length=expected_output_length,
    )


def concatenate_converted_npz(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    """Load compatible derived NPZ files and concatenate their sequences."""
    if not paths:
        raise ValueError("At least one converted NPZ path is required")
    chunks: Dict[str, list[np.ndarray]] = {}
    expected_keys: Optional[Tuple[str, ...]] = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(sorted(archive.files))
            if expected_keys is None:
                expected_keys = keys
                chunks = {key: [] for key in keys}
            elif keys != expected_keys:
                raise ValueError(f"Converted NPZ schema mismatch: {path}")
            for key in keys:
                chunks[key].append(np.asarray(archive[key]))
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def prepare_output_directory(
    output_dir: Path,
    read_only_source_roots: Sequence[Path],
) -> None:
    for source_root in read_only_source_roots:
        if _is_relative_to(output_dir, source_root):
            raise ValueError(
                f"Refusing to write training output {output_dir} inside read-only "
                f"source dataset root {source_root}"
            )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Training output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_masked_training_model(
    base_model: Any,
    head_loss_weights: Optional[Mapping[str, float]] = None,
) -> Any:
    """Wrap a Functional baseline graph with exact per-head masked train steps."""
    tf, keras = _tensorflow_modules()
    head_names = tuple(base_model.output_names)
    if head_names != tuple(OFFICIAL_HEAD_ORDER):
        raise ValueError(
            f"Unexpected model head order: {head_names}; expected "
            f"{tuple(OFFICIAL_HEAD_ORDER)}"
        )
    weights = {
        head: float((head_loss_weights or {}).get(head, 1.0)) for head in head_names
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("Head loss weights cannot be negative")

    class MaskedMultiHeadModel(keras.Model):
        def __init__(self) -> None:
            super().__init__(
                inputs=base_model.inputs,
                outputs=base_model.outputs,
                name=f"{base_model.name}_masked",
            )
            self._head_names = head_names
            self._head_loss_weights = weights
            self._loss_trackers = {
                head: keras.metrics.Mean(name=f"{head}_loss") for head in head_names
            }
            self._accuracy_trackers = {
                head: keras.metrics.Mean(name=f"{head}_accuracy") for head in head_names
            }

        @property
        def metrics(self) -> list[Any]:
            return list(self._loss_trackers.values()) + list(
                self._accuracy_trackers.values()
            )

        @staticmethod
        def _unpack(data: Any) -> Tuple[Any, Any, Any]:
            if not isinstance(data, (tuple, list)) or len(data) != 3:
                raise ValueError(
                    "Masked training requires (inputs, targets, per-head masks)"
                )
            return data[0], data[1], data[2]

        @staticmethod
        def _labels(target: Any) -> Any:
            target = tf.convert_to_tensor(target)
            if target.shape.rank is not None and target.shape.rank >= 1:
                if target.shape[-1] == 1:
                    target = tf.squeeze(target, axis=-1)
            return tf.cast(target, tf.int32)

        def _batch_losses(
            self,
            targets: Mapping[str, Any],
            masks: Mapping[str, Any],
            predictions: Sequence[Any],
        ) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            predictions_by_head = dict(zip(self._head_names, predictions))
            batch_losses: Dict[str, Any] = {}
            per_frame_losses: Dict[str, Any] = {}
            labels_by_head: Dict[str, Any] = {}
            for head in self._head_names:
                labels = self._labels(targets[head])
                mask = tf.cast(masks[head], tf.float32)
                per_frame = keras.losses.sparse_categorical_crossentropy(
                    labels,
                    predictions_by_head[head],
                    from_logits=True,
                )
                numerator = tf.reduce_sum(per_frame * mask)
                denominator = tf.reduce_sum(mask)
                batch_losses[head] = tf.math.divide_no_nan(
                    numerator,
                    denominator,
                )
                per_frame_losses[head] = per_frame
                labels_by_head[head] = labels
            total = tf.add_n(
                [
                    batch_losses[head] * self._head_loss_weights[head]
                    for head in self._head_names
                ]
            )
            if self.losses:
                total += tf.add_n(self.losses)
            return total, batch_losses, per_frame_losses, labels_by_head

        def _update_metrics(
            self,
            targets: Mapping[str, Any],
            masks: Mapping[str, Any],
            predictions: Sequence[Any],
            per_frame_losses: Mapping[str, Any],
            labels_by_head: Mapping[str, Any],
        ) -> None:
            predictions_by_head = dict(zip(self._head_names, predictions))
            for head in self._head_names:
                mask = tf.cast(masks[head], tf.float32)
                predicted = tf.argmax(
                    predictions_by_head[head],
                    axis=-1,
                    output_type=tf.int32,
                )
                correct = tf.cast(
                    tf.equal(predicted, labels_by_head[head]),
                    tf.float32,
                )
                self._loss_trackers[head].update_state(
                    per_frame_losses[head],
                    sample_weight=mask,
                )
                self._accuracy_trackers[head].update_state(
                    correct,
                    sample_weight=mask,
                )

        def _result_logs(self) -> Dict[str, Any]:
            logs = {metric.name: metric.result() for metric in self.metrics}
            logs["loss"] = tf.add_n(
                [
                    self._loss_trackers[head].result() * self._head_loss_weights[head]
                    for head in self._head_names
                ]
            )
            logs["monitored_loss"] = tf.add_n(
                [
                    self._loss_trackers[head].result()
                    for head in OFFICIAL_MONITORED_HEADS
                ]
            )
            logs["monitored_accuracy"] = tf.add_n(
                [
                    self._accuracy_trackers[head].result()
                    for head in OFFICIAL_MONITORED_HEADS
                ]
            ) / float(len(OFFICIAL_MONITORED_HEADS))
            return logs

        def train_step(self, data: Any) -> Dict[str, Any]:
            inputs, targets, masks = self._unpack(data)
            with tf.GradientTape() as tape:
                predictions = self(inputs, training=True)
                total, _, per_frame, labels = self._batch_losses(
                    targets,
                    masks,
                    predictions,
                )
            gradients = tape.gradient(total, self.trainable_variables)
            gradient_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, self.trainable_variables)
                if gradient is not None
            ]
            self.optimizer.apply_gradients(gradient_pairs)
            self._update_metrics(
                targets,
                masks,
                predictions,
                per_frame,
                labels,
            )
            return self._result_logs()

        def test_step(self, data: Any) -> Dict[str, Any]:
            inputs, targets, masks = self._unpack(data)
            predictions = self(inputs, training=False)
            _, _, per_frame, labels = self._batch_losses(
                targets,
                masks,
                predictions,
            )
            self._update_metrics(
                targets,
                masks,
                predictions,
                per_frame,
                labels,
            )
            return self._result_logs()

    return MaskedMultiHeadModel()


def build_controlled_baseline(
    prepared: PreparedBaselineArrays,
    vocabularies: Mapping[str, Sequence[str]],
) -> Any:
    sizes = head_sizes_from_vocabularies(vocabularies)
    if prepared.architecture == "AugmentedNet":
        shapes = {
            name: tuple(value.shape[1:]) for name, value in prepared.inputs.items()
        }
        return build_augmentednet(shapes, sizes)
    input_value = prepared.inputs["BassChromagram70"]
    return build_micchi2020(tuple(input_value.shape[1:]), sizes)


def _jsonable_config(config: MaskedTrainingConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir.resolve())
    payload["read_only_source_roots"] = [
        str(path.resolve()) for path in config.read_only_source_roots
    ]
    return payload


def _dataset_summary(prepared: PreparedBaselineArrays) -> Dict[str, Any]:
    return {
        "architecture": prepared.architecture,
        "sequences": prepared.sequences,
        "input_sequence_length": prepared.input_sequence_length,
        "output_sequence_length": prepared.output_sequence_length,
        "input_shapes": {
            name: list(value.shape) for name, value in prepared.inputs.items()
        },
        "target_shapes": {
            name: list(value.shape) for name, value in prepared.targets.items()
        },
        "supervised_frames_by_head": {
            name: int(mask.sum()) for name, mask in prepared.masks.items()
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_history_csv(path: Path, history: Mapping[str, Sequence[Any]]) -> None:
    keys = list(history)
    rows = max((len(values) for values in history.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"] + keys)
        writer.writeheader()
        for epoch in range(rows):
            row: Dict[str, Any] = {"epoch": epoch + 1}
            for key in keys:
                if epoch < len(history[key]):
                    row[key] = float(history[key][epoch])
            writer.writerow(row)


def _best_finite_epoch(values: Sequence[float]) -> Tuple[int, float]:
    """Return the zero-based epoch and value of the best finite observation."""
    monitored = np.asarray(values, dtype=np.float64)
    if monitored.ndim != 1 or monitored.size == 0:
        raise RuntimeError("Validation history is missing val_monitored_loss")
    finite_indices = np.flatnonzero(np.isfinite(monitored))
    if finite_indices.size == 0:
        raise RuntimeError("Validation history has no finite val_monitored_loss")
    finite_values = monitored[finite_indices]
    best_index = int(finite_indices[int(np.argmin(finite_values))])
    return best_index, float(monitored[best_index])


def _nonfinite_history_epochs(
    history: Mapping[str, Sequence[float]],
) -> Dict[str, list[int]]:
    """List one-based epochs containing non-finite values for each metric."""
    report: Dict[str, list[int]] = {}
    for name, values in history.items():
        epochs = [
            epoch
            for epoch, value in enumerate(values, start=1)
            if not np.isfinite(float(value))
        ]
        if epochs:
            report[name] = epochs
    return report


def _json_safe_history(
    history: Mapping[str, Sequence[float]],
) -> Dict[str, list[Optional[float]]]:
    """Represent non-finite history values as JSON null, not invalid NaN tokens."""
    return {
        name: [float(value) if np.isfinite(float(value)) else None for value in values]
        for name, values in history.items()
    }


def decoded_validation_examples(
    model: Any,
    prepared: PreparedBaselineArrays,
    vocabularies: Mapping[str, Sequence[str]],
    *,
    batch_size: int,
    max_examples_per_head: int,
) -> Dict[str, Any]:
    predictions = model.predict(
        prepared.inputs,
        batch_size=batch_size,
        verbose=0,
    )
    if not isinstance(predictions, list):
        predictions = [predictions]
    predictions_by_head = dict(zip(model.output_names, predictions))
    report: Dict[str, Any] = {
        "architecture": prepared.architecture,
        "frame_downsample": 1 if prepared.architecture == "AugmentedNet" else 4,
        "heads": {},
    }
    for head in OFFICIAL_HEAD_ORDER:
        component = HEAD_TO_COMPONENT[head]
        classes = tuple(vocabularies[component])
        target = prepared.targets[head][..., 0]
        mask = prepared.masks[head].astype(bool)
        predicted = np.argmax(predictions_by_head[head], axis=-1)
        positions = np.argwhere(mask)[:max_examples_per_head]
        examples = []
        for sequence_index, frame_index in positions:
            target_index = int(target[sequence_index, frame_index])
            prediction_index = int(predicted[sequence_index, frame_index])
            examples.append(
                {
                    "sequence_index": int(sequence_index),
                    "output_frame_index": int(frame_index),
                    "source_frame_index": int(
                        frame_index
                        * (1 if prepared.architecture == "AugmentedNet" else 4)
                    ),
                    "target_index": target_index,
                    "target_label": classes[target_index],
                    "prediction_index": prediction_index,
                    "prediction_label": classes[prediction_index],
                    "correct": target_index == prediction_index,
                }
            )
        report["heads"][head] = {
            "component": component,
            "supervised_frames": int(mask.sum()),
            "examples": examples,
        }
    return report


def fit_masked_baseline(
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    vocabularies: Mapping[str, Sequence[str]],
    config: MaskedTrainingConfig,
) -> Dict[str, Any]:
    """Train one controlled baseline run and persist auditable artifacts."""
    config.validate()
    prepare_output_directory(config.output_dir, config.read_only_source_roots)
    tf, keras = _tensorflow_modules()
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    memory_growth_status: Dict[str, str] = {}
    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            memory_growth_status[device.name] = "configured_before_initialization"
        except RuntimeError:
            memory_growth_status[device.name] = "device_already_initialized"
    os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    train = prepare_baseline_arrays(
        train_arrays,
        vocabularies,
        config.architecture,
    )
    validation = prepare_baseline_arrays(
        validation_arrays,
        vocabularies,
        config.architecture,
    )
    if (
        train.input_sequence_length != validation.input_sequence_length
        or train.output_sequence_length != validation.output_sequence_length
    ):
        raise ValueError("Train/validation sequence shapes do not match")

    base_model = build_controlled_baseline(train, vocabularies)
    model = build_masked_training_model(base_model)
    steps_per_epoch = max(1, train.sequences // config.batch_size)
    step_boundaries = [
        boundary * steps_per_epoch for boundary in config.learning_rate_boundaries
    ]
    schedule = (
        keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=step_boundaries,
            values=list(config.learning_rate_values),
        )
        if step_boundaries
        else config.learning_rate_values[0]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule),
        run_eagerly=config.run_eagerly,
    )

    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoints / "best.weights.h5"
    last_checkpoint = checkpoints / "last.weights.h5"
    run_config = {
        "training": _jsonable_config(config),
        "runtime": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "visible_gpus": [
                {
                    "name": device.name,
                    "details": tf.config.experimental.get_device_details(device),
                    "memory_growth": tf.config.experimental.get_memory_growth(device),
                    "memory_growth_status": memory_growth_status[device.name],
                }
                for device in gpu_devices
            ],
        },
        "head_order": list(OFFICIAL_HEAD_ORDER),
        "monitored_heads": list(OFFICIAL_MONITORED_HEADS),
        "head_sizes": head_sizes_from_vocabularies(vocabularies),
        "parameter_count": int(base_model.count_params()),
        "head_loss_weights": {head: 1.0 for head in OFFICIAL_HEAD_ORDER},
        "loss_normalization": "per-head sum(masked CE) / sum(head mask)",
        "total_loss": "equal-weight sum of all 11 normalized head losses",
        "monitored_loss": "sum of the six official monitored-head losses",
        "monitored_accuracy": "mean of the six monitored-head accuracies",
        "steps_per_epoch": steps_per_epoch,
        "learning_rate_step_boundaries": step_boundaries,
        "train_data": _dataset_summary(train),
        "validation_data": _dataset_summary(validation),
        "source_data_policy": "read-only; outputs rejected inside source roots",
        "data_provenance": config.data_provenance,
    }
    _write_json(config.output_dir / "run_config.json", run_config)
    _write_json(
        config.output_dir / "ontology.json",
        {component: list(classes) for component, classes in vocabularies.items()},
    )
    (config.output_dir / "model_graph.json").write_text(
        base_model.to_json() + "\n",
        encoding="utf-8",
    )
    summary_lines: list[str] = []
    base_model.summary(print_fn=summary_lines.append)
    (config.output_dir / "model_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    curves_path = config.output_dir / "curves.csv"

    class FiniteCheckpointGuard(keras.callbacks.Callback):
        """Save only fully finite best/last epochs and stop on invalid metrics."""

        def __init__(self, best_path: Path, last_path: Path) -> None:
            super().__init__()
            self.best_path = best_path
            self.last_path = last_path
            self.best_value = float("inf")
            self.best_epoch: Optional[int] = None
            self.last_finite_epoch: Optional[int] = None

        def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Mapping[str, Any]] = None,
        ) -> None:
            numeric_logs = {
                name: float(value)
                for name, value in (logs or {}).items()
                if value is not None
            }
            if not numeric_logs or not all(
                np.isfinite(value) for value in numeric_logs.values()
            ):
                self.model.stop_training = True
                return
            monitored = numeric_logs.get("val_monitored_loss")
            if monitored is None:
                self.model.stop_training = True
                return
            self.model.save_weights(str(self.last_path))
            self.last_finite_epoch = epoch + 1
            if monitored < self.best_value:
                self.model.save_weights(str(self.best_path))
                self.best_value = monitored
                self.best_epoch = epoch + 1

    finite_checkpoint_guard = FiniteCheckpointGuard(
        best_checkpoint,
        last_checkpoint,
    )
    callbacks = [
        finite_checkpoint_guard,
        keras.callbacks.CSVLogger(str(curves_path), append=False),
        keras.callbacks.TerminateOnNaN(),
    ]
    history_object = model.fit(
        train.inputs,
        train.targets,
        sample_weight=train.masks,
        validation_data=(
            validation.inputs,
            validation.targets,
            validation.masks,
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    _write_json(config.output_dir / "history.json", _json_safe_history(history))
    _write_history_csv(curves_path, history)
    if not best_checkpoint.exists():
        raise RuntimeError("Training completed without writing the best checkpoint")
    if not last_checkpoint.exists():
        raise RuntimeError(
            "Training completed without writing a finite last checkpoint"
        )
    model.load_weights(str(best_checkpoint))
    best_validation_metrics = model.evaluate(
        validation.inputs,
        validation.targets,
        sample_weight=validation.masks,
        batch_size=config.batch_size,
        verbose=0,
        return_dict=True,
    )
    best_validation_metrics = {
        name: float(value) for name, value in best_validation_metrics.items()
    }
    best_validation_metrics_path = config.output_dir / "best_validation_metrics.json"
    _write_json(
        best_validation_metrics_path,
        {
            "architecture": config.architecture,
            "checkpoint": str(best_checkpoint.resolve()),
            "metrics": best_validation_metrics,
            "supervised_frames_by_head": {
                head: int(mask.sum()) for head, mask in validation.masks.items()
            },
        },
    )
    examples = decoded_validation_examples(
        model,
        validation,
        vocabularies,
        batch_size=config.batch_size,
        max_examples_per_head=config.max_validation_examples_per_head,
    )
    _write_json(config.output_dir / "validation_examples.json", examples)

    monitored = history.get("val_monitored_loss", [])
    best_epoch_index, best_monitored_loss = _best_finite_epoch(monitored)
    if finite_checkpoint_guard.best_epoch != best_epoch_index + 1:
        raise RuntimeError(
            "Finite best-checkpoint epoch does not match validation history"
        )
    nonfinite_history_epochs = _nonfinite_history_epochs(history)
    training_status = "failed_nonfinite" if nonfinite_history_epochs else "completed"
    result = {
        "architecture": config.architecture,
        "seed": config.seed,
        "parameter_count": int(base_model.count_params()),
        "training_status": training_status,
        "requested_epochs": config.epochs,
        "epochs_completed": len(monitored),
        "last_finite_epoch": finite_checkpoint_guard.last_finite_epoch,
        "best_epoch": best_epoch_index + 1,
        "best_val_monitored_loss": best_monitored_loss,
        "nonfinite_history_epochs": nonfinite_history_epochs,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "run_config": str((config.output_dir / "run_config.json").resolve()),
        "history": str((config.output_dir / "history.json").resolve()),
        "curves": str((config.output_dir / "curves.csv").resolve()),
        "best_validation_metrics": str(best_validation_metrics_path.resolve()),
        "validation_examples": str(
            (config.output_dir / "validation_examples.json").resolve()
        ),
    }
    _write_json(config.output_dir / "run_summary.json", result)
    if nonfinite_history_epochs:
        affected = ", ".join(
            f"{name}={epochs}" for name, epochs in nonfinite_history_epochs.items()
        )
        raise RuntimeError(
            "Training encountered non-finite history values; the best and last "
            f"finite checkpoints were preserved. Affected metrics: {affected}"
        )
    return result
