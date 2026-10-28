"""Audit local AugN/Mi20 graphs against a pinned official checkout.

This script intentionally runs the two implementations in the same TensorFlow
process. It copies official weights into the local graph before the seeded
forward comparison, isolating graph/computation parity from random
initialization and training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from rngen.baselines.augnet_mi20.model import (
    OFFICIAL_HEAD_ORDER,
    OFFICIAL_INPUTS,
    UPSTREAM_COMMIT,
    UPSTREAM_SOURCE_SHA256,
    augmentednet_parameter_count,
    build_augmentednet,
    build_micchi2020,
    micchi2020_parameter_count,
    official_head_sizes,
)


DEFAULT_ENVIRONMENT = Path(__file__).resolve().parents[1] / "environment.yml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_official_models(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "rngen_pinned_official_augmentednet_models",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape_list(values: Iterable[Any]) -> List[List[int]]:
    return [list(value.shape) for value in values]


def _layer_signature(model: Any) -> List[str]:
    return [layer.__class__.__name__ for layer in model.layers]


def _layer_names(model: Any) -> List[str]:
    return [layer.name for layer in model.layers]


def _output_shapes(model: Any) -> List[List[Any]]:
    shapes = model.output_shape
    if not isinstance(shapes, list):
        shapes = [shapes]
    return [
        [value if value is not None else None for value in shape] for shape in shapes
    ]


def _max_forward_difference(
    official_model: Any,
    local_model: Any,
    inputs: Any,
) -> Dict[str, Any]:
    official_weights = official_model.get_weights()
    local_model.set_weights(official_weights)
    official_outputs = official_model(inputs, training=False)
    local_outputs = local_model(inputs, training=False)
    if not isinstance(official_outputs, (list, tuple)):
        official_outputs = [official_outputs]
        local_outputs = [local_outputs]
    differences = [
        float(np.max(np.abs(left.numpy() - right.numpy())))
        for left, right in zip(official_outputs, local_outputs)
    ]
    return {
        "method": (
            "copy official weights into the local graph, then run identical "
            "seeded float32 inputs with training=False"
        ),
        "per_head_max_abs_difference": dict(zip(OFFICIAL_HEAD_ORDER, differences)),
        "max_abs_difference": max(differences),
    }


def _architecture_result(
    name: str,
    official_model: Any,
    local_model: Any,
    analytic_parameter_count: int,
    inputs: Any,
) -> Dict[str, Any]:
    official_classes = _layer_signature(official_model)
    local_classes = _layer_signature(local_model)
    official_names = _layer_names(official_model)
    local_names = _layer_names(local_model)
    forward = _max_forward_difference(official_model, local_model, inputs)
    result = {
        "architecture": name,
        "official_parameter_count": official_model.count_params(),
        "local_parameter_count": local_model.count_params(),
        "analytic_parameter_count": analytic_parameter_count,
        "official_layer_count": len(official_model.layers),
        "local_layer_count": len(local_model.layers),
        "layer_class_sequence_equal": official_classes == local_classes,
        "official_layer_classes": official_classes,
        "local_layer_classes": local_classes,
        "layer_names_equal": official_names == local_names,
        "layer_name_difference_kind": (
            None
            if official_names == local_names
            else "cosmetic Keras auto-generated names only"
        ),
        "official_layer_names": official_names,
        "local_layer_names": local_names,
        "weight_shapes_equal": (
            _shape_list(official_model.weights) == _shape_list(local_model.weights)
        ),
        "official_weight_shapes": _shape_list(official_model.weights),
        "local_weight_shapes": _shape_list(local_model.weights),
        "output_names_equal": official_model.output_names == local_model.output_names,
        "official_output_names": official_model.output_names,
        "local_output_names": local_model.output_names,
        "output_shapes_equal": (
            _output_shapes(official_model) == _output_shapes(local_model)
        ),
        "official_output_shapes": _output_shapes(official_model),
        "local_output_shapes": _output_shapes(local_model),
        "seeded_forward": forward,
    }
    result["passed"] = all(
        (
            result["official_parameter_count"] == result["local_parameter_count"],
            result["official_parameter_count"] == analytic_parameter_count,
            result["official_layer_count"] == result["local_layer_count"],
            result["layer_class_sequence_equal"],
            result["weight_shapes_equal"],
            result["output_names_equal"],
            result["output_shapes_equal"],
            forward["max_abs_difference"] == 0.0,
        )
    )
    return result


def _installed_versions() -> Mapping[str, str]:
    names = (
        "tensorflow",
        "tensorflow-estimator",
        "numpy",
        "h5py",
        "protobuf",
        "Keras-Preprocessing",
    )
    return {name: importlib.metadata.version(name) for name in names}


def run_audit(
    official_root: Path,
    environment_path: Path,
    sequence_length: int,
    seed: int,
    allow_gpu: bool,
) -> Dict[str, Any]:
    import tensorflow as tf

    source = official_root / "AugmentedNet" / "models.py"
    source_hash = _sha256(source)
    commit = _official_commit(official_root)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Official checkout is at {commit}; expected {UPSTREAM_COMMIT}"
        )
    expected_hash = UPSTREAM_SOURCE_SHA256["AugmentedNet/models.py"]
    if source_hash != expected_hash:
        raise RuntimeError(
            f"Official models.py SHA-256 is {source_hash}; expected {expected_hash}"
        )

    official = _load_official_models(source)
    sizes = official_head_sizes()
    output_descriptors = [
        SimpleNamespace(outputFeatures=sizes[name], shortname=name)
        for name in OFFICIAL_HEAD_ORDER
    ]

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    official_aug_inputs = [
        SimpleNamespace(
            array=np.zeros((1, sequence_length, features), dtype=np.float32),
            name=f"training_X_{name}",
        )
        for name, features in OFFICIAL_INPUTS["AugmentedNet"]
    ]
    official_aug = official.AugmentedNet(
        official_aug_inputs,
        output_descriptors,
    )
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    aug_shapes = {
        name: (sequence_length, features)
        for name, features in OFFICIAL_INPUTS["AugmentedNet"]
    }
    local_aug = build_augmentednet(aug_shapes, sizes)
    aug_rng = np.random.RandomState(seed + 1)
    aug_values = [
        aug_rng.normal(size=(1, sequence_length, features)).astype(np.float32)
        for _, features in OFFICIAL_INPUTS["AugmentedNet"]
    ]
    aug_result = _architecture_result(
        "AugmentedNet",
        official_aug,
        local_aug,
        augmentednet_parameter_count(
            [features for _, features in OFFICIAL_INPUTS["AugmentedNet"]],
            list(sizes.values()),
        ),
        aug_values,
    )

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    mi20_features = OFFICIAL_INPUTS["Micchi2020"][0][1]
    official_mi_inputs = [
        SimpleNamespace(
            array=np.zeros((1, sequence_length, mi20_features), dtype=np.float32),
            name="training_X_BassChromagram70",
        )
    ]
    official_mi20 = official.Micchi2020(
        official_mi_inputs,
        output_descriptors,
    )
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    local_mi20 = build_micchi2020(
        (sequence_length, mi20_features),
        sizes,
    )
    mi20_values = (
        np.random.RandomState(seed + 2)
        .normal(size=(1, sequence_length, mi20_features))
        .astype(np.float32)
    )
    mi20_result = _architecture_result(
        "Micchi2020",
        official_mi20,
        local_mi20,
        micchi2020_parameter_count(mi20_features, list(sizes.values())),
        mi20_values,
    )

    gpu_devices = tf.config.list_physical_devices("GPU")
    visible_gpus = [
        {
            "name": device.name,
            "details": tf.config.experimental.get_device_details(device),
        }
        for device in gpu_devices
    ]
    report = {
        "audit": "AugmentedNet v1.0.0 official/local graph parity",
        "passed": aug_result["passed"] and mi20_result["passed"],
        "sequence_length": sequence_length,
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "executable": os.path.realpath(os.sys.executable),
            "platform": platform.platform(),
            "versions": _installed_versions(),
            "visible_gpus": visible_gpus,
            "device_policy": (
                "GPU-visible parity audit (--allow-gpu)"
                if allow_gpu
                else "CPU-only parity audit (CUDA_VISIBLE_DEVICES cleared)"
            ),
        },
        "environment": {
            "path": str(environment_path.resolve()),
            "sha256": _sha256(environment_path),
        },
        "official_reference": {
            "root": str(official_root.resolve()),
            "commit": commit,
            "models_source": str(source.resolve()),
            "models_source_sha256": source_hash,
        },
        "architectures": {
            "AugmentedNet": aug_result,
            "Micchi2020": mi20_result,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="Local AugmentedNet checkout at the pinned upstream commit.",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=DEFAULT_ENVIRONMENT,
        help="Environment YAML to record in the report (default: AugN/Mi20 environment.yml).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Do not clear CUDA_VISIBLE_DEVICES before TensorFlow import.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = run_audit(
        args.official_root,
        args.environment,
        args.sequence_length,
        args.seed,
        args.allow_gpu,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], "output": str(args.output_json)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
