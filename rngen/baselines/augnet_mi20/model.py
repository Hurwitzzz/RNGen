"""Pinned AugmentedNet v1.0 schema and architecture parity helpers.

The class orders and graph formulas in this module are derived from the MIT-
licensed AugmentedNet v1.0.0 source by Néstor Nápoles López, pinned at commit
709dc59470412453fd9046ef4fb53e5663141144. The external checkout is a reference
only; RNGen does not import from it at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "https://github.com/napulen/AugmentedNet.git"
UPSTREAM_VERSION = "v1.0.0"
UPSTREAM_COMMIT = "709dc59470412453fd9046ef4fb53e5663141144"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_TENSORFLOW_VERSION = "2.4.1"
UPSTREAM_SOURCE_SHA256 = {
    "AugmentedNet/models.py": "3d16b713d2b18ea693f22d7567a8c91f2a5b900f8473c1262d1a47727e68dbca",
    "AugmentedNet/output_representations.py": "09dd332b1ce3b77f74457a71013902af6513c1aa6f1aa1660a7585e0a28159ce",
    "AugmentedNet/feature_representation.py": "f0bb197956e240edd3e9e463d0acc50c757977f238daee846b562d1f4c857eb4",
    "AugmentedNet/train.py": "f07016f59c8a792532abb03e1c53bd8bebc53a838b2ebc35866b4c3005a1eee3",
    "AugmentedNet/evaluate.py": "884fc436c8374ffd8710ae354da418a9d5dd00ac2af6d56c95c1ebb9e0e4bc59",
}

OFFICIAL_INPUTS = {
    "AugmentedNet": (("Bass19", 19), ("Chromagram19", 19)),
    "Micchi2020": (("BassChromagram70", 70),),
}
OFFICIAL_HEAD_ORDER = (
    "LocalKey35",
    "PrimaryDegree22",
    "SecondaryDegree22",
    "Inversion4",
    "ChordQuality15",
    "ChordRoot35",
    "Bass35",
    "RomanNumeral76",
    "TonicizedKey35",
    "PitchClassSet94",
    "HarmonicRhythm2",
)
OFFICIAL_MONITORED_HEADS = OFFICIAL_HEAD_ORDER[:6]
OFFICIAL_AUXILIARY_HEADS = OFFICIAL_HEAD_ORDER[6:]

SPELLINGS = (
    "C--",
    "C-",
    "C",
    "C#",
    "C##",
    "D--",
    "D-",
    "D",
    "D#",
    "D##",
    "E--",
    "E-",
    "E",
    "E#",
    "E##",
    "F--",
    "F-",
    "F",
    "F#",
    "F##",
    "G--",
    "G-",
    "G",
    "G#",
    "G##",
    "A--",
    "A-",
    "A",
    "A#",
    "A##",
    "B--",
    "B-",
    "B",
    "B#",
    "B##",
)
DEGREES = (
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
    "None",
)
KEYS = (
    "F-",
    "C-",
    "G-",
    "D-",
    "A-",
    "E-",
    "B-",
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
    "d-",
    "a-",
    "e-",
    "b-",
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
    "None",
)
CHORD_QUALITIES = (
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
    "None",
)
COMMON_ROMAN_NUMERALS = (
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
    "None",
)
PCSETS = (
    (0, 1, 5, 8),
    (0, 2, 5, 9),
    (0, 2, 6),
    (0, 2, 6, 8),
    (0, 2, 6, 9),
    (0, 3, 5, 8),
    (0, 3, 5, 9),
    (0, 3, 6),
    (0, 3, 6, 8),
    (0, 3, 6, 9),
    (0, 3, 7),
    (0, 3, 7, 8),
    (0, 3, 7, 10),
    (0, 3, 8),
    (0, 3, 9),
    (0, 4, 5, 9),
    (0, 4, 6, 10),
    (0, 4, 7),
    (0, 4, 7, 9),
    (0, 4, 7, 10),
    (0, 4, 7, 11),
    (0, 4, 9),
    (0, 4, 10),
    (0, 5, 8),
    (0, 5, 9),
    (0, 6, 8),
    (0, 6, 9),
    (1, 2, 6, 9),
    (1, 3, 6, 10),
    (1, 3, 7),
    (1, 3, 7, 9),
    (1, 3, 7, 10),
    (1, 4, 6, 9),
    (1, 4, 6, 10),
    (1, 4, 7),
    (1, 4, 7, 9),
    (1, 4, 7, 10),
    (1, 4, 8),
    (1, 4, 8, 9),
    (1, 4, 8, 11),
    (1, 4, 9),
    (1, 4, 10),
    (1, 5, 6, 10),
    (1, 5, 7, 11),
    (1, 5, 8),
    (1, 5, 8, 10),
    (1, 5, 8, 11),
    (1, 5, 10),
    (1, 5, 11),
    (1, 6, 9),
    (1, 6, 10),
    (1, 7, 9),
    (1, 7, 10),
    (2, 3, 7, 10),
    (2, 4, 7, 11),
    (2, 4, 8),
    (2, 4, 8, 10),
    (2, 4, 8, 11),
    (2, 5, 7, 10),
    (2, 5, 7, 11),
    (2, 5, 8),
    (2, 5, 8, 10),
    (2, 5, 8, 11),
    (2, 5, 9),
    (2, 5, 9, 10),
    (2, 5, 10),
    (2, 5, 11),
    (2, 6, 7, 11),
    (2, 6, 9),
    (2, 6, 9, 11),
    (2, 6, 11),
    (2, 7, 10),
    (2, 7, 11),
    (2, 8, 10),
    (2, 8, 11),
    (3, 4, 8, 11),
    (3, 5, 9),
    (3, 5, 9, 11),
    (3, 6, 8, 11),
    (3, 6, 9),
    (3, 6, 9, 11),
    (3, 6, 10),
    (3, 6, 10, 11),
    (3, 6, 11),
    (3, 7, 10),
    (3, 8, 11),
    (3, 9, 11),
    (4, 6, 10),
    (4, 7, 10),
    (4, 7, 11),
    (4, 8, 11),
    (5, 7, 11),
    (5, 8, 11),
    "None",
)

OFFICIAL_HEAD_CLASSES = {
    "LocalKey35": KEYS,
    "PrimaryDegree22": DEGREES,
    "SecondaryDegree22": DEGREES,
    "Inversion4": tuple(range(4)),
    "ChordQuality15": CHORD_QUALITIES,
    "ChordRoot35": SPELLINGS,
    "Bass35": SPELLINGS,
    "RomanNumeral76": COMMON_ROMAN_NUMERALS,
    "TonicizedKey35": KEYS,
    "PitchClassSet94": PCSETS,
    "HarmonicRhythm2": (True, False),
}


def official_head_sizes() -> dict[str, int]:
    """Return the exact upstream output sizes in official head order."""
    return {name: len(OFFICIAL_HEAD_CLASSES[name]) for name in OFFICIAL_HEAD_ORDER}


def _batch_norm_parameters(features: int) -> int:
    # gamma, beta, moving mean, and moving variance
    return 4 * features


def _gru_parameters(input_features: int, units: int) -> int:
    # Keras GRU(reset_after=True): kernel, recurrent kernel, and two bias rows.
    return 3 * units * (input_features + units + 2)


def augmentednet_parameter_count(
    input_features: Sequence[int],
    output_sizes: Sequence[int],
    *,
    blocks: int = 6,
) -> int:
    """Analytically count the parameters in upstream ``AugmentedNet``."""
    total = 0
    branch_widths = []
    for original_width in input_features:
        width = int(original_width)
        for block in range(blocks):
            filters = 2 ** (blocks - 1 - block)
            kernel = 2**block
            total += kernel * width * filters + filters
            total += _batch_norm_parameters(filters)
            width += filters
        branch_widths.append(width)
    width = sum(branch_widths)
    total += width * 64 + 64 + _batch_norm_parameters(64)
    total += 64 * 32 + 32 + _batch_norm_parameters(32)
    total += 2 * _gru_parameters(32, 30) + _batch_norm_parameters(60)
    total += 2 * _gru_parameters(60, 30) + _batch_norm_parameters(60)
    total += sum(60 * size + size for size in output_sizes)
    return total


def micchi2020_parameter_count(
    input_features: int,
    output_sizes: Sequence[int],
) -> int:
    """Analytically count the parameters in upstream ``Micchi2020``."""
    total = 0
    width = int(input_features)
    for _ in range(4):
        total += _batch_norm_parameters(width)
        total += width * 32 + 32
        total += _batch_norm_parameters(32)
        total += 8 * 32 * 8 + 8
        width += 8
    total += _batch_norm_parameters(width)
    total += width * 32 + 32
    total += _batch_norm_parameters(32)
    width = 32
    for _ in range(4):
        total += _batch_norm_parameters(width)
        total += width * 20 + 20
        total += _batch_norm_parameters(20)
        total += 8 * 20 * 5 + 5
        width += 5
    total += _batch_norm_parameters(width)
    total += width * 48 + 48
    total += _batch_norm_parameters(48)
    width = 48
    total += 2 * _gru_parameters(width, 64)
    width = 128
    total += width * 64 + 64
    width = 64
    total += sum(width * size + size for size in output_sizes)
    return total


def official_native_correctness(
    correct_by_head: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reproduce upstream Degree, RN, and RNalt correctness conjunctions."""
    required = set(OFFICIAL_MONITORED_HEADS)
    missing = sorted(required - set(correct_by_head))
    if missing:
        raise ValueError(f"Missing conventional correctness arrays: {missing}")
    degree = np.asarray(correct_by_head["PrimaryDegree22"], dtype=bool) & np.asarray(
        correct_by_head["SecondaryDegree22"], dtype=bool
    )
    roman_numeral = degree.copy()
    for head in (
        "LocalKey35",
        "ChordQuality15",
        "ChordRoot35",
        "Inversion4",
    ):
        roman_numeral &= np.asarray(correct_by_head[head], dtype=bool)
    result = {"Degree": degree, "RomanNumeral": roman_numeral}
    if "RomanNumeral76" in correct_by_head:
        result["AltRomanNumeral"] = (
            np.asarray(correct_by_head["RomanNumeral76"], dtype=bool)
            & np.asarray(correct_by_head["LocalKey35"], dtype=bool)
            & np.asarray(correct_by_head["Inversion4"], dtype=bool)
        )
    return result


def _tensorflow_modules() -> tuple[Any, Any, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to instantiate AugN/Mi20; the pinned "
            "upstream runtime used tensorflow==2.4.1."
        ) from exc
    return tf, keras, layers


def build_augmentednet(
    input_shapes: Mapping[str, tuple[int, int]],
    output_sizes: Mapping[str, int],
    *,
    blocks: int = 6,
):
    """Build the upstream AugmentedNet graph with explicit controlled shapes."""
    _, keras, layers = _tensorflow_modules()
    model_inputs = []
    encoded_inputs = []
    for name, (sequence_length, features) in input_shapes.items():
        branch_input = layers.Input(shape=(sequence_length, features), name=name)
        model_inputs.append(branch_input)
        branch = branch_input
        for block in range(blocks):
            filters = 2 ** (blocks - 1 - block)
            kernel = 2**block
            hidden = layers.Conv1D(filters, kernel, padding="same")(branch)
            hidden = layers.BatchNormalization()(hidden)
            hidden = layers.Activation("relu")(hidden)
            branch = layers.Concatenate()([branch, hidden])
        encoded_inputs.append(branch)
    hidden = (
        layers.Concatenate()(encoded_inputs)
        if len(encoded_inputs) > 1
        else encoded_inputs[0]
    )
    for width in (64, 32):
        hidden = layers.Dense(width)(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Activation("relu")(hidden)
    hidden = layers.Bidirectional(layers.GRU(30, return_sequences=True))(hidden)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Bidirectional(layers.GRU(30, return_sequences=True))(hidden)
    hidden = layers.BatchNormalization()(hidden)
    outputs = [
        layers.Dense(output_sizes[name], name=name)(hidden) for name in output_sizes
    ]
    return keras.Model(inputs=model_inputs, outputs=outputs)


def build_micchi2020(
    input_shape: tuple[int, int],
    output_sizes: Mapping[str, int],
):
    """Build the upstream Micchi2020 graph with explicit controlled shapes."""
    tf, keras, layers = _tensorflow_modules()

    def dense_layer(hidden, blocks: int, features: int, number: int):
        with tf.name_scope(f"denseNet_{number}"):
            for _ in range(blocks):
                branch = layers.BatchNormalization()(hidden)
                branch = layers.Conv1D(4 * features, 1, padding="same")(branch)
                branch = layers.Activation("relu")(branch)
                branch = layers.BatchNormalization()(branch)
                branch = layers.Conv1D(features, 8, padding="same")(branch)
                branch = layers.Activation("relu")(branch)
                hidden = layers.Concatenate()([hidden, branch])
        return hidden

    def pooling_layer(hidden, features: int, stride: int, number: int):
        with tf.name_scope(f"poolingLayer_{number}"):
            hidden = layers.BatchNormalization()(hidden)
            hidden = layers.Conv1D(features, 1, padding="same")(hidden)
            hidden = layers.Activation("relu")(hidden)
            hidden = layers.BatchNormalization()(hidden)
            return layers.MaxPooling1D(stride, stride, padding="same")(hidden)

    notes = layers.Input(shape=input_shape, name="BassChromagram70")
    hidden = dense_layer(notes, 4, 8, 1)
    hidden = pooling_layer(hidden, 32, 2, 1)
    hidden = dense_layer(hidden, 4, 5, 2)
    hidden = pooling_layer(hidden, 48, 2, 1)
    hidden = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.3))(
        hidden
    )
    hidden = layers.TimeDistributed(layers.Dense(64, activation="tanh"))(hidden)
    outputs = [
        layers.Dense(output_sizes[name], name=name)(hidden) for name in output_sizes
    ]
    return keras.Model(inputs=[notes], outputs=outputs)
