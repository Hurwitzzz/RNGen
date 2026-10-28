"""Convert processed P.DCML CSVs into controlled ChordGNN graph examples.

This module reproduces the graph and 83-dimensional note features used by the
public ChordGNN implementation while reusing RNGen's audited harmony
ontology.  Source CSVs are read-only.  Training augmentation is supplied by
the existing materialized 12-transposition corpus; this converter never
transposes a score dynamically.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    AUXILIARY_COMPONENTS,
    COMPONENTS,
    CORE_COMPONENTS,
    SPELLINGS,
    _extract_notes,
    _read_csv,
    build_vocabularies,
    canonicalize_enharmonic_to_classes,
    extract_harmony_events,
    pdcml_to_augnet_spelling,
)


UPSTREAM_COMMIT = "bcbdc72d2850e986c5f577727f800dee0f95f0c4"
FEATURE_DIM = 83
EDGE_TYPES = {
    "onset": 0,
    "consecutive": 1,
    "during": 2,
    "silence": 3,
}
REVERSE_EDGE_TYPES = {
    "consecutive_rev": 4,
    "during_rev": 5,
    "silence_rev": 6,
}
STEP_TO_INDEX = {step: index for index, step in enumerate("CDEFGAB")}
ALTERATIONS = (-3, -2, -1, 0, 1, 2, 3)
CHORD_TEMPLATES = {
    "M/m": (0, 0, 1, 1, 1, 0),
    "sus4": (0, 1, 0, 0, 2, 0),
    "M7": (0, 1, 2, 1, 1, 1),
    "M7wo5": (0, 1, 0, 1, 0, 1),
    "Mmaj7": (1, 0, 1, 2, 2, 0),
    "Mmaj7maj9": (1, 2, 2, 2, 3, 0),
    "M9": (1, 1, 4, 1, 1, 2),
    "M9wo5": (1, 1, 2, 1, 0, 1),
    "m7": (0, 1, 2, 1, 2, 0),
    "m7wo5": (0, 1, 1, 0, 1, 0),
    "m9": (1, 2, 2, 2, 3, 0),
    "m9wo5": (1, 2, 1, 1, 1, 0),
    "m9wo7": (1, 1, 1, 1, 2, 0),
    "mmaj7": (1, 0, 1, 3, 1, 0),
    "Maug": (0, 0, 0, 3, 0, 0),
    "Maug7": (1, 0, 1, 3, 1, 0),
    "mdim": (0, 0, 2, 0, 0, 1),
    "mdim7": (0, 0, 4, 0, 0, 2),
}
INTERVAL_FEATURE_NAMES = tuple(f"interval{i}" for i in range(13))
NOTE_FEATURE_NAMES = (
    tuple(f"int_vec{i}" for i in range(1, 7))
    + INTERVAL_FEATURE_NAMES
    + tuple(CHORD_TEMPLATES)
    + (
        "is_maj_triad",
        "is_pmaj_triad",
        "is_min_triad",
        "ped_note",
        "hv_7",
        "hv_5",
        "hv_3",
        "hv_1",
        "chord_has_2m",
        "chord_has_2M",
    )
)
FEATURE_NAMES = (
    ("pitch",)
    + tuple(f"step_{step}" for step in STEP_TO_INDEX)
    + tuple(f"alter_{alter}" for alter in ALTERATIONS)
    + ("bar_exp_duration",)
    + NOTE_FEATURE_NAMES
    + tuple(f"step_{step}" for step in STEP_TO_INDEX)
    + tuple(f"alter_{alter}" for alter in ALTERATIONS)
    + (
        "min_pitch",
        "max_pitch",
        "is_first_note_of_bar",
        "is_downbeat",
        "is_half_ts_beat",
        "time_until_next_onset",
    )
)
if len(FEATURE_NAMES) != FEATURE_DIM:
    raise AssertionError(f"Unexpected ChordGNN feature count: {len(FEATURE_NAMES)}")


@dataclass(frozen=True)
class ChordGraphExample:
    """Stable, model-independent representation of one ChordGNN graph."""

    relative_path: str
    x: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    node_onset_div: np.ndarray
    onset_representatives: np.ndarray
    onset_quarter: np.ndarray
    labels: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    reference_note_onsets: np.ndarray
    diagnostics: Mapping[str, Any]

    def validate(self, vocabularies: Mapping[str, Sequence[str]] | None = None) -> None:
        n_nodes = int(self.x.shape[0])
        n_onsets = int(self.onset_quarter.shape[0])
        if self.x.shape != (n_nodes, FEATURE_DIM):
            raise ValueError(f"Expected x shape (N, {FEATURE_DIM}), got {self.x.shape}")
        if self.x.dtype != np.float32 or not np.isfinite(self.x).all():
            raise ValueError("ChordGNN features must be finite float32")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_type.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_type must align with edge_index")
        if self.edge_index.size and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node")
        if not set(np.unique(self.edge_type)).issubset(set(EDGE_TYPES.values())):
            raise ValueError("cache must contain only the four forward edge types")
        if self.node_onset_div.shape != (n_nodes,):
            raise ValueError("node_onset_div must have one value per node")
        if self.onset_representatives.shape != (n_onsets,):
            raise ValueError("onset representative count mismatch")
        if n_onsets and not np.array_equal(
            self.node_onset_div[self.onset_representatives],
            np.unique(self.node_onset_div),
        ):
            raise ValueError("onset representatives do not select sorted unique onsets")
        if np.any(np.diff(self.onset_quarter) <= 0):
            raise ValueError("onset_quarter must be strictly increasing")
        for component in COMPONENTS:
            if component not in self.labels or component not in self.masks:
                raise ValueError(f"Missing ChordGNN head {component}")
            labels = np.asarray(self.labels[component])
            mask = np.asarray(self.masks[component])
            if labels.shape != (n_onsets,) or mask.shape != (n_onsets,):
                raise ValueError(f"Head {component} does not align with graph onsets")
            if mask.dtype != np.bool_:
                raise ValueError(f"Head {component} mask must be boolean")
            if vocabularies is not None and mask.any():
                active = labels[mask]
                if int(active.min()) < 0 or int(active.max()) >= len(
                    vocabularies[component]
                ):
                    raise ValueError(f"Head {component} has an out-of-range class")
            if np.any(labels[~mask] != -1):
                raise ValueError(f"Masked head {component} labels must be -1")

    def save_npz(self, output_path: Path) -> None:
        payload: dict[str, Any] = {
            "schema_version": np.asarray("chordgnn-pdcml-v1"),
            "relative_path": np.asarray(self.relative_path),
            "x": self.x,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "node_onset_div": self.node_onset_div,
            "onset_representatives": self.onset_representatives,
            "onset_quarter": self.onset_quarter,
            "reference_note_onsets": self.reference_note_onsets,
            "diagnostics_json": np.asarray(
                json.dumps(self.diagnostics, sort_keys=True, allow_nan=False)
            ),
        }
        for component in COMPONENTS:
            payload[f"y__{component}"] = np.asarray(
                self.labels[component], dtype=np.int64
            )
            payload[f"mask__{component}"] = np.asarray(
                self.masks[component], dtype=np.bool_
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)

    @classmethod
    def load_npz(cls, input_path: Path) -> "ChordGraphExample":
        with np.load(input_path, allow_pickle=False) as arrays:
            schema = str(arrays["schema_version"].item())
            if schema != "chordgnn-pdcml-v1":
                raise ValueError(f"Unsupported ChordGNN cache schema {schema!r}")
            example = cls(
                relative_path=str(arrays["relative_path"].item()),
                x=np.asarray(arrays["x"], dtype=np.float32),
                edge_index=np.asarray(arrays["edge_index"], dtype=np.int64),
                edge_type=np.asarray(arrays["edge_type"], dtype=np.int8),
                node_onset_div=np.asarray(arrays["node_onset_div"], dtype=np.int64),
                onset_representatives=np.asarray(
                    arrays["onset_representatives"], dtype=np.int64
                ),
                onset_quarter=np.asarray(arrays["onset_quarter"], dtype=np.float64),
                labels={
                    component: np.asarray(arrays[f"y__{component}"], dtype=np.int64)
                    for component in COMPONENTS
                },
                masks={
                    component: np.asarray(arrays[f"mask__{component}"], dtype=np.bool_)
                    for component in COMPONENTS
                },
                reference_note_onsets=np.asarray(
                    arrays["reference_note_onsets"], dtype=np.float64
                ),
                diagnostics=json.loads(str(arrays["diagnostics_json"].item())),
            )
        example.validate()
        return example


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_spelling(value: Any) -> tuple[str, int, str]:
    raw = "" if pd.isna(value) else str(value).strip()
    converted = pdcml_to_augnet_spelling(raw)
    if converted not in SPELLINGS:
        converted = canonicalize_enharmonic_to_classes(
            raw, SPELLINGS, preserve_mode_case=False
        )
    if converted not in SPELLINGS:
        raise ValueError(f"Unsupported ChordGNN note spelling {raw!r}")
    step = converted[0].upper()
    accidental = converted[1:]
    alteration = accidental.count("#") - accidental.count("-")
    if alteration not in ALTERATIONS:
        raise ValueError(f"Unsupported ChordGNN alteration {converted!r}")
    return step, alteration, converted


def _tie_consecutive_notes(notes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Reproduce upstream pitch-only consecutive-note tying."""

    columns = (
        "pitch",
        "spelling",
        "onset",
        "release",
        "ts_numerator",
        "ts_denominator",
    )
    missing = [column for column in columns if column not in notes]
    if missing:
        raise ValueError(f"Note rows are missing ChordGNN columns: {missing}")
    records = [
        {
            "pitch": int(round(float(row.pitch))),
            "spelling": str(row.spelling),
            "onset_beat": float(row.onset),
            "duration_beat": float(row.release) - float(row.onset),
            "ts_beats": float(row.ts_numerator),
            "ts_beat_type": float(row.ts_denominator),
        }
        for row in notes[list(columns)].itertuples(index=False)
    ]
    records.sort(key=lambda item: (item["onset_beat"], item["pitch"]))
    skipped: set[int] = set()
    for index in range(max(0, len(records) - 1)):
        if index in skipped:
            continue
        note = records[index]
        while True:
            expected = note["onset_beat"] + note["duration_beat"]
            match = next(
                (
                    candidate
                    for candidate in range(index + 1, len(records))
                    if candidate not in skipped
                    and records[candidate]["onset_beat"] == expected
                    and records[candidate]["pitch"] == note["pitch"]
                ),
                None,
            )
            if match is None:
                break
            note["duration_beat"] += records[match]["duration_beat"]
            skipped.add(match)
    tied = pd.DataFrame(
        [record for index, record in enumerate(records) if index not in skipped]
    )
    if tied.empty:
        raise ValueError("Consecutive-note tying removed every note")
    return tied, len(skipped)


def _create_divisions(notes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    duration_fractions = [
        Fraction(float(value)).limit_denominator(256)
        for value in notes["duration_beat"]
    ]
    onset_fractions = [
        Fraction(float(value)).limit_denominator(256) for value in notes["onset_beat"]
    ]
    divisions = int(
        np.lcm.reduce(
            [
                Fraction(float(value)).limit_denominator(256).denominator
                for value in np.unique(notes["duration_beat"])
            ]
        )
    )
    onset_div = np.asarray(
        [
            int(divisions * value.numerator / value.denominator)
            for value in onset_fractions
        ],
        dtype=np.int64,
    )
    if int(onset_div.min()) < 0:
        onset_div -= int(onset_div.min())
    duration_div = np.asarray(
        [
            int(divisions * value.numerator / value.denominator)
            for value in duration_fractions
        ],
        dtype=np.int64,
    )
    if np.any(duration_div <= 0):
        raise ValueError("ChordGNN division conversion produced a non-positive note")
    return onset_div, duration_div, divisions


def chord_interval_vector(pitches: Sequence[int]) -> tuple[list[int], list[int]]:
    vector = [0] * 6
    pitch_classes = set(int(pitch) % 12 for pitch in pitches)
    for first, second in combinations(pitch_classes, 2):
        interval = abs(first - second)
        index = interval if interval <= 6 else 12 - interval
        if index:
            vector[index - 1] += 1
    return vector, list(pitch_classes)


def _note_features(notes: pd.DataFrame) -> np.ndarray:
    n_notes = len(notes)
    pitches = notes["pitch"].to_numpy(dtype=np.int64)
    onsets = notes["onset_beat"].to_numpy(dtype=np.float64)
    durations = notes["duration_beat"].to_numpy(dtype=np.float64)
    releases = onsets + durations
    time_signatures = notes["ts_beats"].to_numpy(dtype=np.float64)
    spelling = np.zeros((n_notes, 14), dtype=np.float64)
    for index, value in enumerate(notes["spelling"]):
        step, alteration, _ = _parse_spelling(value)
        spelling[index, STEP_TO_INDEX[step]] = 1
        spelling[index, 7 + ALTERATIONS.index(alteration)] = 1

    unique_onsets = np.unique(onsets)
    onset_groups = {
        float(onset): np.flatnonzero(onsets == onset) for onset in unique_onsets
    }
    release_groups = {
        float(release): np.flatnonzero(releases == release)
        for release in np.unique(releases)
    }
    next_distance = np.r_[np.diff(unique_onsets), 0.0]
    next_by_onset = {
        float(onset): float(distance)
        for onset, distance in zip(unique_onsets, next_distance)
    }

    chord_analysis = np.zeros((n_notes, len(NOTE_FEATURE_NAMES)), dtype=np.float64)
    min_max = np.zeros((n_notes, 2), dtype=np.float64)
    metrical = np.zeros((n_notes, 4), dtype=np.float64)
    for onset in unique_onsets:
        current = onset_groups[float(onset)]
        sustained = np.flatnonzero((onsets < onset) & (releases > onset))
        sounding = np.concatenate((current, sustained))
        chord_pitches = pitches[sounding]
        predecessor = release_groups.get(float(onset), np.empty(0, dtype=np.int64))
        predecessor_pitches = pitches[predecessor]
        interval_vector, pitch_classes = chord_interval_vector(chord_pitches)
        chord_flags = [
            int(tuple(interval_vector) == template)
            for template in CHORD_TEMPLATES.values()
        ]
        recentered = sorted(value - min(pitch_classes) for value in pitch_classes)
        is_maj = int(
            bool(chord_flags[0]) and recentered in ([0, 4, 7], [0, 5, 9], [0, 3, 8])
        )
        is_min = int(
            bool(chord_flags[0]) and recentered in ([0, 3, 7], [0, 5, 8], [0, 4, 9])
        )
        chord_min = int(chord_pitches.min())
        chord_max = int(chord_pitches.max())
        is_pmaj = int(
            bool(is_maj)
            and 4 in ((chord_pitches - chord_min) % 12)
            and 7 in ((chord_pitches - chord_min) % 12)
        )
        span_class = (chord_max - chord_min) % 12
        for index in current:
            pitch = int(pitches[index])
            intervals = [
                int(
                    predecessor_pitches.size > 0
                    and (
                        interval in (predecessor_pitches - pitch)
                        or -interval in (predecessor_pitches - pitch)
                    )
                )
                for interval in range(13)
            ]
            tail = [
                is_maj,
                is_pmaj,
                is_min,
                int(durations[index] > time_signatures[index]),
                int(span_class == 10),
                int(span_class == 7),
                int(span_class in (3, 4)),
                int(span_class == 0 and chord_max != chord_min),
                int(pitch - chord_min in (1, -1)),
                int(pitch - chord_min in (2, -2)),
            ]
            chord_analysis[index] = np.asarray(
                interval_vector + intervals + chord_flags + tail,
                dtype=np.float64,
            )
            min_max[index] = (pitch == chord_min, pitch == chord_max)
            ts_beats = time_signatures[index]
            metrical[index] = (
                np.remainder(onset, ts_beats) == 0,
                np.remainder(onset, 1) == 0,
                np.remainder(onset, ts_beats / 2) == 0,
                1 - np.tanh(next_by_onset[float(onset)] / ts_beats),
            )

    duration_feature = (1 - np.tanh(durations / time_signatures)).reshape(n_notes, 1)
    features = np.hstack(
        (
            pitches.reshape(n_notes, 1),
            spelling,
            duration_feature,
            chord_analysis,
            spelling,
            min_max,
            metrical,
        )
    )
    if features.shape != (n_notes, FEATURE_DIM):
        raise AssertionError(f"Unexpected ChordGNN features {features.shape}")
    return features.astype(np.float32)


def _graph_edges(
    onset_div: np.ndarray, duration_div: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    end_div = onset_div + duration_div
    onset_groups = {
        int(value): np.flatnonzero(onset_div == value) for value in np.unique(onset_div)
    }
    sources: list[int] = []
    targets: list[int] = []
    edge_types: list[int] = []
    for source in range(len(onset_div)):
        for target in onset_groups[int(onset_div[source])]:
            if source != int(target):
                sources.append(source)
                targets.append(int(target))
                edge_types.append(EDGE_TYPES["onset"])
        for target in onset_groups.get(int(end_div[source]), ()):
            sources.append(source)
            targets.append(int(target))
            edge_types.append(EDGE_TYPES["consecutive"])
        during = np.flatnonzero(
            (onset_div[source] < onset_div) & (end_div[source] > onset_div)
        )
        for target in during:
            sources.append(source)
            targets.append(int(target))
            edge_types.append(EDGE_TYPES["during"])

    unique_ends = np.sort(np.unique(end_div))
    for end in unique_ends[:-1]:
        if int(end) in onset_groups:
            continue
        future = np.unique(onset_div[onset_div > end])
        if not future.size:
            continue
        for source in np.flatnonzero(end_div == end):
            for target in onset_groups[int(future[0])]:
                sources.append(int(source))
                targets.append(int(target))
                edge_types.append(EDGE_TYPES["silence"])
    return (
        np.asarray([sources, targets], dtype=np.int64),
        np.asarray(edge_types, dtype=np.int8),
    )


def _encode_onset_labels(
    onset_quarter: np.ndarray,
    events: Sequence[Any],
    vocabularies: Mapping[str, Sequence[str]],
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], Counter[str], dict[str, Counter[str]]
]:
    label_to_index = {
        component: {label: index for index, label in enumerate(classes)}
        for component, classes in vocabularies.items()
    }
    normalizations: Counter[str] = Counter()
    auxiliary_oov: dict[str, Counter[str]] = {
        component: Counter() for component in AUXILIARY_COMPONENTS
    }
    labels = {
        component: np.full(len(onset_quarter), -1, dtype=np.int64)
        for component in COMPONENTS
    }
    masks = {
        component: np.zeros(len(onset_quarter), dtype=np.bool_)
        for component in COMPONENTS
    }
    event_onsets = np.asarray([event.onset for event in events], dtype=np.float64)
    for onset_index, onset in enumerate(onset_quarter):
        event_index = int(np.searchsorted(event_onsets, onset, side="right") - 1)
        if event_index < 0:
            continue
        event = events[event_index]
        for component in COMPONENTS:
            value = event.labels[component]
            if component == "harmonic_rhythm":
                value = (
                    "True"
                    if math.isclose(onset, event.onset, abs_tol=1e-7)
                    else "False"
                )
            if (
                component in ("local_key", "chord_root")
                and value not in label_to_index[component]
            ):
                replacement = canonicalize_enharmonic_to_classes(
                    str(value),
                    vocabularies[component],
                    preserve_mode_case=(component == "local_key"),
                )
                if replacement is not None:
                    normalizations[f"{component}:{value}->{replacement}"] += 1
                    value = replacement
            encoded = label_to_index[component].get(value)
            if encoded is None:
                if component in AUXILIARY_COMPONENTS:
                    auxiliary_oov[component][str(value)] += 1
                if component in CORE_COMPONENTS:
                    raise ValueError(
                        f"Core ChordGNN label {component}={value!r} is out of schema"
                    )
                continue
            labels[component][onset_index] = encoded
            masks[component][onset_index] = True
    return labels, masks, normalizations, auxiliary_oov


def convert_piece(
    csv_path: Path,
    *,
    dataset_root: Path,
    split_dir: Path,
    relative_path: str | None = None,
    vocabularies: Mapping[str, Sequence[str]] | None = None,
) -> ChordGraphExample:
    """Convert one source CSV into a validated graph example."""

    csv_path = csv_path.resolve()
    relative = relative_path or str(csv_path.relative_to(dataset_root.resolve()))
    df = pd.read_csv(csv_path, low_memory=False)
    notes, invalid_note_rows = _extract_notes(df)
    if notes.empty:
        raise ValueError(f"No valid note rows in {csv_path}")
    events, event_diagnostics = extract_harmony_events(df, relative)
    if not events:
        raise ValueError(f"No harmony events in {csv_path}")
    if vocabularies is None:
        vocabularies, _, _ = build_vocabularies(dataset_root, split_dir)

    original_note_onsets = np.sort(notes["onset"].unique().astype(np.float64))
    tied, tied_rows = _tie_consecutive_notes(notes)
    onset_div, duration_div, divisions = _create_divisions(tied)
    tied = tied.assign(onset_div=onset_div, duration_div=duration_div)
    x = _note_features(tied)
    edge_index, edge_type = _graph_edges(onset_div, duration_div)
    unique_div, first_indices = np.unique(onset_div, return_index=True)
    onset_quarter = tied["onset_beat"].to_numpy(dtype=np.float64)[first_indices]
    labels, masks, label_normalizations, auxiliary_oov = _encode_onset_labels(
        onset_quarter, events, vocabularies
    )
    edge_counts = Counter(int(value) for value in edge_type)
    diagnostics = {
        "source_sha256": file_sha256(csv_path),
        "invalid_note_rows": invalid_note_rows,
        "source_note_rows": int(len(notes)),
        "tied_note_rows": int(tied_rows),
        "nodes": int(len(tied)),
        "graph_onsets": int(len(unique_div)),
        "reference_note_onsets": int(len(original_note_onsets)),
        "divisions_per_quarter": divisions,
        "edges": int(edge_index.shape[1]),
        "edge_type_counts": {
            name: int(edge_counts[value]) for name, value in EDGE_TYPES.items()
        },
        "label_normalizations": dict(sorted(label_normalizations.items())),
        "auxiliary_oov_masks": {
            component: dict(sorted(auxiliary_oov[component].items()))
            for component in AUXILIARY_COMPONENTS
        },
        "head_supervision": {
            component: int(masks[component].sum()) for component in COMPONENTS
        },
        "event_diagnostics": event_diagnostics,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    example = ChordGraphExample(
        relative_path=relative,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_onset_div=onset_div,
        onset_representatives=first_indices.astype(np.int64),
        onset_quarter=onset_quarter,
        labels=labels,
        masks=masks,
        reference_note_onsets=original_note_onsets,
        diagnostics=diagnostics,
    )
    example.validate(vocabularies)
    return example


def ontology_payload(
    dataset_root: Path, split_dir: Path
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    vocabularies, counters, metadata = build_vocabularies(dataset_root, split_dir)
    payload = {
        "schema_version": "chordgnn-pdcml-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "components": list(COMPONENTS),
        "core_components": list(CORE_COMPONENTS),
        "auxiliary_components": list(AUXILIARY_COMPONENTS),
        "vocabularies": vocabularies,
        "observed_counts": {
            component: dict(sorted(counter.items()))
            for component, counter in counters.items()
        },
        "metadata": metadata,
    }
    return vocabularies, payload
