#!/usr/bin/env python
"""Build Stage 2 fine-grained RNA curriculum SFT annotations.

The generated files keep the normal RNGen SFT annotation shape while adding a
task name and reference input for the curriculum dataset path to format later.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rngen.paths import TEMPLATES_PATH
from rngen.training.metrics_rna_components import (
    canonicalize_primary_function,
    canonicalize_secondary_function,
    canonicalize_quality,
    parse_chord_factors,
    parse_chord_tones,
)

TASKS = (
    "harmony_span_prediction",
    "chord_tone_inventory_prediction",
    "tone_factor_pair_prediction",
    "quality_prediction",
    "inversion_figure_prediction",
    "ps_function_prediction",
)

ALLOWED_QUALITIES = {"M", "m", "Mm7", "o", "o7", "m7", "aug6", "ø7", "M7", "+"}
ALLOWED_INVERSIONS = {"", "6", "64", "7", "65", "43", "2"}
HARMONY_ONSET_RE = re.compile(
    r"^\s*bar\s+(?P<bar>\d+)\s+quarterbeats\s+(?P<qb>[\d.]+)\s*:"
)


class CurriculumBuildError(ValueError):
    """Raised when source annotations cannot be safely converted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RNA Stage 2 curriculum annotation JSON files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing source RNA SFT annotation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated Stage 2 annotation JSON files are written.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=TEMPLATES_PATH,
        help="Path to rngen/config/templates.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to generate.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Source/output filename suffix after the split name; default: train.json, valid.json, test.json.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=TASKS,
        help="Curriculum tasks to generate.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional debug limit for source samples per split.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Optional JSON indent for generated files. Defaults to compact JSON.",
    )
    return parser.parse_args()


def load_stage2_template(template_path: Path) -> Mapping[str, Any]:
    with template_path.open() as f:
        templates = json.load(f)
    try:
        return templates["RNA_CURRICULUM_STAGE2"]
    except KeyError as exc:
        raise CurriculumBuildError(
            f"RNA_CURRICULUM_STAGE2 missing from {template_path}"
        ) from exc


def iter_source_samples(
    source_data: Mapping[str, Any],
    *,
    max_samples: Optional[int],
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    sample_ids = sorted(
        (key for key in source_data.keys() if key != "template"),
        key=lambda value: int(value),
    )
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    for sample_id in sample_ids:
        yield sample_id, source_data[sample_id]


def flatten_output_harmonies(output: Any) -> List[str]:
    if not isinstance(output, Mapping):
        raise CurriculumBuildError("Expected RNA output to be a local-key mapping.")

    harmonies: List[str] = []
    for values in output.values():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise CurriculumBuildError("Expected each RNA output value to be a list.")
        harmonies.extend(str(value) for value in values)
    return harmonies


def parse_onset(harmony: str) -> Tuple[int, Decimal]:
    match = HARMONY_ONSET_RE.match(harmony)
    if match is None:
        raise CurriculumBuildError(f"Could not parse harmony onset: {harmony!r}")
    try:
        qb = Decimal(match.group("qb"))
    except InvalidOperation as exc:
        raise CurriculumBuildError(f"Invalid quarterbeats value: {harmony!r}") from exc
    return int(match.group("bar")), qb


def parse_target_end_bar(sample: Mapping[str, Any]) -> int:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("last_target_bar") is not None:
        return int(metadata["last_target_bar"])

    instruction = str(sample.get("instruction", "")).strip()
    match = re.match(r"^\s*\d+\s*-\s*(\d+)\s*$", instruction)
    if match:
        return int(match.group(1))

    raise CurriculumBuildError(
        "Cannot determine last target bar from metadata.last_target_bar or instruction."
    )


def format_qb(qb: Decimal) -> str:
    if qb == qb.to_integral_value():
        return f"{int(qb)}.0"
    return format(qb.normalize(), "f")


def format_time(bar: int, qb: Decimal) -> str:
    return f"bar {bar} qb {format_qb(qb)}"


def build_spans_and_rna_info(
    sample: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Mapping[str, Any]]]:
    harmonies = flatten_output_harmonies(sample.get("output"))
    rna_info = sample.get("rna_info")
    if not isinstance(rna_info, list):
        raise CurriculumBuildError("Expected sample rna_info to be a list.")
    if len(harmonies) != len(rna_info):
        raise CurriculumBuildError(
            f"Parsed {len(harmonies)} harmony onsets but found {len(rna_info)} rna_info entries."
        )

    ordered_harmonies = []
    for original_index, (harmony, annotation) in enumerate(zip(harmonies, rna_info)):
        bar, qb = parse_onset(harmony)
        ordered_harmonies.append((bar, qb, original_index, annotation))
    ordered_harmonies.sort(key=lambda item: (item[0], item[1], item[2]))

    last_target_bar = parse_target_end_bar(sample)
    final_boundary = (last_target_bar + 1, Decimal("1.0"))

    spans: List[Dict[str, Any]] = []
    ordered_rna_info: List[Mapping[str, Any]] = []
    for idx, (bar, qb, _, annotation) in enumerate(ordered_harmonies):
        end_bar, end_qb = (
            (
                ordered_harmonies[idx + 1][0],
                ordered_harmonies[idx + 1][1],
            )
            if idx + 1 < len(ordered_harmonies)
            else final_boundary
        )
        if (end_bar, end_qb) <= (bar, qb):
            raise CurriculumBuildError(
                f"Non-increasing span boundary at harmony {idx}: {(bar, qb)} -> {(end_bar, end_qb)}"
            )
        spans.append(
            {
                "id": idx,
                "start": format_time(bar, qb),
                "end": format_time(end_bar, end_qb),
            }
        )
        ordered_rna_info.append(annotation)
    return spans, ordered_rna_info


def split_tones_and_factors(
    annotation: Mapping[str, Any]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    tones = parse_chord_tones(annotation)
    factors = parse_chord_factors(annotation)
    if len(tones) != len(factors):
        raise CurriculumBuildError(
            f"Chord-tone/factor length mismatch: {tones!r} vs {factors!r}"
        )
    return tones, factors


def build_reference_spans(spans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(span) for span in spans]


def build_reference_spans_with_keys(
    spans: Sequence[Mapping[str, Any]],
    rna_info: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    reference = []
    for span, annotation in zip(spans, rna_info):
        item = dict(span)
        local_key = str(annotation.get("local_key", "")).strip()
        if not local_key:
            raise CurriculumBuildError(f"Missing local_key for span {span['id']}")
        item["local_key"] = local_key
        reference.append(item)
    return reference


def build_task_output(
    task_name: str,
    spans: Sequence[Mapping[str, Any]],
    rna_info: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for span, annotation in zip(spans, rna_info):
        span_id = int(span["id"])

        if task_name == "harmony_span_prediction":
            output.append(dict(span))
        elif task_name == "chord_tone_inventory_prediction":
            tones, _ = split_tones_and_factors(annotation)
            output.append({"id": span_id, "chord_tones": list(tones)})
        elif task_name == "tone_factor_pair_prediction":
            tones, factors = split_tones_and_factors(annotation)
            output.append({"id": span_id, "tone_factors": dict(zip(tones, factors))})
        elif task_name == "quality_prediction":
            quality = canonicalize_quality(annotation)
            if quality not in ALLOWED_QUALITIES:
                raise CurriculumBuildError(f"Unsupported quality label: {quality!r}")
            output.append({"id": span_id, "quality": quality})
        elif task_name == "inversion_figure_prediction":
            inversion = str(annotation.get("inversion", "")).strip()
            if inversion not in ALLOWED_INVERSIONS:
                raise CurriculumBuildError(
                    f"Unsupported inversion label: {inversion!r}"
                )
            output.append({"id": span_id, "inversion_figure": inversion or "none"})
        elif task_name == "ps_function_prediction":
            primary = canonicalize_primary_function(annotation)
            secondary = canonicalize_secondary_function(annotation)
            if primary is None or secondary is None:
                raise CurriculumBuildError(
                    f"Could not canonicalize P/S functions: {annotation!r}"
                )
            output.append(
                {"id": span_id, "primary": primary, "secondary": secondary or "none"}
            )
        else:
            raise CurriculumBuildError(f"Unsupported task: {task_name}")

    return output


def build_reference_input(
    task_name: str,
    spans: Sequence[Mapping[str, Any]],
    rna_info: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if task_name == "harmony_span_prediction":
        return []
    if task_name == "ps_function_prediction":
        return build_reference_spans_with_keys(spans, rna_info)
    return build_reference_spans(spans)


def build_generated_sample(
    *,
    source_split: str,
    source_sample_id: str,
    source_sample: Mapping[str, Any],
    task_name: str,
    spans: Sequence[Mapping[str, Any]],
    rna_info: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reference_input = build_reference_input(task_name, spans, rna_info)
    output = build_task_output(task_name, spans, rna_info)

    source_metadata = deepcopy(source_sample.get("metadata", {}))
    return {
        "seq_id": source_sample_id,
        "instruction": source_sample["instruction"],
        "task_name": task_name,
        "reference_input": reference_input,
        "output": output,
        "metadata": {
            "task_name": task_name,
            "source_sample_id": source_sample_id,
            "source_split": source_split,
            "source_metadata": source_metadata,
            "target_bars": source_sample.get("target_bars", []),
            "last_target_bar": source_metadata.get("last_target_bar"),
        },
    }


def build_split(
    *,
    split: str,
    source_path: Path,
    stage2_template: Mapping[str, Any],
    tasks: Sequence[str],
    max_samples: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with source_path.open() as f:
        source_data = json.load(f)

    generated: Dict[str, Any] = {"template": stage2_template, "seq_data": {}}
    summary: Dict[str, Any] = {
        "source_path": str(source_path),
        "source_samples": 0,
        "generated_samples": 0,
        "task_counts": Counter(),
        "harmony_count_histogram": Counter(),
    }

    next_id = 0
    for source_sample_id, source_sample in iter_source_samples(
        source_data,
        max_samples=max_samples,
    ):
        try:
            spans, ordered_rna_info = build_spans_and_rna_info(source_sample)
        except Exception as exc:
            raise CurriculumBuildError(
                f"Failed to build spans for {split} sample {source_sample_id}: {exc}"
            ) from exc

        summary["source_samples"] += 1
        summary["harmony_count_histogram"][len(spans)] += 1
        generated["seq_data"][source_sample_id] = {
            "events": source_sample["events"],
            "spelling": source_sample.get("spelling", ""),
            "note_count": source_sample.get("note_count", 0),
            "metadata": deepcopy(source_sample.get("metadata", {})),
            "target_bars": source_sample.get("target_bars", []),
        }

        for task_name in tasks:
            try:
                generated[str(next_id)] = build_generated_sample(
                    source_split=split,
                    source_sample_id=source_sample_id,
                    source_sample=source_sample,
                    task_name=task_name,
                    spans=spans,
                    rna_info=ordered_rna_info,
                )
            except Exception as exc:
                raise CurriculumBuildError(
                    f"Failed to build {task_name} for {split} sample {source_sample_id}: {exc}"
                ) from exc

            summary["task_counts"][task_name] += 1
            next_id += 1

    summary["generated_samples"] = next_id
    summary["task_counts"] = dict(summary["task_counts"])
    summary["harmony_count_histogram"] = {
        str(key): value
        for key, value in sorted(summary["harmony_count_histogram"].items())
    }
    expected_count = summary["source_samples"]
    mismatched_tasks = {
        task: count
        for task, count in summary["task_counts"].items()
        if count != expected_count
    }
    if mismatched_tasks:
        raise CurriculumBuildError(
            f"Unequal task counts in {split}: {mismatched_tasks}"
        )

    return generated, summary


def main() -> None:
    args = parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(TASKS))
    if unknown_tasks:
        raise CurriculumBuildError(f"Unknown tasks: {unknown_tasks}")

    stage2_template = load_stage2_template(args.template_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "suffix": args.suffix,
        "tasks": list(args.tasks),
        "splits": {},
    }

    for split in args.splits:
        source_path = args.source_dir / f"{split}{args.suffix}.json"
        if not source_path.exists():
            raise CurriculumBuildError(f"Missing source annotation file: {source_path}")

        generated, split_summary = build_split(
            split=split,
            source_path=source_path,
            stage2_template=stage2_template,
            tasks=args.tasks,
            max_samples=args.max_samples_per_split,
        )

        output_path = args.output_dir / f"{split}{args.suffix}.json"
        with output_path.open("w") as f:
            json.dump(
                generated,
                f,
                ensure_ascii=False,
                indent=args.indent,
                separators=None if args.indent is not None else (",", ":"),
            )
            f.write("\n")

        summary["splits"][split] = split_summary

    summary_path = args.output_dir / "build_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
