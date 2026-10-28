#!/usr/bin/env python3
"""Quantify dataset-side coverage gaps between RNGen and the RNBERT baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MISSING_TOKENS = {"", "na", "nan", "none", "<na>"}
DEFAULT_SPLITS = ("train", "test")
DEFAULT_AUG6_LABELS = ("it6", "fr6", "ger6", "aug6", "x")


@dataclass
class SplitStats:
    total_onsets: int
    key_labeled_onsets: int
    full_key_counts: Counter[str]
    rnbert_key_counts: Counter[str]
    collapse_counts: Dict[str, Counter[str]]
    quality_counts: Counter[str]
    aug6_counts: Counter[str]
    missing_aug6_quality_onsets: int
    csv_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze dataset-side expanded coverage gaps between RNGen and an "
            "RNBERT-style label space."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing train_paths.txt / test_paths.txt style split files.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory containing the processed RNBERT CSV files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Which split files to analyze. Defaults to train and test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save JSON and CSV reports.",
    )
    return parser.parse_args()


def normalize_token(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    token = str(value).strip()
    if token.lower() in MISSING_TOKENS:
        return None
    return token


def normalize_mode(value: object) -> Optional[str]:
    mode = normalize_token(value)
    if mode is None:
        return None
    return mode


def normalize_key_pc(value: object) -> Optional[str]:
    token = normalize_token(value)
    if token is None:
        return None
    try:
        return str(int(round(float(token))))
    except ValueError:
        return token


def make_full_key_label(key: object, mode: object) -> Optional[str]:
    key_token = normalize_token(key)
    mode_token = normalize_mode(mode)
    if key_token is None or mode_token is None:
        return None
    return f"{key_token}:{mode_token}"


def make_rnbert_key_label(key_pc: object, mode: object) -> Optional[str]:
    key_pc_token = normalize_key_pc(key_pc)
    mode_token = normalize_mode(mode)
    if key_pc_token is None or mode_token is None:
        return None
    return f"{key_pc_token}:{mode_token}"


def canonicalize_quality_label(value: object) -> Optional[str]:
    token = normalize_token(value)
    if token is None:
        return None
    return token


def classify_aug6_quality(value: object) -> Optional[str]:
    token = normalize_token(value)
    if token is None:
        return None

    lowered = token.lower()
    if lowered == "x":
        return "x"
    if "ger" in lowered and "6" in lowered:
        return "Ger6"
    if "fr" in lowered and "6" in lowered:
        return "Fr6"
    if "it" in lowered and "6" in lowered:
        return "It6"
    if lowered in {"ger", "fr", "it"}:
        return token
    if "aug6" in lowered or "augmented6" in lowered or "augmented 6" in lowered:
        return token
    return None


def read_split_paths(split_dir: Path, split_name: str) -> List[str]:
    split_path = split_dir / f"{split_name}_paths.txt"
    with split_path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def _iter_onset_rows(csv_path: Path) -> Iterator[pd.Series]:
    usecols = [
        "onset",
        "type",
        "key",
        "mode",
        "key_pc",
        "quality",
    ]
    frame = pd.read_csv(csv_path, usecols=usecols)
    note_frame = frame.loc[frame["type"] == "note"].copy()
    if note_frame.empty:
        return

    for _, group in note_frame.groupby("onset", sort=False):
        yield group.iloc[0]


def analyze_split(split_paths: Iterable[str], dataset_dir: Path) -> SplitStats:
    full_key_counts: Counter[str] = Counter()
    rnbert_key_counts: Counter[str] = Counter()
    collapse_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    quality_counts: Counter[str] = Counter()
    aug6_counts: Counter[str] = Counter()

    total_onsets = 0
    key_labeled_onsets = 0
    missing_aug6_quality_onsets = 0
    csv_count = 0

    for relative_path in split_paths:
        csv_path = dataset_dir / relative_path
        csv_count += 1

        for onset_row in _iter_onset_rows(csv_path):
            total_onsets += 1

            full_key = make_full_key_label(onset_row.get("key"), onset_row.get("mode"))
            rnbert_key = make_rnbert_key_label(
                onset_row.get("key_pc"),
                onset_row.get("mode"),
            )
            if full_key is not None and rnbert_key is not None:
                key_labeled_onsets += 1
                full_key_counts[full_key] += 1
                rnbert_key_counts[rnbert_key] += 1
                collapse_counts[rnbert_key][full_key] += 1

            quality = canonicalize_quality_label(onset_row.get("quality"))
            if quality is not None:
                quality_counts[quality] += 1
                aug6_label = classify_aug6_quality(quality)
                if aug6_label is not None:
                    aug6_counts[aug6_label] += 1
            else:
                missing_aug6_quality_onsets += 1

    return SplitStats(
        total_onsets=total_onsets,
        key_labeled_onsets=key_labeled_onsets,
        full_key_counts=full_key_counts,
        rnbert_key_counts=rnbert_key_counts,
        collapse_counts=dict(collapse_counts),
        quality_counts=quality_counts,
        aug6_counts=aug6_counts,
        missing_aug6_quality_onsets=missing_aug6_quality_onsets,
        csv_count=csv_count,
    )


def build_summary(stats: SplitStats) -> Dict[str, object]:
    collapse_onsets = 0
    collapsed_groups = 0
    collapsed_full_labels = 0

    for rnbert_key, full_key_counter in stats.collapse_counts.items():
        if len(full_key_counter) > 1:
            collapsed_groups += 1
            collapsed_full_labels += len(full_key_counter)
            collapse_onsets += sum(full_key_counter.values())

    aug6_onsets = sum(stats.aug6_counts.values())
    explicit_aug6_onsets = sum(
        count for label, count in stats.aug6_counts.items() if label.lower() != "x"
    )
    x_aug6_onsets = stats.aug6_counts.get("x", 0)

    return {
        "csv_count": stats.csv_count,
        "total_onsets": stats.total_onsets,
        "key_labeled_onsets": stats.key_labeled_onsets,
        "unique_full_key_labels": len(stats.full_key_counts),
        "unique_rnbert_key_labels": len(stats.rnbert_key_counts),
        "collapsed_rnbert_key_groups": collapsed_groups,
        "full_key_labels_inside_collapsed_groups": collapsed_full_labels,
        "onsets_in_collapsed_key_groups": collapse_onsets,
        "collapsed_key_onset_ratio": (
            collapse_onsets / stats.key_labeled_onsets
            if stats.key_labeled_onsets
            else 0.0
        ),
        "aug6_onsets": aug6_onsets,
        "aug6_onset_ratio": aug6_onsets / stats.total_onsets
        if stats.total_onsets
        else 0.0,
        "explicit_aug6_onsets": explicit_aug6_onsets,
        "x_aug6_onsets": x_aug6_onsets,
        "missing_quality_onsets": stats.missing_aug6_quality_onsets,
    }


def _rows_from_counter(
    counter: Counter[str],
    *,
    total: int,
    label_key: str,
) -> List[Dict[str, object]]:
    rows = []
    for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                label_key: label,
                "onset_count": count,
                "onset_ratio": count / total if total else 0.0,
            }
        )
    return rows


def build_key_collapse_rows(stats: SplitStats) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rnbert_key, full_key_counter in sorted(stats.collapse_counts.items()):
        collapse_group_size = len(full_key_counter)
        collapse_onset_total = sum(full_key_counter.values())
        is_collapsed = collapse_group_size > 1

        for full_key, count in sorted(
            full_key_counter.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            rows.append(
                {
                    "rnbert_key_pc_mode": rnbert_key,
                    "full_local_key_label": full_key,
                    "onset_count": count,
                    "onset_ratio_within_rnbert_key": (
                        count / collapse_onset_total if collapse_onset_total else 0.0
                    ),
                    "collapse_group_size": collapse_group_size,
                    "collapse_onset_total": collapse_onset_total,
                    "is_collapsed": is_collapsed,
                }
            )
    return rows


def build_quality_rows(stats: SplitStats) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = stats.total_onsets
    for quality, count in sorted(
        stats.quality_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        aug6_class = classify_aug6_quality(quality)
        rows.append(
            {
                "quality_label": quality,
                "onset_count": count,
                "onset_ratio": count / total if total else 0.0,
                "is_aug6_like": aug6_class is not None,
                "aug6_category": aug6_class or "",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_reports(output_dir: Path, report: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "rnbert_expanded_coverage_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    for split_name, split_report in report["splits"].items():
        write_csv(
            output_dir / f"rnbert_full_key_distribution_{split_name}.csv",
            split_report["full_key_rows"],
        )
        write_csv(
            output_dir / f"rnbert_key_pc_mode_distribution_{split_name}.csv",
            split_report["rnbert_key_rows"],
        )
        write_csv(
            output_dir / f"rnbert_key_collapse_groups_{split_name}.csv",
            split_report["collapse_rows"],
        )
        write_csv(
            output_dir / f"rnbert_quality_distribution_{split_name}.csv",
            split_report["quality_rows"],
        )
        write_csv(
            output_dir / f"rnbert_aug6_distribution_{split_name}.csv",
            split_report["aug6_rows"],
        )


def main() -> None:
    args = parse_args()
    report: Dict[str, object] = {
        "split_dir": str(args.split_dir),
        "dataset_dir": str(args.dataset_dir),
        "splits": {},
    }

    for split_name in args.splits:
        split_paths = read_split_paths(args.split_dir, split_name)
        stats = analyze_split(split_paths, args.dataset_dir)
        summary = build_summary(stats)
        split_report = {
            "summary": summary,
            "full_key_rows": _rows_from_counter(
                stats.full_key_counts,
                total=stats.key_labeled_onsets,
                label_key="full_local_key_label",
            ),
            "rnbert_key_rows": _rows_from_counter(
                stats.rnbert_key_counts,
                total=stats.key_labeled_onsets,
                label_key="rnbert_key_pc_mode",
            ),
            "collapse_rows": build_key_collapse_rows(stats),
            "quality_rows": build_quality_rows(stats),
            "aug6_rows": _rows_from_counter(
                stats.aug6_counts,
                total=stats.total_onsets,
                label_key="aug6_category",
            ),
        }
        report["splits"][split_name] = split_report

        print(
            f"{split_name}: "
            f"total_onsets={summary['total_onsets']}, "
            f"key_labeled_onsets={summary['key_labeled_onsets']}, "
            f"unique_full_keys={summary['unique_full_key_labels']}, "
            f"unique_rnbert_keys={summary['unique_rnbert_key_labels']}, "
            f"collapsed_key_onset_ratio={summary['collapsed_key_onset_ratio']:.4f}, "
            f"aug6_onset_ratio={summary['aug6_onset_ratio']:.4f}"
        )

    if args.output_dir is not None:
        save_reports(args.output_dir, report)


if __name__ == "__main__":
    main()
