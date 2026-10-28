"""Build ordinary RNA SFT annotations from pre-encoded, segmented CSVs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from rngen.utils.rna import parse_gt_rnbert
from rngen.paths import RNA_SEQUENCE_DIR, RNA_ANNOS_DIR, TEMPLATES_PATH
from rngen.data_prep.build_key_annos import (
    standardize_event_string_bos_eos,
    add_spelling_to_events,
)

# Constants for token processing
TOKEN_BOS = "<s>"
TOKEN_EOS = "</s>"
ORIGINAL_COMPOUND_RATIO = 8
NEW_COMPOUND_RATIO = ORIGINAL_COMPOUND_RATIO + 1


def extract_token_info(token: str) -> Tuple[int, int]:
    """Extracts bar and position from an octuple token."""
    try:
        dim, val = token.strip("<>").split("-")
        return int(dim), int(val)
    except (ValueError, IndexError):
        return -1, -1


def remove_bos_eos(s: str) -> str:
    """Removes the beginning <s> and ending </s> tokens from a string."""
    tokens = s.split()

    # Find core tokens (between <s> and </s>)
    start_idx = 0
    for i, token in enumerate(tokens):
        if token != TOKEN_BOS:
            start_idx = i
            break

    end_idx = len(tokens)
    for i in range(len(tokens) - 1, start_idx - 1, -1):
        if tokens[i] != TOKEN_EOS:
            end_idx = i + 1
            break

    return " ".join(tokens[start_idx:end_idx])


def get_harmony_change_points(
    events_str: str,
    gt_rnbert_str: str,
    target_bars: List[int],
    compound_ratio: int,
) -> Dict[Tuple[int, int], str]:
    """Identifies positions where harmony changes occur within target bars."""
    events = remove_bos_eos(events_str).split()
    gt_rnberts = remove_bos_eos(gt_rnbert_str).split()
    target_bars_set = set(target_bars)

    harmony_points = {}
    current_bar, current_pos = -1, -1

    for i in range(0, len(events), compound_ratio):
        note_idx = i // compound_ratio
        if note_idx >= len(gt_rnberts):
            break

        # Extract bar and position from the first two tokens of the octuple
        dim_bar, bar = extract_token_info(events[i])
        dim_pos, pos = extract_token_info(events[i + 1])

        if dim_bar == 0 and dim_pos == 1:
            current_bar, current_pos = bar, pos
            if current_bar in target_bars_set:
                gt_rnbert = gt_rnberts[note_idx]
                if gt_rnbert != "nan":
                    if (current_bar, current_pos) not in harmony_points:
                        harmony_points[(current_bar, current_pos)] = gt_rnbert

    return harmony_points


def count_notes_in_target_bars(
    events_str: str, target_bars: List[int], compound_ratio: int
) -> int:
    """Counts the number of notes within the target bars."""
    events = events_str.split()
    target_bars_set = set(target_bars)
    note_count = 0
    for i in range(0, len(events), compound_ratio):
        # Extract bar from the first token of the octuple
        dim_bar, bar = extract_token_info(events[i])
        if dim_bar == 0 and bar in target_bars_set:
            note_count += 1
    return note_count


def process_csv_file(csv_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Processes a single segmented CSV file and generates annotation samples."""
    samples = []
    compound_ratio = ORIGINAL_COMPOUND_RATIO
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # 1. Get required columns
            events = row.get("events", "")
            gt_rnbert_str = row.get("gt_rnbert", "")
            keys_str = row.get("key", "")
            spellings = row.get("spelling", "")
            chord_tones_str = row.get("chord_tones", "")
            chord_factors_str = row.get("chord_tone_factors", "")
            target_bars_str = row.get("target_bar_numbers_w_offset", "")

            if not all([events, gt_rnbert_str, keys_str, target_bars_str]):
                raise ValueError(
                    f"Missing required columns in row {idx} of {csv_path}: "
                    f"events, gt_rnbert, key, or target_bar_numbers_w_offset."
                )
            events = standardize_event_string_bos_eos(events, compound_ratio)
            target_bar_list = [int(bar) for bar in target_bars_str.split()]
            # 2. Add accidentals if requested
            if args.include_spelling:
                events = add_spelling_to_events(events, spellings)
                compound_ratio = NEW_COMPOUND_RATIO

            # 3. Identify harmony changes and build ground truth
            harmony_points = get_harmony_change_points(
                events, gt_rnbert_str, target_bar_list, compound_ratio
            )
            keys = remove_bos_eos(keys_str).split()
            chord_tones = remove_bos_eos(chord_tones_str).split()
            chord_factors = remove_bos_eos(chord_factors_str).split()
            event_tokens = remove_bos_eos(events).split()

            gt_by_key = defaultdict(list)
            rna_info_list = []

            sorted_points = sorted(harmony_points.keys())

            for bar, pos in sorted_points:
                # Find the note index for this harmony change
                note_idx = -1
                for i in range(0, len(event_tokens), compound_ratio):
                    _, b = extract_token_info(event_tokens[i])
                    _, p = extract_token_info(event_tokens[i + 1])
                    if b == bar and p == pos:
                        note_idx = i // compound_ratio
                        break

                if note_idx == -1:
                    raise ValueError(
                        f"Could not find note index for bar {bar}, position {pos} in row {idx} of {csv_path}"
                    )

                # Get features for this harmony
                local_key = keys[note_idx] if note_idx < len(keys) else "N/A"
                gt_rnbert = harmony_points[(bar, pos)]
                current_chord_tones = (
                    chord_tones[note_idx] if note_idx < len(chord_tones) else ""
                )
                current_chord_factors = (
                    chord_factors[note_idx] if note_idx < len(chord_factors) else ""
                )

                # Build GT string part
                pos_str = (
                    f"quarterbeats {pos / 16 + 1:.2f}"
                    if args.use_quarterbeats_in_gt
                    else f"position {pos}"
                )
                gt_line = f"bar {bar} {pos_str}: {gt_rnbert}"
                if args.point_out_chord_tones and current_chord_tones != "nan":
                    gt_line += f": {current_chord_tones}"
                if args.point_out_chord_factors and current_chord_factors != "nan":
                    gt_line += f", {current_chord_factors}"
                gt_by_key[local_key].append(gt_line)

                # Derive and store rna_info
                try:
                    parsed_rna = parse_gt_rnbert(gt_rnbert)
                    parsed_rna["local_key"] = local_key
                    parsed_rna["chord_tones"] = (
                        current_chord_tones if current_chord_tones != "nan" else ""
                    )
                    parsed_rna["chord_factors"] = (
                        current_chord_factors if current_chord_factors != "nan" else ""
                    )
                    rna_info_list.append(parsed_rna)
                except ValueError:
                    raise ValueError(
                        f"Could not parse gt_rnbert '{gt_rnbert}' in row {idx} of {csv_path}"
                    )

            # 4. Construct the sample dictionary
            instruction_str = (
                f"{target_bar_list[0]}-{target_bar_list[-1]}" if target_bar_list else ""
            )

            score_id = "/".join(row["csv_path"].split("/")[-2:])
            file_path = "/".join(str(csv_path).split("/")[-4:])

            sample = {
                "instruction": instruction_str,
                "output": gt_by_key,
                "events": events,
                "spelling": remove_bos_eos(spellings),  # remove the leading <s>
                "note_count": count_notes_in_target_bars(
                    events, target_bar_list, compound_ratio
                ),
                "metadata": {
                    "file_path": file_path,
                    "score_id": score_id,
                    "row_idx": idx,
                    "last_target_bar": target_bar_list[-1],
                },
                "rna_info": rna_info_list,
            }
            samples.append(sample)

    return samples


def create_annotation_files(args: argparse.Namespace):
    """Main function to find CSVs and create annotation files."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = Path(args.data_dir) / split
        if not split_dir.exists():
            print(f"Warning: Directory not found for split '{split}': {split_dir}")
            continue

        csv_files = list(split_dir.glob("*.csv"))
        print(f"Processing {len(csv_files)} files for '{split}' split...")

        all_samples = []
        for csv_file in tqdm(csv_files, desc=f"Processing {split} files"):
            all_samples.extend(process_csv_file(csv_file, args))

        # Create final dictionary for the split
        split_samples_dict = {"template": args.template}
        for i, sample in enumerate(all_samples):
            split_samples_dict[str(i)] = sample

        # Save to JSON file
        output_file = output_dir / f"{split}{args.suffix}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_samples_dict, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(all_samples)} samples to {output_file}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RNA_SEQUENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=RNA_ANNOS_DIR)
    parser.add_argument("--templates-path", type=Path, default=TEMPLATES_PATH)
    parser.add_argument("--suffix", default="", help="Optional filename suffix after the split name; default: train.json, valid.json, test.json.")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "valid", "test"),
        default=["train", "valid", "test"],
    )
    parser.add_argument(
        "--include-spelling", action=argparse.BooleanOptionalAction, default=True,
        help="Add the ninth event token per note (default: enabled).",
    )
    parser.add_argument(
        "--point-out-chord-tones", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--point-out-chord-factors", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--use-quarterbeats-in-gt", action=argparse.BooleanOptionalAction, default=True,
        help="Use the historical position / 16 + 1 quarterbeat convention.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    with args.templates_path.open(encoding="utf-8") as f:
        args.template = json.load(f)["RNA_SFT"]
    create_annotation_files(args)


if __name__ == "__main__":
    main()
