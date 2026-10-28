"""Convert pre-encoded sequence CSVs into Key SFT annotation JSON files."""

import argparse
import csv
import json
from tqdm import tqdm
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple, Any

# Support both direct execution and `python -m rngen.data_prep.build_key_annos`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rngen.paths import KEY_SEQUENCE_DIR, KEY_ANNOS_DIR, TEMPLATES_PATH
from rngen.event_vocabulary import get_spelling_event_token

TOKEN_BOS = "<s>"
TOKEN_EOS = "</s>"
ORIGINAL_COMPOUND_RATIO = 8
NEW_COMPOUND_RATIO = ORIGINAL_COMPOUND_RATIO + 1


def extract_token_info(token: str) -> Tuple[int, int]:
    """Extract bar number and position from an octuple token."""
    match = re.match(r"<(\d+)-(\d+)>", token)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def standardize_event_string_bos_eos(events_str: str, compound_ratio: int) -> str:
    """Ensure the events string starts with compound_ratio * <s> and ends with compound_ratio * </s>.

    Args:
        events_str: The input events string.
        compound_ratio: The number of <s> and </s> tokens to add.
    Returns:
        A new string with the required number of <s> and </s> tokens added.
    """
    event_tokens = events_str.split()

    # Find core event tokens (between <s> and </s>)
    music_event_start_idx = 0
    for i, token in enumerate(event_tokens):
        if token != TOKEN_BOS:
            music_event_start_idx = i
            break

    music_event_end_idx = len(event_tokens)
    for i in range(len(event_tokens) - 1, music_event_start_idx - 1, -1):
        if event_tokens[i] != TOKEN_EOS:
            music_event_end_idx = i + 1
            break

    core_event_tokens = event_tokens[music_event_start_idx:music_event_end_idx]

    start_tokens = " ".join([TOKEN_BOS] * compound_ratio)
    end_tokens = " ".join([TOKEN_EOS] * compound_ratio)

    return f"{start_tokens} {' '.join(core_event_tokens)} {end_tokens}"


def add_spelling_to_events(events_str: str, spelling_str: str) -> str:
    """Add spelling tokens to events string, converting 8-token groups to 9-token groups."""
    if not events_str or not spelling_str:
        raise ValueError("Spelling conversion requires non-empty events and spelling.")

    event_tokens = events_str.split()
    spelling_tokens = spelling_str.split()

    # Find core event tokens (between <s> and </s>)
    music_event_start_idx = 0
    for i, token in enumerate(event_tokens):
        if token != TOKEN_BOS:
            music_event_start_idx = i
            break

    music_event_end_idx = len(event_tokens)
    for i in range(len(event_tokens) - 1, music_event_start_idx - 1, -1):
        if event_tokens[i] != TOKEN_EOS:
            music_event_end_idx = i + 1
            break

    core_event_tokens = event_tokens[music_event_start_idx:music_event_end_idx]

    # Extract core spelling tokens (skip <s> if present)
    core_spelling_tokens = (
        spelling_tokens[1:]
        if spelling_tokens and spelling_tokens[0] == TOKEN_BOS
        else spelling_tokens
    )

    # Validate lengths
    if len(core_event_tokens) % ORIGINAL_COMPOUND_RATIO != 0:
        raise ValueError(
            f"Core event tokens length ({len(core_event_tokens)}) is not a multiple of {ORIGINAL_COMPOUND_RATIO}"
        )

    num_event_groups = len(core_event_tokens) // ORIGINAL_COMPOUND_RATIO
    if num_event_groups != len(core_spelling_tokens):
        raise ValueError(
            f"Mismatch: {num_event_groups} event groups but {len(core_spelling_tokens)} spellings"
        )
    if num_event_groups == 0:
        raise ValueError("Spelling conversion requires at least one note.")

    # Combine events with spellings
    new_event_sequence = []
    for i in range(num_event_groups):
        event_group = core_event_tokens[
            i * ORIGINAL_COMPOUND_RATIO : (i + 1) * ORIGINAL_COMPOUND_RATIO
        ]
        spelling_text = core_spelling_tokens[i]

        spelling_token = get_spelling_event_token(spelling_text)
        new_event_sequence.extend(event_group)
        new_event_sequence.append(spelling_token)

    start_tokens = " ".join([TOKEN_BOS] * NEW_COMPOUND_RATIO)
    end_tokens = " ".join([TOKEN_EOS] * NEW_COMPOUND_RATIO)

    return f"{start_tokens} {' '.join(new_event_sequence)} {end_tokens}"


def detect_key_changes(
    events: str,
    compound_ratio: int,
    key_values: str,
    target_bar_masks: str = None,
    target_bars_only: bool = False,
) -> List[Tuple[int, int, int, int, str]]:
    """Detect key changes from events and key values."""
    event_tokens = events.split()
    key_tokens = key_values.split()

    # Filter to only process target bars if specified
    target_bars_set = set()
    if target_bars_only and target_bar_masks:
        masks = [int(mask) for mask in target_bar_masks.split()]

        # Extract the actual bar numbers corresponding to target bars
        bar_indices = []
        for i in range(0, len(event_tokens), compound_ratio):
            if i + 1 >= len(event_tokens):
                break

            bar_token = event_tokens[i]
            bar_match = re.match(r"<0-(\d+)>", bar_token)
            if bar_match:
                bar = int(bar_match.group(1))
                if bar not in bar_indices:
                    bar_indices.append(bar)

        # Map masks to actual bar numbers
        for i, mask in enumerate(masks):
            if mask == 1 and i < len(bar_indices):
                target_bars_set.add(bar_indices[i])

    key_changes = []
    current_key = None
    start_bar = None
    start_pos = None

    for i in range(0, len(event_tokens), compound_ratio):
        if i + 1 >= len(event_tokens):
            break

        # Extract bar and position from tokens
        bar_token = event_tokens[i]
        pos_token = event_tokens[i + 1]

        bar_match = re.match(r"<0-(\d+)>", bar_token)
        pos_match = re.match(r"<1-(\d+)>", pos_token)

        if bar_match and pos_match:
            bar = int(bar_match.group(1))
            pos = int(pos_match.group(1))

            # Skip if we're only processing target bars and this isn't one
            if target_bars_only and target_bars_set and bar not in target_bars_set:
                continue

            # Get key for this note (octuple)
            note_idx = i // compound_ratio
            if note_idx < len(key_tokens):
                key = key_tokens[note_idx]

                if current_key != key:
                    # If we have a previous key, close its segment
                    if current_key is not None and start_bar is not None:
                        key_changes.append(
                            (start_bar, start_pos, prev_bar, prev_pos, current_key)
                        )

                    # Start new key segment
                    current_key = key
                    start_bar = bar
                    start_pos = pos

                prev_bar = bar
                prev_pos = pos

    # Close the last segment
    if current_key is not None and start_bar is not None:
        key_changes.append((start_bar, start_pos, prev_bar, prev_pos, current_key))

    return key_changes


def format_key_detection_response(
    key_changes: List[Tuple[int, int, int, int, str]]
) -> str:
    """Format key changes into the required response format."""
    response = []

    for start_bar, start_pos, end_bar, end_pos, key in key_changes:
        # Format key to proper notation (e.g., "Eb" -> "EbM" or "eb" -> "Ebm")
        if key.islower():
            formatted_key = key  # + " minor"  # Minor key
        else:
            formatted_key = key  # + " Major"  # Major key

        response.append(
            f"bar {start_bar} position {start_pos}, bar {end_bar} position {end_pos}: {formatted_key}"
        )

    return "\n".join(response)


def count_notes_in_range(
    events: str,
    compound_ratio: int,
    start_bar: int,
    start_pos: int,
    end_bar: int,
    end_pos: int,
) -> int:
    """Count the number of notes within a specific bar/position range."""
    event_tokens = events.split()
    note_count = 0

    for i in range(0, len(event_tokens), compound_ratio):
        if i + compound_ratio >= len(event_tokens):
            break

        bar_token = event_tokens[i]
        pos_token = event_tokens[i + 1]

        bar_match = re.match(r"<0-(\d+)>", bar_token)
        pos_match = re.match(r"<1-(\d+)>", pos_token)

        if bar_match and pos_match:
            bar = int(bar_match.group(1))
            pos = int(pos_match.group(1))

            # Check if this note is within the specified range
            if (bar > start_bar or (bar == start_bar and pos >= start_pos)) and (
                bar < end_bar or (bar == end_bar and pos <= end_pos)
            ):
                note_count += 1

    return note_count


def validate_csv_row(row, include_spelling: bool, target_bars_only: bool):
    """Reject malformed rows without repairing or changing annotation semantics."""
    required = {"events", "key", "target_bar_masks", "target_bar_numbers_w_offset", "csv_path"}
    if include_spelling:
        required.add("spelling")
    missing = sorted(name for name in required if not str(row.get(name) or "").strip())
    if missing:
        raise ValueError(f"Missing or empty required columns: {', '.join(missing)}")

    events = row["events"].split()
    while events and events[0] == TOKEN_BOS:
        events.pop(0)
    while events and events[-1] == TOKEN_EOS:
        events.pop()
    if not events or len(events) % ORIGINAL_COMPOUND_RATIO:
        raise ValueError("events must contain complete eight-token note groups.")
    bars = []
    for i in range(0, len(events), ORIGINAL_COMPOUND_RATIO):
        bar = re.fullmatch(r"<0-(\d+)>", events[i])
        pos = re.fullmatch(r"<1-(\d+)>", events[i + 1])
        if bar is None or pos is None:
            raise ValueError(f"Invalid bar/position tokens at note {i // ORIGINAL_COMPOUND_RATIO}.")
        number = int(bar.group(1))
        if number not in bars:
            bars.append(number)

    keys = row["key"].split()
    if keys[-1] == TOKEN_EOS:
        keys = keys[:-1]
    if not keys or keys[0] != TOKEN_BOS or len(keys) != len(events) // ORIGINAL_COMPOUND_RATIO + 1:
        raise ValueError("key must contain one leading <s> and one label per note.")
    try:
        targets = [int(value) for value in row["target_bar_numbers_w_offset"].split()]
        masks = [int(value) for value in row["target_bar_masks"].split()]
    except ValueError as exc:
        raise ValueError("Target bar numbers and masks must be integers.") from exc
    if len(masks) != len(bars) or any(mask not in (0, 1) for mask in masks):
        raise ValueError("target_bar_masks must contain one 0/1 value per event bar.")
    if any(target not in bars for target in targets):
        raise ValueError("Target bar numbers must occur in events.")
    if target_bars_only and targets != [bar for bar, mask in zip(bars, masks) if mask]:
        raise ValueError("Target bar numbers do not match target_bar_masks.")


def process_csv_file_alpaca(
    csv_path: Path, target_bars_only: bool = False, include_spelling: bool = False
) -> List[Dict[str, Any]]:
    """Process a CSV file and extract samples for key change detection in Alpaca format.

    Args:
        csv_path: Path to CSV file
        target_bars_only: If True, only detect key changes in target bars
        include_spelling: If True, add spelling tokens to events

    Returns:
        List of samples in Alpaca format
    """
    samples = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                validate_csv_row(row, include_spelling, target_bars_only)
            except ValueError as exc:
                raise ValueError(f"{csv_path}, row_idx={idx}: {exc}") from exc
            # Only process rows with target bar information
            if "target_bar_masks" in row and "events" in row and "key" in row:
                events = row["events"]
                key_values = row["key"]

                # # Filter out rare keys from data augmentation. For simplicity, we temporarily define rare keys as the new keys introduced by augmentation.
                # rare_keys = ["e#", "E#", "b#", "B#", "fb", "A#", "D#"]
                # found_rare_key = False
                # for rare_key in rare_keys:
                #     if rare_key in key_values:
                #         found_rare_key = True
                #         break

                # if found_rare_key:
                #     continue

                target_bar_masks = row["target_bar_masks"]

                # Add spelling if requested and available
                if include_spelling and "spelling" in row:
                    compound_ratio = NEW_COMPOUND_RATIO
                    try:
                        events = add_spelling_to_events(events, row["spelling"])
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{csv_path}, row_idx={idx}: {exc}") from exc
                else:
                    compound_ratio = ORIGINAL_COMPOUND_RATIO
                    events = standardize_event_string_bos_eos(events, compound_ratio)

                # Get target bar numbers
                target_bars = []
                if "target_bar_numbers_w_offset" in row:
                    target_bars = [
                        int(x) for x in row["target_bar_numbers_w_offset"].split()
                    ]

                # Detect key changes
                key_changes = detect_key_changes(
                    events,
                    compound_ratio,
                    key_values,
                    target_bar_masks=target_bar_masks,
                    target_bars_only=target_bars_only,
                )

                # Format the response
                response = format_key_detection_response(key_changes)

                # Calculate note counts for each key region (for note-level accuracy)
                note_count = 0
                for start_bar, start_pos, end_bar, end_pos, key in key_changes:
                    note_count += count_notes_in_range(
                        events, compound_ratio, start_bar, start_pos, end_bar, end_pos
                    )
                    # note_counts.append({
                    #     "start_bar": start_bar,
                    #     "start_pos": start_pos,
                    #     "end_bar": end_bar,
                    #     "end_pos": end_pos,
                    #     "key": key,
                    #     "note_count": note_count
                    # })

                # Create Alpaca format sample
                bar_str = (
                    ", ".join([str(bar) for bar in target_bars])
                    if target_bars
                    else "all"
                )
                score_id = "/".join(row["csv_path"].split("/")[-2:])
                file_path = "/".join(str(csv_path).split("/")[-4:])

                alpaca_sample = {
                    # Required Alpaca fields
                    "instruction": f"{target_bars[0]}-{target_bars[-1]}",
                    "input": "",
                    "output": response,
                    "system": "",
                    "history": [],
                    # Additional fields for your workflow
                    "events": events,  # Raw events for on-the-fly tokenization
                    "spelling": row.get("spelling", ""),
                    "note_count": note_count,
                    "metadata": {
                        "file_path": file_path,
                        "score_id": score_id,
                        "row_idx": idx,
                        "last_target_bar": target_bars[-1] if target_bars else None,
                    },
                }

                samples.append(alpaca_sample)
            else:
                raise ValueError(
                    f"CSV file {csv_path} is missing required columns: 'target_bar_masks', 'events', 'key'."
                )

    return samples


def create_alpaca_annotation_files(
    data_dir: str,
    output_dir: str,
    target_bars_only: bool = False,
    include_spelling: bool = False,
    *,
    templates_path: Path = TEMPLATES_PATH,
    splits=("train", "valid", "test"),
    suffix: str = "",
):
    """Create Alpaca format annotation files from CSV files in data directories.

    Args:
        data_dir: Directory containing train/valid/test CSV files
        output_dir: Directory to save annotation files
        target_bars_only: If True, only detect key changes in target bars
        include_spelling: If True, add spelling tokens to events
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(templates_path).open() as f:
        template = json.load(f)["KEY_SFT"]

    for split in splits:
        split_dir = Path(data_dir) / split
        split_files = list(split_dir.glob("*.csv"))
        split_files = [f for f in split_files if "metadata" not in f.name]

        if not split_files:
            print(f"Warning: No CSV files found in {split_dir}")
            continue

        print(f"Processing {len(split_files)} {split} files...")
        split_samples = []

        for csv_file in tqdm(split_files, desc=f"Processing {split} files"):
            samples = process_csv_file_alpaca(
                csv_file, target_bars_only, include_spelling
            )
            split_samples.extend(samples)

        # Create dictionary format for Alpaca (indexed by string keys)
        split_samples_dict = {}

        # template = {
        #     "instruction": "Identify local key regions within BAR {} only. Specify key regions using bar and position markers.",
        #     "input": "MusicBERT's output embeddings: <music></music>",
        #     "system": "You are a music theory expert analyzing a piece based on MusicBERT's output. Only identify keys within the specified bar. Other bars are context and must not be labeled. Format output exactly as: bar X position Y, bar Z position W: Key",
        #     "spelling": "Note spellings in the input music: {}",
        # }
        split_samples_dict["template"] = template

        for idx, sample in enumerate(split_samples):
            split_samples_dict[f"{idx}"] = sample

        # Save as JSON file
        output_file = output_dir / f"{split}{suffix}.json"
        print(f"Writing {len(split_samples)} samples to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_samples_dict, f, indent=2, ensure_ascii=False)

        print(f"Created {len(split_samples)} {split} samples")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=KEY_SEQUENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=KEY_ANNOS_DIR)
    parser.add_argument("--templates-path", type=Path, default=TEMPLATES_PATH)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    parser.add_argument("--target-bars-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-spelling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suffix", default="", help="Optional filename suffix after the split name; default: train.json, valid.json, test.json.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    create_alpaca_annotation_files(
        args.data_dir,
        args.output_dir,
        args.target_bars_only,
        args.include_spelling,
        templates_path=args.templates_path,
        splits=args.splits,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
