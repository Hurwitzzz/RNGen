"""Shared event-range helpers for note and onset counting across tasks."""

import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def identify_notes_and_onsets_in_range(
    events_str: str,
    compound_ratio: int,
    start_bar: int,
    start_pos: int,
    end_bar: int,
    end_pos: int,
    left_close: bool = True,
    right_close: bool = True,
) -> Tuple[List[int], Set[Tuple[int, int]]]:
    """Return note indices and unique onsets that fall within a bar/pos range."""
    if not events_str or events_str.strip() == "":
        return [], set()

    tokens = events_str.split()
    note_indices: List[int] = []
    onsets: Set[Tuple[int, int]] = set()

    for i in range(0, len(tokens), compound_ratio):
        if i + compound_ratio > len(tokens):
            break

        bar, pos, is_valid = parse_note_from_event_token(tokens, i)

        if not is_valid:
            continue

        if left_close and right_close:
            if (bar > start_bar or (bar == start_bar and pos >= start_pos)) and (
                bar < end_bar or (bar == end_bar and pos <= end_pos)
            ):
                note_indices.append(i)
                onsets.add((bar, pos))
        elif left_close and not right_close:
            if (bar > start_bar or (bar == start_bar and pos >= start_pos)) and (
                bar < end_bar or (bar == end_bar and pos < end_pos)
            ):
                note_indices.append(i)
                onsets.add((bar, pos))
        elif not left_close and right_close:
            raise NotImplementedError("We should not use this case in RNA")

    return note_indices, onsets


def count_notes_and_onsets_in_bar_position_range(
    events_str: str,
    compound_ratio: int,
    start_bar: int,
    start_pos: int,
    end_bar: int,
    end_pos: int,
) -> Tuple[int, int]:
    """Count notes and distinct onsets within an inclusive bar/pos range."""
    if not events_str or events_str.strip() == "":
        return 0, 0

    tokens = events_str.split()
    note_count = 0
    onsets = set()

    for i in range(0, len(tokens), compound_ratio):
        if i + compound_ratio > len(tokens):
            break

        bar, pos, is_valid = parse_note_from_event_token(tokens, i)

        if not is_valid:
            continue

        if (bar > start_bar or (bar == start_bar and pos >= start_pos)) and (
            bar < end_bar or (bar == end_bar and pos <= end_pos)
        ):
            note_count += 1
            onsets.add((bar, pos))

    return note_count, len(onsets)


def parse_note_from_event_token(
    event_tokens: List[str], start_idx: int
) -> Tuple[int, int, bool]:
    """Parse the bar and position fields from one compound event tuple."""
    if start_idx >= len(event_tokens):
        return 0, 0, False

    bar_token = event_tokens[start_idx]
    pos_token = event_tokens[start_idx + 1] if start_idx + 1 < len(event_tokens) else ""

    bar_match = re.match(r"<0-(\d+)>", bar_token)
    pos_match = re.match(r"<1-(\d+)>", pos_token)

    if bar_match and pos_match:
        bar = int(bar_match.group(1))
        pos = int(pos_match.group(1))
        return bar, pos, True
    if bar_token in ["<s>", "</s>"]:
        return 0, 0, False

    logger.warning(
        "Failed to parse note from tokens: %s",
        event_tokens[start_idx : start_idx + 2],
    )
    return 0, 0, False
