"""Locate released split JSONs, with support for older annotation filenames."""

from pathlib import Path


def resolve_annotation_json_path(jsonl_dir, split, use_spelled_musicbert):
    """Prefer split.json; fall back to the matching historical filename."""
    canonical = Path(jsonl_dir) / f"{split}.json"
    if canonical.is_file():
        return str(canonical)

    suffix = "_target_only_with_spelling" if use_spelled_musicbert else "_target_only"
    legacy = Path(jsonl_dir) / f"{split}{suffix}.json"
    return str(legacy if legacy.is_file() else canonical)
