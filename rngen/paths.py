"""Shared path defaults for repository tools; CLI paths take precedence.

Set the RNGEN_*_ROOT environment variables before starting a process, or edit
the defaults here. Importing this module does not create any directories.
"""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _root(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


DATA_ROOT = _root("RNGEN_DATA_ROOT", PROJECT_ROOT / "data")
OUTPUT_ROOT = _root("RNGEN_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
ANNOS_ROOT = _root("RNGEN_ANNOS_ROOT", DATA_ROOT / "annotations")

# Match the Hugging Face download layout documented in the root README.
RELEASE_DATA_DIR = PROJECT_ROOT / "datasets" / "releases" / "rngen"
PROCESSED_SCORES_DIR = RELEASE_DATA_DIR / "processed_scores"


def baseline_source_roots(source_roots=(), cache_report=None):
    """Protect the published scores and the actual sources of a local cache."""
    roots = [PROCESSED_SCORES_DIR, *(source_roots or ())]
    if cache_report is not None:
        cache_report = Path(cache_report)
        report = json.loads(cache_report.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"Cache report lacks source CSV roots: {cache_report}")
        if isinstance(report.get("source_roots"), dict):
            fields = report["source_roots"]
            keys = ("base", "augmented")
        else:
            fields = report.get("dataset", {})
            keys = ("base_root", "augmented_root")
        if not isinstance(fields, dict) or any(
            not isinstance(fields.get(key), str) or not fields[key].strip()
            for key in keys
        ):
            raise ValueError(f"Cache report lacks source CSV roots: {cache_report}")
        roots.extend(Path(fields[key]) for key in keys)
    return tuple(dict.fromkeys(Path(root).expanduser().resolve() for root in roots))

KEY_SEQUENCE_DIR = (
    DATA_ROOT / "rngen_rna_sft_DcmlCorpora_key_augmented" / "phrase_max500" / "data"
)
KEY_ANNOS_DIR = ANNOS_ROOT / "key"
# Key and RNA annotations consume the same segmented, feature-rich CSVs.
RNA_SEQUENCE_DIR = KEY_SEQUENCE_DIR
RNA_ANNOS_DIR = ANNOS_ROOT / "rna"
TEMPLATES_PATH = PROJECT_ROOT / "rngen" / "config" / "templates.json"
