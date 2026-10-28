"""RNGen package bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MUSICBERT_HF = _REPO_ROOT / "musicbert_hf"

if _LOCAL_MUSICBERT_HF.exists():
    local_musicbert_hf_str = str(_LOCAL_MUSICBERT_HF)
    if local_musicbert_hf_str not in sys.path:
        sys.path.insert(0, local_musicbert_hf_str)
