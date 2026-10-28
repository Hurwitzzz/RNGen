"""Task-to-metrics wiring for the active RNGen training entrypoints."""

from functools import partial

from rngen.training.metrics_key_generation import create_compute_key_metrics_fn
from rngen.training.metrics_rna import create_compute_rna_metrics_fn
from rngen.training.metrics_rna_curriculum_stage2 import (
    create_compute_rna_curriculum_stage2_metrics_fn,
)

METRICS_BUILDERS = {
    "key_sft": partial(create_compute_key_metrics_fn, format_type="region"),
    "rna_sft": create_compute_rna_metrics_fn,
    "rna_curriculum_stage2_sft": create_compute_rna_curriculum_stage2_metrics_fn,
}


def build_compute_metrics_fn(task, tokenizer, compound_ratio, output_dir):
    """Create the compute_metrics callable for a known training task."""
    if task in METRICS_BUILDERS:
        builder = METRICS_BUILDERS[task]
        return builder(
            tokenizer=tokenizer,
            compound_ratio=compound_ratio,
            output_dir=output_dir,
        )

    raise ValueError(f"Unsupported task for metrics: {task}")
