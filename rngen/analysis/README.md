# Result analysis

This directory contains tools for inspecting and scoring existing RNGen and
RNBERT results. Baseline implementations, training, cache preparation, and
baseline evaluation are documented in [../baselines/](../baselines/README.md).
Annotation generators are in [../data_prep/](../data_prep/README.md).

| Directory | Contents |
| --- | --- |
| [rngen/](rngen/) | Saved-prediction scoring and encoder comparison |
| [rnbert/](rnbert/) | RNBERT label-space coverage analysis |
| [reports/](reports/) | Optional analysis of existing results |

[rngen/onset_metric_report.py](rngen/onset_metric_report.py) scores one saved
Key or RNA prediction JSON file. Annotation samples must include an integer
`metadata.last_target_bar`, as written by the annotation generator and read by
the training dataset. Missing target boundaries are reported as errors.

[reports/rna_classwise_distribution_report.py](reports/rna_classwise_distribution_report.py)
combines an existing classwise report with training-label frequencies. It is
not required for training, checkpoint selection, or the primary scores.

```bash
python -m rngen.analysis.rngen.onset_metric_report --help
python -m rngen.analysis.rngen.compare_musicbert_variants --help
python -m rngen.analysis.rnbert.rnbert_expanded_coverage_report --help
python -m rngen.analysis.reports.rna_classwise_distribution_report --help
```

Supply dataset, checkpoint, and output locations as command-line
arguments, and keep generated reports outside the source directories.
