# Legacy experiments

This package retains historical experiments that are outside the paper's
standard RNGen training pipeline.

| File | Purpose |
| --- | --- |
| `spell_warmup.py` | Standalone spelled-pitch warm-up training entrypoint. |
| `spell_warmup_data.py` | Warm-up dataset, collator, spelling labels, and token helpers. |
| `spelled_pitch_warmup.py` | Encoder-only spelling classifier and encoder export. |
| `metrics_dummy_test.py` | Historical simultaneous-note task metrics; not registered as an active SFT task. |

Run the warm-up entrypoint from the repository root:

```bash
python -m rngen.legacy.spell_warmup --help
```

Import legacy helpers directly from their modules. The regular `rngen.data`
and `rngen.modeling` packages do not export them.

Warm-up reuses the maintained encoder implementation in
`rngen/modeling/encoder.py` and the event vocabulary in
`rngen/data/events_incl_spelling_stoi.json`. These shared resources remain part
of the standard SFT pipeline.
