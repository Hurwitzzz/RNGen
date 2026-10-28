# Dataset splits

- `p_dcml/`: P.DCML train, validation, and test manifests used by the controlled
  baselines. `augmented_splits/` contains the corresponding augmented-data
  manifests. The exact configuration used for augmentation is recorded in the
  corresponding configuration file included in the released dataset.
- `rnbert/`: original RNBERT train, validation, and test manifests.

Each manifest lists paths relative to its source dataset root. Reorganizing
these directories does not change the listed paths, split membership, or order.
