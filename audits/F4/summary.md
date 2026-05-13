# F4-lite Covariate Probe Audit

Run ID: `f4_probe_3seed_20260510`
Verdict: **reject**
Recommendation: keep F4 heads opt-in; do not promote presets

Note: `minimal_safe_bio` and `full_bio` may remain available for reproducibility and
manual experiments, but this audit does not support them as useful recommended presets.

## Scope

- Trains baseline and F4-lite covariate-head variants on a fixed Kang IFNB subset.
- Fits held-out logistic probes for donor, batch, condition, and cell type on `z_shared` and `z_private`.
- Missing technical batch is recorded as skipped rows for the standalone batch-shared variant.
- The combined full_bio probe uses donor heads and adds batch-shared only when a real batch key is provided.

## Output

- Metrics rows: `audits/F4/metrics.csv`
- Probe notebooks: `audits/F4/notebooks/`
- Non-skipped probe rows: `72`

## Probe Gates

- `pass` Minimum seed count:
- `reject` Donor accuracy on z_private improves: delta=-0.004, baseline=0.13013, variant=0.126533
- `reject` Donor accuracy on z_shared decreases: delta=+0.011, baseline=0.125489, variant=0.136406
- `reject` full_bio preserves donor signal in z_private: delta=-0.019, baseline=0.13013, variant=0.111531
- `review` full_bio removes donor signal from z_shared: delta=-0.010, baseline=0.125489, variant=0.115535
- `pass` Condition probe on z_shared is reported: delta=-0.019, baseline=0.494444, variant=0.475
- `pass` full_bio preserves cell-type signal in z_shared: delta=+0.009, baseline=0.179606, variant=0.188542
- `skipped` Batch accuracy on z_shared decreases: no real batch_key was provided
- `reject` Cross-seed CV on probe metrics:

## Limitations

- Reconstruction loss and iLISI/kBET gates are not computed by this probe script.
- Preset promotion still requires those broader Kang audit metrics in addition to these probes.
