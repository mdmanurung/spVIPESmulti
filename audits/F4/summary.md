# F4-lite Covariate Probe Audit

Run ID: `f4_probe_smoke_20260510`
Verdict: **informational**

## Scope

- Trains baseline and F4-lite covariate-head variants on a fixed Kang IFNB subset.
- Fits held-out logistic probes for donor, batch, condition, and cell type on `z_shared` and `z_private`.
- Missing technical batch is recorded as skipped rows for the standalone batch-shared variant.
- The combined full_bio probe uses donor heads and adds batch-shared only when a real batch key is provided.

## Output

- Metrics rows: `audits/F4/metrics.csv`
- Non-skipped probe rows: `18`
