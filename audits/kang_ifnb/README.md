# Kang IFNB benchmark audit

This folder is the append-only home for Kang IFNB experiment records and serves
as the general benchmark lane for feature evaluation.

CellDISECT-parity-specific outputs for F10 can be mirrored under `audits/F10/`
while keeping aggregated experiment tracking in this folder.

Use this benchmark to compare every new spVIPESmulti feature against three anchors:

- current spVIPESmulti implementation
- original spVIPES from <https://github.com/nrclaudio/spVIPES>
- contrastiveVAE from scvi-tools
  <https://github.com/scverse/scvi-tools/blob/612157b04320cf13b72e3e500707371b05811f54/src/scvi/external/contrastivevi/_model.py#L49>

## Required comparison settings

- Same Kang IFNB preprocessing path for every model.
- Same train/validation split and random seeds across all models.
- Remove megakaryocytes before evaluation.
- Report mean and standard deviation across at least 3 seeds.

## Runner

Use the append-only benchmark runner:

```bash
python scripts/benchmark_kang_ifnb.py \
  --run-id f1_overhead_20260510 \
  --feature-id F1 \
  --seeds 0,1,2 \
  --models spvipesmulti,spvipes_original,contrastivevae
```

Notes:

- The script always appends rows to `metrics.csv`.
- If original `spVIPES` or `contrastiveVAE` is unavailable in the environment,
  the script still appends a row with `notes` describing the missing baseline.
- Baselines can be scoped for a quick dry run with `--models spvipesmulti`.
- If pertpy cannot download Kang in your environment, use a local dataset copy:

```bash
python scripts/benchmark_kang_ifnb.py \
  --run-id f1_overhead_20260510 \
  --feature-id F1 \
  --seeds 0,1,2 \
  --models spvipesmulti,spvipes_original,contrastivevae \
  --kang-h5ad-path /absolute/path/to/kang_2018.h5ad
```

## Metric schema

Write one row per run to `metrics.csv` with at least these fields:

- `run_id`
- `timestamp`
- `feature_id`
- `model_name`
- `baseline_name`
- `seed`
- `subset`
- `n_cells`
- `n_genes`
- `train_wall_time_sec`
- `iLISI`
- `cLISI`
- `kBET`
- `knn_purity`
- `leiden_ari`
- `silhouette_group`
- `silhouette_label`
- `reconstruction_loss_per_cell`
- `kl_shared`
- `kl_private`
- `orthogonality_within_stratum`
- `orthogonality_worst_stratum`
- `orthogonality_excluded_strata`
- `cycle_consistency_l2`
- `target_decoder_realism`
- `identity_preservation`
- `notes`

For disentanglement-focused experiments, the main comparison signal is the trade-off between lower orthogonality / leakage and preserved or improved integration metrics.

## Baseline interpretation

Prefer a run when it improves over the original spVIPES and contrastiveVAE baselines on the same metric family, not just relative to the current spVIPESmulti model.
