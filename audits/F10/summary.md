# F10b CellDISECT Parity Audit

- run_id: `f10b_smoke_20260513`
- verdict: `informational`
- reason: spVIPESmulti rows are available; external CellDISECT rows were skipped.
- outputs are associative predictions and audit metrics only.

| model | split | metric | value | status | notes |
|---|---|---|---:|---|---|
| spVIPESmulti | cd14_mono | counterfactual_pearson | -0.0204906 | ok | ok; target_cell_type=CD14+ Monocytes; condition=ctrl->stim; train_wall_time_sec=0.2762; n_ctrl=19; n_true=19 |
| spVIPESmulti | cd14_mono | delta_pearson | 0.746105 | ok | ok; target_cell_type=CD14+ Monocytes; condition=ctrl->stim; train_wall_time_sec=0.2762; n_ctrl=19; n_true=19 |
| spVIPESmulti | cd14_mono | top_de_cosine | 0.732405 | ok | ok; target_cell_type=CD14+ Monocytes; condition=ctrl->stim; train_wall_time_sec=0.2762; n_ctrl=19; n_true=19 |
| spVIPESmulti | cd14_mono | wasserstein_mean_all | 7.12873 | ok | ok; target_cell_type=CD14+ Monocytes; condition=ctrl->stim; train_wall_time_sec=0.2762; n_ctrl=19; n_true=19 |
| spVIPESmulti | cd14_mono | wasserstein_mean_top | 45.4553 | ok | ok; target_cell_type=CD14+ Monocytes; condition=ctrl->stim; train_wall_time_sec=0.2762; n_ctrl=19; n_true=19 |
| CellDISECT | cd14_mono | counterfactual_pearson |  | skipped | external install unavailable (celldisect: ModuleNotFoundError: No module named 'celldisect'; CellDISECT: ModuleNotFoundError: No module named 'CellDISECT') |
| CellDISECT | cd14_mono | delta_pearson |  | skipped | external install unavailable (celldisect: ModuleNotFoundError: No module named 'celldisect'; CellDISECT: ModuleNotFoundError: No module named 'CellDISECT') |
| CellDISECT | cd14_mono | top_de_cosine |  | skipped | external install unavailable (celldisect: ModuleNotFoundError: No module named 'celldisect'; CellDISECT: ModuleNotFoundError: No module named 'CellDISECT') |
| CellDISECT | cd14_mono | wasserstein_mean_all |  | skipped | external install unavailable (celldisect: ModuleNotFoundError: No module named 'celldisect'; CellDISECT: ModuleNotFoundError: No module named 'CellDISECT') |
| CellDISECT | cd14_mono | wasserstein_mean_top |  | skipped | external install unavailable (celldisect: ModuleNotFoundError: No module named 'celldisect'; CellDISECT: ModuleNotFoundError: No module named 'CellDISECT') |
