# Disentanglement Validation Results

**Dataset:** `scvi.data.spleen_lymph_cite_seq` (RNA only)  
**Groups (donor):** SLN111 vs SLN208  
**Labels (cell_types):** 26 types kept (>=25 cells per donor)  
**Cells per group:** 1000  HVG: 2000  
**Architecture:** n_hidden=64, n_shared=15, n_private=8  
**Training:** max_epochs=40, batch=256, train_size=0.85, kl_warmup=20  

## z_shared metrics (target: high group-mixing AND high label preservation)

| label                                       |   group_mixing_kbet |   group_mixing_ilisi |   label_purity_knn |   label_clisi |   label_ari |
|:--------------------------------------------|--------------------:|---------------------:|-------------------:|--------------:|------------:|
| preset=off                                  |              0.9269 |               1.9119 |             0.3488 |        3.3103 |      0.2718 |
| preset=shared_only                          |              0.9158 |               1.8966 |             0.3536 |        3.2053 |      0.346  |
| preset=private_only                         |              0.9376 |               1.9195 |             0.3463 |        3.2861 |      0.2487 |
| preset=adversarial_only                     |              0.9217 |               1.901  |             0.3439 |        3.3313 |      0.3597 |
| preset=supervised_only                      |              0.9144 |               1.8932 |             0.3475 |        3.3367 |      0.3653 |
| preset=no_contrastive                       |              0.9364 |               1.9187 |             0.3463 |        3.2576 |      0.2849 |
| preset=full                                 |              0.9266 |               1.9072 |             0.3615 |        3.4128 |      0.3097 |
| full minus disentangle_group_shared_weight  |              0.9314 |               1.9126 |             0.3432 |        3.4612 |      0.294  |
| full minus disentangle_label_shared_weight  |              0.9292 |               1.9092 |             0.3359 |        3.3468 |      0.234  |
| full minus disentangle_group_private_weight |              0.934  |               1.9144 |             0.3413 |        3.3696 |      0.3355 |
| full minus disentangle_label_private_weight |              0.9376 |               1.9204 |             0.3483 |        3.226  |      0.2568 |
| full minus contrastive_weight               |              0.9364 |               1.9187 |             0.3463 |        3.2576 |      0.2849 |

- `group_mixing_kbet` — exp(-mean chi^2). 1.0 = perfect mixing.
- `group_mixing_ilisi` — inverse Simpson on group, k-NN. Closer to n_groups = better.
- `label_purity_knn` — fraction of k-NN with same cell-type label.
- `label_clisi` — inverse Simpson on label, k-NN. *Lower* = better.
- `label_ari` — Leiden(z_shared) vs cell_types ARI.

## z_private metrics (target: high group separability, low label retention)

| label                                       |   group_silhouette |   label_purity_knn |
|:--------------------------------------------|-------------------:|-------------------:|
| preset=off                                  |             0.0321 |             0.288  |
| preset=shared_only                          |             0.0278 |             0.2761 |
| preset=private_only                         |             0.0231 |             0.2713 |
| preset=adversarial_only                     |             0.031  |             0.2784 |
| preset=supervised_only                      |             0.0234 |             0.2862 |
| preset=no_contrastive                       |             0.027  |             0.2566 |
| preset=full                                 |             0.0315 |             0.27   |
| full minus disentangle_group_shared_weight  |             0.0419 |             0.2903 |
| full minus disentangle_label_shared_weight  |             0.0236 |             0.2772 |
| full minus disentangle_group_private_weight |             0.0333 |             0.2744 |
| full minus disentangle_label_private_weight |             0.0274 |             0.2914 |
| full minus contrastive_weight               |             0.027  |             0.2566 |

## Training summary (no divergence, recon not collapsed)

| label                                       |   recon_train_final |   recon_train_drop |   elbo_train_final |   kl_train_final |   recon_train_finite |   train_secs |
|:--------------------------------------------|--------------------:|-------------------:|-------------------:|-----------------:|---------------------:|-------------:|
| preset=off                                  |             3332.06 |            611.89  |            3368.97 |          36.9052 |                    1 |         16.5 |
| preset=shared_only                          |             3304.69 |            655.229 |            3342.5  |          37.8179 |                    1 |         16.8 |
| preset=private_only                         |             3325.33 |            634.68  |            3361.19 |          35.8641 |                    1 |         16.6 |
| preset=adversarial_only                     |             3319.44 |            647.668 |            3355.06 |          35.6163 |                    1 |         31.5 |
| preset=supervised_only                      |             3315.98 |            649.852 |            3352.26 |          36.2798 |                    1 |         18.7 |
| preset=no_contrastive                       |             3325.17 |            622.947 |            3359.91 |          34.7433 |                    1 |         19.7 |
| preset=full                                 |             3316.16 |            632.824 |            3352.13 |          35.9723 |                    1 |         19.2 |
| full minus disentangle_group_shared_weight  |             3284.69 |            682.827 |            3320.39 |          35.7    |                    1 |         18.8 |
| full minus disentangle_label_shared_weight  |             3300.24 |            656.977 |            3337.08 |          36.84   |                    1 |         18.6 |
| full minus disentangle_group_private_weight |             3310.46 |            634.632 |            3348.44 |          37.9728 |                    1 |         18.7 |
| full minus disentangle_label_private_weight |             3290.58 |            663.523 |            3327.21 |          36.6304 |                    1 |         18.5 |
| full minus contrastive_weight               |             3325.17 |            622.947 |            3359.91 |          34.7433 |                    1 |         19   |

- `recon_train_final` is comparable across rows because the seed-fixed train/val split is identical for every preset.
- `recon_train_drop = recon[0] - recon[-1]`; positive = the loss decreased.
- spVIPESmulti' TrainingPlan does not currently log validation_step metrics, so a true held-out NLL is not reported here. Adding `validation_step` to the plan is a straightforward follow-up.

## Verdict

Compare each preset / ablation row to `off`:

- A **healthy** disentanglement objective should *increase* `shared__group_mixing_*` and `shared__label_ari` while not catastrophically degrading `recon_val_final`.
- Removing `disentangle_group_shared_weight` (`full minus q_group_shared`) should be the largest hit to group mixing.
- Removing `disentangle_label_shared_weight` or `contrastive_weight` should be the largest hit to ARI / k-NN purity.
- `private_only` should leave shared metrics ~unchanged from `off`.
