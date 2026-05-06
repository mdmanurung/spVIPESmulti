# ImplementationPlan.md

Purpose: medium-horizon roadmap specs for future candidates only.

Related docs:
- PLAN.md: active execution queue and deferred backlog.
- PROGRESS.md: dated execution history.
- HANDOFF.md: next-session bootstrap.

---

## Active Roadmap Candidates

### C1: CosineAnnealingLR support in SpVIPESmultiTrainingPlan — COMPLETE (2026-05-06)

Size: S  
Source: convergence audit of `malaria_bcells_recommended.ipynb` (2026-05-06)

**Problem:** `ReduceLROnPlateau` with `lr_patience=15` and `check_val_every_n_epoch=5` means the LR is only checked every 75 epochs. Because `reconstruction_loss_validation` was still declining throughout all 700 epochs, the scheduler never fired — the model trained at a flat `lr=5e-4` for its entire run. A scheduled decay is needed.

**Solution:** Add `lr_scheduler_type: Literal["plateau", "cosine"]` kwarg to `SpVIPESmultiTrainingPlan`. When `"cosine"`, configure `CosineAnnealingLR(optimizer, T_max=lr_cosine_T_max, eta_min=lr_min)` instead of `ReduceLROnPlateau`. Pass `max_epochs` from `MultiGroupTrainingMixin.train()` to the training plan as `lr_cosine_T_max` default.

**Locked decisions:**
- Default remains `"plateau"` for backward compatibility.
- `lr_cosine_T_max` defaults to `max_epochs` when not specified.
- `lr_min` reuses existing `plan_kwargs["lr_min"]` (default 1e-5).
- No change to early stopping or validation plumbing.

**Files:**
- `src/spVIPESmulti/model/base/training_mixin.py` — `SpVIPESmultiTrainingPlan.__init__` and `configure_optimizers()`; `MultiGroupTrainingMixin.train()` to thread `max_epochs` into `plan_kwargs` when `lr_scheduler_type="cosine"`.

**Verification:**
```bash
pytest tests/test_lightning_trainer_compat.py -q          # scheduler wiring
python scripts/smoke_vignettes.py --epochs 5 --cells_per_group 300  # smoke
```
Add one test: `train(..., plan_kwargs={"lr_scheduler_type": "cosine"})` completes and LR at last epoch < initial LR.

**Stop criteria:** test passes, smoke passes, no regression in `pytest -q`.

---

### C2: Notebook hyperparameter retuning (malaria_bcells_recommended v3) — COMPLETE (2026-05-06, pending notebook run)

Size: XS  
Source: convergence audit (2026-05-06)

**Problem:** Current `malaria_bcells_recommended.ipynb` (v2) uses `n_hidden=128`, `disentangle_label_shared_weight=5.0`, `batch_size=512`, `max_epochs=700` and did not converge — hit `max_epochs` with reconstruction loss still declining.

**Root causes identified:**
1. `n_hidden=128` compresses 3000 genes through 128 units (the encoder's `fc1`/`fc2` layers). The actual `networks.py` default is 256.
2. `disentangle_label_shared_weight=5.0` creates gradient competition in early epochs. Label-shared curves show it converged by ~epoch 400, so 5× pressure was unnecessary.
3. LR was flat at 5e-4 for all 700 epochs (ReduceLROnPlateau never fired).
4. `batch_size=512` on an L40S leaves VRAM headroom; 1024 reduces gradient noise.

**Changes (notebook constants block only):**

| Constant | v2 | v3 | Rationale |
|---|---|---|---|
| `N_HIDDEN` | 128 | 256 | Match `networks.py` default; wider fc1/fc2 |
| `LABEL_SHARED_W` | 5.0 | 2.0 | Reduce early-epoch gradient competition |
| `BATCH_SIZE` | 512 | 1024 | Lower gradient noise on L40S |
| `MAX_EPOCHS` | 700 | 400 | With fixes above, expect convergence ~epoch 200–350 |
| `lr_scheduler_type` | (plateau) | `"cosine"` | Requires C1 first; scheduled decay throughout training |

Also add `lr_scheduler_type="cosine"` to `plan_kwargs` once C1 is merged.

**Depends on:** C1 (for cosine scheduler); the n_hidden/weight/batch changes are independent.

**Verification:**
- Run notebook to completion; confirm `Trainer.fit stopped: early_stopping` fires (not `max_epochs reached`).
- Confirm final `reconstruction_loss_validation` lower than v2 final value.
- Save model as `results/spvipes_bcells_recommended_v3`.

---

## Prioritization Rules

- Prefer S/M items that reduce repeated notebook boilerplate.
- Only one M/L feature should be active at a time.
- Any new public API must include tests and one example update.

## Verification Baseline

- `pytest -v`
- `python scripts/smoke_vignettes.py`
- `python scripts/validate_disentanglement.py` for disentanglement-adjacent changes

## Last Updated

- 2026-05-06: C1 (CosineAnnealingLR) and C2 (notebook v3 retuning) implemented. C2 pending notebook execution.
- 2026-05-05: R3 (MrVI DA) and R4 (public evaluation API) removed — both complete. Implementation history in PROGRESS.md.
