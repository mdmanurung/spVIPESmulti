# ImplementationPlan.md

Purpose: medium-horizon roadmap specs for future candidates only.

Related docs:
- PLAN.md: active execution queue and deferred backlog (stubs with pointers here).
- PROGRESS.md: dated execution history.
- HANDOFF.md: next-session bootstrap.

---

## Deferred Backlog Specs

### §P-PERF-1. Vectorize `_label_based_poe` reassembly (GPU-CPU sync bottleneck)

Priority: HIGH | Source: Third-pass code audit of `spVIPESmultimodule.py`

**What is wrong:**
In `_label_based_poe` (`src/spVIPESmulti/module/spVIPESmultimodule.py`), the reassembly
block iterates over every cell in a Python loop:

```python
# ~16,384 GPU-CPU syncs per step at batch_size=2048, 8 groups
for i, label_tensor in enumerate(per_group_labels[g]):
    label = label_tensor.item()          # GPU→CPU sync per cell
    for k in stat_keys:
        group_output[k][i] = poe_g_stats[k][tensor_index]
```

**The fix (vectorized, no semantic change):**
```python
for label in all_labels:
    mask = (per_group_labels[g] == label)
    if not mask.any():
        continue
    poe_g = poe_stats_per_label[label][g]
    n_rows = poe_g["logtheta_loc"].shape[0]
    idx = torch.arange(mask.sum(), device=ref_device) % n_rows
    for k in stat_keys:
        group_output[k][mask] = poe_g[k][idx]
```
Cost: O(n_labels) ≈ 10 iterations instead of O(batch × n_groups) ≈ 16K.

**Scope:**
- Edit only the reassembly block in `_label_based_poe` (~15 lines).
- Add regression test: forward pass with `use_labels=True` output numerically identical.
- Validate: `pytest tests/test_multigroup_multimodal.py tests/test_regression_fixes.py`.
- Smoke: `python scripts/smoke_vignettes.py --epochs 5`.

**Success metric:** No `.item()` in PoE hot path (verify with `torch.profiler`).

---

### §P-PERF-2. Low-rank mixer in `LinearDecoderSPVIPE`

Priority: MEDIUM | Source: Third-pass code audit of `src/spVIPESmulti/nn/networks.py`

**What is expensive:**
`LinearDecoderSPVIPE` builds a 296×1000 `mixture` layer (296K params × 8 decoders = 2.37M extra params, 4 matmuls/decoder × 8 decoders = 32 matmuls/step).

**The fix (rank-4 low-rank gate, 296K → 4K params):**
```python
self.mix_down = nn.Linear(n_input_shared + n_input_private, 4)
self.mix_up   = nn.Linear(4, n_output)
# forward:
px_mixing = self.mix_up(F.relu(self.mix_down(z_private_shared)))
```

**Scope:**
- Add `use_low_rank_mixer: bool = False` to `LinearDecoderSPVIPE.__init__` (backward-compatible).
- Run `scripts/validate_disentanglement.py` and compare latent metrics before/after.
- Ablation: 100-epoch B-cell training with default vs. low-rank mixer; compare reconstruction loss and k-NN purity.

---

### §P-PERF-3. `torch.compile` (blocked on P-PERF-1)

Priority: LOW-MEDIUM | Source: Third-pass code audit

**The fix:**
```python
model_spv.module = torch.compile(model_spv.module)
```
**Caveats:** ~30s Triton compilation on first forward pass. Compatible with `"bf16-mixed"`. Blocked until `.item()` loop removed (P-PERF-1).

---

### §P-PERF-4. SiLU activation in encoder

Priority: LOW | Source: Third-pass code audit of `src/spVIPESmulti/nn/networks.py`

**The fix:**
```python
# In Encoder.__init__:
self.relu = nn.SiLU()   # was nn.ReLU()
```
SiLU/Swish reduces epochs to target loss in deep VAEs with no architectural change.

---

### §N5-D. Fix adversarial overreach on z_private

Priority: MEDIUM | Reactivation trigger: after Phase 3 retrain (v4 model)

**Root cause:** `disentangle_label_private_weight=0.5` GRL erases label info from z_private.
Because cell type and antigen group are correlated (Atypical 76% CRXV, Classical 69% CRXV),
the GRL also strips group structure, depressing private silhouette to 0.086.

**Fix:** Reduce `disentangle_label_private_weight` to 0.0–0.1. Validate silhouette increases without harming shared-space iLISI/kBET.

---

### §N5-E. Class-weighted CE for minority cell types

Priority: MEDIUM | Reactivation trigger: after pilot results confirm direction

**Root cause:** All 4 `F.cross_entropy` calls in `module/spVIPESmultimodule.py` use uniform weights. Minority types (Activated MZ n≈200, Transitional n≈300) are dominated by Atypical (n=3155).

**Fix:**
- Compute per-class inverse-frequency tensors at module init.
- Store as `nn.Module` buffers.
- Thread `weight=` into each CE call.

---

### §P6. Multi-covariate generalization

Priority: LOW | Source: CellDISECT (Megas et al., 2025)

**Scope:** Promote `groups_key` to multi-key design; nested covariate metadata in `adata.uns`. Broad refactor across data/model/loss.

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

- 2026-05-08: Removed completed C1/C2 entries (see PROGRESS.md). Added P-PERF-1–4 specs from third-pass code audit. Added N5-D and N5-E specs from baseline analysis. P6 retained.
