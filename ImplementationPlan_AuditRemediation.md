# ImplementationPlan_AuditRemediation.md

**Date:** 2026-05-08
**Owner:** spVIPESmulti maintainers
**Source audits:**
- [audits/2026-05-08-full-package.md](audits/2026-05-08-full-package.md) — primary (referenced as `P§…`)
- [audits/2026-05-08-full-package-2.md](audits/2026-05-08-full-package-2.md) — secondary (referenced as `S§…`)

**Status:** plan only; no production source modified in this pass. Red-stub
regression tests live under [tests/audit_regressions/](tests/audit_regressions/) and
are marked `xfail(strict=True)` so the suite stays green until each fix lands.

---

## 0. How to use this plan

1. Pick the next un-checked work-item from §3 (they are dependency-ordered).
2. Open the corresponding red stub in `tests/audit_regressions/` and convert
   it from `xfail` to a real failing test (the stub already pins the file
   surface and the assertion shape).
3. Implement the minimal fix described in the card.
4. Run `pytest tests/audit_regressions/ -m audit_regression` and confirm the
   target test now passes; flip its xfail off.
5. Run the full suite (`pytest`) and the relevant verification check from
   §5 of the primary audit (cited per card).
6. Tick the card and move to the next.

**Hard rules:**
- No production-source change is merged before its red regression test
  exists in `tests/audit_regressions/`.
- No regression test is removed or weakened to make it pass.
- Q-### items (§7) require user/PI sign-off before any related code change.

---

## 1. Unified findings table

Severity legend: **C** = Critical, **H** = High, **M** = Medium, **A** =
Assumption / domain question.

Disagreements between the two audits are flagged in the *Notes* column.

| ID | Source | Sev | Surface | One-line claim | Verification |
|---|---|---|---|---|---|
| F-NB-LOG1P | P§2.1 + S§3 | C | [src/spVIPESmulti/module/spVIPESmultimodule.py#L1259-L1287](src/spVIPESmulti/module/spVIPESmultimodule.py#L1259-L1287), [src/spVIPESmulti/module/spVIPESmultimodule.py#L1351-L1399](src/spVIPESmulti/module/spVIPESmultimodule.py#L1351-L1399) | NB `log_prob` evaluated at `log1p(x)` targets, outside support | V2.1 |
| F-LIB-LOG1P | S§2.1 | C | [src/spVIPESmulti/module/spVIPESmultimodule.py#L524-L535](src/spVIPESmulti/module/spVIPESmultimodule.py#L524-L535) | Single-modal `library` computed from `log1p` instead of raw counts; multimodal computes correctly | V2.1.lib (S§5) |
| F-POE-ROWWISE | P§2.2 + S§3 | C | [src/spVIPESmulti/module/spVIPESmultimodule.py#L302-L440](src/spVIPESmulti/module/spVIPESmultimodule.py#L302-L440), [src/spVIPESmulti/module/spVIPESmultimodule.py#L614-L622](src/spVIPESmulti/module/spVIPESmultimodule.py#L614-L622) | Across-group PoE pairs cells by minibatch row index | V2.2 |
| F-GAUSSIAN-SIMPLEX | P§2.3 + S§3 | C | [src/spVIPESmulti/nn/networks.py#L357-L365](src/spVIPESmulti/nn/networks.py#L357-L365), [src/spVIPESmulti/module/utils.py#L42-L46](src/spVIPESmulti/module/utils.py#L42-L46) | Gaussian likelihood mean is L1-simplex; cannot represent log-normalised CITE-seq | V2.3 |
| F-DA-NOTEST | P§2.4 + S§3 | C | [src/spVIPESmulti/model/spvipesmulti.py#L786-L867](src/spVIPESmulti/model/spvipesmulti.py#L786-L867) | `differential_abundance` returns a heuristic with no null, no FDR, inverted warning | V2.4 |
| F-GIVE-MEAN | P§2.5 + S§3 | C | [src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432](src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432) | `give_mean=True, normalized=False` returns rsample, not posterior mean | V2.5 |
| F-SILHOUETTE-GLOBAL | P§2.6 + S§3 | H | [src/spVIPESmulti/metrics.py#L194-L228](src/spVIPESmulti/metrics.py#L194-L228), [src/spVIPESmulti/metrics.py#L295-L316](src/spVIPESmulti/metrics.py#L295-L316) | `per_group_silhouette` is global; same scalar replicated across rows | V2.6 |
| F-KBET-LISI | P§2.7 + S§3 | H | [src/spVIPESmulti/metrics.py#L30-L141](src/spVIPESmulti/metrics.py#L30-L141) | `kbet`/`ilisi`/`clisi` are not the published statistics | V2.7 |
| F-JEFFREYS-MEANPOOL | P§3.1 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L601-L622](src/spVIPESmulti/module/spVIPESmultimodule.py#L601-L622) | Jeffreys integ aggregates each batch to one Gaussian by averaging means/vars | V3.1 |
| F-NF-KL-1SAMPLE | P§3.2 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L624-L651](src/spVIPESmulti/module/spVIPESmultimodule.py#L624-L651) | NF-prior KL is 1-sample MC; variance unreported | V3.2 |
| F-NF-PRIOR-GLOBAL | S§2.3 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L295-L305](src/spVIPESmulti/module/spVIPESmultimodule.py#L295-L305) | One global flow shared across groups/modalities despite per-(group, modality) private encoders | V2.3.nf (S§5) |
| F-RECON-POISSON | P§3.3 | M | [src/spVIPESmulti/metrics.py#L494-L512](src/spVIPESmulti/metrics.py#L494-L512) | `reconstruction_error` Poisson NLL drops `log(x!)` and uses `px_rate_shared` only | V3.3 |
| F-RECON-RMSE | P§3.4 | M | [src/spVIPESmulti/metrics.py#L485-L490](src/spVIPESmulti/metrics.py#L485-L490) | RMSE compares simplex to count proportions; scale-invariant by construction | V3.3 |
| F-LATENT-VANISH | P§3.5 | M | [src/spVIPESmulti/metrics.py#L382-L431](src/spVIPESmulti/metrics.py#L382-L431) | `is_vanished` hard-coded at 0.5 std; not a posterior-collapse diagnostic | V3.5 (new) |
| F-BN-EVAL-SMALL | P§3.6 | M | [src/spVIPESmulti/nn/networks.py#L99-L108](src/spVIPESmulti/nn/networks.py#L99-L108), [src/spVIPESmulti/model/spvipesmulti.py#L368-L378](src/spVIPESmulti/model/spvipesmulti.py#L368-L378) | BatchNorm + small final inference batch can give noisy embeddings | V3.6 (new) |
| F-LABEL-DUMMY | P§3.7 (downgraded by S§2.5) | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L826-L879](src/spVIPESmulti/module/spVIPESmultimodule.py#L826-L879) | Single-label-group branch uses `torch.empty`; safe today but fragile (S downgrades P) | V3.7 |
| F-DA-WARN | P§3.8 | M | [src/spVIPESmulti/model/spvipesmulti.py#L786-L797](src/spVIPESmulti/model/spvipesmulti.py#L786-L797) | DA warning fires when alignment off; degeneracy is when alignment is on | (paired with F-DA-NOTEST) |
| F-CE-WEIGHTS | P§3.9 | M | [src/spVIPESmulti/model/spvipesmulti.py#L174-L185](src/spVIPESmulti/model/spvipesmulti.py#L174-L185), [src/spVIPESmulti/module/spVIPESmultimodule.py#L1110-L1166](src/spVIPESmulti/module/spVIPESmultimodule.py#L1110-L1166) | Class-weighted CE with `sum`-then-mean rescale interacts with KL warmup | (paired with F-WARMUP-CE) |
| F-OBS-INDEX | P§3.10 | M | [src/spVIPESmulti/data/prepare_adatas.py#L130-L150](src/spVIPESmulti/data/prepare_adatas.py#L130-L150) | `groups_obs_indices` correctness depends on `pd.Series.values` order assumption | V3.10 (new) |
| F-PROTO-VAL-LEAK | S§2.2 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184](src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184) | Contrastive prototype EMA updated during validation forward passes | V2.2.proto (S§5) |
| F-ENC-POS-LOOKUP | S§2.4 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L550-L558](src/spVIPESmulti/module/spVIPESmultimodule.py#L550-L558) | Per-group encoder lookup is positional; breaks if a minibatch lacks a group | V2.4.idx (S§5) |
| F-WARMUP-CE | S§2.7 | M | [src/spVIPESmulti/module/spVIPESmultimodule.py#L1316-L1321](src/spVIPESmulti/module/spVIPESmultimodule.py#L1316-L1321), [src/spVIPESmulti/module/spVIPESmultimodule.py#L1418-L1422](src/spVIPESmulti/module/spVIPESmultimodule.py#L1418-L1422) | KL warmup also throttles supervised-CE components, not just GRL | V2.7.warmup (S§5) |
| F-BATCH-VS-GROUP | S§2.6 | A | [src/spVIPESmulti/model/spvipesmulti.py#L289-L345](src/spVIPESmulti/model/spvipesmulti.py#L289-L345) | When `batch_key ≡ groups_key`, GRL fights the encoder's own input | V2.6.batch (S§5) |
| Q-LIKELIHOOD-INTENT | P§4.1 | A | doc only | Is Gaussian path meant for log-normalised data, or only `[0,1]`-rescaled? | — |
| Q-POE-INTENT | P§4.2 + S | A | doc only | Is unsupervised PoE a posterior or only a regulariser? | — |
| Q-DA-NULL | P§4.3 | A | doc only | What null distribution should `differential_abundance` quote in print? | — |
| Q-EMBED-DEFAULT | P§4.4 | A | doc only | Should `embed()` return mean or rsample by default? | — |
| Q-RECON-COLS | P§4.5 | A | doc only | Intended interpretation of `reconstruction_error` columns? | — |
| Q-TRAVERSAL | P§4.6 | A | doc only | Is `traverse_latent` interpretable under low-rank mixer? | — |
| Q-DISENT-EQ | P§4.7 | A | doc only | Calibration of GRL adversarial equilibrium under default weights? | — |

---

## 2. Sequencing rationale

Fixes are ordered so that earlier work items do not invalidate the regression
tests of later ones. Concretely:

1. **Posterior-summary semantics first (F-GIVE-MEAN, F-PROTO-VAL-LEAK,
   F-ENC-POS-LOOKUP).** Every downstream metric (DA, integration, recon,
   silhouette) is computed *on* the embedding returned by
   `get_latent_representation`. If that embedding is silently a single
   rsample (F-GIVE-MEAN), every metric test below will be noisy and any
   numerical assertion will be flaky. We pin determinism here first.
2. **Likelihood / target alignment (F-NB-LOG1P, F-LIB-LOG1P,
   F-GAUSSIAN-SIMPLEX).** These change the absolute scale of the ELBO and
   therefore the absolute scale of any `held_out_metrics` numbers. Fixing
   them after evaluation tests would force re-baselining all metric
   thresholds.
3. **PoE pairing semantics (F-POE-ROWWISE, F-LABEL-DUMMY, F-NF-PRIOR-GLOBAL).**
   These change `z_shared` itself; any test that asserts e.g. iLISI/cLISI
   thresholds must be authored against the *post-fix* embedding.
4. **Evaluation layer (F-DA-NOTEST, F-DA-WARN, F-KBET-LISI,
   F-SILHOUETTE-GLOBAL, F-RECON-POISSON, F-RECON-RMSE).** Now that
   embeddings and likelihoods are sound, fix the metrics that consume them.
5. **Numerical robustness (F-JEFFREYS-MEANPOOL, F-NF-KL-1SAMPLE,
   F-LATENT-VANISH, F-BN-EVAL-SMALL, F-CE-WEIGHTS, F-WARMUP-CE,
   F-OBS-INDEX).** Quality-of-life and bias-reduction items that are
   independent of the above.
6. **Documentation / domain Q-### items.** Resolved last, after empirical
   evidence from the fixes is in hand.

---

## 3. TDD work cards

Card template (≤ ~40 lines each; long rationale → §A appendix):

```
### W-### F-XXX  [Sev]  — short title
Owner: <unassigned>          Depends on: <W-IDs or none>
Surface: <file path link>
Red test: tests/audit_regressions/test_<name>.py::<id>
Fix sketch: <≤3 bullets, code-shape only>
Verification check: <V-ref from §5 of primary audit, or "new">
Acceptance:
  - [ ] Red test exists and fails before fix.
  - [ ] Fix is the minimal change satisfying the test.
  - [ ] Verification check passes.
  - [ ] Full pytest suite green.
  - [ ] Doc TODOs (§D) for this W-### resolved or filed.
```

---

### W-001  F-GIVE-MEAN  [C]  — `give_mean=True, normalized=False` returns mean
Depends on: none.
Surface: [src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432](src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432).
Red test: [tests/audit_regressions/test_give_mean.py](tests/audit_regressions/test_give_mean.py)::`test_give_mean_unnormalized_is_deterministic`.
Fix sketch:
- In the un-normalized branch, when `give_mean=True`, append `qz.loc`
  (and the per-group `loc`) instead of `qz.rsample()`.
- Honour `mc_samples` only when `give_mean=False`.
Verification: V2.5.
Acceptance: two consecutive calls with identical args produce
bit-identical embeddings; `mc_samples=k` returns mean over `k` rsamples
when `give_mean=False`.

### W-002  F-PROTO-VAL-LEAK  [M, semantic]  — freeze prototype EMA in eval
Depends on: W-001.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184](src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184).
Red test: `tests/audit_regressions/test_proto_no_val_leak.py`.
Fix sketch:
- Wrap the prototype EMA update in `if self.training:`.
Verification: V2.2.proto (S§5). Property test: `prototypes` buffer
unchanged across `trainer.validate(model)`.

### W-003  F-ENC-POS-LOOKUP  [M]  — encoder lookup by group code, not loop pos
Depends on: W-001.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L550-L558](src/spVIPESmulti/module/spVIPESmultimodule.py#L550-L558).
Red test: `tests/audit_regressions/test_encoder_lookup_by_code.py`.
Fix sketch:
- Replace `for group, (...) in enumerate(zip(x.values(), batch_index))`
  with explicit iteration over the group-code keys of `x`, indexing
  `self.encoders[group_code]` directly.
Verification: V2.4.idx. Property test: latent of group-1-only subset
equals the corresponding rows of full-data inference.

### W-010  F-LIB-LOG1P  [C]  — single-modal library from raw counts
Depends on: W-001.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L524-L535](src/spVIPESmulti/module/spVIPESmultimodule.py#L524-L535).
Red test: `tests/audit_regressions/test_library_from_raw_counts.py`.
Fix sketch:
- Compute `library = log(x_raw.sum(1).clamp(min=1e-6))` *before* the
  `log1p` transform overwrites `x`.
- Mirror the multimodal-path pattern at
  [src/spVIPESmulti/module/spVIPESmultimodule.py#L609-L616](src/spVIPESmulti/module/spVIPESmultimodule.py#L609-L616).
Verification: V2.1.lib. Assert
`library.exp().mean() ≈ x_raw.sum(1).mean()` to within 1%.

### W-011  F-NB-LOG1P  [C]  — NB log_prob on raw counts
Depends on: W-010.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L1259-L1287](src/spVIPESmulti/module/spVIPESmultimodule.py#L1259-L1287),
[src/spVIPESmulti/module/spVIPESmultimodule.py#L1351-L1399](src/spVIPESmulti/module/spVIPESmultimodule.py#L1351-L1399).
Red test: [tests/audit_regressions/test_nb_log1p.py](tests/audit_regressions/test_nb_log1p.py).
Fix sketch:
- Drop the `x_target = log(1+x)` transform on the NB target; pass
  `x_obs` (raw counts) into `px.log_prob`.
- Re-enable the integer-roundness guard in
  `_validate_likelihood_observations` for the NB target.
Verification: V2.1. NB-recovery: simulate `x ~ NB(μ, θ)`, fit, recover
`μ` to within MC noise.

### W-012  F-GAUSSIAN-SIMPLEX  [C]  — un-bound Gaussian mean head
Depends on: W-001.
Surface: [src/spVIPESmulti/nn/networks.py#L357-L365](src/spVIPESmulti/nn/networks.py#L357-L365),
[src/spVIPESmulti/module/utils.py#L42-L46](src/spVIPESmulti/module/utils.py#L42-L46).
Red test: [tests/audit_regressions/test_gaussian_simplex.py](tests/audit_regressions/test_gaussian_simplex.py).
Fix sketch:
- Add an unconstrained linear head for the Gaussian mean (separate from
  the L1-normalised `px_scale` used by NB), or (Q-LIKELIHOOD-INTENT)
  document that protein data must be rescaled to `[0,1]` and validate
  on `setup_anndata`.
Verification: V2.3. Synthetic `y ~ N(μ, 0.5)` with `μ ∈ [-3, 3]` — mean
range covers `y.min()..y.max()`.

### W-020  F-POE-ROWWISE  [C]  — paired-cell semantics for cross-group PoE
Depends on: W-001, W-010, W-011.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L302-L440](src/spVIPESmulti/module/spVIPESmultimodule.py#L302-L440),
[src/spVIPESmulti/module/spVIPESmultimodule.py#L739-L893](src/spVIPESmulti/module/spVIPESmultimodule.py#L739-L893).
Red test: [tests/audit_regressions/test_poe_rowwise.py](tests/audit_regressions/test_poe_rowwise.py).
Fix sketch (decision required — see Q-POE-INTENT):
- Option A (cleanest): drop across-group PoE; the shared latent of cell
  *i* is the encoder posterior of cell *i* alone. Across-group alignment
  remains via the disentangle objective + Jeffreys integration.
- Option B (label-conditional): only PoE-combine cells of the same label
  (already partially the case in `_label_based_poe`); collapse the
  result via expectation, not row-wise product.
- Decision needs Q-POE-INTENT first.
Verification: V2.2. Property test: shuffle minibatch row order in group
B; assert `z_shared` for cells of group A is bit-identical.

### W-021  F-LABEL-DUMMY  [M]  — replace `torch.empty` with `torch.zeros` and assert mask coverage
Depends on: W-020.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L826-L879](src/spVIPESmulti/module/spVIPESmultimodule.py#L826-L879).
Red test: `tests/audit_regressions/test_label_dummy_branch.py`.
Fix sketch:
- Allocate `torch.zeros` for the dummy single-group branch.
- Add `assert mask_union.all()` after reassembly.
Verification: V3.7. Monkeypatch `torch.empty` to return NaN-filled
tensors; loss must remain finite.

### W-022  F-NF-PRIOR-GLOBAL  [M]  — per-(group, modality) flow or documented as global regulariser
Depends on: Q-POE-INTENT (recommended).
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L295-L305](src/spVIPESmulti/module/spVIPESmultimodule.py#L295-L305).
Red test: `tests/audit_regressions/test_nf_prior_per_group.py`.
Fix sketch:
- If per-group: instantiate `nn.ModuleDict` of flows keyed by group (and
  modality in multimodal mode); thread through KL computation.
- If documented-global: emit a `UserWarning` once at construction and
  add a paragraph in the `nf_target` docstring.
Verification: V2.3.nf. Bimodal-private synthetic two-group: with
per-group flows the sample distribution is bimodal; with the global
flow it is unimodal interpolation.

### W-030  F-DA-NOTEST  [C]  — `differential_abundance` returns a calibrated test
Depends on: W-001, W-011, W-020.
Surface: [src/spVIPESmulti/model/spvipesmulti.py#L786-L867](src/spVIPESmulti/model/spvipesmulti.py#L786-L867).
Red test: [tests/audit_regressions/test_da_calibration.py](tests/audit_regressions/test_da_calibration.py).
Fix sketch:
- Return a permutation-null p-value per cell (default `n_permutations=200`)
  alongside the existing score.
- Apply BH FDR control across cells; expose `q_value`.
- Invert the warning per F-DA-WARN: warn when the score is mathematically
  degenerate (best alignment), not when alignment terms are off.
- Decision required: Q-DA-NULL (permutation across what? samples vs cells).
Verification: V2.4. Identical-composition two-group: empirical
distribution of p-values is uniform on `[0, 1]` (KS p > 0.01).

### W-040  F-KBET-LISI  [H]  — replace heuristics with reference statistics
Depends on: W-001, W-020.
Surface: [src/spVIPESmulti/metrics.py#L30-L141](src/spVIPESmulti/metrics.py#L30-L141).
Red test: `tests/audit_regressions/test_kbet_lisi_reference.py`.
Fix sketch:
- `kbet`: return rejection rate at α=0.05 (fraction of cells whose
  neighbourhood χ² test rejects), not `exp(-mean(χ²))`. Optionally vendor
  the original local-χ² formulation.
- `ilisi`/`clisi`: switch to perplexity-tuned weights (Korsunsky 2019);
  optionally call `harmonypy.compute_lisi` if installed.
- Rename current heuristic functions to `*_chi2_proxy` / `*_simpson`
  and keep them as alternate exports for backward compatibility.
Verification: V2.7. On a 5-point synthetic mixing grid, our values must
correlate (ρ > 0.95) with the reference impl values.
Note: optional dep `harmonypy` falls under Q-### (no new heavy deps
without sign-off).

### W-041  F-SILHOUETTE-GLOBAL  [H]  — true per-group silhouette
Depends on: W-001.
Surface: [src/spVIPESmulti/metrics.py#L194-L228](src/spVIPESmulti/metrics.py#L194-L228),
[src/spVIPESmulti/metrics.py#L295-L316](src/spVIPESmulti/metrics.py#L295-L316).
Red test: [tests/audit_regressions/test_silhouette_per_group.py](tests/audit_regressions/test_silhouette_per_group.py).
Fix sketch:
- For each group `g`, compute silhouette on
  `z_private[g]` against labels restricted to cells of group `g`.
- `integration_report` writes the per-group value to row `g`, not the
  global pooled value.
Verification: V2.6. Two-group ablation where group 0 has separable
labels and group 1 has random labels — per-group silhouettes differ by
> 0.1.

### W-042  F-DA-WARN  [M]  — invert warning condition (folded into W-030)
Tracked under W-030; no separate regression test.

### W-043  F-RECON-POISSON  [M]  — calibrated Poisson NLL using mixed rate
Depends on: W-011.
Surface: [src/spVIPESmulti/metrics.py#L494-L512](src/spVIPESmulti/metrics.py#L494-L512).
Red test: `tests/audit_regressions/test_recon_metrics.py::test_poisson_nll_uses_mixed_rate`.
Fix sketch:
- Replace ad-hoc `-x*log(rate) + rate` with
  `-Poisson(rate).log_prob(x)` (or document the dropped `log(x!)`).
- Use `px_mixing` to blend `px_rate_private` and `px_rate_shared`.
Verification: V3.3 (negative control: 10× scaling of counts produces a
proportional change in NLL).

### W-044  F-RECON-RMSE  [M]  — RMSE on counts, not on simplex
Depends on: W-011.
Surface: [src/spVIPESmulti/metrics.py#L485-L490](src/spVIPESmulti/metrics.py#L485-L490).
Red test: `tests/audit_regressions/test_recon_metrics.py::test_rmse_scales_with_counts`.
Fix sketch:
- Compare `px.mean()` (the model's expected counts) to `x_raw`, not
  `px_scale` to `x_raw / library`.
Verification: V3.3.

### W-050  F-JEFFREYS-MEANPOOL  [M]  — per-cell symmetric KL or MMD
Depends on: W-020.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L601-L622](src/spVIPESmulti/module/spVIPESmultimodule.py#L601-L622).
Red test: `tests/audit_regressions/test_jeffreys_per_cell.py`.
Fix sketch:
- Compute symmetric-KL **per cell** between the two posteriors, then
  reduce. Alternatively, compute a sample-MMD between the two batches.
Verification: V3.1. Synthetic `N(0,1)` vs `N(δ,1)`: implemented loss
must scale with `δ²`, not be dominated by intra-batch variance.

### W-051  F-NF-KL-1SAMPLE  [M]  — multi-sample MC KL with reported variance
Depends on: W-022.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L624-L651](src/spVIPESmulti/module/spVIPESmultimodule.py#L624-L651).
Red test: `tests/audit_regressions/test_nf_kl_variance.py`.
Fix sketch:
- Add `n_mc_samples` (default 1 for backward compat) and average
  `log q − log p_flow` across samples.
- Log per-batch variance to `extra_metrics`.
Verification: V3.2.

### W-052  F-LATENT-VANISH  [M]  — KL-collapse diagnostic
Depends on: W-001.
Surface: [src/spVIPESmulti/metrics.py#L382-L431](src/spVIPESmulti/metrics.py#L382-L431).
Red test: `tests/audit_regressions/test_latent_vanish.py`.
Fix sketch:
- Replace 0.5-std heuristic with per-dim `KL(qz_d ‖ N(0,1))` averaged
  over cells; declare collapse when `mean_KL < 0.05`.
Verification: new — synthetic two-dim model with one dim manually KL-
collapsed; only that dim flagged.

### W-053  F-BN-EVAL-SMALL  [M]  — drop_last guidance or layer-swap
Depends on: W-001.
Surface: [src/spVIPESmulti/nn/networks.py#L99-L108](src/spVIPESmulti/nn/networks.py#L99-L108),
[src/spVIPESmulti/model/spvipesmulti.py#L368-L378](src/spVIPESmulti/model/spvipesmulti.py#L368-L378).
Red test: `tests/audit_regressions/test_bn_inference_stable.py`.
Fix sketch:
- Switch encoder/decoder normalisation to LayerNorm by default (BN
  remains opt-in), or document the size-1 trailing batch caveat.
Verification: new — repeated calls with batch sizes `{1, 2, 32}` give
embeddings within 1e-5 of each other.

### W-054  F-CE-WEIGHTS  [M]  — per-batch normalisation of class-weighted CE
Depends on: W-001.
Surface: [src/spVIPESmulti/model/spvipesmulti.py#L174-L185](src/spVIPESmulti/model/spvipesmulti.py#L174-L185),
[src/spVIPESmulti/module/spVIPESmultimodule.py#L1110-L1166](src/spVIPESmulti/module/spVIPESmultimodule.py#L1110-L1166).
Red test: `tests/audit_regressions/test_class_weight_invariance.py`.
Fix sketch:
- Use `reduction="mean"` consistently and let the weights normalise
  across the minibatch.
Verification: new — minibatch composition shuffled; per-cell loss
unchanged within MC noise.

### W-055  F-WARMUP-CE  [M]  — supervised-CE outside KL warmup
Depends on: W-001.
Surface: [src/spVIPESmulti/module/spVIPESmultimodule.py#L1316-L1321](src/spVIPESmulti/module/spVIPESmultimodule.py#L1316-L1321),
[src/spVIPESmulti/module/spVIPESmultimodule.py#L1418-L1422](src/spVIPESmulti/module/spVIPESmultimodule.py#L1418-L1422).
Red test: `tests/audit_regressions/test_warmup_decoupled.py`.
Fix sketch:
- Apply `kl_weight` only to GRL components (1, 4); keep CE components
  (2, 3, 5) at full weight from epoch 0.
Verification: V2.7.warmup.

### W-056  F-OBS-INDEX  [M]  — explicit row-position assembly
Depends on: none.
Surface: [src/spVIPESmulti/data/prepare_adatas.py#L130-L150](src/spVIPESmulti/data/prepare_adatas.py#L130-L150).
Red test: `tests/audit_regressions/test_obs_indices_explicit.py`.
Fix sketch:
- Use `np.flatnonzero(obs["groups"].to_numpy() == k)` and document the
  ordering invariant.
Verification: new — per-group reconstruction round-trip equals identity
under arbitrary `obs` permutation.

---

## 4. Cross-cutting deliverables

### A. Synthetic data harness — `tests/audit_regressions/_synthdata.py`
- `make_nb_counts(n_cells=512, n_genes=200, theta=10.0, lib_loc=8.0, seed=0)`
  returns `(X_raw_int, mu_true, theta_true)`. Used by W-011 and W-043.
- `make_lognorm_protein(n_cells=512, n_proteins=20, mu_range=(-3, 3),
  sigma=0.5, seed=0)` returns `Y` and `mu_true`. Used by W-012.
- `make_paired_two_group(n_cells_per=256, n_genes=200, share_frac=0.5,
  seed=0)` returns `dict[str, AnnData]` where rows of group A and B
  correspond cell-for-cell. Used by W-020 (paired control).
- `make_unpaired_two_group(...)` analogous but with shuffled rows. Used
  by W-020 (negative control: shuffling group B rows must not change
  group A's `z_shared`).
- `make_label_permutation_iter(adata, n_permutations=200, seed=0)`
  yields permuted-group AnnDatas for DA calibration. Used by W-030.
- `make_two_group_bimodal_private(seed=0)`: group 0 ~ N(0,I),
  group 1 ~ N(2·1, I) on private coords. Used by W-022.
- All generators take `seed` and return numpy/torch tensors with shapes
  documented in the docstring.

### B. Calibration test suite — `tests/audit_regressions/test_calibration.py`
- `test_da_pvalue_uniformity`: KS test against `Uniform[0,1]` using
  `make_paired_two_group(share_frac=1.0)` (no true shift). Threshold:
  KS p > 0.01 with 200 permutations × 256 cells. (W-030.)
- `test_kbet_matches_reference`: 5-point mixing grid; assert ρ > 0.95
  vs reference (only runs if `harmonypy` is importable; skipped
  otherwise). (W-040.)
- `test_lisi_matches_reference`: as above. (W-040.)
- `test_nb_recovery`: train one-group spVIPESmulti on
  `make_nb_counts`; assert `corr(px_rate_shared.mean(0), mu_true) > 0.9`.
  (W-011.)

### C. CI integration
- New pytest marker `audit_regression` declared in
  [pyproject.toml](pyproject.toml) (under `[tool.pytest.ini_options]`).
- `--runaudit` CLI flag (via `conftest.py`) gates calibration tests
  that need many permutations or are slow (>30 s).
- `make audit` target (Makefile) running
  `pytest -m audit_regression --runaudit`.
- **Cadence:** `audit_regression` runs on every PR; calibration suite
  (slow, behind `--runaudit`) runs nightly only.

### D. Documentation deltas (TODO list, owner = work-item)

| Doc surface | Current claim that becomes wrong | Owning W-### |
|---|---|---|
| [src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432](src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432) docstring of `get_latent_representation` | "Give mean of distribution or sample from it." | W-001 |
| [docs/notebooks/pbmc_citeseq_tutorial.ipynb](docs/notebooks/pbmc_citeseq_tutorial.ipynb) Gaussian-modality cell narrative | Implies log-normalised protein is a valid input | W-012 / Q-LIKELIHOOD-INTENT |
| [docs/notebooks/multimodal_nf_tutorial.ipynb](docs/notebooks/multimodal_nf_tutorial.ipynb) NF-prior cell | Implies the flow is per-group; it is global | W-022 |
| [CLAUDE.md](CLAUDE.md) §Disentanglement objective | "monotonic, deterministic" warmup symmetric across all 5 components | W-055 |
| [docs/api.md](docs/api.md) `differential_abundance` entry | Documents a "score" but talks about it as if it were a test | W-030 |
| [src/spVIPESmulti/metrics.py#L30-L141](src/spVIPESmulti/metrics.py#L30-L141) docstrings of `kbet`, `ilisi`, `clisi` | Names match published statistics; math does not | W-040 |
| [src/spVIPESmulti/metrics.py#L194-L228](src/spVIPESmulti/metrics.py#L194-L228) `per_group_silhouette` | Returns one global value; column suggests per-group | W-041 |
| [src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184](src/spVIPESmulti/module/spVIPESmultimodule.py#L1170-L1184) prototype EMA comment | Does not mention val/eval semantics | W-002 |

### E. Open scientific questions (from P§4 + S§4)

These require user/PI sign-off **before** any code change associated
with them lands. Each is a yes/no or among-N decision; do not auto-resolve.

- **Q-LIKELIHOOD-INTENT** — Should the Gaussian likelihood path support
  arbitrary log-normalised protein/ADT values, or are users expected to
  rescale to `[0, 1]` before `setup_anndata`? (Determines W-012 scope.)
- **Q-POE-INTENT** — Is the unsupervised PoE intended as a posterior
  per cell, or as a soft regulariser that pulls cells together? Drives
  W-020 between Option A (drop) and Option B (label-conditional).
- **Q-DA-NULL** — For `differential_abundance`, what null distribution
  is appropriate to publish: sample-level permutation, cell-level
  bootstrap, or a parametric null on the score? (Drives W-030.)
- **Q-EMBED-DEFAULT** — Should `embed()` return posterior means or
  rsamples by default? scvi-tools convention is means; current behaviour
  is rsample (W-001 fixes the inconsistency, but the *default* is a
  user-facing choice).
- **Q-RECON-COLS** — Intended interpretation of `reconstruction_error`
  columns relative to the NB-mixture loss actually optimised? (Drives
  the docstring rewrite owned by W-043 / W-044.)
- **Q-TRAVERSAL** — Is `traverse_latent` interpretable under the default
  low-rank mixer (rank=4)? If not, should it be disabled or warn?
- **Q-DISENT-EQ** — Calibration of the GRL adversarial equilibrium under
  default disentangle weights (esp. `disentangle_label_private_weight=0.05`).
- **Q-NF-PER-GROUP** — Is the NF private prior intended to be per-group?
  (Drives W-022 between per-group flows and a documented-global warning.)
- **Q-BATCH-VS-GROUP** — Is `batch_key ≡ groups_key` an expected use
  case? If yes, the GRL story needs documentation (F-BATCH-VS-GROUP);
  if no, `setup_anndata` should warn.
- **Q-HARMONYPY-DEP** — May we add `harmonypy` as an optional dependency
  for W-040 reference comparison? (Per CONSTRAINTS, no new heavy deps
  without flag.)

---

## 5. Self-review checklist

- [x] Every §2.x finding from the primary audit maps to ≥1 work item
  (P§2.1→W-011; P§2.2→W-020; P§2.3→W-012; P§2.4→W-030; P§2.5→W-001;
  P§2.6→W-041; P§2.7→W-040).
- [x] Every §3.x finding maps to ≥1 work item (P§3.1→W-050;
  P§3.2→W-051; P§3.3→W-043; P§3.4→W-044; P§3.5→W-052; P§3.6→W-053;
  P§3.7→W-021; P§3.8→W-030; P§3.9→W-054; P§3.10→W-056).
- [x] Every secondary §2.x finding mapped (S§2.1→W-010; S§2.2→W-002;
  S§2.3→W-022; S§2.4→W-003; S§2.5→W-021; S§2.6→Q-BATCH-VS-GROUP;
  S§2.7→W-055).
- [x] Every §5 verification check from the primary audit maps to ≥1
  test in §B or a card's *Verification* line.
- [x] No work item modifies a file without a regression test referencing
  that file's surface.
- [x] No fix is merged before its red test exists (enforced by §0
  procedure).
- [x] Severity ordering is dependency-respecting (see §2 paragraph).
- [x] Plan is runnable top-to-bottom by an agent except for Q-### items
  in §E.

---

## A. Appendix — long rationale (referenced from cards)

### A.1 Why fix W-001 first
Every regression test under §B asserts a numeric property of an
embedding returned by `get_latent_representation`. While that call
silently returns an rsample, every assertion is at the mercy of MC
noise. The fix is one line plus a test; the payoff is deterministic
fixtures for the entire remediation effort.

### A.2 Why W-020 is decision-blocked behind Q-POE-INTENT
The cleanest fix (drop across-group PoE entirely) is a behaviour change
that affects every published `z_shared`. The user's research narrative
may rely on PoE as a regulariser even if it is not a true posterior.
This is a scientific decision, not an engineering one. The red test
(row-shuffle invariance) is valid under either Option A or Option B —
in Option B, `z_shared` is invariant to shuffling because PoE is
restricted to same-label cells and same-label cells are paired only
within their label sub-batch (still order-dependent unless we further
collapse via expectation, which is Option B's second clause).

### A.3 Why W-040 is `H` and not `C`
The kBET/LISI heuristics are not used inside the loss; they are
post-hoc diagnostics. They mislead reviewers but do not bias the
trained model itself. Critical reserved for items that change what is
optimised.

### A.4 Why the synthetic harness lives under `tests/audit_regressions/`
Per CONSTRAINTS, no edits to existing tests. The harness is new code
exclusively consumed by new tests, so it lives next to its consumers.
