# Phase 3 - Intent-vs-Implementation Findings

## Finding IMPL-001: Latent Representation Return Contract Is Misdocumented

- Severity: MEDIUM
- Status: FIXED in 1.0.1. Historical observation describes the 1.0.0 audit
  baseline.
- Locations: `src/spVIPESmulti/model/spvipesmulti.py:L430`,
  `src/spVIPESmulti/model/spvipesmulti.py:L456` to
  `src/spVIPESmulti/model/spvipesmulti.py:L458`, and
  `src/spVIPESmulti/model/spvipesmulti.py:L1564` to
  `src/spVIPESmulti/model/spvipesmulti.py:L1589`
- scvi-tools version contract checked against: 1.4.2
- Observation: `get_latent_representation` is annotated as returning
  `np.ndarray` and its Returns section says "Low-dimensional topic for each
  cell." `_format_results` returns a dictionary containing shared, private,
  reordered, posterior loc/scale, and optional multimodal-private arrays.
- Risk: users and downstream code can treat the method as array-returning when
  it is dictionary-returning; type checkers and generated docs are misleading.
- Confidence: HIGH, source is direct.
- Suggested fix: update the annotation and numpydoc Returns section to the
  actual dictionary schema, including shape and ordering semantics.

## Finding IMPL-002: `indices` Documentation Claims Subsetting That Is Absent

- Severity: HIGH
- Status: FIXED in 1.0.1. Historical observation describes the 1.0.0 audit
  baseline.
- Locations: `src/spVIPESmulti/model/spvipesmulti.py:L442` to
  `src/spVIPESmulti/model/spvipesmulti.py:L443` and
  `src/spVIPESmulti/model/spvipesmulti.py:L475` to
  `src/spVIPESmulti/model/spvipesmulti.py:L483`
- scvi-tools version contract checked against: 1.4.2
- Observation: the docstring says `indices` selects cells, but implementation
  does not use `indices` at all.
- Risk: this is the user-facing intent mismatch behind INT-002.
- Confidence: HIGH.
- Suggested fix: implement the subset behavior or remove the parameter from the
  public contract after a deprecation cycle.

## Finding IMPL-003: Single-modal and Multimodal Library Handling Diverge

- Severity: MEDIUM
- Status: FIXED in 1.0.1. Historical observation describes the 1.0.0 audit
  baseline.
- Locations: `src/spVIPESmulti/module/spVIPESmultimodule.py:L801` and
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L858` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L859`
- scvi-tools version contract checked against: 1.4.2
- Observation: multimodal inference clamps library sums before `log`, while
  single-modal inference does not.
- Risk: the same biological edge case, an all-zero cell, is handled safely in
  one path and unsafely in the other.
- Confidence: HIGH, confirmed dynamically for the single-modal path.
- Suggested fix: share one small helper for observed library computation across
  both paths.
