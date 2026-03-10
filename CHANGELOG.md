# Changelog

## [0.3.0] - 2026-03-10

### Changed

- Wrap RFModel with `createModelClass` for unified task detection
- `task` parameter (`'classification'` or `'regression'`) is now auto-detected from labels if omitted

## [0.2.0] - 2026-03-03

### Added

- Entropy criterion (`criterion='entropy'`). Shannon entropy as alternative to Gini
  for classification splits.

- Hellinger distance criterion (`criterion='hellinger'`). Robust to class imbalance.
  Cieslak et al. (2012) "Hellinger distance decision trees are robust and
  skew-insensitive." Data Mining and Knowledge Discovery, 24(1), 136-158.

- MAE criterion for regression (`criterion='mae'`). Mean absolute error splitting,
  robust to outliers. Matches sklearn `criterion='absolute_error'`.

- Adjustable sample rate (`sampleRate`, default 1.0). Controls bootstrap/subsample
  size per tree. Equivalent to sklearn `max_samples`.
  Mentch & Zhou (2024) "Randomized Trees and Applications." arXiv:2410.04297.

- Heterogeneous RF (`heterogeneous=1`). Depth-dependent feature weighting that
  downweights features overused at shallow depths, promoting feature diversity
  across trees.
  Klusowski & Tian (2024) "Heterogeneous Random Forests." arXiv:2410.19022.

- OOB-weighted voting (`oobWeighting=1`). Trees weighted by their individual OOB
  performance (accuracy for classification, R2 for regression).

- Cost-complexity pruning (`alphaTrim > 0`). Post-hoc bottom-up pruning that
  collapses subtrees when the impurity gain is less than alpha per leaf.
  Correia & Lecue (2024) "Adaptive Random Forests with Cost-Complexity Pruning."
  arXiv:2408.07151.

- Local linear leaves (`leafModel=1`, regression only). Ridge regression per leaf
  instead of constant mean prediction. Improves R2 on signals with local linearity.
  Horvath et al. (2025) "RaFFLE: Random Forest with Local Linear Extensions."
  arXiv:2502.10185.

### Changed

- Serialization format v2 (backward-compatible read of v1). New header fields for
  criterion, leaf_model, heterogeneous, oob_weighting, sample_rate, alpha_trim.
  Tree weights serialized when oob_weighting is enabled.

## [0.1.0] - 2026-03-01

### Added

- Initial release: C11 Random Forest from scratch.
- Classification (Gini) and regression (MSE).
- Bootstrap aggregating with OOB scoring.
- ExtraTrees mode.
- MDI feature importances.
- Binary serialization (RF01 format).
- WASM build (Emscripten, single-file).
- JS wrapper (`@wlearn/rf`) with wlearn bundle format.
- Python wrapper (`wlearn_rf`) with ctypes FFI.
- 85 C tests, 39 JS tests, 27 Python tests.
