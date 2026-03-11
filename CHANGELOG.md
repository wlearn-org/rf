# Changelog

## [0.3.0] - 2026-03-11

### Added

- Missing value handling (NaN). At each split, both left and right NaN directions
  are tried; the direction that maximizes gain is learned and stored per node.
  At prediction time, NaN values follow the learned direction. Works for both
  classification and regression.

- Monotonic constraints (`monotonicCst` / `monotonic_cst`). Per-feature constraints
  enforced via bound propagation: upper/lower prediction bounds are inherited from
  ancestor nodes and tightened at each split. Supports both regression and binary
  classification (P(class=1) is constrained).
  Bartley et al. (2020) "An Improved Method for Monotone Constraints." arXiv:2011.00986.

- Quantile regression forests (`predictQuantile` / `predict_quantile`). Stores
  training sample indices per leaf in CSR format. At prediction time, aggregates
  co-leaf samples across trees and computes weighted empirical quantiles via linear
  interpolation. A single fitted model serves all quantiles. Auto-enabled for
  regression tasks (`storeLeafSamples` defaults to 1 for regression).
  Meinshausen (2006) "Quantile Regression Forests." JMLR, 7(35):983-999.

- Conformal prediction intervals (`predictInterval` / `predict_interval`).
  Jackknife+-after-Bootstrap (J+ab): computes OOB absolute residuals during fit,
  then constructs symmetric intervals around point predictions using the
  `(1-alpha)(1+1/n)`-th quantile of OOB residuals. Provides distribution-free
  coverage guarantees with no additional computation beyond standard OOB scoring.
  Barber et al. (2021) "Predictive Inference with the Jackknife+." Annals of Statistics.

### Changed

- Wrap RFModel with `createModelClass` for unified task detection
- `task` parameter (`'classification'` or `'regression'`) is now auto-detected from labels if omitted
- Serialization format v3 (backward-compatible read of v1/v2). Adds 1 byte per
  node for learned NaN direction (29 bytes per node, up from 28 in v2).

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
