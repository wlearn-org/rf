# @wlearn/rf

Random Forest classifier and regressor, written in C11 from scratch. No upstream dependencies.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)). Runs as native C, WASM (browser + Node), with JS and Python wrappers.

## Features

- Classification (Gini, entropy, Hellinger) and regression (MSE, MAE)
- Bootstrap aggregating with adjustable sample rate
- ExtraTrees mode (random threshold per feature)
- Heterogeneous RF (depth-dependent feature weighting)
- OOB scoring with optional per-tree weighting
- MDI feature importances
- Cost-complexity pruning
- Local linear leaf models (regression)
- Missing value handling (learned NaN direction per split)
- Monotonic constraints (bound propagation)
- Quantile regression forests (per-leaf sample storage)
- Conformal prediction intervals (Jackknife+-after-Bootstrap)
- Deterministic (LCG PRNG, fixed seed)
- Binary serialization (RF01/RF02/RF03, save/load)
- 85 C tests, 39 JS tests, 27 Python tests, 74 parity tests

Default parameters match sklearn's `RandomForestClassifier` / `RandomForestRegressor` exactly.

## Installation

### JavaScript (npm)

```
npm install @wlearn/rf
```

### Python

```bash
# Build the C library
mkdir build && cd build && cmake .. && make
# Set library path
export RF_LIB_PATH=build/librf.so
```

### C

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make
./test_rf  # run tests
```

## Quick Start

### JavaScript

```js
const { RFModel } = require('@wlearn/rf')

// Classification
const clf = await RFModel.create({ nEstimators: 100, seed: 42 })
clf.fit(X_train, y_train)
const predictions = clf.predict(X_test)
const accuracy = clf.score(X_test, y_test)
const probabilities = clf.predictProba(X_test)
clf.dispose()

// Regression
const reg = await RFModel.create({
  nEstimators: 100, task: 'regression', seed: 42
})
reg.fit(X_train, y_train)
const r2 = reg.score(X_test, y_test)

// Quantile predictions
const median = reg.predictQuantile(X_test, 0.5)
const [q10, q90] = reg.predictQuantile(X_test, [0.1, 0.9])

// Prediction intervals (90% coverage)
const { lower, upper } = reg.predictInterval(X_test, 0.1)
reg.dispose()
```

### Python

```python
from wlearn_rf import RFModel

# Classification
clf = RFModel({'n_estimators': 100, 'seed': 42})
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
probabilities = clf.predict_proba(X_test)
clf.dispose()

# Regression
reg = RFModel({'n_estimators': 100, 'task': 'regression', 'seed': 42})
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)

# Quantile predictions
median = reg.predict_quantile(X_test, quantile=0.5)

# Prediction intervals (90% coverage)
lower, upper = reg.predict_interval(X_test, alpha=0.1)
reg.dispose()
```

### C

```c
#include "rf.h"

rf_params_t params;
rf_params_init(&params);
params.n_estimators = 100;
params.seed = 42;
params.task = 0;  // 0 = classification, 1 = regression

rf_forest_t *forest = rf_fit(X, nrow, ncol, y, &params);

double *preds = malloc(nrow * sizeof(double));
rf_predict(forest, X, nrow, ncol, preds);

// Quantile predictions (requires store_leaf_samples=1)
params.store_leaf_samples = 1;
rf_forest_t *qrf = rf_fit(X, nrow, ncol, y, &params);
double quantiles[] = {0.1, 0.5, 0.9};
double *qout = malloc(nrow * 3 * sizeof(double));
rf_predict_quantile(qrf, X_test, ntest, ncol, quantiles, 3, qout);

// Conformal prediction intervals (requires bootstrap + OOB)
double *lower = malloc(ntest * sizeof(double));
double *upper = malloc(ntest * sizeof(double));
rf_predict_interval(forest, X_test, ntest, ncol, 0.1, lower, upper);

rf_free(forest);
rf_free(qrf);
```

## API Reference

### Parameters

All parameters have defaults that match sklearn. New features are opt-in and do not change default behavior.

| Parameter | JS name | Python name | C field | Default | Description |
|-----------|---------|-------------|---------|---------|-------------|
| Number of trees | `nEstimators` | `n_estimators` | `n_estimators` | 100 | Number of trees in the forest |
| Max depth | `maxDepth` | `max_depth` | `max_depth` | 0 (unlimited) | Maximum tree depth. 0 = grow until pure or min_samples |
| Min samples split | `minSamplesSplit` | `min_samples_split` | `min_samples_split` | 2 | Minimum samples to split an internal node |
| Min samples leaf | `minSamplesLeaf` | `min_samples_leaf` | `min_samples_leaf` | 1 | Minimum samples per leaf |
| Max features | `maxFeatures` | `max_features` | `max_features` | 0 (auto) | Features per split. 0 = sqrt(n) for cls, n/3 for reg. Also: `'sqrt'`, `'log2'`, `'third'` |
| Max leaf nodes | `maxLeafNodes` | `max_leaf_nodes` | `max_leaf_nodes` | 0 (unlimited) | Maximum number of leaf nodes per tree |
| Bootstrap | `bootstrap` | `bootstrap` | `bootstrap` | 1 (true) | Whether to bootstrap sample |
| Compute OOB | `computeOob` | `compute_oob` | `compute_oob` | 1 (true) | Whether to compute OOB score |
| ExtraTrees | `extraTrees` | `extra_trees` | `extra_trees` | 0 (false) | ExtraTrees mode (random threshold) |
| Seed | `seed` | `seed` | `seed` | 42 | Random seed for reproducibility |
| Task | `task` | `task` | `task` | `'classification'` / 0 | `'classification'` or `'regression'` (JS); 0 or 1 (C/Python) |

### v0.2 Parameters

| Parameter | JS name | Python name | C field | Default | Description |
|-----------|---------|-------------|---------|---------|-------------|
| Criterion | `criterion` | `criterion` | `criterion` | 0 / `'gini'` | Split criterion. Classification: `'gini'`(0), `'entropy'`(1), `'hellinger'`(2). Regression: `'mse'`(0), `'mae'`(1) |
| Sample rate | `sampleRate` | `sample_rate` | `sample_rate` | 1.0 | Fraction of samples per tree. Like sklearn `max_samples` |
| Heterogeneous | `heterogeneous` | `heterogeneous` | `heterogeneous` | 0 | Depth-dependent feature weighting. 1 = enabled |
| OOB weighting | `oobWeighting` | `oob_weighting` | `oob_weighting` | 0 | Weight tree votes by OOB performance. 1 = enabled |
| Alpha trim | `alphaTrim` | `alpha_trim` | `alpha_trim` | 0.0 | Cost-complexity pruning penalty. 0 = no pruning |
| Leaf model | `leafModel` | `leaf_model` | `leaf_model` | 0 | Leaf prediction model. 0 = constant (mean), 1 = local linear (regression only) |

### v0.3 Parameters

| Parameter | JS name | Python name | C field | Default | Description |
|-----------|---------|-------------|---------|---------|-------------|
| Store leaf samples | `storeLeafSamples` | `store_leaf_samples` | `store_leaf_samples` | auto | Store per-leaf sample indices for quantile prediction. Auto-enabled for regression |
| Monotonic constraints | `monotonicCst` | `monotonic_cst` | `monotonic_cst` | null | Per-feature constraints: -1 = decreasing, 0 = none, +1 = increasing. Array of length n_features |

### Methods

#### JavaScript (`RFModel`)

```js
// Construction
const model = await RFModel.create(params)

// Training
model.fit(X, y)                // X: number[][] or Float64Array, y: number[]

// Prediction
model.predict(X)               // Returns Float64Array
model.predictProba(X)          // Classification only. Returns Float64Array (flat, row-major)
model.score(X, y)              // Accuracy (cls) or R2 (reg)

// Quantile prediction (regression, requires store_leaf_samples)
model.predictQuantile(X, 0.5)           // Single quantile: returns Float64Array
model.predictQuantile(X, [0.1, 0.9])   // Multiple: returns [Float64Array, Float64Array]

// Conformal prediction intervals (regression, requires bootstrap + OOB)
model.predictInterval(X, alpha)  // Returns { lower: Float64Array, upper: Float64Array }

// Inspection
model.featureImportances()     // Returns Float64Array (normalized MDI)
model.oobScore()               // OOB accuracy (cls) or R2 (reg)
model.nFeatures                // Number of features
model.nClasses                 // Number of classes (0 for regression)
model.nTrees                   // Number of trees
model.isFitted                 // Whether model is fitted
model.capabilities             // { classifier, regressor, predictProba, featureImportances, ... }

// Persistence
const bundle = model.save()    // Returns Uint8Array (wlearn bundle format)
const loaded = await RFModel.load(bundle)

// Cleanup (required)
model.dispose()

// AutoML
RFModel.defaultSearchSpace()   // Returns search space IR for hyperparameter tuning

// Params
model.getParams()              // Returns current params
model.setParams({ maxDepth: 5 })
```

#### Python (`RFModel`)

```python
# Construction
model = RFModel(params_dict)

# Training
model.fit(X, y)                # X: numpy 2D array, y: numpy 1D array

# Prediction
model.predict(X)               # Returns numpy array
model.predict_proba(X)         # Classification only. Returns (n, n_classes) array
model.score(X, y)              # Accuracy (cls) or R2 (reg)

# Quantile prediction (regression)
model.predict_quantile(X, quantile=0.5)    # Returns numpy array

# Conformal prediction intervals (regression)
lower, upper = model.predict_interval(X, alpha=0.1)  # Returns (lower, upper) arrays

# Inspection
model.feature_importances()    # Returns numpy array
model.oob_score()              # OOB score
model.n_features               # Number of features
model.n_classes                # Number of classes
model.n_trees                  # Number of trees
model.is_fitted                # Whether model is fitted

# Persistence
blob = model.save_raw()        # Returns bytes (RF01/RF02/RF03 binary)
loaded = RFModel.load_raw(blob, params={'task': 'regression'})

# Cleanup
model.dispose()
```

#### C API

```c
// Initialize params with defaults
void rf_params_init(rf_params_t *params);

// Fit a forest. Returns NULL on error.
rf_forest_t *rf_fit(const double *X, int32_t nrow, int32_t ncol,
                     const double *y, const rf_params_t *params);

// Predict. out must be pre-allocated (nrow doubles). Returns 0 on success.
int rf_predict(const rf_forest_t *forest, const double *X,
               int32_t nrow, int32_t ncol, double *out);

// Predict probabilities (classification). out: nrow * n_classes doubles.
int rf_predict_proba(const rf_forest_t *forest, const double *X,
                     int32_t nrow, int32_t ncol, double *out);

// Predict quantiles (requires store_leaf_samples=1).
// out: nrow * n_quantiles doubles, row-major.
int rf_predict_quantile(const rf_forest_t *forest, const double *X,
                        int32_t nrow, int32_t ncol,
                        const double *quantiles, int32_t n_quantiles,
                        double *out);

// Predict conformal intervals (requires bootstrap + OOB).
// out_lower, out_upper: nrow doubles each.
int rf_predict_interval(const rf_forest_t *forest, const double *X,
                        int32_t nrow, int32_t ncol, double alpha,
                        double *out_lower, double *out_upper);

// Serialize / deserialize
int rf_save(const rf_forest_t *forest, char **out_buf, int32_t *out_len);
rf_forest_t *rf_load(const char *buf, int32_t len);

// Free resources
void rf_free(rf_forest_t *forest);
void rf_free_buffer(void *ptr);

// Error message
const char *rf_get_error(void);
```

## Advanced Usage Examples

### Missing values (NaN)

NaN values are handled automatically. At each split, both NaN directions (left/right) are tried and the best is learned. At prediction time, NaN values follow the learned direction.

```js
// JS: NaN values work transparently
const X = [[1.0, NaN], [2.0, 3.0], [NaN, 4.0]]
const y = [0, 1, 0]
model.fit(X, y)
model.predict([[NaN, 2.0]])  // NaN follows learned direction
```

```python
# Python: NaN values work transparently
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
y = np.array([0, 1, 0])
model.fit(X, y)
model.predict(np.array([[np.nan, 2.0]]))
```

### Monotonic constraints

Enforce monotonic relationships between features and predictions. Useful when domain knowledge dictates that increasing a feature should never decrease the prediction.

```js
const model = await RFModel.create({
  task: 'regression',
  monotonicCst: [1, -1, 0],  // feature 0: increasing, 1: decreasing, 2: unconstrained
  seed: 42
})
model.fit(X, y)
```

```python
model = RFModel({
    'task': 'regression',
    'monotonic_cst': [1, -1, 0],
    'seed': 42
})
model.fit(X, y)
```

### Quantile regression forests

Predict any quantile of the conditional distribution. A single fitted model serves all quantiles.

```js
const model = await RFModel.create({
  task: 'regression', nEstimators: 200, seed: 42
})
model.fit(X_train, y_train)

// 90% prediction interval via quantiles
const [q05, q95] = model.predictQuantile(X_test, [0.05, 0.95])
model.dispose()
```

```python
model = RFModel({'task': 'regression', 'n_estimators': 200, 'seed': 42})
model.fit(X_train, y_train)

q05 = model.predict_quantile(X_test, quantile=0.05)
q95 = model.predict_quantile(X_test, quantile=0.95)
```

### Conformal prediction intervals

Distribution-free prediction intervals with finite-sample coverage guarantees via Jackknife+-after-Bootstrap. Uses OOB residuals computed during training.

```js
const model = await RFModel.create({
  task: 'regression', nEstimators: 200, seed: 42
})
model.fit(X_train, y_train)

// 90% prediction intervals (alpha=0.1)
const { lower, upper } = model.predictInterval(X_test, 0.1)
model.dispose()
```

```python
model = RFModel({'task': 'regression', 'n_estimators': 200, 'seed': 42})
model.fit(X_train, y_train)

lower, upper = model.predict_interval(X_test, alpha=0.1)
```

### Entropy criterion with sample rate

```js
const model = await RFModel.create({
  nEstimators: 200,
  criterion: 'entropy',
  sampleRate: 0.7,
  seed: 42
})
model.fit(X, y)
```

### Heterogeneous RF for high-dimensional data

Depth-dependent feature weighting promotes diversity. Features overused at shallow
depths are downweighted at deeper levels, giving more features a chance to contribute.

```python
model = RFModel({
    'n_estimators': 200,
    'heterogeneous': 1,
    'max_features': 0,  # auto (sqrt)
    'task': 'regression',
    'seed': 42
})
model.fit(X, y)
```

### Local linear leaves for regression

Instead of predicting the leaf mean, fit a ridge regression per leaf.
Best combined with shallow trees (`max_depth=3-7`) so each leaf has enough
samples for a meaningful linear fit.

```python
model = RFModel({
    'n_estimators': 100,
    'task': 'regression',
    'leaf_model': 1,      # local linear
    'max_depth': 5,       # shallow trees
    'seed': 42
})
model.fit(X, y)
r2 = model.score(X_test, y_test)
```

### Hellinger distance for imbalanced classification

Hellinger distance is robust to class imbalance -- it measures divergence between
class distributions at each split rather than impurity reduction.

```js
const model = await RFModel.create({
  nEstimators: 200,
  criterion: 'hellinger',
  seed: 42
})
model.fit(X_imbalanced, y_imbalanced)
```

### Cost-complexity pruning

Post-hoc pruning reduces overfitting by collapsing subtrees whose impurity gain
is less than `alpha` per leaf.

```js
const model = await RFModel.create({
  nEstimators: 100,
  alphaTrim: 0.01,
  seed: 42
})
model.fit(X, y)
```

### OOB-weighted voting

Weight each tree's vote by its individual OOB performance. Trees that generalize
better get more influence.

```python
model = RFModel({
    'n_estimators': 200,
    'oob_weighting': 1,
    'seed': 42
})
model.fit(X, y)
```

### Combined configuration

```js
const model = await RFModel.create({
  nEstimators: 200,
  task: 'regression',
  criterion: 'mse',
  leafModel: 1,          // local linear leaves
  maxDepth: 5,            // shallow trees for linear leaves
  heterogeneous: 1,       // depth-dependent feature weighting
  oobWeighting: 1,        // weight trees by OOB performance
  sampleRate: 0.8,        // subsample 80%
  monotonicCst: [1, 0, -1, 0, 0],  // monotonic constraints
  seed: 42
})
model.fit(X, y)
```

## Benchmark / Ablation Study

Benchmark on synthetic data (n=500, 100 trees, seed=42). OOB scores are the most
meaningful metric since they estimate generalization without a separate test set.

### Classification (10 features, 3 classes, overlapping distributions)

| Configuration                  | Train Acc |  OOB Acc | Fit (ms) |
| ------------------------------ | --------: | -------: | -------: |
| Default (Gini)                 |    1.0000 |   0.9700 |       54 |
| Entropy                        |    1.0000 |   0.9680 |       29 |
| Hellinger                      |    1.0000 |   0.9980 |       23 |
| Sample rate 0.7                |    1.0000 |   0.9740 |       15 |
| Heterogeneous RF               |    1.0000 |   0.9820 |       22 |
| OOB weighting                  |    1.0000 |   0.9700 |       22 |
| Alpha trim 0.01                |    1.0000 |   0.9700 |       22 |
| ExtraTrees                     |    1.0000 |   0.9820 |        4 |
| Entropy + HRF + OOB wt         |    1.0000 |   0.9680 |       24 |

Hellinger distance achieves the best OOB accuracy (0.998) on this multiclass problem.
ExtraTrees and HRF also improve over the default.

### Regression (10 features, linear signal with noise)

| Configuration                  | Train R2  |   OOB R2 | Fit (ms) |
| ------------------------------ | --------: | -------: | -------: |
| Default (MSE)                  |    0.9762 |   0.8236 |       86 |
| MAE                            |    0.9731 |   0.8011 |      178 |
| Sample rate 0.7                |    0.9536 |   0.8140 |       57 |
| Heterogeneous RF               |    0.9687 |   0.7664 |       91 |
| OOB weighting                  |    0.9771 |   0.8236 |       88 |
| Alpha trim 0.01                |    0.9684 |   0.8144 |       89 |
| Linear leaves                  |    0.9762 |   0.8236 |       87 |
| Linear leaves (depth=5)        |    0.9913 |   0.9677 |       55 |
| ExtraTrees                     |    0.9744 |   0.8083 |       25 |
| Linear + HRF + OOB wt          |    0.9706 |   0.7664 |       95 |

Linear leaves with shallow trees (depth=5) dramatically improve OOB R2 from 0.82 to
0.97 on this linear signal, confirming that local linear models capture within-leaf
trends that constant leaves miss. The fit time is also lower due to fewer nodes.

## Serialization

Models are serialized in a compact binary format (RF01 magic, format version 3).
The format is backward-compatible: v3 readers can load v1/v2 files (new fields get defaults).
v3 adds 1 byte per node for learned NaN direction.

```js
// JS: wlearn bundle format (wraps RF binary in WLRN container)
const bundle = model.save()       // Uint8Array
const loaded = await RFModel.load(bundle)
```

```python
# Python: raw RF binary format
blob = model.save_raw()           # bytes
loaded = RFModel.load_raw(blob)
```

```c
// C: raw RF binary format
char *buf; int32_t len;
rf_save(forest, &buf, &len);
rf_forest_t *loaded = rf_load(buf, len);
rf_free_buffer(buf);
```

## Building

### Native (C)

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make
./test_rf                         # 85 tests
```

### WASM

Requires Emscripten.

```bash
bash scripts/build-wasm.sh        # outputs wasm/rf.js
bash scripts/verify-exports.sh    # verifies 22 exports
```

### JS Tests

```bash
node test/test.js                 # 39 tests
```

### Python Tests

```bash
RF_LIB_PATH=build/librf.so python test/test_python.py       # 27 tests
RF_LIB_PATH=build/librf.so python -m pytest test/test_parity.py -v  # 74 tests
```

### Benchmark

```bash
gcc -O2 -std=c11 -o build/benchmark test/benchmark.c csrc/rf.c -Icsrc -lm
./build/benchmark
```

## References

- Meinshausen (2006). "Quantile Regression Forests." JMLR, 7(35):983-999.
- Barber et al. (2021). "Predictive Inference with the Jackknife+." Annals of
  Statistics, 49(1):486-507.
- Bartley et al. (2020). "An Improved Method for Monotone Constraints in Gradient
  Boosting." arXiv:2011.00986.
- Cieslak et al. (2012). "Hellinger distance decision trees are robust and
  skew-insensitive." Data Mining and Knowledge Discovery, 24(1), 136-158.
- Klusowski & Tian (2024). "Heterogeneous Random Forests." arXiv:2410.19022.
- Mentch & Zhou (2024). "Randomized Trees and Applications." arXiv:2410.04297.
- Correia & Lecue (2024). "Adaptive Random Forests with Cost-Complexity Pruning."
  arXiv:2408.07151.
- Horvath et al. (2025). "RaFFLE: Random Forest with Local Linear Extensions."
  arXiv:2502.10185.

## License

MIT
