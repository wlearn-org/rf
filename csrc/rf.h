/*
 * rf.h -- Random Forest (C11, from scratch)
 *
 * Classification and regression with:
 * - CART splits (Gini / entropy / Hellinger for cls; MSE / MAE for reg)
 * - Bootstrap aggregating (bagging) with adjustable sample rate
 * - Per-split feature subsampling (uniform or depth-weighted HRF)
 * - ExtraTrees mode (random threshold per feature)
 * - OOB scoring with optional per-tree weighting
 * - MDI feature importance
 * - Cost-complexity pruning (alpha_trim)
 * - Local linear leaf models (regression)
 * - Missing value handling (learned NaN direction per split)
 * - Monotonic constraints (bound propagation)
 * - Quantile regression forests (leaf sample storage)
 * - Conformal prediction intervals (J+ab via OOB residuals)
 * - Binary serialization (RF01/RF02/RF03 format)
 */

#ifndef RF_H
#define RF_H

#include <stdint.h>
#include <stddef.h>

/* ---------- PRNG (LCG matching @wlearn/core rng.js) ---------- */

typedef struct {
    uint32_t state;
} rf_rng_t;

static inline uint32_t rf_rng_next(rf_rng_t *rng) {
    rng->state = (rng->state * 1664525u + 1013904223u) & 0x7FFFFFFFu;
    return rng->state;
}

static inline double rf_rng_uniform(rf_rng_t *rng) {
    return (double)rf_rng_next(rng) / (double)0x7FFFFFFFu;
}

static inline int32_t rf_rng_int(rf_rng_t *rng, int32_t n) {
    return (int32_t)(rf_rng_uniform(rng) * n);
}

/* ---------- Tree node ---------- */

typedef struct {
    int32_t  feature;     /* split feature index, -1 for leaf */
    double   threshold;   /* split threshold */
    int32_t  left;        /* left child index, -1 if none */
    int32_t  right;       /* right child index, -1 if none */
    int32_t  n_samples;   /* samples reaching this node */
    double   impurity;    /* Gini (cls) or MSE (reg) */
    int32_t  leaf_idx;    /* index into leaf_data, -1 for internal */
    int8_t   nan_dir;     /* NaN direction: 0=left, 1=right */
} rf_node_t;

/* ---------- Single decision tree ---------- */

typedef struct {
    rf_node_t *nodes;
    int32_t    n_nodes;
    int32_t    capacity;
    double    *leaf_data;  /* cls: n_leaves * n_classes; reg: n_leaves */
    int32_t    n_leaves;
    int32_t    leaf_cap;
    /* Quantile RF: per-leaf sample indices (CSR format) */
    int32_t   *leaf_samples;   /* concatenated sample indices per leaf */
    int32_t   *leaf_offsets;   /* offsets[leaf_id] .. offsets[leaf_id+1] */
    int32_t    n_leaf_samples; /* total length of leaf_samples */
} rf_tree_t;

/* ---------- Forest ---------- */

typedef struct {
    rf_tree_t *trees;
    int32_t    n_trees;
    int32_t    n_features;
    int32_t    n_classes;   /* 0 for regression */
    int32_t    task;        /* 0 = classification, 1 = regression */
    double    *feature_importances;  /* normalized, length n_features */
    double     oob_score;   /* OOB accuracy (cls) or R2 (reg), NaN if not computed */
    double    *tree_weights;  /* per-tree OOB weights, NULL if uniform */
    /* Stored hyperparams for serialization */
    int32_t    max_depth;
    int32_t    min_samples_split;
    int32_t    min_samples_leaf;
    int32_t    max_features;
    int32_t    max_leaf_nodes;
    int32_t    bootstrap;
    int32_t    compute_oob;
    int32_t    extra_trees;
    uint32_t   seed;
    /* New params (v0.2) */
    int32_t    criterion;      /* cls: 0=gini,1=entropy,2=hellinger; reg: 0=mse,1=mae */
    int32_t    heterogeneous;  /* depth-dependent feature weighting */
    int32_t    oob_weighting;  /* per-tree OOB weighting */
    int32_t    leaf_model;     /* 0=constant, 1=local linear (regression only) */
    double     sample_rate;    /* bootstrap/subsample fraction */
    double     alpha_trim;     /* cost-complexity pruning penalty */
    /* v0.3 fields */
    int32_t   *monotonic_cst; /* per-feature monotonic constraints (-1/0/+1), NULL if none */
    int32_t    store_leaf_samples; /* 1 if quantile RF leaf storage is enabled */
    /* Conformal: OOB residuals stored during fit */
    double    *oob_predictions; /* per-training-sample OOB predictions, length n_train */
    int32_t   *oob_counts;      /* per-training-sample OOB tree counts, length n_train */
    int32_t    n_train;          /* number of training samples (for conformal) */
    const double *y_train;       /* pointer to training labels (NOT owned, set during fit) */
    double    *y_train_copy;     /* owned copy of training labels for conformal */
    double    *sample_weight_copy; /* owned copy of sample weights (for weighted quantile) */
    /* v0.4 fields */
    double    *jarf_rotation;  /* ncol * ncol rotation matrix, NULL if no JARF */
    int32_t    jarf_ncol;      /* columns in rotation matrix (must == n_features) */
} rf_forest_t;

/* ---------- Histogram binning ---------- */

typedef struct {
    uint8_t  *binned;     /* nrow * ncol, row-major: binned[i*ncol+j] = bin index for sample i, feature j */
    double   *bin_edges;  /* ncol * (max_bins-1) thresholds: bin_edges[j*(max_bins-1)+k] */
    int32_t  *n_bins;     /* per-feature actual bin count */
    int32_t   max_bins;
    int32_t   ncol;
    int32_t   nrow;
} rf_bins_t;

/* ---------- Hyperparameters ---------- */

typedef struct {
    int32_t  n_estimators;       /* default 100 */
    int32_t  max_depth;          /* 0 = unlimited */
    int32_t  min_samples_split;  /* default 2 */
    int32_t  min_samples_leaf;   /* default 1 */
    int32_t  max_features;       /* 0 = auto (sqrt for cls, n/3 for reg) */
    int32_t  max_leaf_nodes;     /* 0 = unlimited */
    int32_t  bootstrap;          /* default 1 (true) */
    int32_t  compute_oob;        /* default 1 (true) */
    int32_t  extra_trees;        /* default 0 (false); 1 = ExtraTrees */
    uint32_t seed;               /* default 42 */
    int32_t  task;               /* 0 = classification, 1 = regression */
    /* New params (v0.2) -- defaults preserve v0.1 behavior */
    int32_t  criterion;          /* cls: 0=gini,1=entropy,2=hellinger; reg: 0=mse,1=mae */
    int32_t  heterogeneous;      /* 0=off, 1=depth-dependent feature weighting */
    int32_t  oob_weighting;      /* 0=off, 1=weight trees by OOB performance */
    int32_t  leaf_model;         /* 0=constant, 1=local linear (regression only) */
    int32_t  store_leaf_samples; /* 0=off, 1=on (enables quantile prediction) */
    double   sample_rate;        /* bootstrap/subsample fraction, default 1.0 */
    double   alpha_trim;         /* cost-complexity pruning penalty, default 0.0 */
    /* v0.3 params */
    int32_t *monotonic_cst;      /* per-feature constraints: -1=decreasing, 0=none, +1=increasing */
    int32_t  n_monotonic_cst;    /* length of monotonic_cst array (must equal ncol) */
    /* v0.4 params */
    double  *sample_weight;      /* per-sample weights (NULL = uniform), length nrow */
    int32_t  n_sample_weight;    /* length of sample_weight array (must equal nrow) */
    int32_t  histogram;          /* 0=off, 1=histogram binning for split search */
    int32_t  max_bins;           /* max histogram bins (2-256), default 256 */
    int32_t  jarf;               /* 0=off, 1=JARF rotation (Jacobian Aligned RF) */
    int32_t  jarf_n_estimators;  /* surrogate RF trees, default 50 */
    int32_t  jarf_max_depth;     /* surrogate RF max depth, default 6 */
} rf_params_t;

/* ---------- Public API ---------- */

/* Initialize params to defaults */
void rf_params_init(rf_params_t *params);

/* Build a random forest.
 * X: row-major float64, nrow * ncol
 * y: float64 (classification: integer-valued class labels 0..K-1;
 *              regression: continuous values)
 * Returns NULL on error (check rf_get_error()) */
rf_forest_t *rf_fit(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y,
    const rf_params_t *params
);

/* Predict class labels (classification) or values (regression).
 * out: float64 array of length nrow, caller-allocated.
 * Returns 0 on success, -1 on error. */
int rf_predict(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Predict class probabilities (classification only).
 * out: float64 array of length nrow * n_classes, caller-allocated.
 * Row-major: out[i * n_classes + c] = P(class c | X[i])
 * Returns 0 on success, -1 on error. */
int rf_predict_proba(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Serialize forest to binary blob (RF01 format).
 * *out_buf receives malloc'd buffer (caller must free with rf_free_buffer).
 * *out_len receives byte length.
 * Returns 0 on success, -1 on error. */
int rf_save(const rf_forest_t *forest, char **out_buf, int32_t *out_len);

/* Deserialize forest from RF01 binary blob.
 * Returns NULL on error (check rf_get_error()). */
rf_forest_t *rf_load(const char *buf, int32_t len);

/* Predict quantiles (quantile regression forest).
 * Requires store_leaf_samples=1 during fit.
 * quantiles: array of quantile values in [0,1], length n_quantiles.
 * out: float64 array of length nrow * n_quantiles, caller-allocated.
 *   out[i * n_quantiles + q] = predicted quantile q for sample i.
 * Returns 0 on success, -1 on error. */
int rf_predict_quantile(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    const double *quantiles, int32_t n_quantiles,
    double *out
);

/* Predict conformal intervals (Jackknife+-after-Bootstrap).
 * Requires bootstrap=1, compute_oob=1 during fit.
 * alpha: miscoverage rate (e.g. 0.1 for 90% intervals).
 * out_lower, out_upper: float64 arrays of length nrow, caller-allocated.
 * Returns 0 on success, -1 on error. */
int rf_predict_interval(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    double alpha,
    double *out_lower,
    double *out_upper
);

/* Permutation feature importance (model-agnostic, unbiased).
 * For each feature, shuffles it n_repeats times, re-predicts, measures score drop.
 * out: float64 array of length ncol, caller-allocated.
 * Returns 0 on success, -1 on error. */
int rf_permutation_importance(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    const double *y, int32_t n_repeats, uint32_t seed,
    double *out
);

/* Compute proximity matrix.
 * For each pair of samples, counts how often they land in the same leaf.
 * out: float64 array of length nrow * nrow, caller-allocated.
 * Row-major: out[i * nrow + j] = proximity(i, j) in [0, 1].
 * Returns 0 on success, -1 on error. */
int rf_proximity(
    const rf_forest_t *forest,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Free forest and all owned memory. */
void rf_free(rf_forest_t *forest);

/* Free a buffer returned by rf_save. */
void rf_free_buffer(void *ptr);

/* Get last error message. */
const char *rf_get_error(void);

#endif /* RF_H */
