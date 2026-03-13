/*
 * wl_api.c -- C ABI wrapper for RF (WASM/FFI boundary)
 *
 * All params as primitives (no struct passing across ABI).
 * Follows the nanoflann-wasm wl_api pattern.
 */

#include "rf.h"

const char *wl_rf_get_last_error(void) {
    return rf_get_error();
}

rf_forest_t *wl_rf_fit(
    const double *X, int nrow, int ncol,
    const double *y,
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int max_leaf_nodes,
    int bootstrap,
    int compute_oob,
    int extra_trees,
    int seed,
    int task,
    int criterion,
    double sample_rate,
    int heterogeneous,
    int oob_weighting,
    double alpha_trim,
    int leaf_model,
    int store_leaf_samples,
    const int32_t *monotonic_cst,
    int n_monotonic_cst,
    const double *sample_weight,
    int n_sample_weight,
    int histogram,
    int max_bins,
    int jarf,
    int jarf_n_estimators,
    int jarf_max_depth
) {
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = n_estimators;
    params.max_depth = max_depth;
    params.min_samples_split = min_samples_split;
    params.min_samples_leaf = min_samples_leaf;
    params.max_features = max_features;
    params.max_leaf_nodes = max_leaf_nodes;
    params.bootstrap = bootstrap;
    params.compute_oob = compute_oob;
    params.extra_trees = extra_trees;
    params.seed = (uint32_t)seed;
    params.task = task;
    params.criterion = criterion;
    params.sample_rate = sample_rate;
    params.heterogeneous = heterogeneous;
    params.oob_weighting = oob_weighting;
    params.alpha_trim = alpha_trim;
    params.leaf_model = leaf_model;
    params.store_leaf_samples = store_leaf_samples;
    params.monotonic_cst = (int32_t *)monotonic_cst;
    params.n_monotonic_cst = n_monotonic_cst;
    params.sample_weight = (double *)sample_weight;
    params.n_sample_weight = n_sample_weight;
    params.histogram = histogram;
    params.max_bins = max_bins;
    params.jarf = jarf;
    params.jarf_n_estimators = jarf_n_estimators;
    params.jarf_max_depth = jarf_max_depth;
    return rf_fit(X, (int32_t)nrow, (int32_t)ncol, y, &params);
}

int wl_rf_predict(const rf_forest_t *f, const double *X, int nrow, int ncol, double *out) {
    return rf_predict(f, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_rf_predict_proba(const rf_forest_t *f, const double *X, int nrow, int ncol, double *out) {
    return rf_predict_proba(f, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_rf_save(const rf_forest_t *f, char **out_buf, int *out_len) {
    int32_t len32;
    int ret = rf_save(f, out_buf, &len32);
    if (ret == 0 && out_len) *out_len = (int)len32;
    return ret;
}

rf_forest_t *wl_rf_load(const char *buf, int len) {
    return rf_load(buf, (int32_t)len);
}

void wl_rf_free(rf_forest_t *f) {
    rf_free(f);
}

void wl_rf_free_buffer(void *ptr) {
    rf_free_buffer(ptr);
}

int wl_rf_get_n_trees(const rf_forest_t *f) {
    return f ? f->n_trees : 0;
}

int wl_rf_get_n_features(const rf_forest_t *f) {
    return f ? f->n_features : 0;
}

int wl_rf_get_n_classes(const rf_forest_t *f) {
    return f ? f->n_classes : 0;
}

int wl_rf_get_task(const rf_forest_t *f) {
    return f ? f->task : -1;
}

double wl_rf_get_oob_score(const rf_forest_t *f) {
    return f ? f->oob_score : 0.0;
}

double wl_rf_get_feature_importance(const rf_forest_t *f, int i) {
    if (!f || !f->feature_importances || i < 0 || i >= f->n_features) return 0.0;
    return f->feature_importances[i];
}

int wl_rf_get_criterion(const rf_forest_t *f) {
    return f ? f->criterion : 0;
}

double wl_rf_get_sample_rate(const rf_forest_t *f) {
    return f ? f->sample_rate : 1.0;
}

int wl_rf_get_heterogeneous(const rf_forest_t *f) {
    return f ? f->heterogeneous : 0;
}

int wl_rf_get_oob_weighting(const rf_forest_t *f) {
    return f ? f->oob_weighting : 0;
}

double wl_rf_get_alpha_trim(const rf_forest_t *f) {
    return f ? f->alpha_trim : 0.0;
}

int wl_rf_get_leaf_model(const rf_forest_t *f) {
    return f ? f->leaf_model : 0;
}

int wl_rf_predict_quantile(const rf_forest_t *f, const double *X, int nrow, int ncol,
                           const double *quantiles, int n_quantiles, double *out) {
    return rf_predict_quantile(f, X, (int32_t)nrow, (int32_t)ncol,
                               quantiles, (int32_t)n_quantiles, out);
}

int wl_rf_predict_interval(const rf_forest_t *f, const double *X, int nrow, int ncol,
                           double alpha, double *out_lower, double *out_upper) {
    return rf_predict_interval(f, X, (int32_t)nrow, (int32_t)ncol,
                               alpha, out_lower, out_upper);
}

int wl_rf_permutation_importance(const rf_forest_t *f, const double *X, int nrow, int ncol,
                                  const double *y, int n_repeats, int seed, double *out) {
    return rf_permutation_importance(f, X, (int32_t)nrow, (int32_t)ncol,
                                      y, (int32_t)n_repeats, (uint32_t)seed, out);
}

int wl_rf_proximity(const rf_forest_t *f, const double *X, int nrow, int ncol, double *out) {
    return rf_proximity(f, X, (int32_t)nrow, (int32_t)ncol, out);
}
