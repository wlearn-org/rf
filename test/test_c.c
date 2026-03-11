/*
 * test_c.c -- Smoke tests for RF C core
 */
#include "rf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

static void test_classification(void) {
    printf("=== Classification ===\n");

    /* Simple 2D dataset: class 0 if x0 < 0.5, class 1 otherwise */
    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.max_features = d;  /* try all features so importance test is meaningful */
    params.seed = 42;
    params.task = 0;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "rf_fit returns non-NULL");
    ASSERT(forest->n_trees == 20, "20 trees");
    ASSERT(forest->n_classes == 2, "2 classes");
    ASSERT(!isnan(forest->oob_score), "OOB score computed");
    printf("  OOB accuracy: %.3f\n", forest->oob_score);
    ASSERT(forest->oob_score > 0.7, "OOB accuracy > 0.7");

    /* Predict */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    int ret = rf_predict(forest, X, n, d, preds);
    ASSERT(ret == 0, "predict returns 0");

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  Train accuracy: %.3f\n", acc);
    ASSERT(acc > 0.85, "train accuracy > 0.85");

    /* predict_proba */
    double *proba = (double *)malloc((size_t)n * 2 * sizeof(double));
    ret = rf_predict_proba(forest, X, n, d, proba);
    ASSERT(ret == 0, "predict_proba returns 0");

    /* Check row sums */
    int valid_proba = 1;
    for (int i = 0; i < n; i++) {
        double sum = proba[i * 2] + proba[i * 2 + 1];
        if (fabs(sum - 1.0) > 1e-10) { valid_proba = 0; break; }
    }
    ASSERT(valid_proba, "proba rows sum to 1");

    /* Feature importance */
    double imp_sum = 0;
    for (int f = 0; f < d; f++) imp_sum += forest->feature_importances[f];
    printf("  Importances: [%.4f, %.4f] sum=%.4f\n",
           forest->feature_importances[0], forest->feature_importances[1], imp_sum);
    ASSERT(fabs(imp_sum - 1.0) < 1e-10, "importances sum to 1");
    ASSERT(forest->feature_importances[0] > forest->feature_importances[1],
           "feature 0 more important");

    /* Save/load round-trip */
    char *blob = NULL;
    int32_t blob_len = 0;
    ret = rf_save(forest, &blob, &blob_len);
    ASSERT(ret == 0, "save returns 0");
    ASSERT(blob_len > 0, "blob non-empty");
    ASSERT(memcmp(blob, "RF01", 4) == 0, "RF01 magic");

    rf_forest_t *loaded = rf_load(blob, blob_len);
    ASSERT(loaded != NULL, "load returns non-NULL");
    ASSERT(loaded->n_trees == 20, "loaded 20 trees");

    double *preds2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(loaded, X, n, d, preds2);
    int match = 1;
    for (int i = 0; i < n; i++) {
        if (preds[i] != preds2[i]) { match = 0; break; }
    }
    ASSERT(match, "loaded predictions match");

    rf_free_buffer(blob);
    rf_free(loaded);
    free(preds2);
    free(proba);
    free(preds);
    rf_free(forest);
    free(X);
    free(y);
}

static void test_regression(void) {
    printf("=== Regression ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 123 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = 3.0 * X[i * 2] + 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.max_features = d;  /* try all features so importance test is meaningful */
    params.seed = 123;
    params.task = 1;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "rf_fit regression");
    ASSERT(!isnan(forest->oob_score), "OOB R2 computed");
    printf("  OOB R2: %.3f\n", forest->oob_score);

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);

    /* Compute R2 */
    double y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  Train R2: %.3f\n", r2);
    ASSERT(r2 > 0.5, "R2 > 0.5");

    /* Feature 0 should be more important */
    ASSERT(forest->feature_importances[0] > forest->feature_importances[1],
           "feature 0 more important (regression)");

    /* predict_proba should fail */
    double proba[2];
    int ret = rf_predict_proba(forest, X, 1, d, proba);
    ASSERT(ret == -1, "predict_proba fails for regression");

    free(preds);
    rf_free(forest);
    free(X);
    free(y);
}

static void test_extra_trees(void) {
    printf("=== ExtraTrees ===\n");

    int n = 80, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 77 };

    for (int i = 0; i < n; i++) {
        X[i * 3] = rf_rng_uniform(&rng);
        X[i * 3 + 1] = rf_rng_uniform(&rng);
        X[i * 3 + 2] = rf_rng_uniform(&rng);
        y[i] = (X[i * 3] + X[i * 3 + 1] > 1.0) ? 1.0 : 0.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 30;
    params.seed = 77;
    params.task = 0;
    params.extra_trees = 1;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "ExtraTrees fit");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  ExtraTrees accuracy: %.3f\n", acc);
    ASSERT(acc > 0.7, "ExtraTrees accuracy > 0.7");

    free(preds);
    rf_free(forest);
    free(X);
    free(y);
}

static void test_determinism(void) {
    printf("=== Determinism ===\n");

    int n = 50, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 99 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] > 0.5) ? 1.0 : 0.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 10;
    params.seed = 99;
    params.task = 0;

    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params);
    ASSERT(f1 && f2, "both fits succeed");

    double *p1 = (double *)malloc((size_t)n * sizeof(double));
    double *p2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, p1);
    rf_predict(f2, X, n, d, p2);

    int match = 1;
    for (int i = 0; i < n; i++) {
        if (p1[i] != p2[i]) { match = 0; break; }
    }
    ASSERT(match, "same seed -> same predictions");

    free(p1); free(p2);
    rf_free(f1); rf_free(f2);
    free(X); free(y);
}

/* === New v0.2 feature tests === */

static void test_entropy_criterion(void) {
    printf("=== Entropy Criterion ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 42;
    params.task = 0;
    params.criterion = 1;  /* entropy */

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "entropy fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  Entropy accuracy: %.3f\n", acc);
    ASSERT(acc > 0.85, "entropy accuracy > 0.85");

    free(preds);
    rf_free(forest);
    free(X); free(y);
}

static void test_hellinger_criterion(void) {
    printf("=== Hellinger Criterion ===\n");

    /* Imbalanced dataset: 90% class 0, 10% class 1 */
    int n = 200, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 55 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        if (i < 20) {
            y[i] = 1.0;
            X[i * 2] = 0.7 + rf_rng_uniform(&rng) * 0.3;
        } else {
            y[i] = 0.0;
        }
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 30;
    params.seed = 55;
    params.task = 0;
    params.criterion = 2;  /* hellinger */

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "hellinger fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  Hellinger accuracy: %.3f\n", acc);
    ASSERT(acc > 0.8, "hellinger accuracy > 0.8");

    free(preds);
    rf_free(forest);
    free(X); free(y);
}

static void test_mae_criterion(void) {
    printf("=== MAE Criterion ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 88 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = 3.0 * X[i * 2] + 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 88;
    params.task = 1;
    params.criterion = 1;  /* mae */

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "mae fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);
    double y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  MAE R2: %.3f\n", r2);
    ASSERT(r2 > 0.5, "mae R2 > 0.5");

    free(preds);
    rf_free(forest);
    free(X); free(y);
}

static void test_sample_rate(void) {
    printf("=== Sample Rate ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    /* sample_rate=0.5 with bootstrap */
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 42;
    params.task = 0;
    params.sample_rate = 0.5;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "sample_rate=0.5 fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  sample_rate=0.5 accuracy: %.3f\n", acc);
    ASSERT(acc > 0.8, "sample_rate=0.5 accuracy > 0.8");

    /* sample_rate=0.5 without bootstrap */
    params.bootstrap = 0;
    rf_forest_t *forest2 = rf_fit(X, n, d, y, &params);
    ASSERT(forest2 != NULL, "sample_rate=0.5 no-bootstrap fit succeeds");

    rf_predict(forest2, X, n, d, preds);
    correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    acc = (double)correct / n;
    printf("  sample_rate=0.5 no-bootstrap accuracy: %.3f\n", acc);
    ASSERT(acc > 0.8, "sample_rate=0.5 no-bootstrap accuracy > 0.8");

    free(preds);
    rf_free(forest);
    rf_free(forest2);
    free(X); free(y);
}

static void test_heterogeneous_rf(void) {
    printf("=== Heterogeneous RF ===\n");

    int n = 150, d = 4;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 33 };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = rf_rng_uniform(&rng);
        y[i] = 5.0 * X[i * d] + 2.0 * X[i * d + 1];
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 30;
    params.seed = 33;
    params.task = 1;
    params.heterogeneous = 1;
    params.max_features = d;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "HRF fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, preds);
    double y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  HRF R2: %.3f\n", r2);
    ASSERT(r2 > 0.5, "HRF R2 > 0.5");

    free(preds);
    rf_free(forest);
    free(X); free(y);
}

static void test_oob_weighting(void) {
    printf("=== OOB Weighting ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 42;
    params.task = 0;
    params.oob_weighting = 1;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    ASSERT(forest != NULL, "oob_weighting fit succeeds");
    ASSERT(forest->tree_weights != NULL, "tree_weights allocated");

    /* Check tree_weights are reasonable */
    double tw_sum = 0;
    for (int t = 0; t < forest->n_trees; t++) {
        ASSERT(forest->tree_weights[t] >= 0, "tree weight >= 0");
        tw_sum += forest->tree_weights[t];
    }
    printf("  tree_weights sum: %.3f (expect %d)\n", tw_sum, forest->n_trees);
    ASSERT(fabs(tw_sum - forest->n_trees) < 1e-6, "tree_weights sum to n_trees");

    /* Save/load preserves tree_weights */
    char *blob = NULL;
    int32_t blob_len = 0;
    rf_save(forest, &blob, &blob_len);
    rf_forest_t *loaded = rf_load(blob, blob_len);
    ASSERT(loaded != NULL, "oob_weighting load succeeds");
    ASSERT(loaded->tree_weights != NULL, "loaded tree_weights non-NULL");
    ASSERT(loaded->oob_weighting == 1, "loaded oob_weighting == 1");

    int tw_match = 1;
    for (int t = 0; t < forest->n_trees; t++) {
        if (fabs(forest->tree_weights[t] - loaded->tree_weights[t]) > 1e-10) {
            tw_match = 0;
            break;
        }
    }
    ASSERT(tw_match, "tree_weights preserved after save/load");

    rf_free_buffer(blob);
    rf_free(loaded);
    rf_free(forest);
    free(X); free(y);
}

static void test_alpha_trim(void) {
    printf("=== Alpha Trim ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    /* Without pruning */
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 42;
    params.task = 0;

    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);
    int32_t nodes_no_prune = 0;
    for (int t = 0; t < f1->n_trees; t++) nodes_no_prune += f1->trees[t].n_nodes;

    /* With pruning */
    params.alpha_trim = 0.05;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params);
    ASSERT(f2 != NULL, "alpha_trim fit succeeds");
    int32_t nodes_pruned = 0;
    for (int t = 0; t < f2->n_trees; t++) nodes_pruned += f2->trees[t].n_nodes;

    printf("  Nodes without pruning: %d, with pruning: %d\n", nodes_no_prune, nodes_pruned);
    /* Pruning should reduce or maintain tree size */
    ASSERT(nodes_pruned <= nodes_no_prune, "pruning reduces or maintains node count");

    /* Should still produce valid predictions */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f2, X, n, d, preds);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  Alpha trim accuracy: %.3f\n", acc);
    ASSERT(acc > 0.7, "alpha_trim accuracy > 0.7");

    free(preds);
    rf_free(f1);
    rf_free(f2);
    free(X); free(y);
}

static void test_leaf_model_linear(void) {
    printf("=== Local Linear Leaves ===\n");

    /* Linear signal: y = 3*x0 + 2*x1 + 1 */
    int n = 200, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 66 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng) * 4 - 2;
        X[i * 2 + 1] = rf_rng_uniform(&rng) * 4 - 2;
        y[i] = 3.0 * X[i * 2] + 2.0 * X[i * 2 + 1] + 1.0;
    }

    /* Constant leaves -- use shallow trees so constant model struggles */
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 66;
    params.task = 1;
    params.leaf_model = 0;
    params.max_depth = 3;  /* shallow: forces larger leaves */

    rf_forest_t *f_const = rf_fit(X, n, d, y, &params);
    double *p_const = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f_const, X, n, d, p_const);
    double y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    double ss_res_c = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        ss_res_c += (y[i] - p_const[i]) * (y[i] - p_const[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2_const = 1.0 - ss_res_c / ss_tot;

    /* Linear leaves */
    params.leaf_model = 1;
    rf_forest_t *f_lin = rf_fit(X, n, d, y, &params);
    ASSERT(f_lin != NULL, "leaf_model=1 fit succeeds");

    double *p_lin = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f_lin, X, n, d, p_lin);
    double ss_res_l = 0;
    for (int i = 0; i < n; i++) {
        ss_res_l += (y[i] - p_lin[i]) * (y[i] - p_lin[i]);
    }
    double r2_lin = 1.0 - ss_res_l / ss_tot;

    printf("  Constant R2: %.4f, Linear R2: %.4f\n", r2_const, r2_lin);
    ASSERT(r2_lin > r2_const, "linear leaves improve R2 on linear signal");
    ASSERT(r2_lin > 0.95, "linear leaves R2 > 0.95 on pure linear signal");

    /* Save/load round-trip */
    char *blob = NULL;
    int32_t blob_len = 0;
    rf_save(f_lin, &blob, &blob_len);
    rf_forest_t *loaded = rf_load(blob, blob_len);
    ASSERT(loaded != NULL, "linear leaf load succeeds");
    ASSERT(loaded->leaf_model == 1, "loaded leaf_model == 1");

    double *p_loaded = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(loaded, X, n, d, p_loaded);
    int match = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(p_lin[i] - p_loaded[i]) > 1e-10) { match = 0; break; }
    }
    ASSERT(match, "linear leaf predictions preserved after save/load");

    rf_free_buffer(blob);
    rf_free(loaded);
    free(p_loaded);
    free(p_const); free(p_lin);
    rf_free(f_const); rf_free(f_lin);
    free(X); free(y);
}

static void test_defaults_unchanged(void) {
    printf("=== Defaults Unchanged ===\n");

    /* Verify that default params produce same results as before v0.2 */
    rf_params_t params;
    rf_params_init(&params);
    ASSERT(params.criterion == 0, "default criterion == 0 (gini/mse)");
    ASSERT(params.heterogeneous == 0, "default heterogeneous == 0");
    ASSERT(params.oob_weighting == 0, "default oob_weighting == 0");
    ASSERT(params.leaf_model == 0, "default leaf_model == 0");
    ASSERT(fabs(params.sample_rate - 1.0) < 1e-15, "default sample_rate == 1.0");
    ASSERT(fabs(params.alpha_trim) < 1e-15, "default alpha_trim == 0.0");
}

static void test_v2_save_load(void) {
    printf("=== V3 Save/Load ===\n");

    int n = 80, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 15;
    params.seed = 42;
    params.task = 0;
    params.criterion = 1;  /* entropy */
    params.sample_rate = 0.8;

    rf_forest_t *forest = rf_fit(X, n, d, y, &params);
    char *blob = NULL;
    int32_t blob_len = 0;
    rf_save(forest, &blob, &blob_len);

    /* Check version byte */
    uint32_t ver;
    memcpy(&ver, blob + 4, 4);
    ASSERT(ver == 3, "saved format version == 3");

    rf_forest_t *loaded = rf_load(blob, blob_len);
    ASSERT(loaded != NULL, "v2 load succeeds");
    ASSERT(loaded->criterion == 1, "loaded criterion == 1");
    ASSERT(fabs(loaded->sample_rate - 0.8) < 1e-10, "loaded sample_rate == 0.8");

    /* Predictions should match */
    double *p1 = (double *)malloc((size_t)n * sizeof(double));
    double *p2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(forest, X, n, d, p1);
    rf_predict(loaded, X, n, d, p2);
    int match = 1;
    for (int i = 0; i < n; i++) {
        if (p1[i] != p2[i]) { match = 0; break; }
    }
    ASSERT(match, "v2 predictions match after save/load");

    free(p1); free(p2);
    rf_free_buffer(blob);
    rf_free(loaded);
    rf_free(forest);
    free(X); free(y);
}

int main(void) {
    test_classification();
    test_regression();
    test_extra_trees();
    test_determinism();
    test_entropy_criterion();
    test_hellinger_criterion();
    test_mae_criterion();
    test_sample_rate();
    test_heterogeneous_rf();
    test_oob_weighting();
    test_alpha_trim();
    test_leaf_model_linear();
    test_defaults_unchanged();
    test_v2_save_load();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
