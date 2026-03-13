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
    ASSERT(ver == 4, "saved format version == 4");

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

static void test_sample_weight_cls(void) {
    printf("=== Sample Weight (Classification) ===\n");

    /* Dataset where class 0 has 80 samples, class 1 has 20 samples.
     * Without weights: biased toward class 0.
     * With weights: class 1 upweighted 4x to balance. */
    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *sw = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    /* Class 0: x0 < 0.5, class 1: x0 >= 0.5 */
    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = (X[i * 2] < 0.5) ? 0.0 : 1.0;
        /* Upweight class 1 samples */
        sw[i] = (y[i] == 1.0) ? 4.0 : 1.0;
    }

    /* Fit without weights */
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 20;
    params.seed = 42;
    params.task = 0;

    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);
    ASSERT(f1 != NULL, "sw cls: fit without weights");

    /* Fit with weights */
    params.sample_weight = sw;
    params.n_sample_weight = n;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params);
    ASSERT(f2 != NULL, "sw cls: fit with weights");

    /* Both should produce valid predictions */
    double *preds1 = (double *)malloc((size_t)n * sizeof(double));
    double *preds2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, preds1);
    rf_predict(f2, X, n, d, preds2);

    int correct1 = 0, correct2 = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds1[i] == (int)y[i]) correct1++;
        if ((int)preds2[i] == (int)y[i]) correct2++;
    }
    ASSERT(correct1 > 80, "sw cls: unweighted accuracy > 0.8");
    ASSERT(correct2 > 80, "sw cls: weighted accuracy > 0.8");
    printf("  Unweighted acc: %.3f, Weighted acc: %.3f\n",
           (double)correct1 / n, (double)correct2 / n);

    /* predict_proba should still sum to 1 */
    double *proba = (double *)malloc((size_t)n * 2 * sizeof(double));
    rf_predict_proba(f2, X, n, d, proba);
    int valid_proba = 1;
    for (int i = 0; i < n; i++) {
        double sum = proba[i * 2] + proba[i * 2 + 1];
        if (fabs(sum - 1.0) > 1e-10) { valid_proba = 0; break; }
    }
    ASSERT(valid_proba, "sw cls: proba rows sum to 1");

    /* Feature importance should still sum to 1 */
    double imp_sum = f2->feature_importances[0] + f2->feature_importances[1];
    ASSERT(fabs(imp_sum - 1.0) < 1e-10 || imp_sum == 0.0,
           "sw cls: importances sum to 1");

    /* OOB score should be computed */
    ASSERT(!isnan(f2->oob_score), "sw cls: OOB score computed");

    free(proba);
    free(preds1); free(preds2);
    rf_free(f1); rf_free(f2);
    free(X); free(y); free(sw);
}

static void test_sample_weight_reg(void) {
    printf("=== Sample Weight (Regression) ===\n");

    /* Linear dataset with noise. Weights upweight low-noise samples. */
    int n = 100, d = 1;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *sw = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        X[i] = rf_rng_uniform(&rng) * 10.0;
        double noise = (rf_rng_uniform(&rng) - 0.5) * 2.0;
        y[i] = X[i] * 2.0 + 1.0 + noise;
        sw[i] = (i < 50) ? 5.0 : 1.0;  /* first 50 samples get 5x weight */
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 50;
    params.seed = 42;
    params.task = 1;

    /* Fit without weights */
    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);
    ASSERT(f1 != NULL, "sw reg: fit without weights");
    ASSERT(!isnan(f1->oob_score), "sw reg: unweighted OOB computed");

    /* Fit with weights */
    params.sample_weight = sw;
    params.n_sample_weight = n;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params);
    ASSERT(f2 != NULL, "sw reg: fit with weights");
    ASSERT(!isnan(f2->oob_score), "sw reg: weighted OOB computed");
    printf("  Unweighted OOB R2: %.3f, Weighted OOB R2: %.3f\n",
           f1->oob_score, f2->oob_score);

    /* Both should predict reasonably */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f2, X, n, d, preds);
    double ss_res = 0, ss_tot = 0, ymean = 0;
    for (int i = 0; i < n; i++) ymean += y[i];
    ymean /= n;
    for (int i = 0; i < n; i++) {
        double d2 = y[i] - preds[i]; ss_res += d2 * d2;
        double dm = y[i] - ymean; ss_tot += dm * dm;
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  Weighted train R2: %.3f\n", r2);
    ASSERT(r2 > 0.8, "sw reg: weighted train R2 > 0.8");

    free(preds);
    rf_free(f1); rf_free(f2);
    free(X); free(y); free(sw);
}

static void test_sample_weight_uniform(void) {
    printf("=== Sample Weight (Uniform=No Weight) ===\n");

    /* Uniform weights of 1.0 should produce identical results to no weights */
    int n = 50, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *sw = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = X[i * 2] * 3.0 + X[i * 2 + 1] * 2.0;
        sw[i] = 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 10;
    params.seed = 42;
    params.task = 1;

    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);

    params.sample_weight = sw;
    params.n_sample_weight = n;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params);

    ASSERT(f1 != NULL && f2 != NULL, "sw uniform: both fits succeed");

    /* Predictions should be identical */
    double *p1 = (double *)malloc((size_t)n * sizeof(double));
    double *p2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, p1);
    rf_predict(f2, X, n, d, p2);
    int match = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(p1[i] - p2[i]) > 1e-10) { match = 0; break; }
    }
    ASSERT(match, "sw uniform: predictions match no-weight case");

    free(p1); free(p2);
    rf_free(f1); rf_free(f2);
    free(X); free(y); free(sw);
}

static void test_sample_weight_zero(void) {
    printf("=== Sample Weight (Zero weights) ===\n");

    /* Samples with weight 0 should be effectively ignored */
    int n = 60, d = 1;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *sw = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    /* First 30: true signal y = x. Weight = 1.
     * Last 30: garbage y = -100. Weight = 0. */
    for (int i = 0; i < 30; i++) {
        X[i] = rf_rng_uniform(&rng) * 10.0;
        y[i] = X[i];
        sw[i] = 1.0;
    }
    for (int i = 30; i < 60; i++) {
        X[i] = rf_rng_uniform(&rng) * 10.0;
        y[i] = -100.0;  /* garbage */
        sw[i] = 0.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 50;
    params.seed = 42;
    params.task = 1;
    params.sample_weight = sw;
    params.n_sample_weight = n;

    rf_forest_t *f = rf_fit(X, n, d, y, &params);
    ASSERT(f != NULL, "sw zero: fit succeeds");

    /* Predictions on the good samples should be reasonable */
    double *preds = (double *)malloc(30 * sizeof(double));
    rf_predict(f, X, 30, d, preds);
    double max_err = 0;
    for (int i = 0; i < 30; i++) {
        double err = fabs(preds[i] - y[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max error on good samples: %.3f\n", max_err);
    /* Zero-weight garbage samples shouldn't affect leaf predictions too much */
    ASSERT(max_err < 5.0, "sw zero: max error on good samples < 5");

    free(preds);
    rf_free(f);
    free(X); free(y); free(sw);
}

static void test_histogram_regression(void) {
    printf("=== Histogram Binning (Regression) ===\n");

    int n = 500, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = rf_rng_uniform(&rng) * 10.0 - 5.0;
        y[i] = X[i * d] * 2.0 + X[i * d + 1] + rf_rng_uniform(&rng) * 0.5;
    }

    /* Standard */
    rf_params_t p1;
    rf_params_init(&p1);
    p1.n_estimators = 50;
    p1.seed = 42;
    p1.task = 1;
    rf_forest_t *f1 = rf_fit(X, n, d, y, &p1);

    /* Histogram */
    rf_params_t p2;
    rf_params_init(&p2);
    p2.n_estimators = 50;
    p2.seed = 42;
    p2.task = 1;
    p2.histogram = 1;
    p2.max_bins = 256;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &p2);

    ASSERT(f1 && f2, "hist reg: both fits succeed");

    double *pred1 = (double *)malloc((size_t)n * sizeof(double));
    double *pred2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, pred1);
    rf_predict(f2, X, n, d, pred2);

    /* R2 should be similar */
    double ss_res1 = 0, ss_res2 = 0, ss_tot = 0, ymean = 0;
    for (int i = 0; i < n; i++) ymean += y[i];
    ymean /= n;
    for (int i = 0; i < n; i++) {
        ss_res1 += (y[i] - pred1[i]) * (y[i] - pred1[i]);
        ss_res2 += (y[i] - pred2[i]) * (y[i] - pred2[i]);
        ss_tot += (y[i] - ymean) * (y[i] - ymean);
    }
    double r2_std = 1.0 - ss_res1 / ss_tot;
    double r2_bin = 1.0 - ss_res2 / ss_tot;
    printf("  Standard R2: %.3f, Histogram R2: %.3f\n", r2_std, r2_bin);
    ASSERT(r2_bin > 0.9, "hist reg: R2 > 0.9");
    ASSERT(fabs(r2_std - r2_bin) < 0.05, "hist reg: R2 close to standard");

    free(pred1); free(pred2);
    rf_free(f1); rf_free(f2);
    free(X); free(y);
}

static void test_histogram_classification(void) {
    printf("=== Histogram Binning (Classification) ===\n");

    int n = 300, d = 4;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = rf_rng_uniform(&rng) * 4.0 - 2.0;
        y[i] = (X[i * d] + X[i * d + 1] > 0) ? 1.0 : 0.0;
    }

    rf_params_t p;
    rf_params_init(&p);
    p.n_estimators = 50;
    p.seed = 42;
    p.task = 0;
    p.histogram = 1;
    rf_forest_t *f = rf_fit(X, n, d, y, &p);
    ASSERT(f != NULL, "hist cls: fit succeeds");

    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f, X, n, d, preds);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  Histogram cls accuracy: %.3f\n", acc);
    ASSERT(acc > 0.9, "hist cls: accuracy > 0.9");

    free(preds); rf_free(f); free(X); free(y);
}

static void test_histogram_save_load(void) {
    printf("=== Histogram Save/Load ===\n");

    int n = 100, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng);
        X[i * 2 + 1] = rf_rng_uniform(&rng);
        y[i] = X[i * 2] * 3.0 + X[i * 2 + 1];
    }

    rf_params_t p;
    rf_params_init(&p);
    p.n_estimators = 10;
    p.seed = 42;
    p.task = 1;
    p.histogram = 1;
    rf_forest_t *f1 = rf_fit(X, n, d, y, &p);
    ASSERT(f1 != NULL, "hist save: fit succeeds");

    char *buf = NULL;
    int32_t buf_len = 0;
    int ret = rf_save(f1, &buf, &buf_len);
    ASSERT(ret == 0, "hist save: save succeeds");

    rf_forest_t *f2 = rf_load(buf, buf_len);
    ASSERT(f2 != NULL, "hist save: load succeeds");

    double *p1 = (double *)malloc((size_t)n * sizeof(double));
    double *p2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, p1);
    rf_predict(f2, X, n, d, p2);

    int match = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(p1[i] - p2[i]) > 1e-10) { match = 0; break; }
    }
    ASSERT(match, "hist save: predictions match after load");

    free(p1); free(p2);
    rf_free_buffer(buf);
    rf_free(f1); rf_free(f2);
    free(X); free(y);
}

static void test_histogram_binary_feature(void) {
    printf("=== Histogram Binary Feature ===\n");

    int n = 500, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        X[i * 2] = rf_rng_uniform(&rng) * 2.0 - 1.0;
        X[i * 2 + 1] = (rf_rng_uniform(&rng) > 0.5) ? 1.0 : 0.0;
        y[i] = X[i * 2] + X[i * 2 + 1] * 3.0 + rf_rng_uniform(&rng) * 0.5;
    }

    rf_params_t p;
    rf_params_init(&p);
    p.n_estimators = 50;
    p.seed = 42;
    p.task = 1;
    p.histogram = 1;
    rf_forest_t *f = rf_fit(X, n, d, y, &p);
    ASSERT(f != NULL, "hist binary: fit succeeds");

    /* Both features should have positive importance */
    double imp0 = f->feature_importances[0];
    double imp1 = f->feature_importances[1];
    printf("  Importances: f0=%.4f, f1=%.4f\n", imp0, imp1);
    ASSERT(imp1 > 0.01, "hist binary: binary feature has importance");

    /* R2 should be high */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f, X, n, d, preds);
    double ss_res = 0, ss_tot = 0, ymean = 0;
    for (int i = 0; i < n; i++) ymean += y[i];
    ymean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - ymean) * (y[i] - ymean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  R2 with binary feature: %.3f\n", r2);
    ASSERT(r2 > 0.8, "hist binary: R2 > 0.8");

    free(preds); rf_free(f); free(X); free(y);
}

static void test_permutation_importance(void) {
    printf("=== Permutation Importance ===\n");

    /* Feature 0 is signal, features 1-3 are noise */
    int n = 300, d = 4;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = rf_rng_uniform(&rng) * 10.0 - 5.0;
        y[i] = X[i * d] * 3.0 + rf_rng_uniform(&rng) * 0.5;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 50;
    params.seed = 42;
    params.task = 1;
    rf_forest_t *f = rf_fit(X, n, d, y, &params);
    ASSERT(f != NULL, "perm imp: fit succeeds");

    double imp[4];
    int ret = rf_permutation_importance(f, X, n, d, y, 5, 42, imp);
    ASSERT(ret == 0, "perm imp: returns 0");

    printf("  Perm imp: [%.4f, %.4f, %.4f, %.4f]\n", imp[0], imp[1], imp[2], imp[3]);

    /* Signal feature should have highest importance */
    ASSERT(imp[0] > imp[1] && imp[0] > imp[2] && imp[0] > imp[3],
           "perm imp: signal feature has highest importance");
    /* Signal feature should have positive importance */
    ASSERT(imp[0] > 0.01, "perm imp: signal feature importance > 0.01");

    rf_free(f);
    free(X); free(y);
}

static void test_permutation_importance_cls(void) {
    printf("=== Permutation Importance (Classification) ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = rf_rng_uniform(&rng) * 4.0 - 2.0;
        y[i] = (X[i * d] > 0) ? 1.0 : 0.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 50;
    params.seed = 42;
    params.task = 0;
    rf_forest_t *f = rf_fit(X, n, d, y, &params);
    ASSERT(f != NULL, "perm imp cls: fit succeeds");

    double imp[3];
    int ret = rf_permutation_importance(f, X, n, d, y, 5, 42, imp);
    ASSERT(ret == 0, "perm imp cls: returns 0");

    printf("  Perm imp cls: [%.4f, %.4f, %.4f]\n", imp[0], imp[1], imp[2]);
    ASSERT(imp[0] > imp[1] && imp[0] > imp[2],
           "perm imp cls: signal feature has highest importance");

    rf_free(f);
    free(X); free(y);
}

static void test_jarf_rotation(void) {
    printf("=== JARF Rotation ===\n");

    /* Diagonal boundary: y = (x0 + x1 > 0) -- standard RF struggles,
     * JARF should rotate features to align with this boundary */
    int n = 300, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 123 };

    for (int i = 0; i < n; i++) {
        X[i * d + 0] = rf_rng_uniform(&rng) * 4.0 - 2.0;
        X[i * d + 1] = rf_rng_uniform(&rng) * 4.0 - 2.0;
        y[i] = (X[i * d + 0] + X[i * d + 1] > 0) ? 1.0 : 0.0;
    }

    /* Standard RF */
    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 100;
    params.seed = 42;
    params.task = 0;
    rf_forest_t *f1 = rf_fit(X, n, d, y, &params);
    ASSERT(f1 != NULL, "jarf: standard fit succeeds");

    double *pred1 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f1, X, n, d, pred1);
    int correct1 = 0;
    for (int i = 0; i < n; i++) if ((int)pred1[i] == (int)y[i]) correct1++;
    double acc1 = (double)correct1 / n;

    /* JARF RF */
    rf_params_t params2;
    rf_params_init(&params2);
    params2.n_estimators = 100;
    params2.seed = 42;
    params2.task = 0;
    params2.jarf = 1;
    params2.jarf_n_estimators = 50;
    params2.jarf_max_depth = 6;
    rf_forest_t *f2 = rf_fit(X, n, d, y, &params2);
    ASSERT(f2 != NULL, "jarf: JARF fit succeeds");
    ASSERT(f2->jarf_rotation != NULL, "jarf: rotation matrix stored");
    ASSERT(f2->jarf_ncol == d, "jarf: rotation ncol matches");

    double *pred2 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f2, X, n, d, pred2);
    int correct2 = 0;
    for (int i = 0; i < n; i++) if ((int)pred2[i] == (int)y[i]) correct2++;
    double acc2 = (double)correct2 / n;

    printf("  Standard acc: %.3f, JARF acc: %.3f\n", acc1, acc2);
    ASSERT(acc2 >= acc1 - 0.05, "jarf: JARF not significantly worse");

    /* Save/load round-trip */
    char *buf = NULL;
    int32_t len = 0;
    int ret = rf_save(f2, &buf, &len);
    ASSERT(ret == 0, "jarf: save succeeds");

    rf_forest_t *f3 = rf_load(buf, len);
    ASSERT(f3 != NULL, "jarf: load succeeds");
    ASSERT(f3->jarf_rotation != NULL, "jarf: rotation survives load");

    double *pred3 = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f3, X, n, d, pred3);
    int match = 1;
    for (int i = 0; i < n; i++) if (pred2[i] != pred3[i]) { match = 0; break; }
    ASSERT(match, "jarf: save/load predictions match");

    rf_free_buffer(buf);
    rf_free(f1); rf_free(f2); rf_free(f3);
    free(X); free(y); free(pred1); free(pred2); free(pred3);
}

static void test_jarf_regression(void) {
    printf("=== JARF Regression ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 99 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = rf_rng_uniform(&rng) * 4.0 - 2.0;
        y[i] = X[i * d + 0] + X[i * d + 1] + 0.1 * (rf_rng_uniform(&rng) - 0.5);
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 100;
    params.seed = 42;
    params.task = 1;
    params.jarf = 1;
    rf_forest_t *f = rf_fit(X, n, d, y, &params);
    ASSERT(f != NULL, "jarf reg: fit succeeds");
    ASSERT(f->jarf_rotation != NULL, "jarf reg: rotation matrix stored");

    double *pred = (double *)malloc((size_t)n * sizeof(double));
    rf_predict(f, X, n, d, pred);
    double ss_res = 0, ss_tot = 0, y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  JARF regression R2: %.3f\n", r2);
    ASSERT(r2 > 0.9, "jarf reg: good R2");

    rf_free(f);
    free(X); free(y); free(pred);
}

static void test_proximity(void) {
    printf("=== Proximity Matrix ===\n");

    int n = 50, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    rf_rng_t rng = { 42 };

    /* Two clusters: class 0 around (-1,-1), class 1 around (1,1) */
    for (int i = 0; i < n; i++) {
        double cx = (i < n/2) ? -1.0 : 1.0;
        double cy = (i < n/2) ? -1.0 : 1.0;
        X[i * d + 0] = cx + 0.3 * (rf_rng_uniform(&rng) - 0.5);
        X[i * d + 1] = cy + 0.3 * (rf_rng_uniform(&rng) - 0.5);
        y[i] = (i < n/2) ? 0.0 : 1.0;
    }

    rf_params_t params;
    rf_params_init(&params);
    params.n_estimators = 50;
    params.seed = 42;
    params.task = 0;
    rf_forest_t *f = rf_fit(X, n, d, y, &params);
    ASSERT(f != NULL, "proximity: fit succeeds");

    double *prox = (double *)calloc((size_t)n * n, sizeof(double));
    int ret = rf_proximity(f, X, n, d, prox);
    ASSERT(ret == 0, "proximity: returns 0");

    /* Diagonal should be 1.0 */
    ASSERT(fabs(prox[0] - 1.0) < 1e-10, "proximity: diagonal is 1.0");

    /* Symmetric */
    int sym = 1;
    for (int i = 0; i < n && sym; i++)
        for (int j = i+1; j < n && sym; j++)
            if (fabs(prox[i*n+j] - prox[j*n+i]) > 1e-10) sym = 0;
    ASSERT(sym, "proximity: matrix is symmetric");

    /* Within-class proximity > between-class proximity */
    double within = 0, between = 0;
    int nw = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if ((int)y[i] == (int)y[j]) { within += prox[i*n+j]; nw++; }
            else { between += prox[i*n+j]; nb++; }
        }
    }
    within /= nw; between /= nb;
    printf("  Within-class prox: %.3f, Between-class prox: %.3f\n", within, between);
    ASSERT(within > between, "proximity: within-class > between-class");

    rf_free(f);
    free(X); free(y); free(prox);
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
    test_sample_weight_cls();
    test_sample_weight_reg();
    test_sample_weight_uniform();
    test_sample_weight_zero();
    test_histogram_regression();
    test_histogram_classification();
    test_histogram_save_load();
    test_histogram_binary_feature();
    test_permutation_importance();
    test_permutation_importance_cls();
    test_jarf_rotation();
    test_jarf_regression();
    test_proximity();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
