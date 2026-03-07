/*
 * benchmark.c -- Ablation study comparing RF configurations
 *
 * Runs classification and regression benchmarks with each new feature
 * vs the default (sklearn-matching) configuration. Reports accuracy/R2,
 * OOB score, and timing.
 */
#define _POSIX_C_SOURCE 200809L
#include "rf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* Dataset sizes */
#define CLS_N 500
#define CLS_D 10
#define CLS_K 3
#define REG_N 500
#define REG_D 10

static double elapsed_ms(struct timespec *start, struct timespec *end) {
    double s = (double)(end->tv_sec - start->tv_sec) * 1000.0;
    s += (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s;
}

static void make_cls_data(double *X, double *y, int n, int d, int k, uint32_t seed) {
    /* Overlapping class distributions for a harder problem */
    rf_rng_t rng = { seed };
    for (int i = 0; i < n; i++) {
        int label = i % k;
        for (int j = 0; j < d; j++) {
            /* Smaller separation, more noise -- harder to classify */
            X[i * d + j] = label * 0.8 + (rf_rng_uniform(&rng) - 0.5) * 2.0;
        }
        y[i] = (double)label;
    }
}

static void make_reg_data(double *X, double *y, int n, int d, uint32_t seed) {
    rf_rng_t rng = { seed };
    for (int i = 0; i < n; i++) {
        double target = 0;
        for (int j = 0; j < d; j++) {
            double v = rf_rng_uniform(&rng) * 4.0 - 2.0;
            X[i * d + j] = v;
            target += v * (j + 1);
        }
        y[i] = target + (rf_rng_uniform(&rng) - 0.5) * 0.5;
    }
}

static double accuracy(const double *preds, const double *y, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((int)preds[i] == (int)y[i]) correct++;
    }
    return (double)correct / n;
}

static double r2_score(const double *preds, const double *y, int n) {
    double y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
}

typedef struct {
    const char *name;
    rf_params_t params;
} config_t;

static void run_cls_benchmark(const char *label, config_t *configs, int n_configs,
                               const double *X, const double *y, int n, int d) {
    printf("\n### %s (n=%d, d=%d, k=%d)\n\n", label, n, d, CLS_K);
    printf("| %-30s | Train Acc | OOB Acc  | Fit (ms) |\n", "Configuration");
    printf("| %-30s | --------- | -------- | -------- |\n", "------------------------------");

    double *preds = (double *)malloc((size_t)n * sizeof(double));

    for (int c = 0; c < n_configs; c++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        rf_forest_t *forest = rf_fit(X, n, d, y, &configs[c].params);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (!forest) {
            printf("| %-30s | FAILED    |          |          |\n", configs[c].name);
            continue;
        }

        rf_predict(forest, X, n, d, preds);
        double acc = accuracy(preds, y, n);
        double oob = forest->oob_score;
        double ms = elapsed_ms(&t0, &t1);

        printf("| %-30s | %9.4f | %8.4f | %8.1f |\n",
               configs[c].name, acc, oob, ms);

        rf_free(forest);
    }

    free(preds);
}

static void run_reg_benchmark(const char *label, config_t *configs, int n_configs,
                               const double *X, const double *y, int n, int d) {
    printf("\n### %s (n=%d, d=%d)\n\n", label, n, d);
    printf("| %-30s | Train R2  | OOB R2   | Fit (ms) |\n", "Configuration");
    printf("| %-30s | --------- | -------- | -------- |\n", "------------------------------");

    double *preds = (double *)malloc((size_t)n * sizeof(double));

    for (int c = 0; c < n_configs; c++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        rf_forest_t *forest = rf_fit(X, n, d, y, &configs[c].params);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (!forest) {
            printf("| %-30s | FAILED    |          |          |\n", configs[c].name);
            continue;
        }

        rf_predict(forest, X, n, d, preds);
        double r2 = r2_score(preds, y, n);
        double oob = forest->oob_score;
        double ms = elapsed_ms(&t0, &t1);

        printf("| %-30s | %9.4f | %8.4f | %8.1f |\n",
               configs[c].name, r2, oob, ms);

        rf_free(forest);
    }

    free(preds);
}

int main(void) {
    printf("# RF Benchmark / Ablation Study\n");

    /* Generate datasets */
    double *X_cls = (double *)malloc((size_t)CLS_N * CLS_D * sizeof(double));
    double *y_cls = (double *)malloc((size_t)CLS_N * sizeof(double));
    make_cls_data(X_cls, y_cls, CLS_N, CLS_D, CLS_K, 42);

    double *X_reg = (double *)malloc((size_t)REG_N * REG_D * sizeof(double));
    double *y_reg = (double *)malloc((size_t)REG_N * sizeof(double));
    make_reg_data(X_reg, y_reg, REG_N, REG_D, 42);

    /* Base params */
    rf_params_t base;
    rf_params_init(&base);
    base.n_estimators = 100;
    base.seed = 42;

    /* === Classification configs === */
    config_t cls_configs[10];
    int ci = 0;

    /* Default (sklearn) */
    cls_configs[ci].name = "Default (Gini)";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    ci++;

    /* Entropy */
    cls_configs[ci].name = "Entropy";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.criterion = 1;
    ci++;

    /* Hellinger */
    cls_configs[ci].name = "Hellinger";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.criterion = 2;
    ci++;

    /* Sample rate 0.7 */
    cls_configs[ci].name = "Sample rate 0.7";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.sample_rate = 0.7;
    ci++;

    /* HRF */
    cls_configs[ci].name = "Heterogeneous RF";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.heterogeneous = 1;
    ci++;

    /* OOB weighting */
    cls_configs[ci].name = "OOB weighting";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.oob_weighting = 1;
    ci++;

    /* Alpha trim */
    cls_configs[ci].name = "Alpha trim 0.01";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.alpha_trim = 0.01;
    ci++;

    /* ExtraTrees */
    cls_configs[ci].name = "ExtraTrees";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.extra_trees = 1;
    ci++;

    /* Combined: entropy + HRF + OOB weighting */
    cls_configs[ci].name = "Entropy + HRF + OOB wt";
    cls_configs[ci].params = base;
    cls_configs[ci].params.task = 0;
    cls_configs[ci].params.criterion = 1;
    cls_configs[ci].params.heterogeneous = 1;
    cls_configs[ci].params.oob_weighting = 1;
    ci++;

    run_cls_benchmark("Classification", cls_configs, ci, X_cls, y_cls, CLS_N, CLS_D);

    /* === Regression configs === */
    config_t reg_configs[10];
    int ri = 0;

    /* Default (sklearn) */
    reg_configs[ri].name = "Default (MSE)";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    ri++;

    /* MAE */
    reg_configs[ri].name = "MAE";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.criterion = 1;
    ri++;

    /* Sample rate 0.7 */
    reg_configs[ri].name = "Sample rate 0.7";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.sample_rate = 0.7;
    ri++;

    /* HRF */
    reg_configs[ri].name = "Heterogeneous RF";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.heterogeneous = 1;
    ri++;

    /* OOB weighting */
    reg_configs[ri].name = "OOB weighting";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.oob_weighting = 1;
    ri++;

    /* Alpha trim */
    reg_configs[ri].name = "Alpha trim 0.01";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.alpha_trim = 0.01;
    ri++;

    /* Linear leaves */
    reg_configs[ri].name = "Linear leaves";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.leaf_model = 1;
    ri++;

    /* Linear + shallow */
    reg_configs[ri].name = "Linear leaves (depth=5)";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.leaf_model = 1;
    reg_configs[ri].params.max_depth = 5;
    ri++;

    /* ExtraTrees */
    reg_configs[ri].name = "ExtraTrees";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.extra_trees = 1;
    ri++;

    /* Combined: linear + HRF + OOB wt */
    reg_configs[ri].name = "Linear + HRF + OOB wt";
    reg_configs[ri].params = base;
    reg_configs[ri].params.task = 1;
    reg_configs[ri].params.leaf_model = 1;
    reg_configs[ri].params.heterogeneous = 1;
    reg_configs[ri].params.oob_weighting = 1;
    ri++;

    run_reg_benchmark("Regression", reg_configs, ri, X_reg, y_reg, REG_N, REG_D);

    free(X_cls); free(y_cls);
    free(X_reg); free(y_reg);

    return 0;
}
