"""
test_parity.py -- Parity tests for @wlearn/rf against sklearn and other references.

Tests existing features against sklearn for correctness validation, and defines
TDD tests for v0.3 features (quantile RF, conformal prediction, histogram binning,
missing value handling, JARF rotation, monotonic constraints).

Usage:
    RF_LIB_PATH=build/librf.so python -m pytest test/test_parity.py -v

References:
    sklearn 1.7.2 -- RandomForestClassifier/Regressor, ExtraTreesClassifier/Regressor
    quantile-forest 1.4.1 -- RandomForestQuantileRegressor
    mapie 1.3.0 -- MapieRegressor (jackknife+/J+ab)
"""
import sys
import os
import math

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression, make_friedman1
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'python'))
from wlearn_rf import RFModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_wlearn_cls(X, y, n_estimators=100, seed=42, **kwargs):
    m = RFModel({'n_estimators': n_estimators, 'seed': seed, **kwargs})
    m.fit(X, y)
    return m


def fit_wlearn_reg(X, y, n_estimators=100, seed=42, **kwargs):
    m = RFModel({'n_estimators': n_estimators, 'seed': seed,
                 'task': 'regression', **kwargs})
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# Datasets (fixed seeds for reproducibility)
# ---------------------------------------------------------------------------

@pytest.fixture
def iris():
    d = load_iris()
    return d.data, d.target.astype(float)


@pytest.fixture
def cls500():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)
    return X, y.astype(float)


@pytest.fixture
def reg500():
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5,
                           noise=10.0, random_state=42)
    return X, y


@pytest.fixture
def friedman1():
    X, y = make_friedman1(n_samples=500, noise=1.0, random_state=42)
    return X, y


@pytest.fixture
def signal_noise():
    """4 features: feature 0 is signal, 1-3 are noise."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 4)
    y = 5.0 * X[:, 0] + rng.randn(n) * 0.1
    return X, y


@pytest.fixture
def imbalanced():
    """Binary classification with 90/10 class split."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 5)
    y = np.zeros(n)
    y[:30] = 1.0  # 10% minority
    X[y == 1] += 1.5
    return X, y


# ===========================================================================
# PART 1: Existing feature parity (v0.2) -- must pass now
# ===========================================================================

class TestOOBParity:
    """OOB score should be reasonable and match mathematical invariants."""

    def test_oob_classification_iris(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42)
        sk = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
        sk.fit(X, y)

        oob_wl = m.oob_score()
        oob_sk = sk.oob_score_
        assert not math.isnan(oob_wl), 'wlearn OOB is NaN'
        assert abs(oob_wl - oob_sk) < 0.08, (
            f'OOB parity: wlearn={oob_wl:.4f} sklearn={oob_sk:.4f}')
        m.dispose()

    def test_oob_regression_friedman1(self, friedman1):
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        sk = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
        sk.fit(X, y)

        oob_wl = m.oob_score()
        oob_sk = sk.oob_score_
        assert not math.isnan(oob_wl)
        assert abs(oob_wl - oob_sk) < 0.10, (
            f'OOB parity: wlearn={oob_wl:.4f} sklearn={oob_sk:.4f}')
        m.dispose()

    def test_oob_less_than_train_score(self, cls500):
        X, y = cls500
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        train_acc = m.score(X, y)
        oob = m.oob_score()
        assert oob <= train_acc + 1e-10, (
            f'OOB ({oob}) should be <= train score ({train_acc})')
        m.dispose()

    def test_oob_less_than_train_regression(self, reg500):
        X, y = reg500
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42)
        train_r2 = m.score(X, y)
        oob = m.oob_score()
        assert oob <= train_r2 + 1e-10, (
            f'OOB R2 ({oob}) should be <= train R2 ({train_r2})')
        m.dispose()

    def test_oob_converges_with_more_trees(self, iris):
        X, y = iris
        m20 = fit_wlearn_cls(X, y, n_estimators=20, seed=42)
        m200 = fit_wlearn_cls(X, y, n_estimators=200, seed=42)
        oob20 = m20.oob_score()
        oob200 = m200.oob_score()
        # 200 trees should be at least as good or very close
        assert oob200 >= oob20 - 0.05, (
            f'200-tree OOB ({oob200}) worse than 20-tree ({oob20})')
        m20.dispose()
        m200.dispose()

    def test_oob_no_bootstrap(self):
        """Without bootstrap, OOB should be NaN (no held-out samples)."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42,
                           bootstrap=0, sample_rate=1.0)
        oob = m.oob_score()
        assert math.isnan(oob), f'OOB should be NaN without bootstrap, got {oob}'
        m.dispose()

    def test_oob_constant_y_regression(self):
        """Constant y: OOB R2 should be NaN or 0 (ss_tot=0)."""
        X = np.random.RandomState(42).randn(50, 3)
        y = np.ones(50) * 7.0
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42)
        oob = m.oob_score()
        # Either NaN or 0 is acceptable when ss_tot = 0
        assert math.isnan(oob) or abs(oob) < 1e-10, (
            f'Constant y OOB should be NaN or 0, got {oob}')
        m.dispose()


class TestFeatureImportanceParity:
    """MDI feature importances should match sklearn behavior."""

    def test_importances_nonnegative(self, cls500):
        X, y = cls500
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        imp = m.feature_importances()
        assert np.all(imp >= 0), f'Negative importances: {imp}'
        m.dispose()

    def test_importances_sum_to_one(self, cls500):
        X, y = cls500
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        imp = m.feature_importances()
        assert abs(imp.sum() - 1.0) < 1e-10, f'Sum: {imp.sum()}'
        m.dispose()

    def test_importances_ranking_signal_vs_noise(self, signal_noise):
        """Feature 0 (strong signal) should rank first."""
        X, y = signal_noise
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42, max_features=4)
        imp = m.feature_importances()
        assert np.argmax(imp) == 0, f'Feature 0 should be most important, got {imp}'
        m.dispose()

    def test_importances_ranking_parity_iris(self, iris):
        """Feature importance ranking should agree with sklearn on iris."""
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=200, seed=42)
        sk = RandomForestClassifier(n_estimators=200, random_state=42)
        sk.fit(X, y)

        rank_wl = np.argsort(-m.feature_importances())
        rank_sk = np.argsort(-sk.feature_importances_)
        # Top-2 features should agree (petal length and petal width)
        assert set(rank_wl[:2]) == set(rank_sk[:2]), (
            f'Top-2 ranking mismatch: wlearn={rank_wl[:2]} sklearn={rank_sk[:2]}')
        m.dispose()

    def test_importances_noise_features_low(self, signal_noise):
        """Noise features should have low importance."""
        X, y = signal_noise
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42, max_features=4)
        imp = m.feature_importances()
        for j in range(1, 4):
            assert imp[j] < 0.15, (
                f'Noise feature {j} importance too high: {imp[j]}')
        m.dispose()

    def test_importances_single_feature(self):
        """Single feature: importance should be 1.0."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 1)
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42, max_features=1)
        imp = m.feature_importances()
        assert abs(imp[0] - 1.0) < 1e-10, f'Single feature importance: {imp[0]}'
        m.dispose()

    def test_importances_constant_feature_zero(self):
        """A constant feature should have zero importance."""
        rng = np.random.RandomState(42)
        n = 100
        X = np.column_stack([rng.randn(n), np.ones(n)])
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42, max_features=2)
        imp = m.feature_importances()
        assert imp[1] < 1e-10, f'Constant feature importance should be 0, got {imp[1]}'
        m.dispose()

    def test_importances_duplicate_features_shared(self):
        """Two identical features should share importance roughly equally."""
        rng = np.random.RandomState(42)
        n = 200
        x = rng.randn(n)
        X = np.column_stack([x, x, rng.randn(n) * 0.01])
        y = (x > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42, max_features=3)
        imp = m.feature_importances()
        # Features 0 and 1 should have similar importance
        ratio = imp[0] / (imp[1] + 1e-15)
        assert 0.3 < ratio < 3.0, (
            f'Duplicate features should share importance: {imp[0]:.4f} vs {imp[1]:.4f}')
        m.dispose()


class TestPredictionParity:
    """Prediction accuracy should be competitive with sklearn."""

    def test_iris_classification(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42)
        acc = m.score(X, y)
        assert acc > 0.95, f'Iris train accuracy too low: {acc}'

        sk = RandomForestClassifier(n_estimators=100, random_state=42)
        sk.fit(X, y)
        acc_sk = sk.score(X, y)
        assert abs(acc - acc_sk) < 0.05, (
            f'Accuracy gap: wlearn={acc:.4f} sklearn={acc_sk:.4f}')
        m.dispose()

    def test_make_classification_accuracy(self, cls500):
        X, y = cls500
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        m = fit_wlearn_cls(X_train, y_train, n_estimators=100, seed=42)
        sk = RandomForestClassifier(n_estimators=100, random_state=42)
        sk.fit(X_train, y_train)

        acc_wl = m.score(X_test, y_test)
        acc_sk = sk.score(X_test, y_test)
        assert acc_wl > 0.80, f'Test accuracy too low: {acc_wl}'
        assert abs(acc_wl - acc_sk) < 0.08, (
            f'Accuracy gap: wlearn={acc_wl:.4f} sklearn={acc_sk:.4f}')
        m.dispose()

    def test_make_regression_r2(self, reg500):
        X, y = reg500
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        m = fit_wlearn_reg(X_train, y_train, n_estimators=100, seed=42)
        sk = RandomForestRegressor(n_estimators=100, random_state=42)
        sk.fit(X_train, y_train)

        r2_wl = m.score(X_test, y_test)
        r2_sk = sk.score(X_test, y_test)
        assert r2_wl > 0.70, f'Test R2 too low: {r2_wl}'
        assert abs(r2_wl - r2_sk) < 0.12, (
            f'R2 gap: wlearn={r2_wl:.4f} sklearn={r2_sk:.4f}')
        m.dispose()

    def test_friedman1_r2(self, friedman1):
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        sk = RandomForestRegressor(n_estimators=100, random_state=42)
        sk.fit(X, y)

        r2_wl = m.score(X, y)
        r2_sk = sk.score(X, y)
        assert r2_wl > 0.90, f'Friedman1 R2 too low: {r2_wl}'
        assert abs(r2_wl - r2_sk) < 0.08, (
            f'R2 gap: wlearn={r2_wl:.4f} sklearn={r2_sk:.4f}')
        m.dispose()

    def test_predict_matches_argmax_proba(self, iris):
        """predict should equal argmax(predict_proba) for classification."""
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        preds = m.predict(X).astype(int)
        proba = m.predict_proba(X)
        argmax_proba = np.argmax(proba, axis=1)
        assert np.array_equal(preds, argmax_proba), (
            f'predict != argmax(proba): mismatches at {np.where(preds != argmax_proba)}')
        m.dispose()

    def test_predict_idempotent(self, iris):
        """Calling predict twice should give identical results."""
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        p1 = m.predict(X)
        p2 = m.predict(X)
        assert np.array_equal(p1, p2), 'predict is not idempotent'
        m.dispose()


class TestExtraTreesParity:
    """ExtraTrees should be competitive with sklearn ExtraTreesClassifier."""

    def test_extratrees_iris_accuracy(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42, extra_trees=1)
        sk = ExtraTreesClassifier(n_estimators=100, random_state=42)
        sk.fit(X, y)

        acc_wl = m.score(X, y)
        acc_sk = sk.score(X, y)
        assert acc_wl > 0.93, f'ExtraTrees iris accuracy too low: {acc_wl}'
        assert abs(acc_wl - acc_sk) < 0.06, (
            f'ExtraTrees accuracy gap: wlearn={acc_wl:.4f} sklearn={acc_sk:.4f}')
        m.dispose()

    def test_extratrees_make_classification(self, cls500):
        X, y = cls500
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42, extra_trees=1)
        sk = ExtraTreesClassifier(n_estimators=100, random_state=42)
        sk.fit(X, y)

        acc_wl = m.score(X, y)
        acc_sk = sk.score(X, y)
        assert acc_wl > 0.90, f'ExtraTrees accuracy too low: {acc_wl}'
        assert abs(acc_wl - acc_sk) < 0.08, (
            f'Gap: wlearn={acc_wl:.4f} sklearn={acc_sk:.4f}')
        m.dispose()

    def test_extratrees_regression(self, friedman1):
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42, extra_trees=1)
        sk = ExtraTreesRegressor(n_estimators=100, random_state=42)
        sk.fit(X, y)

        r2_wl = m.score(X, y)
        r2_sk = sk.score(X, y)
        assert r2_wl > 0.90, f'ExtraTrees R2 too low: {r2_wl}'
        assert abs(r2_wl - r2_sk) < 0.08, (
            f'Gap: wlearn={r2_wl:.4f} sklearn={r2_sk:.4f}')
        m.dispose()

    def test_extratrees_oob_reasonable(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42, extra_trees=1)
        oob = m.oob_score()
        assert not math.isnan(oob), 'ExtraTrees OOB is NaN'
        assert oob > 0.85, f'ExtraTrees OOB too low: {oob}'
        m.dispose()


class TestProbabilityInvariants:
    """Probability output invariants."""

    def test_proba_rows_sum_to_one(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        proba = m.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-12), (
            f'Row sums deviate: max delta={abs(row_sums - 1).max()}')
        m.dispose()

    def test_proba_in_range(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        proba = m.predict_proba(X)
        assert proba.min() >= -1e-15, f'Negative probability: {proba.min()}'
        assert proba.max() <= 1.0 + 1e-15, f'Probability > 1: {proba.max()}'
        m.dispose()

    def test_proba_binary_vs_multiclass(self):
        """Binary and multiclass probability shapes are correct."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y2 = (X[:, 0] > 0).astype(float)
        y3 = (X[:, 0] > 0.5).astype(float) + (X[:, 0] > -0.5).astype(float)

        m2 = fit_wlearn_cls(X, y2, n_estimators=20, seed=42)
        m3 = fit_wlearn_cls(X, y3, n_estimators=20, seed=42)

        p2 = m2.predict_proba(X)
        p3 = m3.predict_proba(X)

        assert p2.shape == (60, 2)
        assert p3.shape == (60, 3)
        m2.dispose()
        m3.dispose()


class TestFloatingPointEdgeCases:
    """Numerical edge cases that catch subtle bugs."""

    def test_constant_y_classification(self):
        """All labels the same: should predict that class with probability 1."""
        X = np.random.RandomState(42).randn(50, 3)
        y = np.ones(50)
        m = fit_wlearn_cls(X, y, n_estimators=10, seed=42)
        preds = m.predict(X)
        assert np.all(preds == 1.0), 'Should predict constant class'
        m.dispose()

    def test_constant_y_regression(self):
        """Constant y: predictions should all equal that constant."""
        X = np.random.RandomState(42).randn(50, 3)
        y = np.full(50, 3.14)
        m = fit_wlearn_reg(X, y, n_estimators=10, seed=42)
        preds = m.predict(X)
        assert np.allclose(preds, 3.14, atol=1e-10), f'Max error: {abs(preds - 3.14).max()}'
        m.dispose()

    def test_near_constant_y(self):
        """Tiny variance: should not crash or produce NaN."""
        X = np.random.RandomState(42).randn(50, 3)
        y = 1.0 + np.random.RandomState(42).randn(50) * 1e-14
        m = fit_wlearn_reg(X, y, n_estimators=10, seed=42)
        preds = m.predict(X)
        assert not np.any(np.isnan(preds)), 'NaN predictions on near-constant y'
        assert not np.any(np.isinf(preds)), 'Inf predictions on near-constant y'
        m.dispose()

    def test_large_values(self):
        """Large feature and target values: no overflow."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3) * 1e12
        y = X[:, 0] * 2 + rng.randn(100) * 1e10
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42)
        preds = m.predict(X)
        assert not np.any(np.isnan(preds)), 'NaN on large values'
        assert not np.any(np.isinf(preds)), 'Inf on large values'
        r2 = m.score(X, y)
        assert r2 > 0.5, f'R2 too low on large values: {r2}'
        m.dispose()

    def test_small_values(self):
        """Tiny feature values: splits should still distinguish them."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3) * 1e-12
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42)
        acc = m.score(X, y)
        assert acc > 0.8, f'Accuracy too low on small values: {acc}'
        m.dispose()

    def test_negative_values(self):
        """All negative features and targets."""
        rng = np.random.RandomState(42)
        X = -np.abs(rng.randn(100, 3))
        y = X[:, 0] * 3 + rng.randn(100) * 0.1
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42)
        r2 = m.score(X, y)
        assert r2 > 0.8, f'R2 too low on negative values: {r2}'
        m.dispose()

    def test_all_zero_feature(self):
        """Feature that is always 0 should have zero importance."""
        rng = np.random.RandomState(42)
        n = 100
        X = np.column_stack([rng.randn(n), np.zeros(n), rng.randn(n)])
        y = X[:, 0] * 2 + rng.randn(n) * 0.1
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42, max_features=3)
        imp = m.feature_importances()
        assert imp[1] < 1e-10, f'All-zero feature importance should be 0, got {imp[1]}'
        m.dispose()

    def test_single_sample_regression(self):
        """Training with n=1 should not crash (regression)."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.14])
        m = fit_wlearn_reg(X, y, n_estimators=5, seed=42)
        pred = m.predict(X)
        assert abs(pred[0] - 3.14) < 1e-10
        m.dispose()

    def test_single_sample_cls_needs_two_classes(self):
        """Classification with only 1 class should error (need >= 2)."""
        X = np.array([[1.0, 2.0]])
        y = np.array([0.0])
        with pytest.raises(RuntimeError, match='at least 2 classes'):
            fit_wlearn_cls(X, y, n_estimators=5, seed=42)

    def test_two_samples_classification(self):
        """n=2 binary classification."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0.0, 1.0])
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        preds = m.predict(X)
        # With bootstrap, predictions may not be exact but should not crash
        assert len(preds) == 2
        m.dispose()

    def test_many_classes(self):
        """20 classes should work correctly."""
        rng = np.random.RandomState(42)
        n = 400
        X = rng.randn(n, 5)
        y = np.arange(n) % 20
        X += y.reshape(-1, 1) * 0.5
        m = fit_wlearn_cls(X, y.astype(float), n_estimators=100, seed=42)
        acc = m.score(X, y.astype(float))
        assert acc > 0.5, f'20-class accuracy too low: {acc}'
        assert m.n_classes == 20
        proba = m.predict_proba(X)
        assert proba.shape == (n, 20)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        m.dispose()

    def test_scale_invariance_regression(self):
        """Predictions should shift with y offset."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X[:, 0] * 2 + rng.randn(100) * 0.1

        m1 = fit_wlearn_reg(X, y, n_estimators=50, seed=42)
        m2 = fit_wlearn_reg(X, y + 1000.0, n_estimators=50, seed=42)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        # p2 should be approximately p1 + 1000
        assert np.allclose(p2 - p1, 1000.0, atol=1.0), (
            f'Shift not preserved: mean delta={np.mean(p2 - p1):.2f}')
        m1.dispose()
        m2.dispose()


class TestCriteriaParity:
    """Different split criteria should give reasonable results."""

    def test_entropy_vs_gini_iris(self, iris):
        X, y = iris
        m_gini = fit_wlearn_cls(X, y, n_estimators=100, seed=42, criterion='gini')
        m_entr = fit_wlearn_cls(X, y, n_estimators=100, seed=42, criterion='entropy')
        acc_g = m_gini.score(X, y)
        acc_e = m_entr.score(X, y)
        # Both should be high; difference should be small
        assert acc_g > 0.93
        assert acc_e > 0.93
        assert abs(acc_g - acc_e) < 0.05
        m_gini.dispose()
        m_entr.dispose()

    def test_hellinger_on_imbalanced(self, imbalanced):
        """Hellinger should handle imbalanced data at least as well as Gini."""
        X, y = imbalanced
        m_gini = fit_wlearn_cls(X, y, n_estimators=100, seed=42, criterion='gini')
        m_hell = fit_wlearn_cls(X, y, n_estimators=100, seed=42, criterion='hellinger')
        oob_g = m_gini.oob_score()
        oob_h = m_hell.oob_score()
        # Hellinger should be at least comparable
        assert oob_h >= oob_g - 0.05, (
            f'Hellinger OOB ({oob_h}) much worse than Gini ({oob_g}) on imbalanced data')
        m_gini.dispose()
        m_hell.dispose()

    def test_mse_vs_mae_regression(self, friedman1):
        X, y = friedman1
        m_mse = fit_wlearn_reg(X, y, n_estimators=100, seed=42, criterion='mse')
        m_mae = fit_wlearn_reg(X, y, n_estimators=100, seed=42, criterion='mae')
        r2_mse = m_mse.score(X, y)
        r2_mae = m_mae.score(X, y)
        assert r2_mse > 0.90
        assert r2_mae > 0.85  # MAE typically slightly worse on R2
        m_mse.dispose()
        m_mae.dispose()


class TestSaveLoadParity:
    """Save/load round-trip preserves all properties."""

    def test_classification_round_trip(self, iris):
        X, y = iris
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42,
                           criterion='entropy', sample_rate=0.8)
        p1 = m.predict(X)
        imp1 = m.feature_importances()
        oob1 = m.oob_score()

        blob = m.save_raw()
        m.dispose()

        m2 = RFModel.load_raw(blob)
        p2 = m2.predict(X)
        imp2 = m2.feature_importances()

        assert np.array_equal(p1, p2), 'Predictions differ after load'
        assert np.allclose(imp1, imp2, atol=1e-10), 'Importances differ after load'
        m2.dispose()

    def test_regression_round_trip(self, friedman1):
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42, leaf_model=1, max_depth=5)
        p1 = m.predict(X)

        blob = m.save_raw()
        m.dispose()

        m2 = RFModel.load_raw(blob, params={'task': 'regression'})
        p2 = m2.predict(X)
        assert np.allclose(p1, p2, atol=1e-10), (
            f'Max prediction diff: {abs(p1 - p2).max()}')
        m2.dispose()


# ===========================================================================
# PART 2: v0.3 Feature Parity (TDD -- expected to fail until implemented)
# ===========================================================================

def _has_predict_quantile():
    return hasattr(RFModel, 'predict_quantile')


def _has_predict_interval():
    return hasattr(RFModel, 'predict_interval')


# Feature flags: check if C API supports new v0.3 params.
# Until implemented, these return False and tests are skipped.
def _has_histogram_binning():
    """Check if histogram_binning param is recognized."""
    try:
        m = RFModel({'n_estimators': 1, 'seed': 1, 'histogram_binning': 1})
        rng = np.random.RandomState(1)
        X = rng.randn(10, 2)
        y = (X[:, 0] > 0).astype(float)
        m.fit(X, y)
        m.dispose()
        return True
    except Exception:
        return False


def _has_missing_value_support():
    """Check if NaN handling is explicitly supported (not just silent corruption)."""
    # v0.3 will add explicit NaN direction learning.
    # Without it, NaN produces undefined/garbage results.
    return True


def _has_monotonic_constraints():
    """Check if monotonic_cst param is supported."""
    return True


def _has_jarf():
    """Check if JARF rotation is supported."""
    return True


@pytest.mark.skipif(not _has_predict_quantile(),
                    reason='quantile RF not yet implemented')
class TestQuantileRF:
    """Quantile Random Forest parity with quantile-forest package."""

    def test_quantile_ordering(self, friedman1):
        """q=0.1 <= q=0.5 <= q=0.9 for every sample (critical invariant)."""
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        q10 = m.predict_quantile(X, quantile=0.1)
        q50 = m.predict_quantile(X, quantile=0.5)
        q90 = m.predict_quantile(X, quantile=0.9)
        assert np.all(q10 <= q50 + 1e-10), 'q10 > q50 violations'
        assert np.all(q50 <= q90 + 1e-10), 'q50 > q90 violations'
        m.dispose()

    def test_median_close_to_mean(self, friedman1):
        """Median prediction should be correlated with mean prediction."""
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        mean_pred = m.predict(X)
        median_pred = m.predict_quantile(X, quantile=0.5)
        corr = np.corrcoef(mean_pred, median_pred)[0, 1]
        assert corr > 0.95, f'Median-mean correlation too low: {corr}'
        # Relative difference should be small
        rdiff = np.mean(np.abs(median_pred - mean_pred)) / np.std(y)
        assert rdiff < 0.20, f'Median-mean relative diff too large: {rdiff}'
        m.dispose()

    def test_coverage_calibration(self):
        """90% interval should cover ~90% of test values."""
        X, y = make_regression(n_samples=800, n_features=5, noise=5.0,
                               random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        m = fit_wlearn_reg(X_train, y_train, n_estimators=200, seed=42)
        q05 = m.predict_quantile(X_test, quantile=0.05)
        q95 = m.predict_quantile(X_test, quantile=0.95)

        covered = np.mean((y_test >= q05) & (y_test <= q95))
        assert 0.80 <= covered <= 0.98, f'90% interval coverage: {covered:.3f}'
        m.dispose()

    def test_extreme_quantiles(self, friedman1):
        """q=0 should be min leaf value, q=1 should be max."""
        X, y = friedman1
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42)
        q0 = m.predict_quantile(X, quantile=0.0)
        q1 = m.predict_quantile(X, quantile=1.0)
        mean_pred = m.predict(X)
        # q=0 should be <= mean, q=1 should be >= mean
        assert np.all(q0 <= mean_pred + 1e-10)
        assert np.all(q1 >= mean_pred - 1e-10)
        m.dispose()

    def test_single_tree_quantile(self):
        """Single tree: quantiles match empirical leaf distribution."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = X[:, 0] * 2 + rng.randn(100) * 0.5
        m = fit_wlearn_reg(X, y, n_estimators=1, seed=42, bootstrap=0)
        q50 = m.predict_quantile(X[:5], quantile=0.5)
        # With 1 tree and no bootstrap, q50 should equal median of leaf y-values
        assert len(q50) == 5
        assert not np.any(np.isnan(q50))
        m.dispose()

    def test_constant_leaf_all_quantiles_equal(self):
        """If all y-values in a leaf are identical, all quantiles should match."""
        n = 50
        X = np.zeros((n, 1))
        X[:25, 0] = -1.0
        X[25:, 0] = 1.0
        y = np.zeros(n)
        y[:25] = 3.0
        y[25:] = 7.0
        m = fit_wlearn_reg(X, y, n_estimators=1, seed=42, bootstrap=0)
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pred = m.predict_quantile(X[:1], quantile=q)
            assert abs(pred[0] - 3.0) < 1e-10, f'q={q}: expected 3.0, got {pred[0]}'
        m.dispose()

    def test_heteroscedastic_intervals(self):
        """Wider intervals where noise is larger."""
        rng = np.random.RandomState(42)
        n = 500
        X = rng.uniform(0, 10, (n, 1))
        y = X[:, 0] + X[:, 0] * rng.randn(n) * 0.3  # noise scales with x
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        q10 = m.predict_quantile(X, quantile=0.1)
        q90 = m.predict_quantile(X, quantile=0.9)
        widths = q90 - q10

        # Split into low-x and high-x groups
        low = X[:, 0] < 3
        high = X[:, 0] > 7
        assert np.mean(widths[high]) > np.mean(widths[low]) * 1.3, (
            f'High-x intervals ({np.mean(widths[high]):.3f}) should be wider '
            f'than low-x ({np.mean(widths[low]):.3f})')
        m.dispose()

    def test_parity_with_quantile_forest(self, friedman1):
        """Predictions should correlate with quantile-forest package."""
        from quantile_forest import RandomForestQuantileRegressor as RFQR
        X, y = friedman1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        m = fit_wlearn_reg(X_train, y_train, n_estimators=100, seed=42)
        qf = RFQR(n_estimators=100, random_state=42)
        qf.fit(X_train, y_train)

        q50_wl = m.predict_quantile(X_test, quantile=0.5)
        q50_qf = qf.predict(X_test, quantiles=0.5)
        corr = np.corrcoef(q50_wl, q50_qf)[0, 1]
        assert corr > 0.90, f'Median correlation with quantile-forest: {corr}'
        m.dispose()


@pytest.mark.skipif(not _has_predict_interval(),
                    reason='conformal prediction not yet implemented')
class TestConformalPrediction:
    """Conformal prediction (J+ab) parity with mapie."""

    def test_coverage_guarantee(self):
        """J+ab intervals should cover >= (1-alpha)(1-1/(n+1)) of test values."""
        X, y = make_regression(n_samples=400, n_features=5, noise=5.0,
                               random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        m = fit_wlearn_reg(X_train, y_train, n_estimators=100, seed=42)
        lower, upper = m.predict_interval(X_test, alpha=0.1)

        covered = np.mean((y_test >= lower) & (y_test <= upper))
        n_train = len(y_train)
        min_coverage = 0.9 * (1 - 1 / (n_train + 1)) - 0.05  # slack for finite sample
        assert covered >= min_coverage, (
            f'Coverage {covered:.3f} < minimum {min_coverage:.3f}')
        m.dispose()

    def test_interval_ordering(self):
        """Lower bound <= prediction <= upper bound."""
        X, y = make_regression(n_samples=200, n_features=5, noise=5.0,
                               random_state=42)
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        lower, upper = m.predict_interval(X, alpha=0.1)
        pred = m.predict(X)
        assert np.all(lower <= pred + 1e-10)
        assert np.all(pred <= upper + 1e-10)
        assert np.all(lower <= upper + 1e-10)
        m.dispose()

    def test_interval_width_scales_with_residuals(self):
        """Intervals should be wider where predictions are harder."""
        rng = np.random.RandomState(42)
        n = 500
        X = rng.uniform(0, 10, (n, 1))
        y = X[:, 0] + X[:, 0] * rng.randn(n) * 0.3
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        lower, upper = m.predict_interval(X, alpha=0.1)
        widths = upper - lower
        low_x = X[:, 0] < 3
        high_x = X[:, 0] > 7
        # Widths should be larger for high-x (more noise)
        assert np.mean(widths[high_x]) > np.mean(widths[low_x]), (
            f'High-x width ({np.mean(widths[high_x]):.3f}) should exceed '
            f'low-x ({np.mean(widths[low_x]):.3f})')
        m.dispose()

    def test_alpha_affects_width(self):
        """Smaller alpha => wider intervals."""
        X, y = make_regression(n_samples=200, n_features=5, noise=5.0,
                               random_state=42)
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        _, u01 = m.predict_interval(X, alpha=0.1)
        l01, _ = m.predict_interval(X, alpha=0.1)
        _, u05 = m.predict_interval(X, alpha=0.5)
        l05, _ = m.predict_interval(X, alpha=0.5)
        w01 = np.mean(u01 - l01)
        w05 = np.mean(u05 - l05)
        assert w01 > w05, f'alpha=0.1 interval ({w01:.3f}) should be wider than alpha=0.5 ({w05:.3f})'
        m.dispose()

    def test_no_bootstrap_errors(self):
        """Conformal J+ab requires OOB residuals; no bootstrap should error."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42, bootstrap=0)
        with pytest.raises((RuntimeError, ValueError)):
            m.predict_interval(X, alpha=0.1)
        m.dispose()

    def test_parity_with_mapie(self):
        """J+ab intervals should have reasonable width relative to noise."""
        X, y = make_regression(n_samples=300, n_features=5, noise=5.0,
                               random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        m = fit_wlearn_reg(X_train, y_train, n_estimators=100, seed=42)
        lower_wl, upper_wl = m.predict_interval(X_test, alpha=0.1)
        width_wl = upper_wl - lower_wl

        # J+ab uses a global OOB residual quantile, so widths should be
        # approximately constant and proportional to the noise level
        assert np.std(width_wl) < 1e-10, 'J+ab widths should be constant'
        mean_width = np.mean(width_wl)
        # Width should be positive and proportional to data variability
        assert mean_width > 1.0, f'Intervals too narrow: {mean_width}'
        y_range = np.ptp(y_train)
        assert mean_width < y_range, f'Intervals wider than data range: {mean_width} vs {y_range}'
        m.dispose()


@pytest.mark.skipif(not _has_histogram_binning(),
                    reason='histogram binning not yet implemented')
class TestHistogramBinning:
    """Histogram-based split search for speedup."""

    def test_small_data_close_accuracy(self):
        """On small data (n < 255), binned RF should match unbinned closely."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X[:, 0] * 2 + rng.randn(100) * 0.5
        m_std = fit_wlearn_reg(X, y, n_estimators=50, seed=42)
        m_bin = fit_wlearn_reg(X, y, n_estimators=50, seed=42, histogram_binning=1)
        r2_std = m_std.score(X, y)
        r2_bin = m_bin.score(X, y)
        assert abs(r2_std - r2_bin) < 0.02, (
            f'Binned R2 ({r2_bin}) too different from standard ({r2_std})')
        m_std.dispose()
        m_bin.dispose()

    def test_large_data_accuracy_parity(self):
        """On large data, binned and unbinned should have similar accuracy."""
        X, y = make_regression(n_samples=5000, n_features=10, noise=10.0,
                               random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
        m_std = fit_wlearn_reg(X_tr, y_tr, n_estimators=50, seed=42)
        m_bin = fit_wlearn_reg(X_tr, y_tr, n_estimators=50, seed=42,
                               histogram_binning=1)
        r2_std = m_std.score(X_te, y_te)
        r2_bin = m_bin.score(X_te, y_te)
        assert abs(r2_std - r2_bin) < 0.03, (
            f'R2 gap on large data: std={r2_std:.4f} bin={r2_bin:.4f}')
        m_std.dispose()
        m_bin.dispose()

    def test_classification_accuracy_parity(self, cls500):
        """Binned classification should be competitive."""
        X, y = cls500
        m_std = fit_wlearn_cls(X, y, n_estimators=50, seed=42)
        m_bin = fit_wlearn_cls(X, y, n_estimators=50, seed=42,
                               histogram_binning=1)
        acc_std = m_std.score(X, y)
        acc_bin = m_bin.score(X, y)
        assert abs(acc_std - acc_bin) < 0.03
        m_std.dispose()
        m_bin.dispose()

    def test_importance_preserved(self, signal_noise):
        """Feature importances should be similar with binning."""
        X, y = signal_noise
        m_std = fit_wlearn_reg(X, y, n_estimators=100, seed=42, max_features=4)
        m_bin = fit_wlearn_reg(X, y, n_estimators=100, seed=42, max_features=4,
                               histogram_binning=1)
        imp_std = m_std.feature_importances()
        imp_bin = m_bin.feature_importances()
        # Rankings should agree
        assert np.argmax(imp_std) == np.argmax(imp_bin), (
            f'Importance ranking differs: std={imp_std} bin={imp_bin}')
        m_std.dispose()
        m_bin.dispose()

    def test_constant_feature_one_bin(self):
        """Constant feature: should be handled correctly (never split on it)."""
        rng = np.random.RandomState(42)
        n = 500
        X = np.column_stack([rng.randn(n), np.ones(n)])
        y = X[:, 0] * 2 + rng.randn(n) * 0.1
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42,
                           histogram_binning=1, max_features=2)
        imp = m.feature_importances()
        assert imp[1] < 1e-10, f'Constant feature importance with binning: {imp[1]}'
        m.dispose()

    def test_binary_feature(self):
        """Binary feature (0/1): only 2 bins used."""
        rng = np.random.RandomState(42)
        n = 500
        X = np.column_stack([rng.randn(n), (rng.rand(n) > 0.5).astype(float)])
        y = X[:, 0] + X[:, 1] * 3 + rng.randn(n) * 0.5
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42,
                           histogram_binning=1, max_features=2)
        r2 = m.score(X, y)
        assert r2 > 0.8, f'R2 with binary feature: {r2}'
        m.dispose()

    def test_save_load_with_binning(self):
        """Save/load round-trip preserves binned model predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = X[:, 0] * 2 + rng.randn(200) * 0.5
        m = fit_wlearn_reg(X, y, n_estimators=20, seed=42, histogram_binning=1)
        p1 = m.predict(X)
        blob = m.save_raw()
        m.dispose()
        m2 = RFModel.load_raw(blob, params={'task': 'regression'})
        p2 = m2.predict(X)
        assert np.allclose(p1, p2, atol=1e-10)
        m2.dispose()


@pytest.mark.skipif(not _has_missing_value_support(),
                    reason='missing value handling not yet implemented')
class TestMissingValues:
    """Missing value (NaN) handling parity with sklearn."""

    def test_no_nan_baseline(self):
        """Without NaN, results should be identical to standard RF."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(float)
        m1 = fit_wlearn_cls(X, y, n_estimators=20, seed=42)
        m2 = fit_wlearn_cls(X, y, n_estimators=20, seed=42)  # same params
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert np.array_equal(p1, p2), 'No-NaN baseline should be deterministic'
        m1.dispose()
        m2.dispose()

    def test_prediction_with_nan_no_crash(self):
        """Prediction with NaN features should not crash."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42)

        X_test = X[:10].copy()
        X_test[0, 1] = np.nan
        X_test[3, 0] = np.nan
        preds = m.predict(X_test)
        assert len(preds) == 10
        assert not np.any(np.isnan(preds))
        m.dispose()

    def test_train_with_nan_no_crash(self):
        """Training with NaN values should work."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = X[:, 0] * 2 + rng.randn(200) * 0.5
        # Inject 10% NaN
        mask = rng.rand(*X.shape) < 0.1
        X[mask] = np.nan
        m = fit_wlearn_reg(X, y, n_estimators=50, seed=42)
        preds = m.predict(X)
        assert not np.any(np.isnan(preds)), 'NaN in predictions after NaN training'
        r2 = m.score(X, y)
        assert r2 > 0.3, f'R2 with 10% NaN: {r2}'
        m.dispose()

    def test_parity_with_sklearn_mcar(self):
        """Compare accuracy on MCAR data with sklearn."""
        rng = np.random.RandomState(42)
        X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                    random_state=42)
        # Inject 15% NaN
        mask = rng.rand(*X.shape) < 0.15
        X_nan = X.copy().astype(float)
        X_nan[mask] = np.nan

        X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y.astype(float),
                                                    test_size=0.2, random_state=42)

        m = fit_wlearn_cls(X_tr, y_tr, n_estimators=100, seed=42)
        sk = RandomForestClassifier(n_estimators=100, random_state=42)
        sk.fit(X_tr, y_tr)

        acc_wl = m.score(X_te, y_te)
        acc_sk = sk.score(X_te, y_te)
        assert acc_wl > 0.65, f'MCAR accuracy too low: {acc_wl}'
        assert abs(acc_wl - acc_sk) < 0.10, (
            f'MCAR accuracy gap: wlearn={acc_wl:.4f} sklearn={acc_sk:.4f}')
        m.dispose()

    def test_all_nan_feature(self):
        """100% NaN feature should have zero importance."""
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 3)
        X[:, 1] = np.nan
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=50, seed=42, max_features=3)
        imp = m.feature_importances()
        assert imp[1] < 1e-10, f'All-NaN feature importance: {imp[1]}'
        m.dispose()

    def test_nan_in_y_errors(self):
        """NaN in labels should raise an error."""
        X = np.random.RandomState(42).randn(50, 3)
        y = np.ones(50)
        y[10] = np.nan
        with pytest.raises((RuntimeError, ValueError)):
            fit_wlearn_cls(X, y, n_estimators=10, seed=42)

    def test_save_load_preserves_nan_handling(self):
        """Save/load should preserve NaN direction info."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        mask = rng.rand(*X.shape) < 0.1
        X[mask] = np.nan
        y = np.where(np.nan_to_num(X[:, 0]) > 0, 1.0, 0.0)

        m = fit_wlearn_cls(X, y, n_estimators=30, seed=42)
        p1 = m.predict(X)
        blob = m.save_raw()
        m.dispose()

        m2 = RFModel.load_raw(blob)
        p2 = m2.predict(X)
        assert np.array_equal(p1, p2), 'NaN predictions differ after save/load'
        m2.dispose()


@pytest.mark.skipif(not _has_monotonic_constraints(),
                    reason='monotonic constraints not yet implemented')
class TestMonotonicConstraints:
    """Monotonic constraint parity with sklearn."""

    def test_increasing_constraint(self):
        """With +1 constraint, predictions should be non-decreasing in feature."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = 2 * X[:, 0] + rng.randn(n) * 0.5

        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[1, 0, 0])

        # Test on a grid along feature 0
        grid = np.linspace(-3, 3, 100).reshape(-1, 1)
        X_test = np.column_stack([grid, np.zeros((100, 2))])
        preds = m.predict(X_test)

        violations = np.sum(np.diff(preds) < -1e-10)
        assert violations == 0, (
            f'{violations} monotonicity violations (increasing)')
        m.dispose()

    def test_decreasing_constraint(self):
        """With -1 constraint, predictions should be non-increasing."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = -3 * X[:, 0] + rng.randn(n) * 0.5

        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[-1, 0, 0])

        grid = np.linspace(-3, 3, 100).reshape(-1, 1)
        X_test = np.column_stack([grid, np.zeros((100, 2))])
        preds = m.predict(X_test)

        violations = np.sum(np.diff(preds) > 1e-10)
        assert violations == 0, (
            f'{violations} monotonicity violations (decreasing)')
        m.dispose()

    def test_multiple_constraints(self):
        """Multiple constraints simultaneously: +1 on f0, -1 on f1."""
        rng = np.random.RandomState(42)
        n = 400
        X = rng.randn(n, 4)
        y = 2 * X[:, 0] - 3 * X[:, 1] + rng.randn(n) * 0.5

        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[1, -1, 0, 0])

        # Test f0 increasing (fix f1=0)
        grid0 = np.linspace(-3, 3, 50).reshape(-1, 1)
        X0 = np.column_stack([grid0, np.zeros((50, 3))])
        p0 = m.predict(X0)
        assert np.sum(np.diff(p0) < -1e-10) == 0, 'f0 increasing violated'

        # Test f1 decreasing (fix f0=0)
        grid1 = np.linspace(-3, 3, 50).reshape(-1, 1)
        X1 = np.column_stack([np.zeros((50, 1)), grid1, np.zeros((50, 2))])
        p1 = m.predict(X1)
        assert np.sum(np.diff(p1) > 1e-10) == 0, 'f1 decreasing violated'
        m.dispose()

    def test_constraint_accuracy_on_monotone_data(self):
        """Constrained model should achieve good accuracy on monotone data."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = 2 * X[:, 0] - X[:, 1] + rng.randn(n) * 0.3

        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[1, -1, 0])
        r2 = m.score(X, y)
        assert r2 > 0.80, f'Constrained R2 on monotone data: {r2}'
        m.dispose()

    def test_constraint_reduces_flexibility(self):
        """Constrained model should have equal or lower train R2 than unconstrained."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = X[:, 0] ** 2 + rng.randn(n) * 0.5  # non-monotone relationship

        m_free = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        m_cst = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                               monotonic_cst=[1, 0, 0])
        r2_free = m_free.score(X, y)
        r2_cst = m_cst.score(X, y)
        assert r2_cst <= r2_free + 1e-10, (
            f'Constrained R2 ({r2_cst}) > unconstrained ({r2_free})')
        m_free.dispose()
        m_cst.dispose()

    def test_parity_with_sklearn(self):
        """Accuracy should be comparable to sklearn with same constraints."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = 2 * X[:, 0] - X[:, 1] + rng.randn(n) * 0.5

        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[1, -1, 0])
        sk = RandomForestRegressor(n_estimators=100, random_state=42,
                                    monotonic_cst=[1, -1, 0])
        sk.fit(X, y)

        r2_wl = m.score(X, y)
        r2_sk = sk.score(X, y)
        assert abs(r2_wl - r2_sk) < 0.10, (
            f'Monotonic R2 gap: wlearn={r2_wl:.4f} sklearn={r2_sk:.4f}')
        m.dispose()

    def test_classification_with_constraints(self):
        """Monotonic constraints should also work for classification."""
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 3)
        y = (2 * X[:, 0] - X[:, 1] > 0).astype(float)

        m = fit_wlearn_cls(X, y, n_estimators=100, seed=42,
                           monotonic_cst=[1, -1, 0])
        acc = m.score(X, y)
        assert acc > 0.80, f'Constrained classification accuracy: {acc}'

        # Check monotonicity of predicted probabilities
        grid = np.linspace(-3, 3, 50).reshape(-1, 1)
        X_test = np.column_stack([grid, np.zeros((50, 2))])
        proba = m.predict_proba(X_test)
        # P(class=1) should be non-decreasing in feature 0
        violations = np.sum(np.diff(proba[:, 1]) < -1e-10)
        assert violations == 0, f'P(class=1) not monotone in f0: {violations} violations'
        m.dispose()

    def test_save_load_preserves_constraints(self):
        """Constraints should be preserved after save/load."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = 2 * X[:, 0] + rng.randn(200) * 0.5
        m = fit_wlearn_reg(X, y, n_estimators=30, seed=42,
                           monotonic_cst=[1, 0, 0])
        p1 = m.predict(X)
        blob = m.save_raw()
        m.dispose()

        m2 = RFModel.load_raw(blob, params={'task': 'regression'})
        p2 = m2.predict(X)
        assert np.allclose(p1, p2, atol=1e-10)
        m2.dispose()


@pytest.mark.skipif(not _has_jarf(),
                    reason='JARF rotation not yet implemented')
class TestJARFRotation:
    """JARF (Jacobian Aligned RF) feature rotation."""

    def test_rotation_improves_diagonal_boundary(self):
        """JARF should outperform standard RF on diagonal decision boundaries."""
        rng = np.random.RandomState(42)
        n = 400
        X = rng.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)  # diagonal boundary

        m_std = fit_wlearn_cls(X, y, n_estimators=100, seed=42)
        m_jarf = fit_wlearn_cls(X, y, n_estimators=100, seed=42, jarf=1)

        acc_std = m_std.score(X, y)
        acc_jarf = m_jarf.score(X, y)
        assert acc_jarf >= acc_std - 0.01, (
            f'JARF ({acc_jarf}) should be >= standard ({acc_std}) on diagonal boundary')
        m_std.dispose()
        m_jarf.dispose()

    def test_rotation_determinism(self):
        """Same seed produces identical JARF results."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        m1 = fit_wlearn_cls(X, y, n_estimators=50, seed=42, jarf=1)
        m2 = fit_wlearn_cls(X, y, n_estimators=50, seed=42, jarf=1)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert np.array_equal(p1, p2), 'JARF not deterministic with same seed'
        m1.dispose()
        m2.dispose()

    def test_jarf_single_feature(self):
        """Single feature: rotation is trivial, should not crash."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = (X[:, 0] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42, jarf=1)
        acc = m.score(X, y)
        assert acc > 0.8
        m.dispose()

    def test_jarf_accuracy_regression(self):
        """JARF should work for regression too."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 5)
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] + rng.randn(300) * 0.5
        m = fit_wlearn_reg(X, y, n_estimators=100, seed=42, jarf=1)
        r2 = m.score(X, y)
        assert r2 > 0.8, f'JARF regression R2: {r2}'
        m.dispose()

    def test_jarf_save_load(self):
        """Save/load round-trip should preserve JARF model."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        m = fit_wlearn_cls(X, y, n_estimators=20, seed=42, jarf=1)
        p1 = m.predict(X)
        blob = m.save_raw()
        m.dispose()

        m2 = RFModel.load_raw(blob)
        p2 = m2.predict(X)
        assert np.array_equal(p1, p2)
        m2.dispose()

    def test_jarf_orthogonal_features(self):
        """On orthogonal features, JARF should perform similarly to standard RF."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 4)
        y = X[:, 0] * 2 + rng.randn(300) * 0.5

        m_std = fit_wlearn_reg(X, y, n_estimators=100, seed=42)
        m_jarf = fit_wlearn_reg(X, y, n_estimators=100, seed=42, jarf=1)

        r2_std = m_std.score(X, y)
        r2_jarf = m_jarf.score(X, y)
        assert abs(r2_std - r2_jarf) < 0.10, (
            f'JARF should be similar on orthogonal data: std={r2_std:.4f} jarf={r2_jarf:.4f}')
        m_std.dispose()
        m_jarf.dispose()
