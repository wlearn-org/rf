"""
test_python.py -- Python tests for wlearn_rf
"""
import sys
import os
import math

# Add the Python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'python'))

import numpy as np
from wlearn_rf import RFModel

tests_run = 0
tests_passed = 0


def test(name, fn):
    global tests_run, tests_passed
    tests_run += 1
    try:
        fn()
        print(f'  PASS: {name}')
        tests_passed += 1
    except Exception as e:
        print(f'  FAIL: {name}')
        print(f'        {e}')


def lcg(seed):
    """LCG matching @wlearn/core rng.js"""
    s = seed & 0x7FFFFFFF
    def _next():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    return _next


def make_cls_data(seed, n, d, n_classes=2):
    rng = lcg(seed)
    X = np.zeros((n, d))
    y = np.zeros(n)
    for i in range(n):
        label = i % n_classes
        for j in range(d):
            X[i, j] = label * 2 + (rng() - 0.5) * 0.5
        y[i] = label
    return X, y


def make_reg_data(seed, n, d):
    rng = lcg(seed)
    X = np.zeros((n, d))
    y = np.zeros(n)
    for i in range(n):
        target = 0
        for j in range(d):
            v = rng() * 4 - 2
            X[i, j] = v
            target += v * (j + 1)
        y[i] = target + (rng() - 0.5) * 0.1
    return X, y


# === Classification ===
print('\n=== Classification ===')


def test_binary_cls():
    X, y = make_cls_data(100, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 100})
    m.fit(X, y)
    assert m.is_fitted
    assert m.n_features == 2
    assert m.n_classes == 2
    assert m.n_trees == 20
    acc = m.score(X, y)
    assert acc > 0.8, f'accuracy too low: {acc}'
    m.dispose()

test('Binary classification', test_binary_cls)


def test_multiclass_cls():
    X, y = make_cls_data(200, 90, 3, 3)
    m = RFModel({'n_estimators': 30, 'seed': 200})
    m.fit(X, y)
    assert m.n_classes == 3
    acc = m.score(X, y)
    assert acc > 0.7, f'accuracy too low: {acc}'
    m.dispose()

test('Multiclass classification', test_multiclass_cls)


def test_predict_proba():
    X, y = make_cls_data(300, 60, 2)
    m = RFModel({'n_estimators': 20, 'seed': 300})
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (60, 2), f'shape: {proba.shape}'
    # Row sums should be 1
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f'row sums: {row_sums}'
    # All values in [0, 1]
    assert proba.min() >= 0 and proba.max() <= 1
    m.dispose()

test('predict_proba', test_predict_proba)


def test_proba_fails_regression():
    X, y = make_reg_data(301, 30, 2)
    m = RFModel({'n_estimators': 10, 'task': 'regression', 'seed': 301})
    m.fit(X, y)
    try:
        m.predict_proba(X)
        assert False, 'should have raised'
    except RuntimeError:
        pass
    m.dispose()

test('predict_proba fails for regression', test_proba_fails_regression)


def test_extra_trees():
    X, y = make_cls_data(302, 80, 3)
    m = RFModel({'n_estimators': 30, 'seed': 302, 'extra_trees': 1})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.7, f'ExtraTrees accuracy too low: {acc}'
    m.dispose()

test('ExtraTrees classification', test_extra_trees)


# === Regression ===
print('\n=== Regression ===')


def test_regression():
    X, y = make_reg_data(400, 80, 2)
    m = RFModel({'n_estimators': 20, 'task': 'regression', 'seed': 400})
    m.fit(X, y)
    assert m.n_classes == 0
    r2 = m.score(X, y)
    assert r2 > 0.3, f'R2 too low: {r2}'
    m.dispose()

test('Regression basic', test_regression)


# === Feature importances + OOB ===
print('\n=== Feature importances + OOB ===')


def test_importances_sum():
    X, y = make_cls_data(500, 80, 3)
    m = RFModel({'n_estimators': 20, 'seed': 500, 'max_features': 3})
    m.fit(X, y)
    imp = m.feature_importances()
    assert len(imp) == 3
    assert abs(imp.sum() - 1.0) < 1e-10, f'sum: {imp.sum()}'
    m.dispose()

test('Feature importances sum to 1', test_importances_sum)


def test_dominant_feature():
    rng = lcg(501)
    n = 200
    X = np.zeros((n, 4))
    y = np.zeros(n)
    rng2 = lcg(999)
    rng3 = lcg(777)
    rng4 = lcg(333)
    for i in range(n):
        X[i, 0] = rng()
        X[i, 1] = rng2()
        X[i, 2] = rng3()
        X[i, 3] = rng4()
        y[i] = 5.0 * X[i, 0]
    m = RFModel({'n_estimators': 100, 'task': 'regression', 'seed': 501, 'max_features': 4})
    m.fit(X, y)
    imp = m.feature_importances()
    for j in range(1, 4):
        assert imp[0] > imp[j], f'feature 0 ({imp[0]}) should be > feature {j} ({imp[j]})'
    m.dispose()

test('Dominant feature has highest importance', test_dominant_feature)


def test_oob_cls():
    X, y = make_cls_data(502, 80, 2)
    m = RFModel({'n_estimators': 30, 'seed': 502})
    m.fit(X, y)
    oob = m.oob_score()
    assert not math.isnan(oob), 'OOB is NaN'
    assert oob > 0.5, f'OOB too low: {oob}'
    m.dispose()

test('OOB score (classification)', test_oob_cls)


def test_oob_reg():
    X, y = make_reg_data(503, 80, 2)
    m = RFModel({'n_estimators': 30, 'task': 'regression', 'seed': 503})
    m.fit(X, y)
    oob = m.oob_score()
    assert not math.isnan(oob), 'OOB is NaN'
    m.dispose()

test('OOB score (regression)', test_oob_reg)


# === Save/load ===
print('\n=== Save/load ===')


def test_save_load_cls():
    X, y = make_cls_data(800, 50, 2)
    m = RFModel({'n_estimators': 15, 'seed': 800})
    m.fit(X, y)
    p1 = m.predict(X)
    blob = m.save_raw()
    m.dispose()

    assert len(blob) > 0
    assert blob[:4] == b'RF01', f'magic: {blob[:4]}'

    m2 = RFModel.load_raw(blob)
    assert m2.is_fitted
    assert m2.n_trees == 15
    p2 = m2.predict(X)
    assert np.allclose(p1, p2), 'predictions differ after load'
    m2.dispose()

test('Classification save/load', test_save_load_cls)


def test_save_load_reg():
    X, y = make_reg_data(801, 50, 2)
    m = RFModel({'n_estimators': 15, 'task': 'regression', 'seed': 801})
    m.fit(X, y)
    p1 = m.predict(X)
    blob = m.save_raw()
    m.dispose()

    m2 = RFModel.load_raw(blob, params={'task': 'regression'})
    p2 = m2.predict(X)
    assert np.allclose(p1, p2, atol=1e-10), 'predictions differ after load'
    m2.dispose()

test('Regression save/load', test_save_load_reg)


def test_importances_preserved():
    rng = lcg(803)
    n = 80
    X = np.zeros((n, 3))
    y = np.zeros(n)
    for i in range(n):
        X[i, 0] = rng()
        X[i, 1] = rng()
        X[i, 2] = rng()
        y[i] = 1 if X[i, 0] > 0.5 else 0
    m = RFModel({'n_estimators': 30, 'seed': 803, 'max_features': 3})
    m.fit(X, y)
    imp1 = m.feature_importances()
    blob = m.save_raw()
    m.dispose()

    m2 = RFModel.load_raw(blob)
    imp2 = m2.feature_importances()
    assert np.allclose(imp1, imp2, atol=1e-10), f'importances differ: {imp1} vs {imp2}'
    m2.dispose()

test('Feature importances preserved after load', test_importances_preserved)


# === Determinism ===
print('\n=== Determinism ===')


def test_determinism():
    X, y = make_cls_data(700, 60, 2)
    m1 = RFModel({'n_estimators': 15, 'seed': 700})
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = RFModel({'n_estimators': 15, 'seed': 700})
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2), 'same seed should give same predictions'
    m1.dispose()
    m2.dispose()

test('Same seed -> same predictions', test_determinism)


# === Disposal ===
print('\n=== Disposal ===')


def test_dispose():
    X, y = make_cls_data(900, 20, 2)
    m = RFModel({'n_estimators': 10, 'seed': 900})
    m.fit(X, y)
    m.dispose()
    assert not m.is_fitted
    try:
        m.predict(X)
        assert False, 'should have raised'
    except RuntimeError:
        pass

test('dispose prevents further use', test_dispose)


def test_double_dispose():
    m = RFModel({'n_estimators': 10})
    m.dispose()
    m.dispose()  # should not raise

test('double dispose is safe', test_double_dispose)


# === New v0.2 features ===
print('\n=== Entropy Criterion ===')


def test_entropy_cls():
    X, y = make_cls_data(600, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 600, 'criterion': 'entropy'})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.8, f'entropy accuracy too low: {acc}'
    m.dispose()

test('Entropy classification', test_entropy_cls)


def test_hellinger_cls():
    X, y = make_cls_data(601, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 601, 'criterion': 'hellinger'})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.8, f'hellinger accuracy too low: {acc}'
    m.dispose()

test('Hellinger classification', test_hellinger_cls)


print('\n=== MAE Criterion ===')


def test_mae_reg():
    X, y = make_reg_data(602, 80, 2)
    m = RFModel({'n_estimators': 20, 'task': 'regression', 'seed': 602, 'criterion': 'mae'})
    m.fit(X, y)
    r2 = m.score(X, y)
    assert r2 > 0.3, f'mae R2 too low: {r2}'
    m.dispose()

test('MAE regression', test_mae_reg)


print('\n=== Sample Rate ===')


def test_sample_rate():
    X, y = make_cls_data(603, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 603, 'sample_rate': 0.5})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.8, f'sample_rate=0.5 accuracy too low: {acc}'
    m.dispose()

test('Sample rate 0.5', test_sample_rate)


def test_sample_rate_no_bootstrap():
    X, y = make_cls_data(604, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 604, 'sample_rate': 0.5, 'bootstrap': 0})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.8, f'sample_rate no-bootstrap accuracy too low: {acc}'
    m.dispose()

test('Sample rate 0.5 no bootstrap', test_sample_rate_no_bootstrap)


print('\n=== Heterogeneous RF ===')


def test_heterogeneous():
    X, y = make_reg_data(605, 150, 4)
    m = RFModel({'n_estimators': 30, 'task': 'regression', 'seed': 605,
                 'heterogeneous': 1, 'max_features': 4})
    m.fit(X, y)
    r2 = m.score(X, y)
    assert r2 > 0.3, f'HRF R2 too low: {r2}'
    m.dispose()

test('Heterogeneous RF', test_heterogeneous)


print('\n=== OOB Weighting ===')


def test_oob_weighting():
    X, y = make_cls_data(606, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 606, 'oob_weighting': 1})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.8, f'oob_weighting accuracy too low: {acc}'
    # Save/load should preserve
    blob = m.save_raw()
    m.dispose()
    m2 = RFModel.load_raw(blob)
    assert m2.is_fitted
    p2 = m2.predict(X)
    assert len(p2) == 80
    m2.dispose()

test('OOB weighted voting', test_oob_weighting)


print('\n=== Alpha Trim ===')


def test_alpha_trim():
    X, y = make_cls_data(607, 80, 2)
    m = RFModel({'n_estimators': 20, 'seed': 607, 'alpha_trim': 0.05})
    m.fit(X, y)
    acc = m.score(X, y)
    assert acc > 0.7, f'alpha_trim accuracy too low: {acc}'
    m.dispose()

test('Cost-complexity pruning', test_alpha_trim)


print('\n=== Local Linear Leaves ===')


def test_linear_leaves():
    rng = lcg(608)
    n = 200
    X = np.zeros((n, 2))
    y = np.zeros(n)
    for i in range(n):
        X[i, 0] = rng() * 4 - 2
        X[i, 1] = rng() * 4 - 2
        y[i] = 3.0 * X[i, 0] + 2.0 * X[i, 1] + 1.0

    # Constant leaves with shallow trees
    m_const = RFModel({'n_estimators': 20, 'task': 'regression', 'seed': 608,
                        'max_depth': 3, 'leaf_model': 0})
    m_const.fit(X, y)
    r2_const = m_const.score(X, y)

    # Linear leaves with shallow trees
    m_lin = RFModel({'n_estimators': 20, 'task': 'regression', 'seed': 608,
                      'max_depth': 3, 'leaf_model': 1})
    m_lin.fit(X, y)
    r2_lin = m_lin.score(X, y)

    assert r2_lin > r2_const, f'linear R2 ({r2_lin}) should exceed constant R2 ({r2_const})'
    assert r2_lin > 0.95, f'linear R2 too low: {r2_lin}'

    # Save/load round-trip
    blob = m_lin.save_raw()
    m_loaded = RFModel.load_raw(blob, params={'task': 'regression'})
    p_orig = m_lin.predict(X)
    p_loaded = m_loaded.predict(X)
    assert np.allclose(p_orig, p_loaded, atol=1e-10), 'linear leaf predictions differ after load'

    m_const.dispose()
    m_lin.dispose()
    m_loaded.dispose()

test('Local linear leaves', test_linear_leaves)


print('\n=== V2 Serialization ===')


def test_v2_round_trip():
    X, y = make_cls_data(609, 60, 2)
    m = RFModel({'n_estimators': 15, 'seed': 609, 'criterion': 'entropy',
                 'sample_rate': 0.8})
    m.fit(X, y)
    p1 = m.predict(X)
    blob = m.save_raw()

    # Version field should be 2
    import struct
    ver = struct.unpack('<I', blob[4:8])[0]
    assert ver == 2, f'expected format version 2, got {ver}'

    m2 = RFModel.load_raw(blob)
    p2 = m2.predict(X)
    assert np.array_equal(p1, p2), 'v2 predictions differ after load'
    m.dispose()
    m2.dispose()

test('V2 save/load round-trip', test_v2_round_trip)


def test_defaults_unchanged():
    # Default params should produce bit-identical results
    X, y = make_cls_data(100, 80, 2)
    m1 = RFModel({'n_estimators': 20, 'seed': 100})
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = RFModel({'n_estimators': 20, 'seed': 100,
                   'criterion': 0, 'sample_rate': 1.0, 'heterogeneous': 0,
                   'oob_weighting': 0, 'alpha_trim': 0.0, 'leaf_model': 0})
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2), 'explicit defaults should match implicit defaults'
    m1.dispose()
    m2.dispose()

test('Explicit defaults match implicit defaults', test_defaults_unchanged)


# === Summary ===
print(f'\n{tests_run} tests: {tests_passed} passed, {tests_run - tests_passed} failed\n')
sys.exit(0 if tests_passed == tests_run else 1)
