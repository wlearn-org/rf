"""
wlearn_rf -- Random Forest for Python via C11 core.

Classification and regression with CART splits (Gini/entropy/Hellinger for
classification; MSE/MAE for regression), bootstrap with adjustable sample rate,
ExtraTrees, heterogeneous RF, OOB scoring with optional per-tree weighting,
cost-complexity pruning, local linear leaves, and MDI feature importances.
"""

import ctypes
import numpy as np
from ._ffi import _load_lib, RFParams


_CLS_CRITERIA = {'gini': 0, 'entropy': 1, 'hellinger': 2}
_REG_CRITERIA = {'mse': 0, 'squared_error': 0, 'mae': 1, 'absolute_error': 1}


def _resolve_criterion(value, task):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if task == 0:
            return _CLS_CRITERIA.get(value.lower(), 0)
        else:
            return _REG_CRITERIA.get(value.lower(), 0)
    return 0


class RFModel:
    """Random Forest classifier/regressor backed by C11 core."""

    def __init__(self, params=None):
        self._handle = None
        self._fitted = False
        self._disposed = False
        self._params = dict(params or {})
        self._n_features = 0
        self._n_classes = 0

    def fit(self, X, y):
        """Fit the random forest on training data."""
        if self._disposed:
            raise RuntimeError('RFModel has been disposed.')

        lib = _load_lib()

        # Dispose previous if refitting
        if self._handle:
            lib.rf_free(self._handle)
            self._handle = None

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape
        if y.shape[0] != nrow:
            raise ValueError(f'y length ({y.shape[0]}) does not match X rows ({nrow})')

        task = 1 if self._params.get('task') == 'regression' else 0

        # Resolve max_features
        max_features = self._params.get('max_features', 0)
        if isinstance(max_features, str):
            if max_features == 'sqrt':
                max_features = max(1, round(ncol ** 0.5))
            elif max_features == 'log2':
                max_features = max(1, round(np.log2(ncol)))
            elif max_features == 'third':
                max_features = max(1, round(ncol / 3))
            else:
                max_features = 0
        if not isinstance(max_features, int):
            max_features = int(max_features) if max_features else 0

        # Build params struct
        params = RFParams()
        lib.rf_params_init(ctypes.byref(params))
        params.n_estimators = self._params.get('n_estimators', 100)
        params.max_depth = self._params.get('max_depth', 0)
        params.min_samples_split = self._params.get('min_samples_split', 2)
        params.min_samples_leaf = self._params.get('min_samples_leaf', 1)
        params.max_features = max_features
        params.max_leaf_nodes = self._params.get('max_leaf_nodes', 0)
        params.bootstrap = self._params.get('bootstrap', 1)
        params.compute_oob = self._params.get('compute_oob', 1)
        params.extra_trees = self._params.get('extra_trees', 0)
        params.seed = self._params.get('seed', 42)
        params.task = task
        params.criterion = _resolve_criterion(self._params.get('criterion', 0), task)
        params.heterogeneous = self._params.get('heterogeneous', 0)
        params.oob_weighting = self._params.get('oob_weighting', 0)
        params.leaf_model = self._params.get('leaf_model', 0)
        params.store_leaf_samples = self._params.get(
            'store_leaf_samples', 1 if task == 1 else 0)
        params.sample_rate = self._params.get('sample_rate', 1.0)
        params.alpha_trim = self._params.get('alpha_trim', 0.0)

        # Monotonic constraints
        mono = self._params.get('monotonic_cst', None)
        if mono is not None:
            mono_arr = np.asarray(mono, dtype=np.int32)
            if mono_arr.shape[0] == ncol:
                self._mono_buf = mono_arr  # prevent GC
                params.monotonic_cst = mono_arr.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int32))
                params.n_monotonic_cst = ncol
            else:
                params.monotonic_cst = None
                params.n_monotonic_cst = 0
        else:
            params.monotonic_cst = None
            params.n_monotonic_cst = 0

        handle = lib.rf_fit(
            X.ctypes.data, nrow, ncol,
            y.ctypes.data,
            ctypes.byref(params)
        )

        if not handle:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_fit failed: {msg}')

        self._handle = handle
        self._fitted = True
        self._n_features = lib.wl_rf_get_n_features(handle)
        self._n_classes = lib.wl_rf_get_n_classes(handle)

        return self

    def predict(self, X):
        """Predict class labels (classification) or values (regression)."""
        self._ensure_fitted()
        lib = _load_lib()

        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape

        out = np.zeros(nrow, dtype=np.float64)
        ret = lib.rf_predict(self._handle, X.ctypes.data, nrow, ncol, out.ctypes.data)
        if ret != 0:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_predict failed: {msg}')

        return out

    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        self._ensure_fitted()
        if self._params.get('task') == 'regression':
            raise RuntimeError('predict_proba is only available for classification')

        lib = _load_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape

        nc = self._n_classes
        out = np.zeros(nrow * nc, dtype=np.float64)
        ret = lib.rf_predict_proba(self._handle, X.ctypes.data, nrow, ncol, out.ctypes.data)
        if ret != 0:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_predict_proba failed: {msg}')

        return out.reshape(nrow, nc)

    def predict_quantile(self, X, quantile=0.5):
        """Predict quantiles (quantile regression forest).

        Requires store_leaf_samples=1 during fit.
        """
        self._ensure_fitted()
        lib = _load_lib()

        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape

        quantiles = np.ascontiguousarray([quantile], dtype=np.float64)
        out = np.zeros(nrow, dtype=np.float64)
        ret = lib.rf_predict_quantile(
            self._handle, X.ctypes.data, nrow, ncol,
            quantiles.ctypes.data, 1, out.ctypes.data)
        if ret != 0:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_predict_quantile failed: {msg}')

        return out

    def predict_interval(self, X, alpha=0.1):
        """Predict conformal intervals (Jackknife+-after-Bootstrap).

        Requires bootstrap=1, compute_oob=1 during fit.
        Returns (lower, upper) arrays.
        """
        self._ensure_fitted()
        lib = _load_lib()

        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape

        out_lower = np.zeros(nrow, dtype=np.float64)
        out_upper = np.zeros(nrow, dtype=np.float64)
        ret = lib.rf_predict_interval(
            self._handle, X.ctypes.data, nrow, ncol,
            float(alpha), out_lower.ctypes.data, out_upper.ctypes.data)
        if ret != 0:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_predict_interval failed: {msg}')

        return out_lower, out_upper

    def score(self, X, y):
        """Compute accuracy (classification) or R2 (regression)."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)

        if self._params.get('task') == 'regression':
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            return np.mean(preds.astype(int) == y.astype(int))

    def feature_importances(self):
        """Return normalized MDI feature importances."""
        self._ensure_fitted()
        lib = _load_lib()
        result = np.zeros(self._n_features, dtype=np.float64)
        for i in range(self._n_features):
            result[i] = lib.wl_rf_get_feature_importance(self._handle, i)
        return result

    def oob_score(self):
        """Return OOB accuracy (classification) or R2 (regression)."""
        self._ensure_fitted()
        lib = _load_lib()
        return lib.wl_rf_get_oob_score(self._handle)

    def save_raw(self):
        """Save the forest to RF01 binary format."""
        self._ensure_fitted()
        lib = _load_lib()

        out_buf = ctypes.c_void_p()
        out_len = ctypes.c_int32()
        ret = lib.rf_save(self._handle, ctypes.byref(out_buf), ctypes.byref(out_len))
        if ret != 0:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_save failed: {msg}')

        buf_len = out_len.value
        result = bytes((ctypes.c_char * buf_len).from_address(out_buf.value))
        lib.rf_free_buffer(out_buf)
        return result

    @classmethod
    def load_raw(cls, data, params=None):
        """Load a forest from RF01 binary format."""
        lib = _load_lib()
        buf = ctypes.create_string_buffer(data)
        handle = lib.rf_load(buf, len(data))
        if not handle:
            err = lib.rf_get_error()
            msg = err.decode('utf-8') if err else 'unknown error'
            raise RuntimeError(f'rf_load failed: {msg}')

        obj = cls(params)
        obj._handle = handle
        obj._fitted = True
        obj._n_features = lib.wl_rf_get_n_features(handle)
        obj._n_classes = lib.wl_rf_get_n_classes(handle)
        return obj

    def dispose(self):
        """Free native resources."""
        if self._disposed:
            return
        self._disposed = True
        if self._handle:
            lib = _load_lib()
            lib.rf_free(self._handle)
            self._handle = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        return self

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_trees(self):
        if not self._handle or self._disposed:
            return 0
        lib = _load_lib()
        return lib.wl_rf_get_n_trees(self._handle)

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_fitted(self):
        if self._disposed:
            raise RuntimeError('RFModel has been disposed.')
        if not self._fitted:
            raise RuntimeError('RFModel is not fitted. Call fit() first.')

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and not self._disposed:
            try:
                self.dispose()
            except Exception:
                pass
