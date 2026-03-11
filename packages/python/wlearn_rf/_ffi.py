"""
_ffi.py -- ctypes wrapper around librf shared library.

Loads librf.so / librf.dylib and exposes the C API as Python functions.
"""

import ctypes
import ctypes.util
import os
import platform

_lib = None


def _find_lib():
    """Find librf shared library."""
    # 1. Check RF_LIB_PATH environment variable
    env_path = os.environ.get('RF_LIB_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check relative to this file (development layout)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()

    if system == 'Darwin':
        lib_name = 'librf.dylib'
    elif system == 'Windows':
        lib_name = 'rf.dll'
    else:
        lib_name = 'librf.so'

    search_paths = [
        os.path.join(this_dir, lib_name),
        os.path.join(this_dir, '..', '..', '..', '..', 'build', lib_name),
    ]

    for path in search_paths:
        resolved = os.path.abspath(path)
        if os.path.isfile(resolved):
            return resolved

    # 3. Try system library path
    found = ctypes.util.find_library('rf')
    if found:
        return found

    return None


def _load_lib():
    """Load the shared library and declare function signatures."""
    global _lib
    if _lib is not None:
        return _lib

    lib_path = _find_lib()
    if not lib_path:
        raise RuntimeError(
            'Could not find librf shared library. '
            'Set RF_LIB_PATH or build the C core first: '
            'mkdir build && cd build && cmake .. && make'
        )

    _lib = ctypes.CDLL(lib_path)

    # rf_params_init
    _lib.rf_params_init.argtypes = [ctypes.c_void_p]
    _lib.rf_params_init.restype = None

    # rf_fit
    _lib.rf_fit.argtypes = [
        ctypes.c_void_p,   # X
        ctypes.c_int32,    # nrow
        ctypes.c_int32,    # ncol
        ctypes.c_void_p,   # y
        ctypes.c_void_p,   # params
    ]
    _lib.rf_fit.restype = ctypes.c_void_p

    # rf_predict
    _lib.rf_predict.argtypes = [
        ctypes.c_void_p,   # forest
        ctypes.c_void_p,   # X
        ctypes.c_int32,    # nrow
        ctypes.c_int32,    # ncol
        ctypes.c_void_p,   # out
    ]
    _lib.rf_predict.restype = ctypes.c_int

    # rf_predict_proba
    _lib.rf_predict_proba.argtypes = [
        ctypes.c_void_p,   # forest
        ctypes.c_void_p,   # X
        ctypes.c_int32,    # nrow
        ctypes.c_int32,    # ncol
        ctypes.c_void_p,   # out
    ]
    _lib.rf_predict_proba.restype = ctypes.c_int

    # rf_save
    _lib.rf_save.argtypes = [
        ctypes.c_void_p,                           # forest
        ctypes.POINTER(ctypes.c_void_p),           # out_buf
        ctypes.POINTER(ctypes.c_int32),            # out_len
    ]
    _lib.rf_save.restype = ctypes.c_int

    # rf_load
    _lib.rf_load.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    _lib.rf_load.restype = ctypes.c_void_p

    # rf_free
    _lib.rf_free.argtypes = [ctypes.c_void_p]
    _lib.rf_free.restype = None

    # rf_free_buffer
    _lib.rf_free_buffer.argtypes = [ctypes.c_void_p]
    _lib.rf_free_buffer.restype = None

    # rf_get_error
    _lib.rf_get_error.argtypes = []
    _lib.rf_get_error.restype = ctypes.c_char_p

    # --- wl_rf_* getters (stable ABI, no pointer arithmetic) ---

    _lib.wl_rf_get_n_trees.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_n_trees.restype = ctypes.c_int

    _lib.wl_rf_get_n_features.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_n_features.restype = ctypes.c_int

    _lib.wl_rf_get_n_classes.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_n_classes.restype = ctypes.c_int

    _lib.wl_rf_get_task.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_task.restype = ctypes.c_int

    _lib.wl_rf_get_oob_score.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_oob_score.restype = ctypes.c_double

    _lib.wl_rf_get_feature_importance.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.wl_rf_get_feature_importance.restype = ctypes.c_double

    _lib.wl_rf_get_criterion.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_criterion.restype = ctypes.c_int

    _lib.wl_rf_get_sample_rate.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_sample_rate.restype = ctypes.c_double

    _lib.wl_rf_get_heterogeneous.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_heterogeneous.restype = ctypes.c_int

    _lib.wl_rf_get_oob_weighting.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_oob_weighting.restype = ctypes.c_int

    _lib.wl_rf_get_alpha_trim.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_alpha_trim.restype = ctypes.c_double

    _lib.wl_rf_get_leaf_model.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_leaf_model.restype = ctypes.c_int

    # rf_predict_quantile
    _lib.rf_predict_quantile.argtypes = [
        ctypes.c_void_p,   # forest
        ctypes.c_void_p,   # X
        ctypes.c_int32,    # nrow
        ctypes.c_int32,    # ncol
        ctypes.c_void_p,   # quantiles
        ctypes.c_int32,    # n_quantiles
        ctypes.c_void_p,   # out
    ]
    _lib.rf_predict_quantile.restype = ctypes.c_int

    # rf_predict_interval
    _lib.rf_predict_interval.argtypes = [
        ctypes.c_void_p,   # forest
        ctypes.c_void_p,   # X
        ctypes.c_int32,    # nrow
        ctypes.c_int32,    # ncol
        ctypes.c_double,   # alpha
        ctypes.c_void_p,   # out_lower
        ctypes.c_void_p,   # out_upper
    ]
    _lib.rf_predict_interval.restype = ctypes.c_int

    return _lib


# --- rf_params_t struct (matches C layout) ---

class RFParams(ctypes.Structure):
    _fields_ = [
        ('n_estimators', ctypes.c_int32),
        ('max_depth', ctypes.c_int32),
        ('min_samples_split', ctypes.c_int32),
        ('min_samples_leaf', ctypes.c_int32),
        ('max_features', ctypes.c_int32),
        ('max_leaf_nodes', ctypes.c_int32),
        ('bootstrap', ctypes.c_int32),
        ('compute_oob', ctypes.c_int32),
        ('extra_trees', ctypes.c_int32),
        ('seed', ctypes.c_uint32),
        ('task', ctypes.c_int32),
        ('criterion', ctypes.c_int32),
        ('heterogeneous', ctypes.c_int32),
        ('oob_weighting', ctypes.c_int32),
        ('leaf_model', ctypes.c_int32),
        ('store_leaf_samples', ctypes.c_int32),
        ('sample_rate', ctypes.c_double),
        ('alpha_trim', ctypes.c_double),
        ('monotonic_cst', ctypes.POINTER(ctypes.c_int32)),
        ('n_monotonic_cst', ctypes.c_int32),
    ]
