const { getWasm, loadRF } = require('./wasm.js')
const {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} = require('@wlearn/core')

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/rf: Model was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_rf_get_last_error', 'string', [], [])
}

function resolveMaxFeatures(value, ncol, task) {
  if (typeof value === 'string') {
    const lower = value.toLowerCase()
    if (lower === 'sqrt') return Math.max(1, Math.round(Math.sqrt(ncol)))
    if (lower === 'log2') return Math.max(1, Math.round(Math.log2(ncol)))
    if (lower === 'third') return Math.max(1, Math.round(ncol / 3))
  }
  if (typeof value === 'number' && value > 0) return value
  // auto: sqrt for classification, n/3 for regression
  if (task === 1) return Math.max(1, Math.round(ncol / 3))
  return Math.max(1, Math.round(Math.sqrt(ncol)))
}

function resolveCriterion(value, task) {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const lower = value.toLowerCase()
    if (task === 0) {
      if (lower === 'gini') return 0
      if (lower === 'entropy') return 1
      if (lower === 'hellinger') return 2
    } else {
      if (lower === 'mse' || lower === 'squared_error') return 0
      if (lower === 'mae' || lower === 'absolute_error') return 1
    }
  }
  return 0
}

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- RFModel ---

class RFModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #fitted = false
  #nFeatures = 0
  #nClasses = 0

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      // Internal: created by load() / _fromBundle()
      this.#handle = params
      this.#params = extra.params || {}
      this.#nFeatures = extra.nFeatures || 0
      this.#nClasses = extra.nClasses || 0
      this.#fitted = true
    } else {
      // Normal construction (from create())
      this.#handle = null
      this.#params = handle || {}
    }

    this.#freed = false
    if (this.#handle) {
      this.#registerLeak()
    }
  }

  static async create(params = {}) {
    await loadRF()
    return new RFModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    // Dispose previous model if refitting
    if (this.#handle) {
      wasm._wl_rf_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const task = this.#taskEnum()
    const maxFeatures = resolveMaxFeatures(this.#params.maxFeatures, cols, task)
    const criterion = resolveCriterion(this.#params.criterion, task)

    // Allocate X on WASM heap
    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    // Allocate y on WASM heap
    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const modelPtr = wasm._wl_rf_fit(
      xPtr, rows, cols,
      yPtr,
      this.#params.nEstimators ?? 100,
      this.#params.maxDepth ?? 0,
      this.#params.minSamplesSplit ?? 2,
      this.#params.minSamplesLeaf ?? 1,
      maxFeatures,
      this.#params.maxLeafNodes ?? 0,
      this.#params.bootstrap ?? 1,
      this.#params.computeOob ?? 1,
      this.#params.extraTrees ?? 0,
      this.#params.seed ?? 42,
      task,
      criterion,
      this.#params.sampleRate ?? 1.0,
      this.#params.heterogeneous ?? 0,
      this.#params.oobWeighting ?? 0,
      this.#params.alphaTrim ?? 0.0,
      this.#params.leafModel ?? 0
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nClasses = wasm._wl_rf_get_n_classes(modelPtr)

    this.#registerLeak()
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_rf_predict(this.#handle, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows))
    wasm._free(outPtr)
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#taskEnum() !== 0) {
      throw new Error('predictProba is only available for classification')
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nClasses = this.#nClasses

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * nClasses * 8)

    const ret = wasm._wl_rf_predict_proba(this.#handle, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows * nClasses)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows * nClasses))
    wasm._free(outPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#taskEnum() === 1) {
      // R-squared (regression)
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    } else {
      // Accuracy (classification)
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
      }
      return correct / preds.length
    }
  }

  // --- Feature importances ---

  featureImportances() {
    this.#ensureFitted()
    const wasm = getWasm()
    const n = this.#nFeatures
    const result = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      result[i] = wasm._wl_rf_get_feature_importance(this.#handle, i)
    }
    return result
  }

  oobScore() {
    this.#ensureFitted()
    const wasm = getWasm()
    return wasm._wl_rf_get_oob_score(this.#handle)
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#saveRaw()
    const task = this.#params.task || 'classification'
    const typeId = task === 'regression'
      ? 'wlearn.rf.regressor@1'
      : 'wlearn.rf.classifier@1'

    const metadata = {
      nFeatures: this.#nFeatures,
      nClasses: this.#nClasses
    }

    return encodeBundle(
      { typeId, params: this.getParams(), metadata },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return RFModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadRF()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    // Copy blob to WASM heap
    const bufPtr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, bufPtr)
    const modelPtr = wasm._wl_rf_load(bufPtr, raw.length)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    const params = manifest.params || {}
    const metadata = manifest.metadata || {}
    const nFeatures = metadata.nFeatures || wasm._wl_rf_get_n_features(modelPtr)
    const nClasses = metadata.nClasses || wasm._wl_rf_get_n_classes(modelPtr)

    return new RFModel(LOAD_SENTINEL, modelPtr, {
      params, nFeatures, nClasses
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_rf_free(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      nEstimators: { type: 'int_uniform', low: 50, high: 500 },
      maxDepth: { type: 'int_uniform', low: 3, high: 30 },
      maxFeatures: { type: 'categorical', values: ['sqrt', 'log2', 'third'] },
      minSamplesSplit: { type: 'int_uniform', low: 2, high: 20 },
      minSamplesLeaf: { type: 'int_uniform', low: 1, high: 10 },
      extraTrees: { type: 'categorical', values: [0, 1] },
      criterion: { type: 'categorical', values: ['gini', 'entropy'] },
      sampleRate: { type: 'uniform', low: 0.5, high: 1.0 },
      heterogeneous: { type: 'categorical', values: [0, 1] },
      oobWeighting: { type: 'categorical', values: [0, 1] },
      alphaTrim: { type: 'uniform', low: 0.0, high: 0.1 },
      leafModel: { type: 'categorical', values: [0, 1] }
    }
  }

  // --- Inspection ---

  get nFeatures() {
    return this.#nFeatures
  }

  get nClasses() {
    return this.#nClasses
  }

  get nTrees() {
    if (!this.#handle || this.#freed) return 0
    const wasm = getWasm()
    return wasm._wl_rf_get_n_trees(this.#handle)
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const isClassifier = this.#taskEnum() === 0
    return {
      classifier: isClassifier,
      regressor: !isClassifier,
      predictProba: isClassifier,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
      featureImportances: true
    }
  }

  // --- Private helpers ---

  #taskEnum() {
    return (this.#params.task || 'classification') === 'regression' ? 1 : 0
  }

  #normalizeX(X) {
    return normalizeX(X, 'auto')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('RFModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('RFModel is not fitted. Call fit() first.')
  }

  #registerLeak() {
    this.#ptrRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_rf_free(h)
      }, this)
    }
  }

  #saveRaw() {
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_rf_save(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_rf_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.rf.classifier@1', (m, t, b) => RFModel._fromBundle(m, t, b))
register('wlearn.rf.regressor@1', (m, t, b) => RFModel._fromBundle(m, t, b))

module.exports = { RFModel }
