let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// --- Deterministic LCG PRNG (matches @wlearn/core rng.js) ---
function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeClassificationData(rng, nSamples, nFeatures, nClasses = 2) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % nClasses
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

async function main() {

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadRF } = require('../src/wasm.js')
const wasm = await loadRF()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_rf_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// RFModel basics
// ============================================================
console.log('\n=== RFModel ===')

const { RFModel } = require('../src/model.js')

await test('create() returns model', async () => {
  const model = await RFModel.create({ nEstimators: 10 })
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

// ============================================================
// Classification
// ============================================================
console.log('\n=== Classification ===')

await test('Binary classification', async () => {
  const rng = makeLCG(100)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 100
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nFeatures === 2, `nFeatures: ${model.nFeatures}`)
  assert(model.nClasses === 2, `nClasses: ${model.nClasses}`)
  assert(model.nTrees === 20, `nTrees: ${model.nTrees}`)

  const preds = model.predict(X)
  assert(preds.length === 80, `preds length: ${preds.length}`)

  const accuracy = model.score(X, y)
  assert(accuracy > 0.8, `accuracy too low: ${accuracy}`)

  model.dispose()
})

await test('Multiclass classification', async () => {
  const rng = makeLCG(200)
  const { X, y } = makeClassificationData(rng, 90, 3, 3)
  const model = await RFModel.create({
    nEstimators: 30, task: 'classification', seed: 200
  })
  model.fit(X, y)

  assert(model.nClasses === 3, `nClasses: ${model.nClasses}`)

  const accuracy = model.score(X, y)
  assert(accuracy > 0.7, `accuracy too low: ${accuracy}`)

  model.dispose()
})

await test('predictProba returns valid probabilities', async () => {
  const rng = makeLCG(300)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 300
  })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba.length === 60 * 2, `proba length: ${proba.length}, expected ${60 * 2}`)

  // Each row should sum to 1
  for (let i = 0; i < 60; i++) {
    const rowSum = proba[i * 2] + proba[i * 2 + 1]
    assertClose(rowSum, 1.0, 1e-10, `row ${i} sum: ${rowSum}`)
  }

  // Probabilities should be in [0, 1]
  for (let i = 0; i < proba.length; i++) {
    assert(proba[i] >= 0 && proba[i] <= 1, `proba[${i}] = ${proba[i]} out of range`)
  }

  model.dispose()
})

await test('predictProba fails for regression', async () => {
  const rng = makeLCG(301)
  const { X, y } = makeRegressionData(rng, 30, 2)
  const model = await RFModel.create({
    nEstimators: 10, task: 'regression', seed: 301
  })
  model.fit(X, y)

  let threw = false
  try {
    model.predictProba(X)
  } catch (e) {
    threw = true
    assert(e.message.includes('classification'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw for regression')
  model.dispose()
})

await test('ExtraTrees classification', async () => {
  const rng = makeLCG(302)
  const { X, y } = makeClassificationData(rng, 80, 3)
  const model = await RFModel.create({
    nEstimators: 30, task: 'classification', seed: 302, extraTrees: 1
  })
  model.fit(X, y)

  const accuracy = model.score(X, y)
  assert(accuracy > 0.7, `ExtraTrees accuracy too low: ${accuracy}`)

  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('Regression basic', async () => {
  const rng = makeLCG(400)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'regression', seed: 400
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nClasses === 0, 'regressor should have 0 classes')

  const preds = model.predict(X)
  assert(preds.length === 80, `preds length: ${preds.length}`)

  const r2 = model.score(X, y)
  assert(r2 > 0.3, `R2 too low: ${r2}`)

  model.dispose()
})

await test('ExtraTrees regression', async () => {
  const rng = makeLCG(401)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 30, task: 'regression', seed: 401, extraTrees: 1
  })
  model.fit(X, y)

  const r2 = model.score(X, y)
  assert(r2 > 0.3, `ExtraTrees R2 too low: ${r2}`)

  model.dispose()
})

await test('Single-feature regression', async () => {
  const rng = makeLCG(402)
  const { X, y } = makeRegressionData(rng, 60, 1)
  const model = await RFModel.create({
    nEstimators: 20, task: 'regression', seed: 402
  })
  model.fit(X, y)

  const r2 = model.score(X, y)
  assert(r2 > 0, `R2 should be positive: ${r2}`)

  model.dispose()
})

// ============================================================
// Feature importances + OOB
// ============================================================
console.log('\n=== Feature importances + OOB ===')

await test('Feature importances sum to 1', async () => {
  const rng = makeLCG(500)
  const { X, y } = makeClassificationData(rng, 80, 3)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 500, maxFeatures: 3
  })
  model.fit(X, y)

  const imp = model.featureImportances()
  assert(imp.length === 3, `importances length: ${imp.length}`)

  let sum = 0
  for (let i = 0; i < imp.length; i++) sum += imp[i]
  assertClose(sum, 1.0, 1e-10, `importances sum: ${sum}`)

  model.dispose()
})

await test('Dominant feature has highest importance', async () => {
  // Regression: y = 5*x0 + 0*x1 + 0*x2 + 0*x3
  // Feature 0 fully determines target, others are noise
  // Use regression for cleaner MDI signal (MSE splits)
  const rng = makeLCG(501)
  const n = 200, d = 4
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < d; j++) row.push(rng())
    X.push(row)
    y.push(5.0 * row[0])
  }
  const model = await RFModel.create({
    nEstimators: 100, task: 'regression', seed: 501, maxFeatures: d
  })
  model.fit(X, y)

  const imp = model.featureImportances()
  for (let j = 1; j < d; j++) {
    assert(imp[0] > imp[j], `feature 0 (${imp[0]}) should be > feature ${j} (${imp[j]})`)
  }
  assert(imp[0] > imp[2], `feature 0 (${imp[0]}) should be > feature 2 (${imp[2]})`)

  model.dispose()
})

await test('OOB score (classification) is reasonable', async () => {
  const rng = makeLCG(502)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 30, task: 'classification', seed: 502
  })
  model.fit(X, y)

  const oob = model.oobScore()
  assert(!isNaN(oob), 'OOB should not be NaN')
  assert(oob > 0.5, `OOB too low: ${oob}`)

  model.dispose()
})

await test('OOB score (regression) is reasonable', async () => {
  const rng = makeLCG(503)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 30, task: 'regression', seed: 503
  })
  model.fit(X, y)

  const oob = model.oobScore()
  assert(!isNaN(oob), 'OOB should not be NaN')

  model.dispose()
})

// ============================================================
// Edge cases
// ============================================================
console.log('\n=== Edge cases ===')

await test('max_depth=1 (stumps)', async () => {
  const rng = makeLCG(600)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await RFModel.create({
    nEstimators: 50, task: 'classification', seed: 600, maxDepth: 1
  })
  model.fit(X, y)

  const preds = model.predict(X)
  assert(preds.length === 60, 'should predict')

  model.dispose()
})

await test('setParams / getParams round-trip', async () => {
  const model = await RFModel.create({ nEstimators: 10, maxDepth: 5 })
  const params = model.getParams()
  assert(params.nEstimators === 10, 'nEstimators should be 10')
  assert(params.maxDepth === 5, 'maxDepth should be 5')

  model.setParams({ nEstimators: 50 })
  assert(model.getParams().nEstimators === 50, 'nEstimators should be updated')

  model.dispose()
})

await test('capabilities reflect task', async () => {
  const clf = await RFModel.create({ task: 'classification' })
  assert(clf.capabilities.classifier === true, 'classifier')
  assert(clf.capabilities.predictProba === true, 'predictProba')
  assert(clf.capabilities.featureImportances === true, 'featureImportances')
  clf.dispose()

  const reg = await RFModel.create({ task: 'regression' })
  assert(reg.capabilities.regressor === true, 'regressor')
  assert(reg.capabilities.predictProba === false, 'no predictProba for regression')
  assert(reg.capabilities.featureImportances === true, 'featureImportances')
  reg.dispose()
})

await test('Deterministic: same seed -> same predictions', async () => {
  const rng1 = makeLCG(700)
  const { X, y } = makeClassificationData(rng1, 60, 2)

  const m1 = await RFModel.create({
    nEstimators: 15, task: 'classification', seed: 700
  })
  m1.fit(X, y)
  const p1 = m1.predict(X)

  const m2 = await RFModel.create({
    nEstimators: 15, task: 'classification', seed: 700
  })
  m2.fit(X, y)
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred mismatch at ${i}: ${p1[i]} vs ${p2[i]}`)
  }

  m1.dispose()
  m2.dispose()
})

// ============================================================
// Persistence (save / load)
// ============================================================
console.log('\n=== Persistence ===')

await test('Classification save/load round-trip', async () => {
  const rng = makeLCG(800)
  const { X, y } = makeClassificationData(rng, 50, 2)
  const model = await RFModel.create({
    nEstimators: 15, task: 'classification', seed: 800
  })
  model.fit(X, y)

  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  assert(bundle instanceof Uint8Array, 'save should return Uint8Array')
  assert(bundle.length > 0, 'bundle should not be empty')

  const model2 = await RFModel.load(bundle)
  assert(model2.isFitted, 'loaded model should be fitted')
  assert(model2.nFeatures === 2, `nFeatures: ${model2.nFeatures}`)
  assert(model2.nClasses === 2, `nClasses: ${model2.nClasses}`)
  assert(model2.nTrees === 15, `nTrees: ${model2.nTrees}`)

  const preds2 = model2.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `pred mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }

  model2.dispose()
})

await test('Regression save/load round-trip', async () => {
  const rng = makeLCG(801)
  const { X, y } = makeRegressionData(rng, 50, 2)
  const model = await RFModel.create({
    nEstimators: 15, task: 'regression', seed: 801
  })
  model.fit(X, y)

  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const model2 = await RFModel.load(bundle)
  const preds2 = model2.predict(X)

  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `pred mismatch at ${i}`)
  }

  model2.dispose()
})

await test('Bundle blob has RF01 header', async () => {
  const { decodeBundle } = require('@wlearn/core')

  const rng = makeLCG(802)
  const { X, y } = makeClassificationData(rng, 20, 2)
  const model = await RFModel.create({
    nEstimators: 10, task: 'classification', seed: 802
  })
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()

  const { manifest, toc, blobs } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.rf.classifier@1', `typeId: ${manifest.typeId}`)
  assert(manifest.metadata.nFeatures === 2, `nFeatures: ${manifest.metadata.nFeatures}`)
  assert(manifest.metadata.nClasses === 2, `nClasses: ${manifest.metadata.nClasses}`)

  const entry = toc.find(e => e.id === 'model')
  assert(entry, 'should have model artifact')

  const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
  const magic = String.fromCharCode(blob[0], blob[1], blob[2], blob[3])
  assert(magic === 'RF01', `magic: ${magic}`)
})

await test('Feature importances preserved after load', async () => {
  const rng = makeLCG(803)
  const n = 80
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x0 = rng()
    X.push([x0, rng(), rng()])
    y.push(x0 > 0.5 ? 1 : 0)
  }
  const model = await RFModel.create({
    nEstimators: 30, task: 'classification', seed: 803, maxFeatures: 3
  })
  model.fit(X, y)

  const imp1 = model.featureImportances()
  const bundle = model.save()
  model.dispose()

  const model2 = await RFModel.load(bundle)
  const imp2 = model2.featureImportances()

  for (let i = 0; i < imp1.length; i++) {
    assertClose(imp1[i], imp2[i], 1e-10, `importance mismatch at ${i}`)
  }

  model2.dispose()
})

await test('Regressor bundle typeId', async () => {
  const { decodeBundle } = require('@wlearn/core')

  const rng = makeLCG(804)
  const { X, y } = makeRegressionData(rng, 20, 2)
  const model = await RFModel.create({
    nEstimators: 10, task: 'regression', seed: 804
  })
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()

  const { manifest } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.rf.regressor@1', `typeId: ${manifest.typeId}`)
})

// ============================================================
// Disposal & error handling
// ============================================================
console.log('\n=== Disposal ===')

await test('dispose() prevents further use', async () => {
  const rng = makeLCG(900)
  const { X, y } = makeClassificationData(rng, 20, 2)
  const model = await RFModel.create({
    nEstimators: 10, task: 'classification', seed: 900
  })
  model.fit(X, y)
  model.dispose()

  assert(!model.isFitted, 'should not be fitted after dispose')

  let threw = false
  try { model.predict(X) } catch (e) {
    threw = true
    assert(e.message.includes('disposed'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw after dispose')
})

await test('double dispose is safe', async () => {
  const model = await RFModel.create({ nEstimators: 10 })
  model.dispose()
  model.dispose() // should not throw
})

await test('predict before fit throws NotFittedError', async () => {
  const model = await RFModel.create({ task: 'classification' })
  let threw = false
  try { model.predict([[1, 2]]) } catch (e) {
    threw = true
    assert(e.message.includes('not fitted'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw')
  model.dispose()
})

// ============================================================
// defaultSearchSpace
// ============================================================
console.log('\n=== Search space ===')

await test('defaultSearchSpace returns valid IR', async () => {
  const space = RFModel.defaultSearchSpace()
  assert(space.nEstimators, 'should have nEstimators')
  assert(space.nEstimators.type === 'int_uniform', 'nEstimators type')
  assert(space.maxDepth, 'should have maxDepth')
  assert(space.maxFeatures, 'should have maxFeatures')
  assert(space.maxFeatures.type === 'categorical', 'maxFeatures type')
  assert(space.extraTrees, 'should have extraTrees')
})

// ============================================================
// New v0.2 features
// ============================================================
console.log('\n=== Entropy Criterion ===')

await test('Entropy classification', async () => {
  const rng = makeLCG(1001)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1001, criterion: 'entropy'
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.8, `entropy accuracy too low: ${acc}`)
  model.dispose()
})

await test('Hellinger classification', async () => {
  const rng = makeLCG(1002)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1002, criterion: 'hellinger'
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.8, `hellinger accuracy too low: ${acc}`)
  model.dispose()
})

console.log('\n=== MAE Criterion ===')

await test('MAE regression', async () => {
  const rng = makeLCG(1003)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'regression', seed: 1003, criterion: 'mae'
  })
  model.fit(X, y)
  const r2 = model.score(X, y)
  assert(r2 > 0.3, `MAE R2 too low: ${r2}`)
  model.dispose()
})

console.log('\n=== Sample Rate ===')

await test('Sample rate 0.5', async () => {
  const rng = makeLCG(1004)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1004, sampleRate: 0.5
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.8, `sample_rate=0.5 accuracy too low: ${acc}`)
  model.dispose()
})

await test('Sample rate 0.5 no bootstrap', async () => {
  const rng = makeLCG(1005)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1005,
    sampleRate: 0.5, bootstrap: 0
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.8, `sample_rate no-bootstrap accuracy too low: ${acc}`)
  model.dispose()
})

console.log('\n=== Heterogeneous RF ===')

await test('HRF regression', async () => {
  const rng = makeLCG(1006)
  const { X, y } = makeRegressionData(rng, 150, 4)
  const model = await RFModel.create({
    nEstimators: 30, task: 'regression', seed: 1006,
    heterogeneous: 1, maxFeatures: 4
  })
  model.fit(X, y)
  const r2 = model.score(X, y)
  assert(r2 > 0.3, `HRF R2 too low: ${r2}`)
  model.dispose()
})

console.log('\n=== OOB Weighting ===')

await test('OOB weighted voting', async () => {
  const rng = makeLCG(1007)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1007, oobWeighting: 1
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.8, `oob weighted accuracy too low: ${acc}`)

  // Save/load round-trip
  const bundle = model.save()
  model.dispose()
  const m2 = await RFModel.load(bundle)
  assert(m2.isFitted, 'loaded model should be fitted')
  const p2 = m2.predict(X)
  assert(p2.length === 80, 'loaded prediction length')
  m2.dispose()
})

console.log('\n=== Alpha Trim ===')

await test('Cost-complexity pruning', async () => {
  const rng = makeLCG(1008)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await RFModel.create({
    nEstimators: 20, task: 'classification', seed: 1008, alphaTrim: 0.05
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.7, `alpha_trim accuracy too low: ${acc}`)
  model.dispose()
})

console.log('\n=== Local Linear Leaves ===')

await test('Linear leaves improve on linear signal', async () => {
  const rng1 = makeLCG(1009)
  const n = 200, d = 2
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x0 = rng1() * 4 - 2
    const x1 = rng1() * 4 - 2
    X.push([x0, x1])
    y.push(3.0 * x0 + 2.0 * x1 + 1.0)
  }

  // Constant leaves (shallow)
  const mc = await RFModel.create({
    nEstimators: 20, task: 'regression', seed: 1009, maxDepth: 3, leafModel: 0
  })
  mc.fit(X, y)
  const r2c = mc.score(X, y)

  // Linear leaves (shallow)
  const ml = await RFModel.create({
    nEstimators: 20, task: 'regression', seed: 1009, maxDepth: 3, leafModel: 1
  })
  ml.fit(X, y)
  const r2l = ml.score(X, y)

  assert(r2l > r2c, `linear R2 (${r2l}) should exceed constant R2 (${r2c})`)
  assert(r2l > 0.95, `linear R2 too low: ${r2l}`)

  // Save/load
  const bundle = ml.save()
  const ml2 = await RFModel.load(bundle)
  const p1 = ml.predict(X)
  const p2 = ml2.predict(X)
  for (let i = 0; i < n; i++) {
    assertClose(p1[i], p2[i], 1e-10, `linear leaf pred mismatch at ${i}`)
  }

  mc.dispose()
  ml.dispose()
  ml2.dispose()
})

console.log('\n=== Defaults Unchanged ===')

await test('Explicit defaults match implicit defaults', async () => {
  const rng = makeLCG(100)
  const { X, y } = makeClassificationData(rng, 80, 2)

  const m1 = await RFModel.create({ nEstimators: 20, seed: 100 })
  m1.fit(X, y)
  const p1 = m1.predict(X)

  const m2 = await RFModel.create({
    nEstimators: 20, seed: 100,
    criterion: 0, sampleRate: 1.0, heterogeneous: 0,
    oobWeighting: 0, alphaTrim: 0.0, leafModel: 0
  })
  m2.fit(X, y)
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `defaults mismatch at ${i}: ${p1[i]} vs ${p2[i]}`)
  }

  m1.dispose()
  m2.dispose()
})

// ============================================================
// Reproducibility (JS matches C)
// ============================================================
console.log('\n=== Reproducibility ===')

await test('Same seed produces identical models across create() calls', async () => {
  const rng1 = makeLCG(950)
  const { X, y } = makeRegressionData(rng1, 40, 2)

  const m1 = await RFModel.create({
    nEstimators: 10, task: 'regression', seed: 950
  })
  m1.fit(X, y)

  const m2 = await RFModel.create({
    nEstimators: 10, task: 'regression', seed: 950
  })
  m2.fit(X, y)

  const p1 = m1.predict(X)
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-10, `regression pred mismatch at ${i}`)
  }

  m1.dispose()
  m2.dispose()
})

// ============================================================
// Sample weight tests
// ============================================================

await test('sample weight -- weighted classification', async () => {
  const rng = makeLCG(900)
  const X = [], y = []
  // 90 class-0, 10 class-1
  for (let i = 0; i < 90; i++) {
    X.push([rng() * 2, rng() * 2])
    y.push(0)
  }
  for (let i = 0; i < 10; i++) {
    X.push([3 + rng() * 2, 3 + rng() * 2])
    y.push(1)
  }

  // Without weights: should work fine
  const m1 = await RFModel.create({ task: 'classification', nEstimators: 50, seed: 42 })
  m1.fit(X, y)
  const acc1 = m1.score(X, y)
  assert(acc1 > 0.9, `unweighted acc ${acc1} should be > 0.9`)

  // With 4x weight on minority class
  const sw = new Float64Array(100)
  for (let i = 0; i < 90; i++) sw[i] = 1.0
  for (let i = 90; i < 100; i++) sw[i] = 4.0

  const m2 = await RFModel.create({
    task: 'classification', nEstimators: 50, seed: 42, sampleWeight: sw
  })
  m2.fit(X, y)
  const acc2 = m2.score(X, y)
  assert(acc2 > 0.9, `weighted acc ${acc2} should be > 0.9`)

  m1.dispose()
  m2.dispose()
})

await test('sample weight -- classWeight balanced', async () => {
  const rng = makeLCG(901)
  const X = [], y = []
  for (let i = 0; i < 90; i++) {
    X.push([rng() * 2, rng() * 2])
    y.push(0)
  }
  for (let i = 0; i < 10; i++) {
    X.push([3 + rng() * 2, 3 + rng() * 2])
    y.push(1)
  }

  const m = await RFModel.create({
    task: 'classification', nEstimators: 50, seed: 42, classWeight: 'balanced'
  })
  m.fit(X, y)
  const acc = m.score(X, y)
  assert(acc > 0.9, `balanced acc ${acc} should be > 0.9`)
  m.dispose()
})

await test('sample weight -- uniform weights match no weights', async () => {
  const rng = makeLCG(902)
  const { X, y } = makeRegressionData(rng, 60, 3)

  const m1 = await RFModel.create({ task: 'regression', nEstimators: 20, seed: 77 })
  m1.fit(X, y)
  const p1 = m1.predict(X)

  const sw = new Float64Array(60).fill(1.0)
  const m2 = await RFModel.create({
    task: 'regression', nEstimators: 20, seed: 77, sampleWeight: sw
  })
  m2.fit(X, y)
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-10, `pred mismatch at ${i}`)
  }
  m1.dispose()
  m2.dispose()
})

await test('sample weight -- zero weight garbage ignored', async () => {
  const rng = makeLCG(903)
  const X = [], y = []
  // 50 clean samples: y = 2*x[0] + x[1]
  for (let i = 0; i < 50; i++) {
    const x0 = rng() * 4, x1 = rng() * 4
    X.push([x0, x1])
    y.push(2 * x0 + x1)
  }
  // 50 garbage with zero weight
  for (let i = 0; i < 50; i++) {
    X.push([rng() * 4, rng() * 4])
    y.push(999.0)
  }

  const sw = new Float64Array(100)
  for (let i = 0; i < 50; i++) sw[i] = 1.0
  // sw[50..99] = 0.0

  const m = await RFModel.create({
    task: 'regression', nEstimators: 50, seed: 42, sampleWeight: sw
  })
  m.fit(X, y)

  // Predict on clean samples
  const Xtest = X.slice(0, 50)
  const ytest = y.slice(0, 50)
  const preds = m.predict(Xtest)

  let ssRes = 0, ssTot = 0, yMean = 0
  for (let i = 0; i < 50; i++) yMean += ytest[i]
  yMean /= 50
  for (let i = 0; i < 50; i++) {
    ssRes += (ytest[i] - preds[i]) ** 2
    ssTot += (ytest[i] - yMean) ** 2
  }
  const r2 = 1 - ssRes / ssTot
  assert(r2 > 0.8, `zero-weight r2 ${r2} should be > 0.8 (garbage ignored)`)
  m.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)

}

main()
