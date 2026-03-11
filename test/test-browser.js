#!/usr/bin/env node
// Browser smoke test for IIFE + ESM bundles
// Generic: auto-discovers package name and exports from package.json + src/index.js

const { chromium } = require('playwright')
const path = require('path')
const http = require('http')
const fs = require('fs')

const ROOT = path.resolve(__dirname, '..')
const pkg = require(path.join(ROOT, 'package.json'))
const NAME = pkg.name.split('/').pop()
const EXPORTS = Object.keys(require(path.join(ROOT, 'src', 'index.js')))

const bundles = [
  { name: 'IIFE', file: `dist/${NAME}.js`,  type: 'iife', global: NAME },
  { name: 'ESM',  file: `dist/${NAME}.mjs`, type: 'esm' },
]

function makeIifeHtml(jsPath, globalName, exportKeys) {
  return `<!DOCTYPE html><html><body>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    var lib = ${globalName}
    var expected = ${JSON.stringify(exportKeys)}
    var missing = expected.filter(function(k) { return !(k in lib) })
    if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
    var types = {}
    expected.forEach(function(k) { types[k] = typeof lib[k] })
    return { ok: true, exports: expected.length, types: types }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

function makeFunctionalHtml(jsPath, globalName) {
  return `<!DOCTYPE html><html><body>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    var lib = ${globalName}
    var results = []

    // --- Classification with NaN ---
    var clf = await lib.RFModel.create({ nEstimators: 10, seed: 42 })
    var X = [[1,2],[3,4],[5,6],[7,8],[2,3],[4,5],[6,7],[8,9],
             [1.5,NaN],[NaN,5]]
    var y = [0,1,0,1,0,1,0,1,0,1]
    clf.fit(X, y)
    var preds = clf.predict([[NaN, 3], [4, NaN]])
    results.push({ name: 'NaN handling', ok: preds.length === 2 })
    clf.dispose()

    // --- Monotonic constraints ---
    var X2 = [], y2 = []
    for (var i = 0; i < 100; i++) {
      var x = i / 10.0
      X2.push([x, Math.random()])
      y2.push(x + Math.random() * 0.5)
    }
    var reg = await lib.RFModel.create({
      task: 'regression', nEstimators: 50, seed: 42,
      monotonicCst: [1, 0]
    })
    reg.fit(X2, y2)
    var grid = []
    for (var i = 0; i < 20; i++) grid.push([i * 0.5, 0.5])
    var mono_preds = reg.predict(grid)
    var mono_ok = true
    for (var i = 1; i < mono_preds.length; i++) {
      if (mono_preds[i] < mono_preds[i-1] - 1e-10) mono_ok = false
    }
    results.push({ name: 'monotonic constraints', ok: mono_ok })
    reg.dispose()

    // --- Quantile prediction ---
    var qrf = await lib.RFModel.create({
      task: 'regression', nEstimators: 50, seed: 42
    })
    qrf.fit(X2, y2)
    var q50 = qrf.predictQuantile(grid, 0.5)
    results.push({ name: 'predictQuantile single', ok: q50.length === 20 })
    var qmulti = qrf.predictQuantile(grid, [0.1, 0.9])
    results.push({ name: 'predictQuantile multi', ok: Array.isArray(qmulti) && qmulti.length === 2 })
    // q10 <= q90
    var q_order_ok = true
    for (var i = 0; i < 20; i++) {
      if (qmulti[0][i] > qmulti[1][i] + 1e-10) q_order_ok = false
    }
    results.push({ name: 'quantile ordering', ok: q_order_ok })

    // --- Conformal prediction intervals ---
    var iv = qrf.predictInterval(grid, 0.1)
    results.push({ name: 'predictInterval shape', ok: iv.lower.length === 20 && iv.upper.length === 20 })
    var iv_ok = true
    for (var i = 0; i < 20; i++) {
      if (iv.lower[i] > iv.upper[i] + 1e-10) iv_ok = false
    }
    results.push({ name: 'interval lower <= upper', ok: iv_ok })
    qrf.dispose()

    var allOk = results.every(function(r) { return r.ok })
    var summary = results.map(function(r) { return (r.ok ? 'PASS' : 'FAIL') + ': ' + r.name }).join('\\n')
    return { ok: allOk, results: results, summary: summary, count: results.length }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

function makeEsmHtml(jsPath, exportKeys) {
  const imports = exportKeys.join(', ')
  return `<!DOCTYPE html><html><body>
<script type="module">
import { ${imports} } from '${jsPath}'
async function runTest() {
  try {
    var types = {}
    var exports = [${exportKeys.map(k => `['${k}', ${k}]`).join(', ')}]
    exports.forEach(function(e) { types[e[0]] = typeof e[1] })
    return { ok: true, exports: ${exportKeys.length}, types: types }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

async function main() {
  const server = http.createServer((req, res) => {
    const fp = path.join(ROOT, decodeURIComponent(req.url.slice(1)))
    if (!fs.existsSync(fp)) { res.writeHead(404); res.end('Not found: ' + req.url); return }
    const ext = path.extname(fp)
    const ct = ext === '.html' ? 'text/html' : 'application/javascript'
    res.writeHead(200, { 'Content-Type': ct })
    res.end(fs.readFileSync(fp))
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  const port = server.address().port
  const base = `http://127.0.0.1:${port}`

  const browser = await chromium.launch({ headless: true })
  let passed = 0, failed = 0

  for (const b of bundles) {
    const htmlName = `_test_${b.name}.html`
    const htmlPath = path.join(ROOT, 'dist', htmlName)
    const jsUrl = '/' + b.file

    if (b.type === 'iife') {
      fs.writeFileSync(htmlPath, makeIifeHtml(jsUrl, b.global, EXPORTS))
    } else {
      fs.writeFileSync(htmlPath, makeEsmHtml(jsUrl, EXPORTS))
    }

    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(e.message))

    try {
      await page.goto(`${base}/dist/${htmlName}`, { timeout: 30000 })
      await page.waitForFunction(() => window.__testResult, { timeout: 30000 })
      const result = await page.evaluate(() => window.__testResult)

      if (result && result.ok) {
        console.log(`  PASS: ${b.name} -- ${result.exports} exports`)
        passed++
      } else {
        console.log(`  FAIL: ${b.name} -- ${result ? result.error : 'no result'}`)
        if (result && result.stack) console.log(`        ${result.stack.split('\n')[1]}`)
        failed++
      }
    } catch (e) {
      console.log(`  FAIL: ${b.name} -- ${e.message}`)
      if (errors.length) console.log(`        page errors: ${errors.join('; ')}`)
      failed++
    }

    await page.close()
    fs.unlinkSync(htmlPath)
  }

  // Functional test: fit + predict + v0.3 features in browser
  {
    const htmlName = '_test_functional.html'
    const htmlPath = path.join(ROOT, 'dist', htmlName)
    const jsUrl = '/' + bundles[0].file  // use IIFE bundle
    fs.writeFileSync(htmlPath, makeFunctionalHtml(jsUrl, bundles[0].global))

    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(e.message))

    try {
      await page.goto(`${base}/dist/${htmlName}`, { timeout: 60000 })
      await page.waitForFunction(() => window.__testResult, { timeout: 60000 })
      const result = await page.evaluate(() => window.__testResult)

      if (result && result.ok) {
        for (const r of result.results) {
          console.log(`  PASS: browser ${r.name}`)
          passed++
        }
      } else if (result && result.results) {
        for (const r of result.results) {
          if (r.ok) {
            console.log(`  PASS: browser ${r.name}`)
            passed++
          } else {
            console.log(`  FAIL: browser ${r.name}`)
            failed++
          }
        }
      } else {
        console.log(`  FAIL: browser functional -- ${result ? result.error : 'no result'}`)
        if (result && result.stack) console.log(`        ${result.stack}`)
        failed++
      }
    } catch (e) {
      console.log(`  FAIL: browser functional -- ${e.message}`)
      if (errors.length) console.log(`        page errors: ${errors.join('; ')}`)
      failed++
    }

    await page.close()
    fs.unlinkSync(htmlPath)
  }

  await browser.close()
  server.close()

  console.log(`\n=== ${passed} passed, ${failed} failed ===`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(e => { console.error(e); process.exit(1) })
