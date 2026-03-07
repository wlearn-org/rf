// WASM loader -- loads the RF WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadRF(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    // Emscripten output is CJS, use createRequire for ESM compatibility
    const require = createRequire(import.meta.url)
    const createRF = require('../wasm/rf.cjs')
    wasmModule = await createRF(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadRF() first')
  return wasmModule
}
