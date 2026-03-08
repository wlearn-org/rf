// WASM loader -- loads the RF WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadRF(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createRF = require('../wasm/rf.js')
    wasmModule = await createRF(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadRF() first')
  return wasmModule
}

module.exports = { loadRF, getWasm }
